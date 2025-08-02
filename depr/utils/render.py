import torch
import torch.nn as nn


class LaplaceDensity(nn.Module):
    def __init__(self, beta: float = 0.1, beta_min: float = 0.0001):
        super().__init__()
        self.register_buffer("beta_min", torch.tensor(beta_min))
        self.register_buffer("beta", torch.tensor(beta))

    def forward(self, sdf: torch.Tensor, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        return alpha * (0.5 + 0.5 * sdf.sign() * torch.expm1(-sdf.abs() / beta))

    def get_beta(self) -> torch.Tensor:
        beta = self.beta.abs() + self.beta_min
        return beta


class UniformSampler(nn.Module):
    def __init__(self, N_samples):
        super().__init__()
        self.N_samples = N_samples

    def near_far_from_cube(self, rays_o, rays_d, bound):
        tmin = (-bound - rays_o) / (rays_d + 1e-15)  # [B, N, 3]
        tmax = (bound - rays_o) / (rays_d + 1e-15)
        near = torch.where(tmin < tmax, tmin, tmax).max(dim=-1, keepdim=True)[0]
        far = torch.where(tmin > tmax, tmin, tmax).min(dim=-1, keepdim=True)[0]
        # if far < near, means no intersection, set both near and far to inf (1e9 here)
        mask = far < near
        near[mask] = 0
        far[mask] = 0
        # restrict near to a minimal value
        # near = torch.clamp(near, min=self.near).cuda()
        # far = torch.clamp(far, max=self.far).cuda()
        return near, far, mask

    def get_z_vals(self, ray_dirs, cam_loc, bound):
        near, far, mask = self.near_far_from_cube(cam_loc, ray_dirs, bound=bound)

        t_vals = torch.linspace(0.0, 1.0, steps=self.N_samples).to(ray_dirs.device)
        z_vals = near * (1.0 - t_vals) + far * (t_vals)

        return z_vals, mask


class Renderer(nn.Module):

    def __init__(
        self,
        sampler: UniformSampler,
        density: LaplaceDensity,
        height: int,
        width: int,
        device="cuda",
    ):
        super().__init__()
        self.sampler = sampler
        self.density = density
        self.height = height
        self.width = width
        self._device = device

    def render_depth(
        self,
        intrinsic: torch.Tensor,
        cam2obj: torch.Tensor,
        model: nn.Module,
        triplane: torch.Tensor,
        depth_scale: float,
        render_mask=None,
    ) -> torch.Tensor:
        ray_dirs = self.get_rays(intrinsic, cam2obj)
        num_pixels, _ = ray_dirs.shape
        cam_loc = cam2obj[:3, 3]
        cam_loc = cam_loc.unsqueeze(0).repeat(num_pixels, 1)
        z_vals, mask = self.sampler.get_z_vals(ray_dirs, cam_loc, bound=1)
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        mask = mask.view(-1)
        if render_mask is not None:
            mask = mask | (~render_mask.view(-1))  # mask out
        query_points = points[~mask].view(-1, 3)
        sdf_values = torch.zeros((num_pixels, self.sampler.N_samples)).to(self._device)

        query_sdf_values = model.sdf_model(
            triplane.unsqueeze(0), query_points.unsqueeze(0)
        ).squeeze(0)
        sdf_values[~mask] = query_sdf_values.view(-1, self.sampler.N_samples)
        sdf_values = sdf_values.view(-1)
        # return points.view(self.height, self.width, -1, 3), sdf_values.view(self.height, self.width, -1), ray_dirs.view(self.height, self.width, -1, 3), z_vals.view(self.height, self.width, -1)
        weights = self.volume_rendering(z_vals, sdf_values)
        depth_values = torch.sum(weights * z_vals, 1, keepdims=True) / (
            weights.sum(dim=1, keepdims=True) + 1e-8
        )

        depth_values[mask] = 0  # mask out
        # we should scale rendered distance to depth along z direction
        depth_values = depth_scale * depth_values
        depth_map = depth_values.view(self.height, self.width)
        return depth_map

    def get_rays(self, intrinsic: torch.Tensor, cam2obj: torch.Tensor) -> torch.Tensor:
        pos_pix = torch.meshgrid(
            torch.arange(self.width).to(self._device),  # 648
            torch.arange(self.height).to(self._device),  # 484
            indexing="ij",
        )
        pos_pix = (
            torch.stack(pos_pix, dim=-1).float().to(self._device)
        )  # (648 * 484, 2)
        pos_pix = pos_pix.view(self.width, self.height, 2).transpose(
            0, 1
        )  # (484, 648, 2)
        # (0, 0), (1, 0), ....,(648, 0)
        # (0, 1), (1, 1), ....,(648, 1)
        # ...
        # (0, 484), (1, 484), ....,(648, 484)
        pos_pix = pos_pix.reshape(-1, 2)
        pos_pix = torch.cat([pos_pix, torch.ones_like(pos_pix[:, :1])], dim=1)
        pos_cam = torch.matmul(pos_pix, intrinsic.inverse().transpose(0, 1).float())
        pos_cam = pos_cam / pos_cam[:, 2:]
        pos_cam = torch.cat([pos_cam, torch.ones_like(pos_cam[:, :1])], dim=1)
        pos_obj = torch.matmul(pos_cam, cam2obj.transpose(0, 1).float())
        pos_obj = pos_obj[:, :3] / pos_obj[:, 3:]
        ray_dirs = pos_obj - cam2obj[:3, 3]
        ray_dirs = ray_dirs / ray_dirs.norm(dim=-1, keepdim=True)
        return ray_dirs

    def volume_rendering(self, z_vals, sdf) -> torch.Tensor:
        density_flat = self.density(sdf)
        density = density_flat.reshape(
            -1, z_vals.shape[1]
        )  # (batch_size * num_pixels) x N_samples

        dists = z_vals[:, 1:] - z_vals[:, :-1]
        dists = torch.cat(
            [
                dists,
                torch.tensor([1e10])
                .to(self._device)
                .unsqueeze(0)
                .repeat(dists.shape[0], 1),
            ],
            -1,
        )

        # LOG SPACE
        free_energy = dists * density
        shifted_free_energy = torch.cat(
            [torch.zeros(dists.shape[0], 1).to(self._device), free_energy[:, :-1]],
            dim=-1,
        )  # shift one step
        alpha = 1 - torch.exp(-free_energy)  # probability of it is not empty here
        transmittance = torch.exp(
            -torch.cumsum(shifted_free_energy, dim=-1)
        )  # probability of everything is empty up to now
        weights = alpha * transmittance  # probability of the ray hits something here

        return weights
