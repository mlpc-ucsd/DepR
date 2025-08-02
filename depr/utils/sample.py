import torch
from tqdm import tqdm, trange
import numpy as np
from depr.utils import optim, geom
from .transforms import (
    to_image_plane,
)
from .render import Renderer, UniformSampler, LaplaceDensity


@torch.enable_grad()
def get_scene_transformations(
    source_list,
    target_list,
    intrinsics,
    num_repeats=20,
    num_steps=500,
    lr=5e-2,
    cutoff_steps=300,
    verbose=False,
    return_losses=False,
    loss_3d_weight=100,
    loss_2d_weight=1,
    dof=5,
    enable_global_rotation=True,
    use_open3d_registration=False,
    print_every=50,
    single_directional=False,
):
    print("Beginning scene registration...")
    device = "cuda"
    source_pts = torch.stack(source_list).to(device)
    target_pts = torch.stack(target_list).to(device)
    num_instances = source_pts.shape[0]
    intrinsics = intrinsics.to(device)
    target_pts_2d = to_image_plane(target_pts, intrinsics)

    source_pts, source_center, source_scale = geom.get_normalized_pcd_with_centroid(
        source_pts, return_extra=True
    )
    _, target_center, target_scale = geom.get_normalized_pcd_with_centroid(
        target_pts, return_extra=True
    )

    random_rotation = optim.random_rotation_matrix_y(num_repeats, num_instances, device)
    y_up_rotation = torch.diag(
        torch.as_tensor([-1, -1, 1], device=device, dtype=random_rotation.dtype)
    )
    if use_open3d_registration:
        pre_transforms = y_up_rotation
    else:
        pre_transforms = torch.matmul(y_up_rotation, random_rotation)

        estimator = optim.PoseEstimator(
            num_repeats=num_repeats,
            num_instances=num_instances,
            dof=dof,
            enable_global_rotation=enable_global_rotation,
        )
        estimator = estimator.to(device)
        optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)

        last_loss_3d = None

    pre_transformed_source_pts = torch.matmul(
        source_pts, pre_transforms.transpose(-1, -2)
    )
    pre_transformed_source_pts = (
        pre_transformed_source_pts * target_scale[:, None, None]
        + target_center[:, None, :]
    )

    if use_open3d_registration:
        optimized_transforms = optim.open3d_global_registration(
            pre_transformed_source_pts.cpu().numpy(),
            target_pts.cpu().numpy(),
            verbose=verbose,
        )
        optimized_transforms = optimized_transforms.to(device)
    else:
        pbar = trange if verbose else range
        for step in pbar(num_steps):
            optimizer.zero_grad()
            transformed_source_pts = estimator(pre_transformed_source_pts)
            transformed_source_pts_2d = to_image_plane(
                transformed_source_pts, intrinsics
            )
            loss_3d = estimator.get_cd_loss(
                transformed_source_pts,
                target_pts,
                single_directional=single_directional,
            )
            loss_2d = estimator.get_cd_loss(
                transformed_source_pts_2d,
                target_pts_2d,
                single_directional=single_directional,
            )
            if step < cutoff_steps:
                loss = loss_3d * loss_3d_weight
            else:
                loss = loss_3d * loss_3d_weight + loss_2d * loss_2d_weight

            last_loss_3d = loss_3d.detach()
            loss = loss.mean()

            if verbose and step % print_every == 0:
                mean_loss_3d = loss_3d.mean().item()
                mean_loss_2d = loss_2d.mean().item()
                mean_loss = loss.item()
                if enable_global_rotation:
                    best_loss = last_loss_3d.sum(dim=-1).min().item()
                else:
                    best_loss = last_loss_3d.min(dim=0).values.sum().item()
                print(
                    f"Step {step}: 3D Loss: {mean_loss_3d}, 2D Loss: {mean_loss_2d}, Total Loss: {mean_loss}, Best Loss: {best_loss}"
                )

            loss.backward()
            optimizer.step()

        optimized_transforms = estimator.get_transformation_matrix()

    normalize_transforms = torch.eye(4, device=device, dtype=source_pts.dtype)[
        None
    ].repeat(num_instances, 1, 1)
    normalize_transforms[..., :3, 3] = -source_center
    normalize_transforms[..., :3, :] /= source_scale[:, None, None]
    pre_transforms = optim.augment_transformation_matrix(pre_transforms)
    post_norm_transforms = torch.eye(4, device=device, dtype=source_pts.dtype)[
        None
    ].repeat(num_instances, 1, 1)
    post_norm_transforms[..., :3, :] *= target_scale[:, None, None]
    post_norm_transforms[..., :3, 3] = target_center
    all_transforms = torch.matmul(pre_transforms, normalize_transforms)
    all_transforms = torch.matmul(post_norm_transforms, all_transforms)
    all_transforms = torch.matmul(optimized_transforms, all_transforms)
    print("Scene registration completed.")
    if not use_open3d_registration:
        if enable_global_rotation:
            best_rep = torch.argmin(last_loss_3d.sum(dim=-1))
            best_transforms = all_transforms[best_rep].cpu()
            best_losses = last_loss_3d[best_rep].cpu()
        else:
            best_rep = torch.argmin(last_loss_3d, dim=0)
            best_transforms = all_transforms[
                best_rep, torch.arange(num_instances)
            ].cpu()
            best_losses = last_loss_3d[best_rep, torch.arange(num_instances)].cpu()
        if return_losses:
            return best_transforms, best_losses
    else:
        best_transforms = all_transforms.cpu()

    return best_transforms


@torch.enable_grad()
def get_object_transformations(
    source_list,
    target_list,
    num_repeats=20,
    num_steps=50,
    lr=5e-2,
    verbose=False,
    dof=7,
):
    device = "cuda"
    source_pts = torch.stack(source_list).to(device)
    target_pts = torch.stack(target_list).to(device)
    num_instances = len(source_list)
    estimator = optim.PoseEstimator(
        num_repeats=num_repeats,
        num_instances=num_instances,
        dof=dof,
    )
    estimator = estimator.to(device)
    optimizer = torch.optim.Adam(estimator.parameters(), lr=lr)

    last_loss = None

    random_rotation = optim.random_rotation_matrix_y(num_repeats, num_instances, device)
    transformed_source_pts = torch.matmul(source_pts, random_rotation.transpose(-1, -2))

    pbar = trange if verbose else range
    for _ in pbar(num_steps):
        optimizer.zero_grad()
        loss = estimator(transformed_source_pts, target_pts)
        last_loss = loss.detach()
        loss = loss.mean()
        loss.backward()
        optimizer.step()
        if verbose:
            print(f"Step {_}: Best loss: {last_loss.min().item():.4f}")

    random_rotation = optim.augment_transformation_matrix(random_rotation)
    all_transforms = estimator.get_transformation_matrix()
    all_transforms = torch.matmul(all_transforms, random_rotation)
    best_inds = torch.argmin(last_loss, dim=0)
    best_transforms = all_transforms[best_inds, torch.arange(num_instances)]
    return best_transforms.cpu()


@torch.enable_grad()
def guided_sampling(
    model,
    depth,
    obj2cam_matrices,
    triplane_feature,
    intrinsics,
    masks,
    alpha=10,
    depth_weight=1.0,
    timesteps=200,
    verbose=False,
    init_noise=None,
):
    segment_masks = []
    loss_list = []
    # guided_steps = []
    # original_steps = []
    # Downsample depth to 1 / 2
    depth = depth[::2, ::2]
    masks = masks[:, ::2, ::2]
    intrinsics[:2] /= 2
    device = depth.device

    renderer = Renderer(
        sampler=UniformSampler(32),
        density=LaplaceDensity(0.001),
        height=depth.shape[-2],
        width=depth.shape[-1],
        device=device,
    ).to(device)

    with torch.no_grad():
        for idx, segment_mask in enumerate(masks):
            segment_mask = segment_mask & (depth > 0)
            segment_masks.append(segment_mask)
        cam2obj_matrices = [torch.inverse(t) for t in obj2cam_matrices]

    assert len(triplane_feature) == len(masks) == len(cam2obj_matrices)

    if init_noise is None:
        pred_embeds = torch.randn((len(masks), 2, 32, 96)).to(device)
    else:
        pred_embeds = init_noise
    model.scheduler.set_timesteps(timesteps)
    if verbose:
        pbar = tqdm(model.scheduler.timesteps)
    else:
        pbar = model.scheduler.timesteps
    for t in pbar:
        input_embeds = torch.cat((pred_embeds, triplane_feature), dim=1)
        with torch.no_grad():
            model_output = model.unet(input_embeds, t).sample

        model_output = model_output.detach()
        model_output.requires_grad = True

        step_triplanes = model.triplane_vae.decode(
            model.denormalize_embeds(model_output)
        )
        step_triplanes = model.denormalize_triplanes(step_triplanes)

        losses = []
        for idx, segment_mask in enumerate(segment_masks):
            pred_depth = renderer.render_depth(
                intrinsics, cam2obj_matrices[idx], model, step_triplanes[idx], 1
            )  # , render_mask=segment_mask)

            # MSE depth loss
            # depth_loss = ((pred_depth - depth)[segment_mask] ** 2).mean()

            # scale invariant depth loss
            min_depth = 1e-4
            depth_error = torch.log(pred_depth[segment_mask] + min_depth) - torch.log(
                depth[segment_mask] + min_depth
            )
            depth_loss = ((depth_error - depth_error.mean()) ** 2).mean()
            if (not torch.isnan(depth_loss)) and (not torch.isinf(depth_loss)):
                loss = depth_loss * depth_weight
                losses.append(loss)
            else:
                losses.append(None)

        last_losses = [loss.item() if loss is not None else 0 for loss in losses]
        losses = [loss for loss in losses if loss is not None]
        loss_list.append(last_losses)
        if len(losses) > 0:
            total_loss = torch.sum(torch.stack(losses))
            gradient = torch.autograd.grad(total_loss, model_output)[0]
        else:
            total_loss = torch.tensor(0.0, device=device)
            gradient = torch.zeros_like(model_output)

        if verbose and t % 10 == 0:
            pbar.set_description(
                f"Total: {total_loss.item():.4f}, grad: {gradient.abs().max().item():.4f}, grad mean: {gradient.abs().mean().item():.4f}"
            )
        pred_embeds = model.scheduler.step(
            model_output, t, pred_embeds, classifier_gradient=gradient * alpha
        ).prev_sample

    pred_embeds = model.denormalize_embeds(pred_embeds)
    pred_triplane = model.triplane_vae.decode(pred_embeds)
    return {
        "triplane": pred_triplane.detach(),
        "pred_embeds": pred_embeds.detach(),
        "loss_list": np.array(loss_list).transpose(),
        "last_losses": np.array(last_losses),
    }
