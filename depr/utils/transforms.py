import torch
from sklearn.linear_model import RANSACRegressor, LinearRegression


def apply_transformation_matrix(
    points: torch.Tensor, transformation_matrix: torch.Tensor
) -> torch.Tensor:
    if points.shape[-1] == 3:
        points = torch.cat([points, torch.ones_like(points[:, :1])], dim=1)
    points = torch.matmul(points, transformation_matrix.transpose(0, 1))
    points = points[..., :3] / points[..., 3:]
    return points


def to_image_plane(points: torch.Tensor, intrinsic: torch.Tensor) -> torch.Tensor:
    points = torch.matmul(points, intrinsic.transpose(0, 1))
    points = points[..., :2] / points[..., 2:]
    return points


def align_depth(relative_depth, metric_depth, mask=None, min_samples=0.2):
    regressor = RANSACRegressor(
        estimator=LinearRegression(fit_intercept=True), min_samples=min_samples
    )
    if mask is not None:
        regressor.fit(
            relative_depth[mask].reshape(-1, 1), metric_depth[mask].reshape(-1, 1)
        )
    else:
        regressor.fit(relative_depth.reshape(-1, 1), metric_depth.reshape(-1, 1))
    depth = regressor.predict(relative_depth.reshape(-1, 1)).reshape(
        relative_depth.shape
    )
    return depth
