import numpy as np
import scipy


def pointcloud_neighbor_distances_indices(source_points, target_points):
    target_kdtree = scipy.spatial.cKDTree(target_points)
    distances, indices = target_kdtree.query(source_points)
    return distances, indices


def percent_below(dists, thresh):
    return np.mean((dists**2 <= thresh).astype(np.float32)) * 100.0


def _f_score(a_to_b, b_to_a, thresh):
    OCCNET_FSCORE_EPS = 1e-09
    precision = percent_below(a_to_b, thresh)
    recall = percent_below(b_to_a, thresh)

    return (2 * precision * recall) / (precision + recall + OCCNET_FSCORE_EPS)


def f_score(points1, points2, tau=0.002):
    """Computes the F-Score at tau between two meshes."""
    dist12, _ = pointcloud_neighbor_distances_indices(points1, points2)
    dist21, _ = pointcloud_neighbor_distances_indices(points2, points1)
    f_score_tau = _f_score(dist12, dist21, tau)
    return f_score_tau
