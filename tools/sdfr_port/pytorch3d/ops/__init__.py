import torch

from . import points_alignment  # noqa: F401


def sample_farthest_points(points: torch.Tensor, lengths=None, K: int = 50, random_start_point: bool = False):
    """Iterative farthest point sampling as in pytorch3d.ops.sample_farthest_points (default: start at index 0).
    Returns (selected points (N, K, 3), indices (N, K))."""
    N, P, D = points.shape
    K = min(int(K), P)
    idx = torch.zeros((N, K), dtype=torch.long, device=points.device)
    dists = torch.full((N, P), float("inf"), dtype=points.dtype, device=points.device)
    if random_start_point:
        idx[:, 0] = torch.randint(P, (N,), device=points.device)
    sel = idx[:, 0]
    for k in range(1, K):
        last = points[torch.arange(N, device=points.device), sel]          # (N, D)
        dists = torch.minimum(dists, ((points - last[:, None, :]) ** 2).sum(-1))
        sel = torch.argmax(dists, dim=1)
        idx[:, k] = sel
    return torch.gather(points, 1, idx[..., None].expand(N, K, D)), idx
