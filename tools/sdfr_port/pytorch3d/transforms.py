import torch


def hat(v: torch.Tensor) -> torch.Tensor:
    N = v.shape[0]
    h = v.new_zeros((N, 3, 3))
    x, y, z = v.unbind(1)
    h[:, 0, 1], h[:, 0, 2] = -z, y
    h[:, 1, 0], h[:, 1, 2] = z, -x
    h[:, 2, 0], h[:, 2, 1] = -y, x
    return h


def hat_inv(h: torch.Tensor) -> torch.Tensor:
    return torch.stack((h[:, 2, 1], h[:, 0, 2], h[:, 1, 0]), dim=1)


def so3_rotation_angle(R: torch.Tensor, eps: float = 1e-4, cos_angle: bool = False, cos_bound: float = 1e-4) -> torch.Tensor:
    rot_trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    phi_cos = (rot_trace - 1.0) * 0.5
    if cos_angle:
        return phi_cos
    if cos_bound > 0.0:
        bound = 1.0 - cos_bound
        phi_cos = phi_cos.clamp(-bound, bound)
    return torch.acos(phi_cos)


def so3_exp_map(log_rot: torch.Tensor, eps: float = 0.0001) -> torch.Tensor:
    nrms = (log_rot * log_rot).sum(1)
    rot_angles = torch.clamp(nrms, eps).sqrt()
    rot_angles_inv = 1.0 / rot_angles
    fac1 = rot_angles_inv * rot_angles.sin()
    fac2 = rot_angles_inv * rot_angles_inv * (1.0 - rot_angles.cos())
    skews = hat(log_rot)
    skews_square = torch.bmm(skews, skews)
    return (fac1[:, None, None] * skews + fac2[:, None, None] * skews_square
            + torch.eye(3, dtype=log_rot.dtype, device=log_rot.device)[None])


def so3_log_map(R: torch.Tensor, eps: float = 0.0001, cos_bound: float = 1e-4) -> torch.Tensor:
    phi = so3_rotation_angle(R, cos_bound=cos_bound, eps=eps)
    phi_sin = torch.sin(phi)
    phi_factor = torch.empty_like(phi)
    ok_denom = phi_sin.abs() > (0.5 * eps)
    phi_factor[~ok_denom] = 0.5 + (phi[~ok_denom] ** 2) * (1.0 / 12)
    phi_factor[ok_denom] = phi[ok_denom] / (2.0 * phi_sin[ok_denom])
    log_rot_hat = phi_factor[:, None, None] * (R - R.permute(0, 2, 1))
    return hat_inv(log_rot_hat)


def se3_log_map(*args, **kwargs):  # imported by sdfr.py but not used by SDFR_adam
    raise NotImplementedError("se3_log_map is not provided by the SDFR port")
