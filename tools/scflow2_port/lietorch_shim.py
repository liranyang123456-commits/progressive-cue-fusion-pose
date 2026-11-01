# -*- coding: utf-8 -*-
"""
Pure-PyTorch replacement for the `lietorch` SE3 group class (Teed & Deng, "Tangent Space Backpropagation", 2021).

Why: SCFlow2 (CVPR 2025) depends on lietorch's compiled backend `lietorch_backends`, which builds on Windows/MSVC
but crashes with an access violation on every group operation (CPU and CUDA, with and without optimisation).
The SCFlow2 *test* path uses only a small part of the lietorch API -- SE3(data), SE3.Identity, SE3.exp, .log(),
.inv(), group * group, group * points, .matrix(), .data, .shape, indexing -- all of which are plain tensor
arithmetic. This module implements exactly that with the same conventions as lietorch/include/se3.h:

  data layout  [tx, ty, tz, qx, qy, qz, qw]   (Eigen quaternion coefficient order x, y, z, w)
  tangent      [tau (3), phi (3)]              Exp: R = exp(phi), t = J_l(phi) tau;  Log: phi = log(R), tau = J_l^{-1}(phi) t
  action       X * p = R p + t;  X1 * X2 = (R1 R2, t1 + R1 t2);  inv = (R^T, -R^T t)

Autograd is provided by PyTorch. `lietorch_extras` (the CUDA kernels that build the dense SE3 least-squares
system) is unaffected and keeps being used from the compiled extension.

Install: copy this file over <site-packages>/lietorch/__init__.py (the compiled lietorch_backends stays unused).
"""
from __future__ import annotations

import torch

_EPS = 1e-6


def _hat(v: torch.Tensor) -> torch.Tensor:
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack([torch.stack([o, -z, y], -1), torch.stack([z, o, -x], -1), torch.stack([-y, x, o], -1)], -2)


def _quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    """q = [x, y, z, w] (normalised) -> (..., 3, 3)"""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz, wx, wy, wz = x * y, x * z, y * z, w * x, w * y, w * z
    r0 = torch.stack([1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], -1)
    r1 = torch.stack([2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)], -1)
    r2 = torch.stack([2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)], -1)
    return torch.stack([r0, r1, r2], -2)


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by,
                        aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw,
                        aw * bw - ax * bx - ay * by - az * bz], -1)


def _so3_exp(phi: torch.Tensor) -> torch.Tensor:
    theta = phi.norm(dim=-1, keepdim=True)
    small = theta < _EPS
    half = 0.5 * theta
    # sin(theta/2)/theta with a Taylor fallback
    k = torch.where(small, 0.5 - theta ** 2 / 48.0, torch.sin(half) / theta.clamp_min(1e-12))
    w = torch.where(small, 1.0 - theta ** 2 / 8.0, torch.cos(half))
    return torch.cat([k * phi, w], -1)


def _so3_log(q: torch.Tensor) -> torch.Tensor:
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    v, w = q[..., :3], q[..., 3:4]
    sign = torch.where(w < 0, -torch.ones_like(w), torch.ones_like(w))  # shortest rotation
    v, w = v * sign, w * sign
    n = v.norm(dim=-1, keepdim=True)
    small = n < _EPS
    theta = 2.0 * torch.atan2(n, w)
    k = torch.where(small, 2.0 / w.clamp_min(1e-12), theta / n.clamp_min(1e-12))
    return k * v


def _left_jacobian(phi: torch.Tensor) -> torch.Tensor:
    theta = phi.norm(dim=-1, keepdim=True)[..., None]
    small = theta < _EPS
    Phi = _hat(phi)
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).expand_as(Phi)
    c1 = torch.where(small, 0.5 - theta ** 2 / 24.0, (1 - torch.cos(theta)) / theta.clamp_min(1e-12) ** 2)
    c2 = torch.where(small, 1.0 / 6.0 - theta ** 2 / 120.0, (theta - torch.sin(theta)) / theta.clamp_min(1e-12) ** 3)
    return I + c1 * Phi + c2 * (Phi @ Phi)


def _left_jacobian_inverse(phi: torch.Tensor) -> torch.Tensor:
    theta = phi.norm(dim=-1, keepdim=True)[..., None]
    small = theta < _EPS
    Phi = _hat(phi)
    I = torch.eye(3, dtype=phi.dtype, device=phi.device).expand_as(Phi)
    t = theta.clamp_min(1e-12)
    c = torch.where(small, 1.0 / 12.0 + theta ** 2 / 720.0, 1.0 / t ** 2 - (1 + torch.cos(theta)) / (2 * t * torch.sin(theta).clamp_min(1e-12)))
    return I - 0.5 * Phi + c * (Phi @ Phi)


class SE3:
    """Batched SE3 elements stored as (..., 7) tensors [t, q_xyzw]; lietorch-compatible subset."""
    group_name = "SE3"
    manifold_dim = 6
    embedded_dim = 7

    def __init__(self, data: torch.Tensor):
        if isinstance(data, SE3):
            data = data.data
        if data.shape[-1] != 7:
            raise ValueError(f"SE3 data must have last dimension 7, got {tuple(data.shape)}")
        self.data = data

    # ---- constructors -----------------------------------------------------------------------
    @classmethod
    def Identity(cls, *batch_shape, device=None, dtype=torch.float32, requires_grad=False):
        if len(batch_shape) == 1 and isinstance(batch_shape[0], (tuple, list, torch.Size)):
            batch_shape = tuple(batch_shape[0])
        d = torch.zeros(*batch_shape, 7, device=device, dtype=dtype)
        d[..., 6] = 1.0
        if requires_grad:
            d.requires_grad_(True)
        return cls(d)

    @classmethod
    def InitFromVec(cls, data: torch.Tensor):
        return cls(data)

    @classmethod
    def exp(cls, tau_phi: torch.Tensor):
        tau, phi = tau_phi[..., :3], tau_phi[..., 3:6]
        q = _so3_exp(phi)
        t = (_left_jacobian(phi) @ tau[..., None])[..., 0]
        return cls(torch.cat([t, q], -1))

    # ---- accessors ---------------------------------------------------------------------------
    @property
    def shape(self):
        return self.data.shape[:-1]

    @property
    def device(self):
        return self.data.device

    @property
    def dtype(self):
        return self.data.dtype

    def __getitem__(self, idx):
        return SE3(self.data[idx])

    def __repr__(self):
        return f"SE3(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device})"

    def vec(self):
        return self.data

    def translation(self):
        return torch.cat([self.data[..., :3], torch.ones_like(self.data[..., :1])], -1)

    def detach(self):
        return SE3(self.data.detach())

    def to(self, *args, **kwargs):
        return SE3(self.data.to(*args, **kwargs))

    def cpu(self):
        return SE3(self.data.cpu())

    def cuda(self):
        return SE3(self.data.cuda())

    def float(self):
        return SE3(self.data.float())

    def double(self):
        return SE3(self.data.double())

    def view(self, *shape):
        return SE3(self.data.view(*shape, 7))

    def reshape(self, *shape):
        return SE3(self.data.reshape(*shape, 7))

    # ---- group operations --------------------------------------------------------------------
    def matrix(self):
        R = _quat_to_matrix(self.data[..., 3:7])
        t = self.data[..., :3]
        T = torch.zeros(*self.shape, 4, 4, dtype=self.data.dtype, device=self.data.device)
        T[..., :3, :3] = R
        T[..., :3, 3] = t
        T[..., 3, 3] = 1.0
        return T

    def log(self):
        phi = _so3_log(self.data[..., 3:7])
        tau = (_left_jacobian_inverse(phi) @ self.data[..., :3, None])[..., 0]
        return torch.cat([tau, phi], -1)

    def inv(self):
        q = self.data[..., 3:7]
        q_inv = torch.cat([-q[..., :3], q[..., 3:4]], -1)
        R_inv = _quat_to_matrix(q_inv)
        t_inv = -(R_inv @ self.data[..., :3, None])[..., 0]
        return SE3(torch.cat([t_inv, q_inv], -1))

    def act(self, p: torch.Tensor):
        """p: (..., 3) or (..., 4) homogeneous points, broadcast against the group's batch shape."""
        R = _quat_to_matrix(self.data[..., 3:7])
        t = self.data[..., :3]
        if p.shape[-1] == 4:
            xyz = (R @ p[..., :3, None])[..., 0] + t * p[..., 3:4]
            return torch.cat([xyz, p[..., 3:4]], -1)
        return (R @ p[..., None])[..., 0] + t

    def mul(self, other: "SE3"):
        q1, q2 = self.data[..., 3:7], other.data[..., 3:7]
        R1 = _quat_to_matrix(q1)
        t = self.data[..., :3] + (R1 @ other.data[..., :3, None])[..., 0]
        q = _quat_mul(q1, q2)
        q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        return SE3(torch.cat([t, q], -1))

    def __mul__(self, other):
        if isinstance(other, SE3):
            return self.mul(other)
        return self.act(other)

    def retr(self, a: torch.Tensor):
        return SE3.exp(a) * self

    def adj(self, a=None):
        R = _quat_to_matrix(self.data[..., 3:7])
        tx = _hat(self.data[..., :3])
        Z = torch.zeros_like(R)
        Ad = torch.cat([torch.cat([R, tx @ R], -1), torch.cat([Z, R], -1)], -2)
        return Ad if a is None else (Ad @ a[..., None])[..., 0]

    def adjT(self, a: torch.Tensor):
        return (self.adj().transpose(-1, -2) @ a[..., None])[..., 0]


class SO3:
    """Minimal SO3 (quaternion xyzw) for completeness."""
    group_name = "SO3"
    manifold_dim = 3
    embedded_dim = 4

    def __init__(self, data):
        self.data = data.data if isinstance(data, SO3) else data

    @classmethod
    def Identity(cls, *batch_shape, device=None, dtype=torch.float32):
        d = torch.zeros(*batch_shape, 4, device=device, dtype=dtype)
        d[..., 3] = 1.0
        return cls(d)

    @classmethod
    def exp(cls, phi):
        return cls(_so3_exp(phi))

    def log(self):
        return _so3_log(self.data)

    def matrix(self):
        return _quat_to_matrix(self.data)

    def inv(self):
        return SO3(torch.cat([-self.data[..., :3], self.data[..., 3:4]], -1))

    def __mul__(self, other):
        if isinstance(other, SO3):
            return SO3(_quat_mul(self.data, other.data))
        return (_quat_to_matrix(self.data) @ other[..., None])[..., 0]

    @property
    def shape(self):
        return self.data.shape[:-1]


__all__ = ["SE3", "SO3"]
