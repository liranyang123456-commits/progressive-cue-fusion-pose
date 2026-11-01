# -*- coding: utf-8 -*-
"""
Minimal stand-in for `sksparse.cholmod` (scikit-sparse has no Windows build) covering the two calls that
SCFlow2's RAFT-3D grid smoother makes at inference time:

    sym = cholmod.analyze_AAt(A, ordering_method='best')   # symbolic analysis of A A^T
    chol = sym.cholesky_AAt(A)                              # numeric factorisation
    x = chol(b)                                             # solve (A A^T) x = b

The systems are small (one unknown per cell of the 1/8-resolution feature grid, ~1k unknowns) and sparse, so a
SuperLU factorisation of A A^T (scipy.sparse.linalg.splu) is exact and fast enough; the result is identical to
CHOLMOD's up to floating-point round-off. Install as <site-packages>/sksparse/cholmod.py (+ empty __init__.py).
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class _Factor:
    def __init__(self, A):
        AAt = (A @ A.T).tocsc()
        self._dtype = AAt.dtype
        self._lu = spla.splu(AAt)

    def __call__(self, b):
        b = np.ascontiguousarray(np.asarray(b), dtype=self._dtype)  # SuperLU requires the factor dtype
        return self._lu.solve(b).astype(np.float32)

    def solve_A(self, b):  # CHOLMOD API name
        return self(b)


class _Symbolic:
    def cholesky_AAt(self, A, beta=0.0):
        return _Factor(A)

    def cholesky_AAt_inplace(self, A, beta=0.0):
        return _Factor(A)


def analyze_AAt(A, mode="auto", ordering_method="default", use_long=None):
    return _Symbolic()


def cholesky_AAt(A, beta=0.0, **kwargs):
    return _Factor(A)
