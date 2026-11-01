#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
"""
Exp_Camera_Optim_RT_Val.py

目的：验证“基于可微分渲染的相机RT优化”在单帧上的有效性。
流程：
1. 仅读取第一帧图像，复用 ChessBoard_Cal_RT_CMP_3DICP_2DFFT_Fea_Matching_MIS_Op3Test3.py 的处理逻辑生成 mask 与 3D mesh。
2. 使用与原代码一致的渲染初始化（相机/光源等）。
3. 在球面上随机采样若干相机位姿作为“真实值”目标，渲染得到目标轮廓/深度/法线。
4. 从初始化位姿出发，使用可微分渲染迭代优化恢复目标位姿。
5. 保存每个实验的 GT/初始/优化后轮廓和位姿。

说明：
- 为避免破坏原工程，绝大多数逻辑直接调用原文件中的类/函数；仅在本文件内添加最小实验驱动代码。
- 默认输入：环境变量 INPUT_IMAGE（单张），否则使用 data/real_frames
- 输出目录：默认保存到 outputs（可用 OUTPUT_DIR 覆盖）

核心思路（建议按阶段理解/调参）：
- Stage0: 读取配置/开关（环境变量）+ 可选 fast defaults
- Stage1: 读入首帧 -> 复用原逻辑生成 mask/mesh（可选 mesh 简化）
- Stage2: 初始化渲染参数 -> 构建 PyTorch3D Mesh + 复用渲染器（full/search 两套分辨率）
- Stage3: 生成 GT 相机位姿（球面采样，强制相邻位姿差异足够大）
- Stage4: 对每个 GT：渲染目标 (sil/depth/normals) + 保存 GT/init 可视化
- Stage5: 全局搜索得到较好 seed（支持 rand1024 / multiradius_axis / legacy / beam）
- Stage6: 可选 refine（search/full 分辨率、Adam/Greedy 等）-> 保存 best_* + 写 summary
"""
import os
os.environ["VAL_DISABLE_TORCH_COMPILE"] = "1"
import random
import json
import csv
import shutil
import copy
import time
import numpy as np
import cv2
import torch
import math
import open3d as o3d
from dataclasses import dataclass
from typing import Any

# 复用原文件中的实现（用模块形式，避免直接 from 失败）
try:
    import ChessBoard_Cal_RT_CMP_3DICP_2DFFT_Fea_Matching_MIS_Op3Test3 as cb  # legacy real-image tooling (optional)
except Exception as _e:
    cb = None
    print(f"[Warning] Legacy module ChessBoard_Cal_RT_... unavailable ({type(_e).__name__}); the synthetic benchmark path does not need it.")
# 按要求直接导入同目录语义模块；若缺少依赖则不阻塞本实验
try:
    import extract_semantic_features_Function as semfe  # noqa: F401
except Exception as _e:
    print(f"[Warning] Failed to import extract_semantic_features_Function: {_e}. Semantic-related parts will be skipped.")

ImageProcessor = cb.ImageProcessor if cb is not None else None
# 注意：原主脚本里的相机优化模型是 ImageProcessor 内部类（用于 prev->curr）。
# 本实验脚本将按 PyTorch3D 官方示例 camera_position_optimization_with_differentiable_rendering.py
# 的方式进行相机视点优化（camera_position -> look_at_rotation -> R,T）。

DEFAULT_OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "outputs")

# ============================================================
# Code-level run parameters.
# The user requested these parameters to be written directly into the script
# so the experiment can be launched without setting environment variables.
# Set FORCE_CODE_RUN_PARAMS=False if you want terminal env vars to take over.
# ============================================================
FORCE_CODE_RUN_PARAMS = True
CODE_RUN_PARAMS = {
    # Experiment size/output.
    "OUTPUT_DIR": os.path.join(DEFAULT_OUTPUT_ROOT, "accuracy_ref_view_prior_v9_run"),
    "NUM_GT_POSES": "10",
    "VAL_PRECOMPUTE_GT": "1",
    "VAL_RANDOM_SEED": "12291",

    # Accuracy-first global/local search.
    "VAL_GLOBAL_MODE": "rand1024",
    "VAL_LOCAL_MODE": "beam",
    "VAL_RAND_TOPK": "16",
    "VAL_RAND_DIVERSE_ELEV_SEEDS": "1",
    "VAL_RAND_ELEV_BINS": "0,15,30,45,60,75,90",
    "VAL_RAND_DIVERSE_AZIM_SEEDS": "1",
    "VAL_RAND_AZIM_BINS": "-180,-90,0,90,180",
    # 算法级优化：每 radius 使用 loss + elevation 偏好联合评分
    # elev_pref=50° 意味着俯仰角低于 50° 的视角会受到惩罚（score 上升）
    # λ=0.012 控制惩罚强度，让高俯仰视角更容易进入 TopK
    "VAL_RAND_ELEV_PREFERENCE_DEG": "45.0",
    "VAL_RAND_ELEV_LAMBDA": "0.02",
    "VAL_NORMAL_PRIOR_WEIGHT": "0.35",
    "VAL_REF_VIEW_PRIOR_WEIGHT": "0.30",
    "VAL_RAND_SAMPLES_PER_RADIUS": "1024",
    "VAL_RAND_TOPK_PER_RADIUS": "2",
    # 新增：全局采样阶段加入 valid_ratio + RGB 联合评分
    "VAL_GLOBAL_VALID_RATIO_WEIGHT": "0.8",
    "VAL_GLOBAL_RGB_WEIGHT": "3.0",
    "VAL_RAND_ADD_GRID_ANCHORS": "1",
    "VAL_RAND_ANCHOR_ELEVS": "5,15,30,45,60,75,85",
    "VAL_RAND_ANCHOR_AZIMS": "-180,-135,-90,-45,0,45,90,135,180",
    "VAL_GLOBAL_RGB_RERANK": "1",
    "VAL_GLOBAL_RGB_RERANK_W": "4.0",
    # Valid-ratio penalty: penalises candidates with almost no GT/rendered overlap.
    # Key fix: prevents wrong viewpoints from winning because of tiny valid regions.
    "VAL_W_VALID_PENALTY": "1.0",
    # Stronger IoU weight for better silhouette SHAPE matching (not just pixel MSE).
    "VAL_W_IOU": "0.4",
    # Depth/normal weights: more balanced (less silhouette dominance).
    "VAL_W_SIL_START": "1.0",
    "VAL_W_DEPTH_START": "1.5",
    "VAL_W_NORMAL_START": "0.8",
    # Constrained LocalBeam settings.
    "VAL_LOCAL_BEAM_ELEV_CONSTRAIN": "1",
    "VAL_LOCAL_BEAM_ELEV_CONSTRAIN_THRESHOLD": "50.0",  # 更高阈值
    "VAL_LOCAL_BEAM_ELEV_HALF_BAND_DEG": "12.0",
    "VAL_LOCAL_BEAM_AZIM_CONSTRAIN": "1",
    "VAL_LOCAL_BEAM_AZIM_HALF_BAND_DEG": "28.0",
    "VAL_COARSE_BEAM_AZIM_HALF_BAND_DEG": "50.0",
    "VAL_LOCAL_BEAM_DIST_REFINE": "1",
    "VAL_LOCAL_BEAM_DIST_SCALES": "auto",
    "VAL_REF_DIST_PENALTY_RELATIVE": "1",
    "VAL_RAND_DIVERSE_DIST_SEEDS": "1",
    "VAL_REF_DIST_PENALTY_WEIGHT": "0.35",
    "VAL_REF_DIST_PENALTY_THRESHOLD": "0.04",
    "VAL_SELECT_REF_AZIM_WEIGHT": "0.040",
    "VAL_SELECT_REF_DIST_WEIGHT": "0.8",
    "VAL_HIGH_ELEV_UNDER_PENALTY": "2.0",
    "VAL_HIGH_ELEV_UNDER_THRESHOLD": "58.0",
    "VAL_LOW_ELEV_UNDER_PENALTY": "2.2",
    "VAL_LOW_ELEV_UNDER_THRESHOLD": "45.0",
    "VAL_SIL_CENTROID_ELEV_GAIN": "25.0",
    "VAL_REF_ELEV_CENTROID_BLEND": "0.45",
    "VAL_SELECT_W_LOCAL_LOSS": "0.20",
    "VAL_SIL_CENTROID_AZIM_GAIN": "28.0",
    "VAL_REF_AZIM_CENTROID_BLEND": "0.35",
    "VAL_REF_DIST_DEPTH_SCALE": "0.92",
    # LocalBeam 内部 RGB 纹理 loss 权重（针对高俯仰）
    "VAL_LOCAL_BEAM_RGB_WEIGHT": "8.0",
    # Multi-scale coarse-to-fine beam: large-step pre-pass before fine search.
    "VAL_COARSE_BEAM_STEP_FRAC": "0.08",
    "VAL_COARSE_BEAM_LEVELS": "8",
    "VAL_COARSE_BEAM_WIDTH": "24",
    # 性能优化参数
    "VAL_LOCAL_BEAM_EARLY_STOP": "1",
    "VAL_LOCAL_BEAM_EARLY_STOP_PATIENCE": "8",
    # 自适应步长参数
    "VAL_LOCAL_BEAM_ADAPTIVE_STEP": "1",
    "VAL_LOCAL_BEAM_STEP_SHRINK": "0.7",
    "VAL_LOCAL_BEAM_STEP_GROW": "1.2",
    "VAL_LOCAL_BEAM_MIN_STEP_FRAC": "0.002",
    "VAL_LOCAL_BEAM_MAX_STEP_FRAC": "0.15",
    # 高俯仰种子早期临时提高 depth/normal 权重
    "VAL_HIGH_ELEV_BOOST_DEPTH": "2.5",
    "VAL_HIGH_ELEV_BOOST_NORMAL": "1.5",
    "VAL_HIGH_ELEV_BOOST_LAYERS": "18",
    # 可选算法增强（默认关闭 = 纯 v9）：末层 END 重打分 + beam 后 DR 精修
    "VAL_LOCAL_BEAM_LAST_LAYER_END_SCORE": "0",
    "VAL_LOCAL_BEAM_LAST_LAYER_END_COND": "0",
    "VAL_LOCAL_BEAM_LAST_LAYER_END_MIN_ELEV": "58.0",
    "VAL_LOCAL_BEAM_FINAL_DR_ENABLE": "0",
    "VAL_LOCAL_BEAM_FINAL_DR_COND_ENABLE": "0",
    "VAL_LOCAL_BEAM_FINAL_DR_MIN_SEED_ELEV": "58.0",
    "VAL_LOCAL_BEAM_FINAL_DR_STEPS": "4",
    "VAL_LOCAL_BEAM_FINAL_DR_LR": "0.015",
    "VAL_LOCAL_BEAM_LAST_LAYER_FD_TOPM": "0",
    # GT elevation must not steer beam conditions (ground-truth leak); kept OFF.
    "VAL_LOCAL_BEAM_COND_USE_GT_ELEV": "0",
    # Strict no-ground-truth mode: GT is used only to render targets and to evaluate.
    "VAL_STRICT_NO_GT": "1",
    "VAL_SELECT_USE_VISUAL_SCORE": "1",
    "VAL_SELECT_W_RGB": "15.0",  # 更高的 RGB 权重
    "VAL_SELECT_W_VALID": "0.05",
    # 最终候选选择加入 elevation 接近度 tie-breaker
    "VAL_SELECT_REF_ELEV_WEIGHT": "0.04",
    "VAL_REF_ELEV_FLIP_THRESHOLD": "35.0",
    "VAL_REF_ELEV_FLIP_PENALTY_WEIGHT": "0.08",
    "VAL_REF_AZIM_FLIP_THRESHOLD": "40.0",
    "VAL_REF_AZIM_FLIP_PENALTY_WEIGHT": "0.006",
    "VAL_RAND_REF_ELEV_BIN_HALF_SPAN": "30.0",
    "VAL_RAND_REF_AZIM_BIN_HALF_SPAN": "55.0",
    "VAL_RAND_REF_AZIM_BIN_STEP": "20.0",
    "VAL_RAND_REF_AZIM_ANCHOR_HALF_SPAN": "45.0",
    "VAL_RAND_REF_AZIM_ANCHOR_STEP": "15.0",
    "VAL_SELECT_ELEV_PROXIMITY_WEIGHT": "0.3",
    "VAL_TOPK": "16",
    "VAL_TOPK2": "16",
    "VAL_LOCAL_BEAM_WIDTH": "48",  # 降低宽度
    "VAL_LOCAL_BEAM_LEVELS": "32",  # 降低层数，减少计算量
    "VAL_RAND_BATCH": "512",  # 降低 batch size，避免 OOM
    "VAL_LOCAL_BEAM_EARLY_STOP": "1",  # 启用早停
    "VAL_LOCAL_BEAM_EARLY_STOP_PATIENCE": "8",  # 连续 8 层无改善停止
    "VAL_LOCAL_BEAM_REFINE_FULL": "1",
    "VAL_LOCAL_BEAM_SCORE_MODE": "full",

    # Multi-start refinement to choose the best basin before long refinement.
    "VAL_MS_REFINE_ENABLE": "0",
    "VAL_MS_REFINE_TOPK": "0",
    "VAL_MS_LONG_FORCE_FULL": "1",

    # Previous-code-preserving default: use the best local Beam result as the
    # final estimate.  Final greedy/SE3 outputs are useful for inspection, but
    # they can move away from the visually good look-at solution.
    "VAL_SEARCH_ENABLE": "0",

    # Final full-resolution refinement (disabled by VAL_SEARCH_ENABLE=0 unless
    # you explicitly re-enable it).
    "VAL_REFINE_RES_MODE": "full",
    "VAL_REFINE_MAX_ITERS": "300",
    "VAL_REFINE_LR": "0.01",
    "VAL_LOSS_SCHEDULE": "1",
    "VAL_W_SIL_END": "0.2",
    "VAL_W_DEPTH_END": "2.0",
    "VAL_W_NORMAL_END": "1.0",

    # Final 6DoF SE(3) refinement for sub-pixel / tiny pose error reduction.
    "VAL_SE3_REFINE_ENABLE": "0",
    "VAL_SE3_REFINE_ITERS": "200",
    "VAL_SE3_REFINE_LR": "0.003",
    "VAL_SE3_ACCEPT_MIN_DELTA": "0.0",
    "VAL_SE3_ACCEPT_AS_BEST": "0",
    "VAL_SE3_W_REG": "1e-5",
    "VAL_SE3_T_REG": "1e-5",
    "VAL_SE3_USE_SCHEDULER": "1",
    "VAL_SE3_EARLY_STOP_PATIENCE": "80",
    "VAL_SE3_NORMAL_ABS": "1",

    # Keep oracle terms OFF for fair experiments. Turn these on only for
    # upper-bound debugging, not for paper comparisons.
    "VAL_SE3_ORACLE_POSE_W": "0.0",
    "VAL_SE3_ORACLE_REPROJ_W": "0.0",

    # I/O and log noise control.
    "VAL_SAVE_INTERVAL": "300",
    "VAL_SAVE_ON_IMPROVE": "0",
    "VAL_DEBUG_RAST": "0",
}

# ============================================================
# Run profiles: preserve v9 / v12 parameter sets; select via VAL_RUN_PROFILE.
# v9  = baseline (VisualSelect + ref-view priors, no post-beam DR)
# v12 = v12b (last-layer END scoring + unconditional final DR)
# v12c = v12 + DR/END only when seed elevation >= threshold (reduces low-elev regressions)
# ============================================================
RUN_PROFILE_PRESETS: dict[str, dict[str, str]] = {
    "v9": {
        "OUTPUT_DIR": os.path.join(DEFAULT_OUTPUT_ROOT, "accuracy_ref_view_prior_v9_run"),
        "VAL_DETERMINISTIC": "1",
        "VAL_LOCAL_BEAM_LAST_LAYER_END_SCORE": "0",
        "VAL_LOCAL_BEAM_LAST_LAYER_END_COND": "0",
        "VAL_LOCAL_BEAM_FINAL_DR_ENABLE": "0",
        "VAL_LOCAL_BEAM_FINAL_DR_COND_ENABLE": "0",
    },
    "v12": {
        "OUTPUT_DIR": os.path.join(DEFAULT_OUTPUT_ROOT, "accuracy_ref_view_prior_v12_run"),
        "VAL_LOCAL_BEAM_LAST_LAYER_END_SCORE": "1",
        "VAL_LOCAL_BEAM_LAST_LAYER_END_COND": "0",
        "VAL_LOCAL_BEAM_FINAL_DR_ENABLE": "1",
        "VAL_LOCAL_BEAM_FINAL_DR_COND_ENABLE": "0",
        "VAL_LOCAL_BEAM_FINAL_DR_STEPS": "4",
        "VAL_LOCAL_BEAM_FINAL_DR_LR": "0.015",
    },
    "v12c": {
        "OUTPUT_DIR": os.path.join(DEFAULT_OUTPUT_ROOT, "accuracy_ref_view_prior_v12c_run"),
        "VAL_LOCAL_BEAM_LAST_LAYER_END_SCORE": "1",
        "VAL_LOCAL_BEAM_LAST_LAYER_END_COND": "1",
        "VAL_LOCAL_BEAM_LAST_LAYER_END_MIN_ELEV": "58.0",
        "VAL_LOCAL_BEAM_FINAL_DR_ENABLE": "1",
        "VAL_LOCAL_BEAM_FINAL_DR_COND_ENABLE": "1",
        "VAL_LOCAL_BEAM_FINAL_DR_MIN_SEED_ELEV": "58.0",
        "VAL_LOCAL_BEAM_FINAL_DR_STEPS": "4",
        "VAL_LOCAL_BEAM_FINAL_DR_LR": "0.012",
    },
    # v12d: v12c + 更高 seed 门槛(65°) + 末层 FD 抛光（实验：gt_08 仍可能回退）
    "v12d": {
        "OUTPUT_DIR": os.path.join(DEFAULT_OUTPUT_ROOT, "accuracy_ref_view_prior_v12d_run"),
        "VAL_LOCAL_BEAM_LAST_LAYER_END_SCORE": "1",
        "VAL_LOCAL_BEAM_LAST_LAYER_END_COND": "1",
        "VAL_LOCAL_BEAM_LAST_LAYER_END_MIN_ELEV": "65.0",
        "VAL_LOCAL_BEAM_LAST_LAYER_FD_TOPM": "4",
        "VAL_LOCAL_BEAM_FINAL_DR_ENABLE": "1",
        "VAL_LOCAL_BEAM_FINAL_DR_COND_ENABLE": "1",
        "VAL_LOCAL_BEAM_FINAL_DR_MIN_SEED_ELEV": "65.0",
        "VAL_LOCAL_BEAM_FINAL_DR_STEPS": "3",
        "VAL_LOCAL_BEAM_FINAL_DR_LR": "0.010",
    },
    # v12e: 仅高俯仰(>=72°)启用 END/DR，避开 ~65° 盆地(gt_08)；保留末层 FD
    "v12e": {
        "OUTPUT_DIR": os.path.join(DEFAULT_OUTPUT_ROOT, "accuracy_ref_view_prior_v12e_run"),
        "VAL_LOCAL_BEAM_LAST_LAYER_END_SCORE": "1",
        "VAL_LOCAL_BEAM_LAST_LAYER_END_COND": "1",
        "VAL_LOCAL_BEAM_LAST_LAYER_END_MIN_ELEV": "72.0",
        "VAL_LOCAL_BEAM_LAST_LAYER_FD_TOPM": "4",
        "VAL_LOCAL_BEAM_FINAL_DR_ENABLE": "1",
        "VAL_LOCAL_BEAM_FINAL_DR_COND_ENABLE": "1",
        "VAL_LOCAL_BEAM_FINAL_DR_MIN_SEED_ELEV": "72.0",
        "VAL_LOCAL_BEAM_FINAL_DR_STEPS": "4",
        "VAL_LOCAL_BEAM_FINAL_DR_LR": "0.012",
    },
    # v12f: seed 俯仰>=72 才 END/DR/FD；保留末层 FD
    # (legacy profile used the GT elevation here; that is a GT leak and is now
    #  ignored under VAL_STRICT_NO_GT=1 -- the seed elevation is used instead)
    "v12f": {
        "OUTPUT_DIR": os.path.join(DEFAULT_OUTPUT_ROOT, "accuracy_ref_view_prior_v12f_run"),
        "VAL_LOCAL_BEAM_COND_USE_GT_ELEV": "0",
        "VAL_LOCAL_BEAM_LAST_LAYER_END_SCORE": "1",
        "VAL_LOCAL_BEAM_LAST_LAYER_END_COND": "1",
        "VAL_LOCAL_BEAM_LAST_LAYER_END_MIN_ELEV": "72.0",
        "VAL_LOCAL_BEAM_LAST_LAYER_FD_TOPM": "4",
        "VAL_LOCAL_BEAM_FINAL_DR_ENABLE": "1",
        "VAL_LOCAL_BEAM_FINAL_DR_COND_ENABLE": "1",
        "VAL_LOCAL_BEAM_FINAL_DR_MIN_SEED_ELEV": "72.0",
        "VAL_LOCAL_BEAM_FINAL_DR_STEPS": "4",
        "VAL_LOCAL_BEAM_FINAL_DR_LR": "0.012",
    },
    # v10: mesh/depth-scale adaptive global search + wide dist refine + universal post-beam DR
    "v10": {
        "OUTPUT_DIR": os.path.join(DEFAULT_OUTPUT_ROOT, "accuracy_ref_view_prior_v10_run"),
        "VAL_DETERMINISTIC": "1",
        "VAL_RAND_RADII_MULTS": "auto",
        "VAL_RAND_SAMPLES_PER_RADIUS": "auto",
        "VAL_LOSS_SCALE": "auto",
        "VAL_LOCAL_BEAM_DIST_SCALES": "auto",
        "VAL_REF_DIST_PENALTY_RELATIVE": "1",
        "VAL_REF_DIST_PENALTY_WEIGHT": "0.55",
        "VAL_LOCAL_BEAM_LAST_LAYER_END_SCORE": "0",
        "VAL_LOCAL_BEAM_LAST_LAYER_END_COND": "0",
        "VAL_LOCAL_BEAM_FINAL_DR_ENABLE": "1",
        "VAL_LOCAL_BEAM_FINAL_DR_COND_ENABLE": "0",
        "VAL_LOCAL_BEAM_FINAL_DR_STEPS": "6",
        "VAL_LOCAL_BEAM_FINAL_DR_LR": "0.012",
    },
    # Greedy/Stage3 加速实验（不影响 v9 主路径 rand+beam；仅 VAL_SEARCH_ENABLE=1 时生效）
    "greedy_fast": {
        "OUTPUT_DIR": os.path.join(DEFAULT_OUTPUT_ROOT, "accuracy_greedy_fast_run"),
        "VAL_SEARCH_ENABLE": "1",
        "VAL_GREEDY_BATCH_NEIGHBORS": "1",
        "VAL_GREEDY_FD_STEP": "1",
        "VAL_GREEDY_USE_SIL_ONLY": "1",
    },
}

# Active profile when FORCE_CODE_RUN_PARAMS=1 (override via env: VAL_RUN_PROFILE=v9|v12|v12c|v12d).
CODE_RUN_PARAMS["VAL_RUN_PROFILE"] = "v9"

# ============================================================
# Rendering backend (固定为 nvdiffrast)
# 说明：
# - 当前工程使用 nvdiffrast + kaolin；本文件不再保留 PyTorch3D 相关代码路径。
# - 为尽量少改动调用点，本文件仍提供 PyTorch3D 风格的最小 API（MeshRenderer/MeshRasterizer 等）。
# ============================================================

try:
    import nvdiffrast.torch as dr  # type: ignore
    _NVDIFFRAST_IMPORT_ERROR: Exception | None = None
except Exception as _e:
    dr = None  # type: ignore
    _NVDIFFRAST_IMPORT_ERROR = _e

# nvdiffrast path (no pytorch3d dependency at runtime)
MeshRenderer = Any
MeshRasterizer = Any
SoftSilhouetteShader = Any
RasterizationSettings = Any
BlendParams = Any
PointLights = Any
FoVPerspectiveCameras = Any
Meshes = Any
HardPhongShader = Any

# One-time warning flags (avoid log spam)
_WARNED_SIL_FALLBACK = False


@dataclass
class RasterizationSettings:  # noqa: D401
    """Minimal settings holder compatible with existing call sites."""

    image_size: tuple[int, int]
    blur_radius: float = 0.0
    faces_per_pixel: int = 1
    bin_size: int = 0
    max_faces_per_bin: int = 0


@dataclass
class BlendParams:
    sigma: float = 1e-4
    gamma: float = 1e-4


class SoftSilhouetteShader:
    def __init__(self, blend_params: BlendParams | None = None):
        self.blend_params = blend_params


class HardPhongShader:
    def __init__(self, *args, **kwargs):
        pass


class PointLights:
    def __init__(self, *args, **kwargs):
        pass


class FoVPerspectiveCameras:
    def __init__(self, device: torch.device, R: torch.Tensor | None = None, T: torch.Tensor | None = None):
        self.device = device
        self.R = R
        self.T = T


@dataclass
class MeshData:
    verts: torch.Tensor  # (V,3)
    faces: torch.Tensor  # (F,3) int64/long
    verts_normals: torch.Tensor  # (V,3)
    verts_colors: torch.Tensor | None = None  # (V,3) in [0,1]

    # Keep a PyTorch3D-like minimal API to avoid touching too many call sites.
    def verts_packed(self) -> torch.Tensor:
        return self.verts

    def faces_packed(self) -> torch.Tensor:
        return self.faces

    def verts_normals_packed(self) -> torch.Tensor:
        return self.verts_normals

    def faces_normals_packed(self) -> torch.Tensor:
        """
        Match PyTorch3D Meshes.faces_normals_packed(): returns per-face normals in WORLD space.
        Shape: (F,3), unit-length where possible.
        """
        v = self.verts_packed()
        f = self.faces_packed().long()
        v0 = v[f[:, 0]]
        v1 = v[f[:, 1]]
        v2 = v[f[:, 2]]
        n = torch.cross(v1 - v0, v2 - v0, dim=-1)
        n = torch.nn.functional.normalize(n, dim=-1, eps=1e-8)
        return n

    def extend(self, _B: int) -> "MeshData":
        # PyTorch3D Meshes.extend(B) is used for batching; for nvdiffrast we batch via R/T.
        return self


def look_at_view_transform(  # type: ignore
    dist: torch.Tensor | float,
    elev: torch.Tensor | float,
    azim: torch.Tensor | float,
    at: torch.Tensor,
    device: torch.device | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    nvdiffrast backend replacement for PyTorch3D look_at_view_transform (API-compatible).
    Returns world-to-view (R,T) where p_view = p_world @ R^T + T.
    Coordinate convention follows common OpenGL style (camera looks along -Z).
    """
    if device is None:
        device = at.device
    dist_t = dist if isinstance(dist, torch.Tensor) else torch.tensor([float(dist)], device=device)
    elev_t = elev if isinstance(elev, torch.Tensor) else torch.tensor([float(elev)], device=device)
    azim_t = azim if isinstance(azim, torch.Tensor) else torch.tensor([float(azim)], device=device)
    if dist_t.dim() == 0:
        dist_t = dist_t[None]
    if elev_t.dim() == 0:
        elev_t = elev_t[None]
    if azim_t.dim() == 0:
        azim_t = azim_t[None]
    B = int(dist_t.shape[0])
    at_b = at
    if at_b.dim() == 1:
        at_b = at_b.view(1, 3).expand(B, 3)
    elif at_b.shape[0] != B:
        at_b = at_b.expand(B, 3)

    elev_r = elev_t * math.pi / 180.0
    azim_r = azim_t * math.pi / 180.0
    x = dist_t * torch.cos(elev_r) * torch.sin(azim_r)
    y = dist_t * torch.sin(elev_r)
    z = dist_t * torch.cos(elev_r) * torch.cos(azim_r)
    cam = torch.stack([x, y, z], dim=-1) + at_b  # (B,3)

    up = torch.tensor([0.0, 1.0, 0.0], device=device).view(1, 3).expand(B, 3)
    z_axis = torch.nn.functional.normalize(cam - at_b, dim=-1)  # backward
    x_axis = torch.nn.functional.normalize(torch.cross(up, z_axis, dim=-1), dim=-1)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)
    R = torch.stack([x_axis, y_axis, z_axis], dim=1)  # (B,3,3) rows
    T = -(torch.bmm(R, cam.view(B, 3, 1)).view(B, 3))
    return R, T


def interpolate_face_attributes(  # type: ignore
    pix_to_face: torch.Tensor,
    bary_coords: torch.Tensor,
    face_attributes: torch.Tensor,
) -> torch.Tensor:
    """
    nvdiffrast backend replacement for PyTorch3D interpolate_face_attributes (API-compatible).
    Supports K=1 (faces_per_pixel==1) which is what this script uses for aux/depth/normals.
    Inputs:
      - pix_to_face: (B,H,W,K)
      - bary_coords: (B,H,W,K,3)
      - face_attributes: (F,3,C)
    Returns:
      - out: (B,H,W,K,C)
    """
    if pix_to_face.dim() != 4:
        raise ValueError(f"pix_to_face must be (B,H,W,K), got {tuple(pix_to_face.shape)}")
    if bary_coords.dim() != 5:
        raise ValueError(f"bary_coords must be (B,H,W,K,3), got {tuple(bary_coords.shape)}")
    B, H, W, K = pix_to_face.shape
    if K != 1:
        # This script only uses K=1 in aux rasterization. Keep it simple & explicit.
        raise ValueError("nvdiffrast interpolate_face_attributes only supports K=1 in this script.")
    face_idx = pix_to_face[..., 0].long()  # (B,H,W)
    valid = face_idx >= 0
    face_idx_safe = torch.where(valid, face_idx, torch.zeros_like(face_idx))
    # Safety: clamp face index to valid range in case of unexpected rasterization outputs.
    # This prevents CUDA gather OOB asserts which would poison the CUDA context.
    try:
        F = int(face_attributes.shape[0])
        if F > 0:
            face_idx_safe = face_idx_safe.clamp(0, F - 1)
    except Exception:
        pass
    attrs = face_attributes.index_select(0, face_idx_safe.reshape(-1)).view(B, H, W, 3, -1)  # (B,H,W,3,C)
    bc = bary_coords[..., 0, :].unsqueeze(-1)  # (B,H,W,3,1)
    out = (attrs * bc).sum(dim=3)  # (B,H,W,C)
    out = out.unsqueeze(3)  # (B,H,W,1,C)
    out = out * valid.unsqueeze(-1).unsqueeze(-1)
    return out


class _NVFragments:
    def __init__(
        self,
        *,
        pix_to_face: torch.Tensor,
        bary_coords: torch.Tensor,
        zbuf: torch.Tensor,
        rast: torch.Tensor,
        pos_clip: torch.Tensor,
        tri: torch.Tensor,
    ):
        self.pix_to_face = pix_to_face
        self.bary_coords = bary_coords
        self.zbuf = zbuf
        # Keep raw nvdiffrast outputs for antialias/shading.
        self.rast = rast
        self.pos_clip = pos_clip
        self.tri = tri


def _nv_project(verts_view: torch.Tensor, fov_y_deg: float, aspect: float, znear: float, zfar: float) -> torch.Tensor:
    """
    verts_view: (B,V,3) in a camera coordinate where points in front have negative z (OpenGL style).
    Returns pos_clip: (B,V,4)
    """
    fovy = float(fov_y_deg) * math.pi / 180.0
    f = 1.0 / max(1e-8, math.tan(0.5 * fovy))
    # OpenGL projection
    P = torch.tensor(
        [
            [f / float(aspect), 0.0, 0.0, 0.0],
            [0.0, f, 0.0, 0.0],
            [0.0, 0.0, (zfar + znear) / (znear - zfar), (2.0 * zfar * znear) / (znear - zfar)],
            [0.0, 0.0, -1.0, 0.0],
        ],
        dtype=verts_view.dtype,
        device=verts_view.device,
    )
    ones = torch.ones((*verts_view.shape[:-1], 1), dtype=verts_view.dtype, device=verts_view.device)
    v_h = torch.cat([verts_view, ones], dim=-1)  # (B,V,4)
    pos_clip = torch.matmul(v_h, P.T)
    return pos_clip


class MeshRasterizer:
    def __init__(self, cameras: FoVPerspectiveCameras, raster_settings: RasterizationSettings):
        if dr is None:
            raise ImportError(
                f"nvdiffrast is required but failed to import: {_NVDIFFRAST_IMPORT_ERROR}"
            )
        self.cameras = cameras
        self.raster_settings = raster_settings
        self._ctx = dr.RasterizeCudaContext()

    def __call__(self, meshes_world: MeshData, cameras: FoVPerspectiveCameras | None = None, R: torch.Tensor | None = None, T: torch.Tensor | None = None):
        cams = cameras if cameras is not None else self.cameras
        Rb = R if R is not None else cams.R
        Tb = T if T is not None else cams.T
        if Rb is None or Tb is None:
            raise ValueError("Rasterizer requires R/T either via args or cameras.")
        # verts_world: (V,3) -> (B,V,3)
        verts = meshes_world.verts_packed()
        if verts.dim() != 2:
            raise ValueError("MeshData.verts must be (V,3)")
        B = int(Rb.shape[0])
        V = int(verts.shape[0])
        verts_b = verts.view(1, V, 3).expand(B, V, 3)
        verts_view = torch.bmm(verts_b, Rb.transpose(1, 2)) + Tb.view(B, 1, 3)

        H, W = int(self.raster_settings.image_size[0]), int(self.raster_settings.image_size[1])
        aspect = float(W) / float(max(1, H))
        fov_y_deg = float(os.environ.get("VAL_FOVY_DEG", "60.0"))
        znear = float(os.environ.get("VAL_ZNEAR", "0.01"))
        zfar = float(os.environ.get("VAL_ZFAR", "10000.0"))
        pos_clip = _nv_project(verts_view, fov_y_deg=fov_y_deg, aspect=aspect, znear=znear, zfar=zfar)

        tri = meshes_world.faces_packed().to(torch.int32)
        rast, _db = dr.rasterize(self._ctx, pos_clip, tri, resolution=[H, W])
        # rast: (B,H,W,4) -> [u,v,z,tri_id]
        tri_id_raw = rast[..., 3].long()
        # Normalize nvdiffrast primitive id convention:
        # - Common convention: rast[...,3] = prim_id + 1, background = 0
        # - Some builds may already use background = -1 and prim_id in [0..F-1]
        # We convert everything to: background = -1, prim_id in [0..F-1]
        tri_id = tri_id_raw
        try:
            mn = int(tri_id_raw.min().item())
            # If there is no negative value, treat 0 as background and shift by -1.
            if mn >= 0:
                tri_id = tri_id_raw - 1
        except Exception:
            # fallback: assume common convention
            tri_id = tri_id_raw - 1
        pix_to_face = tri_id.unsqueeze(-1)  # (B,H,W,1)
        u = rast[..., 0]
        v = rast[..., 1]
        w_bc = 1.0 - u - v
        bary = torch.stack([w_bc, u, v], dim=-1).unsqueeze(-2)  # (B,H,W,1,3)
        # Depth buffer = METRIC view-space depth (positive distance along the optical axis,
        # in mesh units), obtained by perspective-correct interpolation of the per-vertex
        # view depth. (rast[...,2] is the non-linear NDC z/w and is NOT a metric depth; it is
        # kept in fr.rast for antialiasing only.)
        try:
            depth_attr = (-verts_view[..., 2:3]).contiguous()  # (B,V,1), camera looks along -Z
            zbuf, _ = dr.interpolate(depth_attr, rast, tri)  # type: ignore  # (B,H,W,1)
            zbuf = zbuf * (tri_id >= 0).to(zbuf.dtype).unsqueeze(-1)
        except Exception:
            zbuf = rast[..., 2].unsqueeze(-1)  # (B,H,W,1) fallback (should not happen)
        if int(os.environ.get("VAL_DEBUG_RAST", "1")) != 0:
            try:
                mn = int(tri_id.min().item())
                mx = int(tri_id.max().item())
                vis = float((tri_id >= 0).float().mean().item())
                print(f"[RastDebug] tri_id range=[{mn},{mx}] vis_ratio={vis:.4f} res=({W}x{H})")
            except Exception:
                pass
        return _NVFragments(pix_to_face=pix_to_face, bary_coords=bary, zbuf=zbuf, rast=rast, pos_clip=pos_clip, tri=tri)


class MeshRenderer:
    def __init__(self, rasterizer: MeshRasterizer, shader: Any | None = None):
        self.rasterizer = rasterizer
        self.shader = shader

    def __call__(self, meshes_world: MeshData, cameras: FoVPerspectiveCameras | None = None, R: torch.Tensor | None = None, T: torch.Tensor | None = None):
        fr = self.rasterizer(meshes_world=meshes_world, cameras=cameras, R=R, T=T)
        tri_id = fr.pix_to_face[..., 0]
        mask = (tri_id >= 0).float()  # (B,H,W)
        B, H, W = mask.shape
        # nvdiffrast antialias:
        # - For silhouette gradients, antialias is usually IMPORTANT (gives useful edge gradients).
        # - For some driver/arch combos, antialias on RGBA may be unstable; keep a dedicated knob.
        sil_aa = int(os.environ.get("VAL_SIL_AA", "1")) != 0
        rgb_aa = int(os.environ.get("VAL_NV_AA", "0")) != 0

        # If this renderer was built for silhouette (SoftSilhouetteShader), output RGBA with alpha.
        if isinstance(self.shader, SoftSilhouetteShader):
            # IMPORTANT:
            # Do NOT use hard mask (tri_id>=0) as silhouette when gradients are needed.
            # Instead, rasterize + interpolate a constant 1 attribute, and (optionally) antialias it.
            rgb = torch.zeros((B, H, W, 3), device=fr.pos_clip.device, dtype=fr.pos_clip.dtype)
            alpha = None
            try:
                # Attribute is ones per-vertex; interpolation produces coverage-like silhouette.
                V = int(fr.pos_clip.shape[1])
                ones = torch.ones((B, V, 1), device=fr.pos_clip.device, dtype=fr.pos_clip.dtype)
                alpha, _db = dr.interpolate(ones, fr.rast, fr.tri)  # type: ignore  # (B,H,W,1)
                alpha = torch.clamp(alpha, 0.0, 1.0)
                if sil_aa:
                    alpha = dr.antialias(alpha, fr.rast, fr.pos_clip, fr.tri)  # type: ignore
                alpha = _postprocess_silhouette_alpha(alpha.squeeze(-1)).unsqueeze(-1)
            except Exception as _e:
                # Fallback (non-differentiable) silhouette.
                global _WARNED_SIL_FALLBACK
                if (not _WARNED_SIL_FALLBACK) and (int(os.environ.get("VAL_WARN_SIL_FALLBACK", "1")) != 0):
                    _WARNED_SIL_FALLBACK = True
                    print(
                        f"[Warning] Silhouette rasterization fell back to hard mask (non-differentiable). "
                        f"Gradients wrt pose may degrade. Error: {_e}"
                    )
                alpha = _postprocess_silhouette_alpha(mask).unsqueeze(-1).to(dtype=rgb.dtype)

            rgba = torch.cat([rgb, alpha], dim=-1)
            return rgba

        # Otherwise, approximate "phong renderer" as vertex-color interpolation for visualization.
        rgb = torch.zeros((B, H, W, 3), device=mask.device, dtype=mask.dtype)
        try:
            vcol = getattr(meshes_world, "verts_colors", None)
            if vcol is not None:
                faces = meshes_world.faces_packed()
                faces_vcol = vcol[faces]  # (F,3,3)
                pix_col = interpolate_face_attributes(fr.pix_to_face, fr.bary_coords, faces_vcol)[..., 0, :][0]  # (H,W,3) for B=1
                if B == 1:
                    rgb = pix_col.unsqueeze(0) * mask.unsqueeze(-1)
                else:
                    # generalized gather for B>1
                    pix_col_b = interpolate_face_attributes(fr.pix_to_face, fr.bary_coords, faces_vcol)[..., 0, :]
                    rgb = pix_col_b * mask.unsqueeze(-1)
        except Exception:
            pass

        alpha = mask.unsqueeze(-1)
        rgba = torch.cat([rgb, alpha], dim=-1)
        if rgb_aa:
            try:
                rgba = dr.antialias(rgba, fr.rast, fr.pos_clip, fr.tri)  # type: ignore
            except Exception:
                pass
        return rgba

# ----------------------------
# Camera helper functions
# ----------------------------

def _camera_position_from_spherical_global(
    dist: torch.Tensor, elev_deg: torch.Tensor, azim_deg: torch.Tensor, at_xyz: torch.Tensor
) -> torch.Tensor:
    """
    Convert spherical (dist, elev_deg, azim_deg) to camera position xyz in world coordinates.
    at_xyz: (3,)
    """
    elev = elev_deg * math.pi / 180.0
    azim = azim_deg * math.pi / 180.0
    x = dist * torch.cos(elev) * torch.sin(azim)
    y = dist * torch.sin(elev)
    z = dist * torch.cos(elev) * torch.cos(azim)
    return torch.stack([x, y, z], dim=0) + at_xyz


def _spherical_from_camera_position_global(cam_xyz: torch.Tensor, at_xyz: torch.Tensor):
    """
    cam_xyz, at_xyz: (3,)
    Return (dist, elev_deg, azim_deg) as python floats.
    """
    v = cam_xyz - at_xyz
    d = torch.norm(v).clamp(min=1e-8)
    x, y, z = v[0] / d, v[1] / d, v[2] / d
    elev = torch.asin(y)
    azim = torch.atan2(x, z)
    return float(d.detach().cpu().item()), float(elev.detach().cpu().item() * 180.0 / math.pi), float(azim.detach().cpu().item() * 180.0 / math.pi)


def _postprocess_silhouette_alpha(alpha: torch.Tensor) -> torch.Tensor:
    """
    Silhouette render modes for DR baseline comparison (see BENCHMARK_BASELINES.md):
      - nvdiffrast_aa (default): interpolate + nvdiffrast antialias (edge gradients)
      - nvdiffrast_hard: hard mask fallback path (VAL_SIL_AA=0)
      - gaussian_soft: extra isotropic blur on alpha (SoftRas-like smooth occupancy)
    """
    mode = os.environ.get("VAL_SIL_RENDER_MODE", "nvdiffrast_aa").strip().lower()
    a = torch.clamp(alpha, 0.0, 1.0)
    if mode in ("gaussian_soft", "softras_like", "soft"):
        sigma = float(os.environ.get("VAL_SIL_SOFT_BLUR_SIGMA", "1.5"))
        if sigma > 1e-6 and a.dim() == 2:
            k = int(max(3, round(sigma * 4) | 1))
            x = torch.arange(k, device=a.device, dtype=a.dtype) - (k - 1) / 2.0
            g = torch.exp(-0.5 * (x / sigma) ** 2)
            g = (g / g.sum()).view(1, 1, k)
            pad = k // 2
            aa = a.unsqueeze(0).unsqueeze(0)
            aa = torch.nn.functional.conv2d(aa, g.unsqueeze(0), padding=(0, pad))
            aa = torch.nn.functional.conv2d(aa, g.unsqueeze(0).transpose(-1, -2), padding=(pad, 0))
            a = aa.squeeze(0).squeeze(0).clamp(0.0, 1.0)
    return a


# Occlusion-aware silhouette term (real-data variant, bop_full6dof_eval.py --occlusion-aware). SIL_SCENE_DEPTH is the
# metric depth of the WHOLE observed scene in the observation frame (same units / orientation as ref_depth). A rendered
# pixel outside the visible mask is excluded from the silhouette MSE / IoU when the observed scene surface lies in
# front of the hypothesised object surface by more than SIL_OCC_MARGIN, i.e. when an occluder explains why the object
# is not visible there. Default None -> no change to any loss (synthetic benchmark unaffected).
SIL_SCENE_DEPTH = None
SIL_OCC_MARGIN = 0.0


def _occlusion_aware_sil(alpha, ref, vis_mask, zbuf, w_iou, loss_scale, dims):
    sc = SIL_SCENE_DEPTH
    if sc is None or tuple(sc.shape[-2:]) != tuple(alpha.shape[-2:]):
        return None
    sc = sc.to(device=alpha.device, dtype=alpha.dtype)
    occl = (vis_mask > 0.5) & (ref < 0.5) & (sc > 0) & (sc < zbuf - float(SIL_OCC_MARGIN))
    keep = (~occl).to(alpha.dtype)
    a, r = alpha * keep, ref * keep
    if dims is None:
        mse = torch.mean((a - r) ** 2)
        inter = (a * r).sum()
        union = a.sum() + r.sum() - inter
    else:
        mse = torch.mean((a - r) ** 2, dim=dims)
        inter = torch.sum(a * r, dim=dims)
        union = torch.sum(a, dim=dims) + torch.sum(r, dim=dims) - inter
    iou = 1.0 - (inter / (union + 1e-8))
    return (1.0 - w_iou) * (mse * loss_scale) + w_iou * iou


def _evaluate_pose_loss_no_grad(
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    at_xyz: torch.Tensor,
    dist_v: float,
    elev_v: float,
    azim_v: float,
    device: torch.device,
    progress: float | None = None,
):
    """
    Evaluate the SAME loss components as training (sil + depth + normals) under no_grad.
    Returns dict: total, sil, depth, normal, valid_ratio
    """
    with torch.no_grad():
        R, T = look_at_view_transform(
            dist=torch.tensor([dist_v], device=device),
            elev=torch.tensor([elev_v], device=device),
            azim=torch.tensor([azim_v], device=device),
            at=at_xyz.view(1, 3),
            device=device,
        )
        # silhouette
        img = silhouette_renderer(meshes_world=mesh, R=R, T=T)
        alpha = _postprocess_silhouette_alpha(torch.clamp(img[0, ..., 3], 0, 1))
        ref = torch.clamp(image_ref_silhouette, 0, 1)
        # Be robust: only treat last channel as alpha when ref is truly RGBA (...,4).
        # (If ref is e.g. (1,H,W) or (H,W,1), indexing [...,3] would crash.)
        if ref.dim() == 3 and ref.shape[-1] == 4:
            ref = ref[..., 3]
        ref = ref.squeeze()
        loss_mse = torch.mean((alpha - ref) ** 2)
        inter = (alpha * ref).sum()
        union = alpha.sum() + ref.sum() - inter
        loss_iou = 1.0 - (inter / (union + 1e-8))
        w_iou = float(os.environ.get("VAL_W_IOU", "0.1"))
        scale_cfg = os.environ.get("VAL_LOSS_SCALE", "1.0")
        if scale_cfg.strip().lower() == "auto":
            h = int(ref.shape[0])
            w = int(ref.shape[1])
            loss_scale = float(h * w)
        else:
            loss_scale = float(scale_cfg)
        loss_sil = (1.0 - w_iou) * (loss_mse * loss_scale) + w_iou * loss_iou

        # depth + normals
        cams = FoVPerspectiveCameras(device=device, R=R, T=T)
        fr = rasterizer(meshes_world=mesh, cameras=cams)
        vis_mask = (fr.pix_to_face[..., 0] >= 0).float()[0]
        depth = (fr.zbuf[..., 0][0] * vis_mask)
        if SIL_SCENE_DEPTH is not None:
            _ls = _occlusion_aware_sil(alpha, ref, vis_mask, fr.zbuf[..., 0][0], w_iou, loss_scale, None)
            if _ls is not None:
                loss_sil = _ls

        ref_d = ref_depth.squeeze()
        ref_n = ref_normals.squeeze()
        valid = (vis_mask > 0.5) & (ref_d > 0)
        valid_ratio = float(valid.float().mean().detach().cpu().item())

        if valid.any():
            beta = float(os.environ.get("VAL_DEPTH_BETA", "0.05"))
            depth_norm_eps = float(os.environ.get("VAL_DEPTH_NORM_EPS", "1e-6"))
            ref_scale = ref_d[valid].mean().detach().clamp(min=depth_norm_eps)
            depth_n = depth / ref_scale
            ref_depth_n = ref_d / ref_scale
            loss_depth = torch.nn.functional.smooth_l1_loss(depth_n[valid], ref_depth_n[valid], beta=beta, reduction="mean")

            verts_normals = mesh.verts_normals_packed()
            faces = mesh.faces_packed()
            verts_normals_view = torch.matmul(verts_normals, R[0].T)
            faces_verts_normals = verts_normals_view[faces]
            pix_normals = interpolate_face_attributes(fr.pix_to_face, fr.bary_coords, faces_verts_normals)
            normals = pix_normals[..., 0, :][0]
            normals = torch.nn.functional.normalize(normals, dim=-1) * vis_mask.unsqueeze(-1)
            cos = (normals * ref_n).sum(dim=-1).clamp(-1, 1)
            loss_normal = (1.0 - cos)[valid].mean()
        else:
            loss_depth = torch.tensor(0.0, device=device)
            loss_normal = torch.tensor(0.0, device=device)

        # progress=None -> v9 START weights; progress=1.0 -> END weights (sil/depth/normal schedule)
        if progress is None:
            w_sil = float(os.environ.get("VAL_W_SIL_START", "2.0"))
            w_depth = float(os.environ.get("VAL_W_DEPTH_START", "0.5"))
            w_normal = float(os.environ.get("VAL_W_NORMAL_START", "0.5"))
        else:
            w_sil, w_depth, w_normal = _loss_weights_at_progress(float(progress))
        # Valid-ratio penalty: penalise poses where the rendered object barely
        # overlaps with the reference visible region.  Without this, a wrong
        # viewpoint whose silhouette does not intersect the GT can get
        # artificially low depth/normal losses (computed on almost no pixels).
        w_vp = float(os.environ.get("VAL_W_VALID_PENALTY", "0.0"))
        valid_penalty = w_vp * (1.0 - float(valid_ratio)) if w_vp > 0 else 0.0
        total = w_sil * loss_sil + w_depth * loss_depth + w_normal * loss_normal + valid_penalty

        return {
            "loss_total": float(total.detach().cpu().item()),
            "loss_sil": float(loss_sil.detach().cpu().item()),
            "loss_depth": float(loss_depth.detach().cpu().item()),
            "loss_normal": float(loss_normal.detach().cpu().item()),
            "valid_ratio": valid_ratio,
            "R": R[0].detach(),
            "T": T[0].detach(),
        }


def _loss_weights_at_progress(p: float) -> tuple[float, float, float]:
    """
    Match SphericalCameraModel's weight schedule in optimize_to_target.
    Returns (w_sil, w_depth, w_normal).
    """
    use_sched = int(os.environ.get("VAL_LOSS_SCHEDULE", "1")) != 0
    if use_sched:
        p = float(np.clip(float(p), 0.0, 1.0))
        w_sil0 = float(os.environ.get("VAL_W_SIL_START", "2.0"))
        w_sil1 = float(os.environ.get("VAL_W_SIL_END", "0.2"))
        w_depth0 = float(os.environ.get("VAL_W_DEPTH_START", "0.5"))
        w_depth1 = float(os.environ.get("VAL_W_DEPTH_END", "2.0"))
        w_norm0 = float(os.environ.get("VAL_W_NORMAL_START", "0.5"))
        w_norm1 = float(os.environ.get("VAL_W_NORMAL_END", "1.0"))
        w_sil = (1.0 - p) * w_sil0 + p * w_sil1
        w_depth = (1.0 - p) * w_depth0 + p * w_depth1
        w_normal = (1.0 - p) * w_norm0 + p * w_norm1
        return float(w_sil), float(w_depth), float(w_normal)
    return (
        float(os.environ.get("VAL_W_SIL", "0.2")),
        float(os.environ.get("VAL_W_DEPTH", "2.0")),
        float(os.environ.get("VAL_W_NORMAL", "1.0")),
    )


def _evaluate_pose_loss_rt_no_grad(
    *,
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    R_w2v: torch.Tensor,
    T_w2v: torch.Tensor,
    device: torch.device,
    progress: float = 1.0,
) -> dict[str, float | torch.Tensor]:
    """
    Evaluate full loss (sil + depth + normals) given explicit world-to-view (R,T).
    This is used for:
    - A: ranking refined candidates without re-imposing look-at
    - E: SE(3) refinement evaluation
    """
    with torch.no_grad():
        Rb = R_w2v.view(1, 3, 3).to(device)
        Tb = T_w2v.view(1, 3).to(device)
        img = silhouette_renderer(meshes_world=mesh, R=Rb, T=Tb)
        alpha = _postprocess_silhouette_alpha(torch.clamp(img[0, ..., 3], 0, 1))
        ref = torch.clamp(image_ref_silhouette, 0, 1)
        if ref.dim() == 3 and ref.shape[-1] == 4:
            ref = ref[..., 3]
        ref = ref.squeeze()
        loss_mse = torch.mean((alpha - ref) ** 2)
        inter = (alpha * ref).sum()
        union = alpha.sum() + ref.sum() - inter
        loss_iou = 1.0 - (inter / (union + 1e-8))
        w_iou = float(os.environ.get("VAL_W_IOU", "0.1"))
        scale_cfg = os.environ.get("VAL_LOSS_SCALE", "1.0")
        if scale_cfg.strip().lower() == "auto":
            loss_scale = float(ref.shape[0] * ref.shape[1])
        else:
            loss_scale = float(scale_cfg)
        loss_sil = (1.0 - w_iou) * (loss_mse * loss_scale) + w_iou * loss_iou

        cams = FoVPerspectiveCameras(device=device, R=Rb, T=Tb)
        fr = rasterizer(meshes_world=mesh, cameras=cams)
        vis_mask = (fr.pix_to_face[..., 0] >= 0).float()[0]
        depth = (fr.zbuf[..., 0][0] * vis_mask)

        ref_d = ref_depth.squeeze()
        ref_n = ref_normals.squeeze()
        valid = (vis_mask > 0.5) & (ref_d > 0)
        valid_ratio = float(valid.float().mean().detach().cpu().item())
        if valid.any():
            beta = float(os.environ.get("VAL_DEPTH_BETA", "0.05"))
            depth_norm_eps = float(os.environ.get("VAL_DEPTH_NORM_EPS", "1e-6"))
            ref_scale = ref_d[valid].mean().detach().clamp(min=depth_norm_eps)
            depth_n = depth / ref_scale
            ref_depth_n = ref_d / ref_scale
            loss_depth = torch.nn.functional.smooth_l1_loss(depth_n[valid], ref_depth_n[valid], beta=beta, reduction="mean")

            verts_normals = mesh.verts_normals_packed()
            faces = mesh.faces_packed()
            verts_normals_view = torch.matmul(verts_normals, Rb[0].T)
            faces_verts_normals = verts_normals_view[faces]
            pix_normals = interpolate_face_attributes(fr.pix_to_face, fr.bary_coords, faces_verts_normals)
            normals = pix_normals[..., 0, :][0]
            normals = torch.nn.functional.normalize(normals, dim=-1) * vis_mask.unsqueeze(-1)
            cos = (normals * ref_n).sum(dim=-1).clamp(-1, 1)
            loss_normal = (1.0 - cos)[valid].mean()
        else:
            loss_depth = torch.tensor(0.0, device=device)
            loss_normal = torch.tensor(0.0, device=device)

        w_sil, w_depth, w_normal = _loss_weights_at_progress(float(progress))
        total = w_sil * loss_sil + w_depth * loss_depth + w_normal * loss_normal
        return {
            "loss_total": float(total.detach().cpu().item()),
            "loss_sil": float(loss_sil.detach().cpu().item()),
            "loss_depth": float(loss_depth.detach().cpu().item()),
            "loss_normal": float(loss_normal.detach().cpu().item()),
            "valid_ratio": float(valid_ratio),
            "R": Rb[0].detach(),
            "T": Tb[0].detach(),
        }


def _evaluate_spherical_visual_score_no_grad(
    *,
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    phong_renderer: MeshRenderer | None,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    image_ref_rgb: torch.Tensor | None,
    at_xyz: torch.Tensor,
    dist_v: float,
    elev_v: float,
    azim_v: float,
    device: torch.device,
    gt_elev: float | None = None,
    ref_elev: float | None = None,
    ref_azim: float | None = None,
    ref_dist: float | None = None,
) -> dict[str, float]:
    """
    Final candidate selector used after LocalBeam.
    It keeps the previous look-at pose family, but re-ranks candidates at full
    resolution with geometry loss plus optional RGB/texture consistency.  This
    helps avoid choosing a visually wrong symmetric basin with a low silhouette
    or depth/normal loss.
    """
    if _strict_no_gt():
        # GT elevation must never act as a tie-breaker in the final selection.
        gt_elev = None
    with torch.no_grad():
        ev = _evaluate_pose_loss_no_grad(
            mesh=mesh,
            silhouette_renderer=silhouette_renderer,
            rasterizer=rasterizer,
            image_ref_silhouette=image_ref_silhouette,
            ref_depth=ref_depth,
            ref_normals=ref_normals,
            at_xyz=at_xyz,
            dist_v=float(dist_v),
            elev_v=float(elev_v),
            azim_v=float(azim_v),
            device=device,
        )
        loss_rgb = torch.tensor(0.0, device=device, dtype=torch.float32)
        if phong_renderer is not None and image_ref_rgb is not None:
            Rb = ev["R"].view(1, 3, 3).to(device)
            Tb = ev["T"].view(1, 3).to(device)
            cams = FoVPerspectiveCameras(device=device, R=Rb, T=Tb)
            rgb = torch.clamp(phong_renderer(meshes_world=mesh, cameras=cams)[0, ..., :3], 0, 1)
            ref_rgb = torch.clamp(image_ref_rgb.to(device), 0, 1)
            if ref_rgb.dim() == 4:
                ref_rgb = ref_rgb[0]
            ref_sil = torch.clamp(image_ref_silhouette, 0, 1)
            if ref_sil.dim() == 3 and ref_sil.shape[-1] == 4:
                ref_sil = ref_sil[..., 3]
            ref_sil = ref_sil.squeeze().to(device)
            rgba = silhouette_renderer(meshes_world=mesh, R=Rb, T=Tb)[0]
            alpha = torch.clamp(rgba[..., 3], 0, 1)
            mask = ((ref_sil > 0.05) | (alpha > 0.05)).float().unsqueeze(-1)
            if mask.sum() > 0:
                loss_rgb = (((rgb - ref_rgb) ** 2) * mask).sum() / (mask.sum() * 3.0 + 1e-8)
        try:
            _mr_sel = float(_mesh_radius_from_verts(mesh.verts_packed(), at_xyz))
        except Exception:
            _mr_sel = _mesh_scale_ref_radius()
        w_rgb = _rgb_weight_for_mesh_radius(_mr_sel, "VAL_SELECT_W_RGB", 8.0)
        w_valid = float(os.environ.get("VAL_SELECT_W_VALID", "0.05"))
        w_elev_proximity = float(os.environ.get("VAL_SELECT_ELEV_PROXIMITY_WEIGHT", "0.3"))
        w_ref_elev = float(os.environ.get("VAL_SELECT_REF_ELEV_WEIGHT", "0.04"))
        w_ref_azim = float(os.environ.get("VAL_SELECT_REF_AZIM_WEIGHT", "0.022"))
        w_ref_dist = float(os.environ.get("VAL_SELECT_REF_DIST_WEIGHT", "0.8"))
        
        # Elevation proximity tie-breaker (GT for debugging, ref-view for inference).
        elev_proximity_bonus = 0.0
        elev_anchor = ref_elev if ref_elev is not None else gt_elev
        if elev_anchor is not None:
            elev_diff = abs(float(elev_v) - float(elev_anchor))
            elev_proximity_bonus = max(0.0, 1.0 - elev_diff / 90.0)
        
        score = float(ev["loss_total"]) + w_rgb * float(loss_rgb.detach().cpu().item()) + w_valid * (1.0 - float(ev.get("valid_ratio", 0.0))) - w_elev_proximity * elev_proximity_bonus
        if ref_elev is not None:
            high_thr = float(os.environ.get("VAL_HIGH_ELEV_UNDER_THRESHOLD", "58.0"))
            low_thr = float(os.environ.get("VAL_LOW_ELEV_UNDER_THRESHOLD", "45.0"))
            if float(ref_elev) >= high_thr:
                score += w_ref_elev * max(0.0, float(ref_elev) - float(elev_v))
            elif float(ref_elev) <= low_thr:
                score += w_ref_elev * max(0.0, float(ref_elev) - float(elev_v)) * 2.0
                score += w_ref_elev * max(0.0, float(elev_v) - float(ref_elev)) * 0.35
            else:
                score += w_ref_elev * abs(float(elev_v) - float(ref_elev))
        if ref_azim is not None:
            score += w_ref_azim * _azim_diff_deg(float(azim_v), float(ref_azim))
        if ref_dist is not None:
            score += w_ref_dist * abs(float(dist_v) - float(ref_dist))
        return {
            "score": float(score),
            "loss_total": float(ev["loss_total"]),
            "loss_sil": float(ev["loss_sil"]),
            "loss_depth": float(ev["loss_depth"]),
            "loss_normal": float(ev["loss_normal"]),
            "loss_rgb": float(loss_rgb.detach().cpu().item()),
            "valid_ratio": float(ev.get("valid_ratio", 0.0)),
            "elev_proximity_bonus": float(elev_proximity_bonus),
        }


def _skew_symmetric(v: torch.Tensor) -> torch.Tensor:
    v = v.view(3)
    return torch.stack(
        [
            torch.stack([torch.tensor(0.0, device=v.device, dtype=v.dtype), -v[2], v[1]]),
            torch.stack([v[2], torch.tensor(0.0, device=v.device, dtype=v.dtype), -v[0]]),
            torch.stack([-v[1], v[0], torch.tensor(0.0, device=v.device, dtype=v.dtype)]),
        ],
        dim=0,
    )


def _so3_exp(w: torch.Tensor) -> torch.Tensor:
    """
    Exponential map from so(3) to SO(3).
    w: (3,) axis-angle vector (radians)
    """
    w = w.view(3)
    theta = torch.norm(w).clamp(min=1e-12)
    K = _skew_symmetric(w / theta)
    I = torch.eye(3, device=w.device, dtype=w.dtype)
    s = torch.sin(theta)
    c = torch.cos(theta)
    R = I + s * K + (1.0 - c) * (K @ K)
    # small-angle fallback (first-order)
    small = theta < 1e-4
    if bool(small.detach().cpu().item()):
        R = I + _skew_symmetric(w)
    return R


def se3_refine_pose(
    *,
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    R_init_w2v: torch.Tensor,
    T_init_w2v: torch.Tensor,
    device: torch.device,
    iters: int = 200,
    lr: float = 0.01,
    progress_for_weights: float = 1.0,
    R_gt_w2v: torch.Tensor | None = None,
    T_gt_w2v: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """
    E. 6DoF SE(3) refinement (left-multiplicative in view coordinates):
      R = R_delta * R0
      T = R_delta * T0 + t_delta
    Returns (R_best, T_best, best_loss).
    """
    iters = max(1, int(iters))
    lr = float(lr)
    R0 = R_init_w2v.to(device=device, dtype=torch.float32)
    T0 = T_init_w2v.to(device=device, dtype=torch.float32).view(3)

    # small updates
    w = torch.zeros((3,), device=device, dtype=torch.float32, requires_grad=True)
    t = torch.zeros((3,), device=device, dtype=torch.float32, requires_grad=True)
    opt = torch.optim.Adam([w, t], lr=lr)
    use_scheduler = int(os.environ.get("VAL_SE3_USE_SCHEDULER", "1")) != 0
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=iters, eta_min=max(lr * 0.05, 1e-6)) if use_scheduler else None

    w_reg = float(os.environ.get("VAL_SE3_W_REG", "1e-3"))
    t_reg = float(os.environ.get("VAL_SE3_T_REG", "1e-3"))
    clip = float(os.environ.get("VAL_CLIP_NORM", "10.0"))
    patience = max(1, int(os.environ.get("VAL_SE3_EARLY_STOP_PATIENCE", "120")))
    min_delta = float(os.environ.get("VAL_SE3_EARLY_STOP_DELTA", "1e-8"))
    use_normal_abs = int(os.environ.get("VAL_SE3_NORMAL_ABS", "1")) != 0
    oracle_pose_w = float(os.environ.get("VAL_SE3_ORACLE_POSE_W", "0.0"))
    oracle_reproj_w = float(os.environ.get("VAL_SE3_ORACLE_REPROJ_W", "0.0"))
    if _strict_no_gt() and (oracle_pose_w > 0.0 or oracle_reproj_w > 0.0):
        raise RuntimeError(
            "[GT-LEAK] se3_refine_pose: oracle pose/reprojection terms use the GT pose and are "
            "not allowed under VAL_STRICT_NO_GT=1."
        )
    if _strict_no_gt():
        R_gt_w2v = None
        T_gt_w2v = None
    oracle_trans_scale = float(os.environ.get("VAL_SE3_ORACLE_TRANS_SCALE", "1.0"))
    oracle_max_verts = max(128, int(os.environ.get("VAL_SE3_ORACLE_REPROJ_VERTS", "4096")))

    best_loss = float("inf")
    best_R = R0.detach().clone()
    best_T = T0.detach().clone()
    stale = 0

    # cached refs
    ref_sil = torch.clamp(image_ref_silhouette, 0, 1)
    if ref_sil.dim() == 3 and ref_sil.shape[-1] == 4:
        ref_sil = ref_sil[..., 3]
    ref_sil = ref_sil.squeeze()
    ref_d = ref_depth.squeeze()
    ref_n = ref_normals.squeeze()

    w_sil, w_depth, w_normal = _loss_weights_at_progress(float(progress_for_weights))

    for _ in range(iters):
        opt.zero_grad(set_to_none=True)
        R_delta = _so3_exp(w)
        R = (R_delta @ R0)
        T = (R_delta @ T0) + t

        # differentiable rendering loss (same family as optimize_to_target)
        img = silhouette_renderer(meshes_world=mesh, R=R.view(1, 3, 3), T=T.view(1, 3))
        alpha = _postprocess_silhouette_alpha(torch.clamp(img[0, ..., 3], 0, 1))
        loss_mse = torch.mean((alpha - ref_sil) ** 2)
        inter = (alpha * ref_sil).sum()
        union = alpha.sum() + ref_sil.sum() - inter
        loss_iou = 1.0 - (inter / (union + 1e-8))
        w_iou = float(os.environ.get("VAL_W_IOU", "0.1"))
        scale_cfg = os.environ.get("VAL_LOSS_SCALE", "1.0")
        if scale_cfg.strip().lower() == "auto":
            loss_scale = float(ref_sil.shape[0] * ref_sil.shape[1])
        else:
            loss_scale = float(scale_cfg)
        loss_sil = (1.0 - w_iou) * (loss_mse * loss_scale) + w_iou * loss_iou

        cams = FoVPerspectiveCameras(device=device, R=R.view(1, 3, 3), T=T.view(1, 3))
        fr = rasterizer(meshes_world=mesh, cameras=cams)
        vis_mask = (fr.pix_to_face[..., 0] >= 0).float()[0]
        depth = (fr.zbuf[..., 0][0] * vis_mask)
        valid = (vis_mask > 0.5) & (ref_d > 0)
        if valid.any():
            beta = float(os.environ.get("VAL_DEPTH_BETA", "0.05"))
            depth_norm_eps = float(os.environ.get("VAL_DEPTH_NORM_EPS", "1e-6"))
            ref_scale = ref_d[valid].mean().detach().clamp(min=depth_norm_eps)
            depth_n = depth / ref_scale
            ref_depth_n = ref_d / ref_scale
            loss_depth = torch.nn.functional.smooth_l1_loss(depth_n[valid], ref_depth_n[valid], beta=beta, reduction="mean")

            verts_normals = mesh.verts_normals_packed()
            faces = mesh.faces_packed()
            verts_normals_view = torch.matmul(verts_normals, R.T)
            faces_verts_normals = verts_normals_view[faces]
            pix_normals = interpolate_face_attributes(fr.pix_to_face, fr.bary_coords, faces_verts_normals)
            normals = pix_normals[..., 0, :][0]
            normals = torch.nn.functional.normalize(normals, dim=-1) * vis_mask.unsqueeze(-1)
            cos = (normals * ref_n).sum(dim=-1).clamp(-1, 1)
            if use_normal_abs:
                cos = torch.abs(cos)
            loss_normal = (1.0 - cos)[valid].mean()
        else:
            loss_depth = torch.tensor(0.0, device=device)
            loss_normal = torch.tensor(0.0, device=device)

        reg = float(w_reg) * (w.pow(2).sum()) + float(t_reg) * (t.pow(2).sum())
        loss_oracle = torch.tensor(0.0, device=device, dtype=torch.float32)
        if oracle_pose_w > 0.0 and R_gt_w2v is not None and T_gt_w2v is not None:
            R_gt = R_gt_w2v.to(device=device, dtype=torch.float32).view(3, 3)
            T_gt = T_gt_w2v.to(device=device, dtype=torch.float32).view(3)
            R_rel = R @ R_gt.T
            trace = torch.trace(R_rel).clamp(-1.0, 3.0)
            rot_cos = ((trace - 1.0) * 0.5).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
            rot_loss = torch.acos(rot_cos).pow(2)
            trans_loss = torch.mean(((T - T_gt) / max(float(oracle_trans_scale), 1e-8)).pow(2))
            loss_oracle = rot_loss + trans_loss
        loss_oracle_reproj = torch.tensor(0.0, device=device, dtype=torch.float32)
        if oracle_reproj_w > 0.0 and R_gt_w2v is not None and T_gt_w2v is not None:
            R_gt = R_gt_w2v.to(device=device, dtype=torch.float32).view(3, 3)
            T_gt = T_gt_w2v.to(device=device, dtype=torch.float32).view(3)
            verts = mesh.verts_packed().to(device=device, dtype=torch.float32)
            if int(verts.shape[0]) > oracle_max_verts:
                step_v = max(1, int(math.ceil(float(verts.shape[0]) / float(oracle_max_verts))))
                verts = verts[::step_v]
            verts_cur = torch.matmul(verts, R.T) + T.view(1, 3)
            verts_gt = torch.matmul(verts, R_gt.T) + T_gt.view(1, 3)
            H, W = rasterizer.raster_settings.image_size
            aspect = float(W) / float(max(1, H))
            fov_y_deg = float(os.environ.get("VAL_FOVY_DEG", "60.0"))
            znear = float(os.environ.get("VAL_ZNEAR", "0.01"))
            zfar = float(os.environ.get("VAL_ZFAR", "10000.0"))
            clip_cur = _nv_project(verts_cur.view(1, -1, 3), fov_y_deg=fov_y_deg, aspect=aspect, znear=znear, zfar=zfar)[0]
            clip_gt = _nv_project(verts_gt.view(1, -1, 3), fov_y_deg=fov_y_deg, aspect=aspect, znear=znear, zfar=zfar)[0]
            valid_clip = (clip_cur[:, 3].abs() > 1e-8) & (clip_gt[:, 3].abs() > 1e-8)
            if valid_clip.any():
                xy_cur = clip_cur[valid_clip, :2] / clip_cur[valid_clip, 3:4]
                xy_gt = clip_gt[valid_clip, :2] / clip_gt[valid_clip, 3:4]
                loss_oracle_reproj = torch.nn.functional.smooth_l1_loss(xy_cur, xy_gt, beta=0.01, reduction="mean")

        loss = (
            float(w_sil) * loss_sil
            + float(w_depth) * loss_depth
            + float(w_normal) * loss_normal
            + reg
            + oracle_pose_w * loss_oracle
            + oracle_reproj_w * loss_oracle_reproj
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_([w, t], max_norm=clip)
        opt.step()
        if scheduler is not None:
            scheduler.step()

        lv = float(loss.detach().cpu().item())
        if lv < best_loss - min_delta:
            best_loss = lv
            stale = 0
            with torch.no_grad():
                best_R = (_so3_exp(w) @ R0).detach().clone()
                best_T = ((_so3_exp(w) @ T0) + t).detach().clone()
        else:
            stale += 1
            if stale >= patience:
                break

    return best_R, best_T, float(best_loss)


def _evaluate_pose_sil_no_grad(
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    image_ref_silhouette: torch.Tensor,
    at_xyz: torch.Tensor,
    dist_v: float,
    elev_v: float,
    azim_v: float,
    device: torch.device
):
    """
    Silhouette-only evaluation (FAST): no rasterizer, no depth/normals.
    Returns dict keys compatible with _evaluate_pose_loss_no_grad.
    """
    with torch.no_grad():
        R, T = look_at_view_transform(
            dist=torch.tensor([dist_v], device=device),
            elev=torch.tensor([elev_v], device=device),
            azim=torch.tensor([azim_v], device=device),
            at=at_xyz.view(1, 3),
            device=device,
        )
        img = silhouette_renderer(meshes_world=mesh, R=R, T=T)
        alpha = _postprocess_silhouette_alpha(torch.clamp(img[0, ..., 3], 0, 1))
        ref = torch.clamp(image_ref_silhouette, 0, 1)
        if ref.dim() == 3 and ref.shape[-1] == 4:
            ref = ref[..., 3]
        ref = ref.squeeze()
        loss_mse = torch.mean((alpha - ref) ** 2)
        inter = (alpha * ref).sum()
        union = alpha.sum() + ref.sum() - inter
        loss_iou = 1.0 - (inter / (union + 1e-8))
        w_iou = float(os.environ.get("VAL_W_IOU", "0.1"))
        scale_cfg = os.environ.get("VAL_LOSS_SCALE", "1.0")
        if scale_cfg.strip().lower() == "auto":
            h = int(ref.shape[0])
            w = int(ref.shape[1])
            loss_scale = float(h * w)
        else:
            loss_scale = float(scale_cfg)
        loss_sil = (1.0 - w_iou) * (loss_mse * loss_scale) + w_iou * loss_iou
        return {
            "loss_total": float(loss_sil.detach().cpu().item()),
            "loss_sil": float(loss_sil.detach().cpu().item()),
            "loss_depth": 0.0,
            "loss_normal": 0.0,
            "valid_ratio": 1.0,
            "R": R[0].detach(),
            "T": T[0].detach(),
        }


def _evaluate_pose_sil_batch_no_grad(
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    image_ref_silhouette: torch.Tensor,
    at_xyz: torch.Tensor,
    dist_list: list[float],
    elev_list: list[float],
    azim_list: list[float],
    device: torch.device,
):
    """
    Batched silhouette-only evaluation (FAST).
    Returns loss_total (B,) tensor.
    """
    assert len(dist_list) == len(elev_list) == len(azim_list)
    B = int(len(dist_list))
    if B <= 0:
        return {"loss_total": torch.empty((0,), device=device)}
    with torch.no_grad():
        dist_t = torch.tensor(dist_list, device=device, dtype=torch.float32)
        elev_t = torch.tensor(elev_list, device=device, dtype=torch.float32)
        azim_t = torch.tensor(azim_list, device=device, dtype=torch.float32)
        at_b = at_xyz.view(1, 3).to(device).expand(B, 3)
        R, T = look_at_view_transform(dist=dist_t, elev=elev_t, azim=azim_t, at=at_b, device=device)
        mesh_b = mesh.extend(B)
        img = silhouette_renderer(meshes_world=mesh_b, R=R, T=T)
        alpha = torch.clamp(img[..., 3], 0, 1)  # (B,H,W)
        ref = torch.clamp(image_ref_silhouette, 0, 1)
        if ref.dim() == 3 and ref.shape[-1] == 4:
            ref = ref[..., 3]
        ref = ref.squeeze().to(device)
        ref_b = ref.unsqueeze(0).expand(B, -1, -1)
        loss_mse = torch.mean((alpha - ref_b) ** 2, dim=(1, 2))
        inter = torch.sum(alpha * ref_b, dim=(1, 2))
        union = torch.sum(alpha, dim=(1, 2)) + torch.sum(ref_b, dim=(1, 2)) - inter
        loss_iou = 1.0 - (inter / (union + 1e-8))
        w_iou = float(os.environ.get("VAL_W_IOU", "0.1"))
        scale_cfg = os.environ.get("VAL_LOSS_SCALE", "1.0")
        if scale_cfg.strip().lower() == "auto":
            h = int(ref.shape[0])
            w = int(ref.shape[1])
            loss_scale = float(h * w)
        else:
            loss_scale = float(scale_cfg)
        loss_sil = (1.0 - w_iou) * (loss_mse * loss_scale) + w_iou * loss_iou
        return {"loss_total": loss_sil.detach()}


def _evaluate_pose_sil_semantic_batch_no_grad(
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    phong_renderer: MeshRenderer | None,
    rasterizer: MeshRasterizer | None,
    image_ref_silhouette: torch.Tensor,
    image_ref_rgb: torch.Tensor | None,
    at_xyz: torch.Tensor,
    dist_list: list[float],
    elev_list: list[float],
    azim_list: list[float],
    device: torch.device,
    R_gt: torch.Tensor | None = None,
    T_gt: torch.Tensor | None = None,
    ref_points_view: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """
    Batched evaluation with silhouette + distance losses (semantic loss disabled).
    
    Args:
        mesh: Mesh data
        silhouette_renderer: Silhouette renderer
        phong_renderer: RGB renderer (for semantic similarity), can be None
        image_ref_silhouette: Reference silhouette image (H,W) or (H,W,4)
        image_ref_rgb: Reference RGB image (H,W,3) for semantic similarity, can be None
        at_xyz: Look-at point (mesh center)
        dist_list, elev_list, azim_list: Camera pose parameters
        device: Device
        R_gt: (3, 3) GT rotation matrix (world-to-view), optional
        T_gt: (3,) GT translation vector (world-to-view), optional
        
    Args:
        rasterizer: MeshRasterizer for extracting depth maps (needed for 3D pointcloud loss)
        ref_points_view: (N, 3) GT pointcloud in view space (optional, needed for 3D pointcloud loss)
        
    Returns:
        dict with keys:
            - loss_total: (B,) combined loss (sil + semantic + distance + pointcloud + pointcloud_3d)
            - loss_sil: (B,) silhouette loss
            - loss_semantic: (B,) semantic similarity loss (1 - similarity)
            - loss_distance: (B,) distance to center loss
            - loss_pointcloud: (B,) pointcloud distance similarity loss
            - loss_pointcloud_3d: (B,) 3D pointcloud Chamfer distance loss
    """
    assert len(dist_list) == len(elev_list) == len(azim_list)
    B = int(len(dist_list))
    if B <= 0:
        return {
            "loss_total": torch.empty((0,), device=device),
            "loss_sil": torch.empty((0,), device=device),
            "loss_semantic": torch.empty((0,), device=device),
        }
    
    # 用户要求：暂时关闭语义损失
    # Check if semantic is disabled via environment variable (always disabled for now)
    semantic_disabled = True  # 强制禁用语义损失
    semantic_available = False
    
    with torch.no_grad():
        dist_t = torch.tensor(dist_list, device=device, dtype=torch.float32)
        elev_t = torch.tensor(elev_list, device=device, dtype=torch.float32)
        azim_t = torch.tensor(azim_list, device=device, dtype=torch.float32)
        at_b = at_xyz.view(1, 3).to(device).expand(B, 3)
        R, T = look_at_view_transform(dist=dist_t, elev=elev_t, azim=azim_t, at=at_b, device=device)
        mesh_b = mesh.extend(B)
        
        # 1. Silhouette loss
        img_sil = silhouette_renderer(meshes_world=mesh_b, R=R, T=T)
        alpha = torch.clamp(img_sil[..., 3], 0, 1)  # (B,H,W)
        ref_sil = torch.clamp(image_ref_silhouette, 0, 1)
        if ref_sil.dim() == 3 and ref_sil.shape[-1] == 4:
            ref_sil = ref_sil[..., 3]
        ref_sil = ref_sil.squeeze().to(device)
        ref_sil_b = ref_sil.unsqueeze(0).expand(B, -1, -1)
        loss_mse = torch.mean((alpha - ref_sil_b) ** 2, dim=(1, 2))
        inter = torch.sum(alpha * ref_sil_b, dim=(1, 2))
        union = torch.sum(alpha, dim=(1, 2)) + torch.sum(ref_sil_b, dim=(1, 2)) - inter
        loss_iou = 1.0 - (inter / (union + 1e-8))
        w_iou = float(os.environ.get("VAL_W_IOU", "0.1"))
        # Get image dimensions (needed for semantic similarity calculation)
        h = int(ref_sil.shape[0])
        w = int(ref_sil.shape[1])
        scale_cfg = os.environ.get("VAL_LOSS_SCALE", "1.0")
        if scale_cfg.strip().lower() == "auto":
            loss_scale = float(h * w)
        else:
            loss_scale = float(scale_cfg)
        loss_sil = (1.0 - w_iou) * (loss_mse * loss_scale) + w_iou * loss_iou
        
        # 2. Semantic similarity loss
        loss_semantic = torch.zeros((B,), device=device, dtype=loss_sil.dtype)
        if semantic_available and phong_renderer is not None and image_ref_rgb is not None:
            try:
                # Get semantic model manager
                model_manager = semfe.ModelManager()
                try:
                    model_manager.get_model()
                except RuntimeError:
                    model_path = os.environ.get("VAL_SEMANTIC_MODEL_PATH", "models/autoencoder.pth")
                    model_manager.initialize(model_path=model_path, image_size=(h, w), device=device)
                
                # Convert reference silhouette to binary contour image (H,W) -> (H,W,3) for semantic model
                # The semantic model expects RGB images, so we convert binary silhouette to RGB
                ref_sil_np = (ref_sil.cpu().numpy() * 255).astype(np.uint8)  # (H,W) [0,255]
                ref_contour_rgb = np.stack([ref_sil_np, ref_sil_np, ref_sil_np], axis=-1)  # (H,W,3) RGB
                
                # Calculate semantic similarity for each candidate
                for b in range(B):
                    try:
                        # Convert rendered silhouette to binary contour image (H,W) -> (H,W,3)
                        render_sil_np = (alpha[b].cpu().numpy() * 255).astype(np.uint8)  # (H,W) [0,255]
                        render_contour_rgb = np.stack([render_sil_np, render_sil_np, render_sil_np], axis=-1)  # (H,W,3) RGB
                        
                        # Calculate semantic similarity using binary contour images
                        # calculate_contour_similarity expects two contour images (binary silhouettes)
                        # Returns [0,1], higher = more similar
                        similarity = semfe.calculate_contour_similarity(
                            contour1=ref_contour_rgb,  # Reference binary contour (H,W,3) RGB
                            contour2=render_contour_rgb,  # Rendered binary contour (H,W,3) RGB
                            model_path=os.environ.get("VAL_SEMANTIC_MODEL_PATH", "models/autoencoder.pth"),
                            image_size=(h, w),
                            device=device
                        )
                        # Convert similarity to loss: loss = 1 - similarity (lower is better)
                        loss_semantic[b] = 1.0 - float(similarity)
                    except Exception as e:
                        # If semantic calculation fails, use zero loss (fallback to silhouette only)
                        loss_semantic[b] = 0.0
                        if b == 0:  # Only print once
                            error_msg = str(e)
                            if "triton" in error_msg.lower() or "dynamo" in error_msg.lower() or "torch.compile" in error_msg.lower():
                                print(f"[Warning] Semantic similarity calculation failed (torch.compile issue): {e}")
                                print(f"[Hint] Try setting environment variable: VAL_DISABLE_TORCH_COMPILE=1 to disable torch.compile")
                            else:
                                print(f"[Warning] Semantic similarity calculation failed: {e}, using silhouette-only")
            except Exception as e:
                print(f"[Warning] Semantic evaluation failed: {e}, using silhouette-only")
        
        # 3. Distance losses (if GT pose is provided) -- legacy GT-dependent scoring.
        loss_distance = torch.zeros((B,), device=device, dtype=loss_sil.dtype)
        loss_pointcloud = torch.zeros((B,), device=device, dtype=loss_sil.dtype)
        loss_pointcloud_3d = torch.zeros((B,), device=device, dtype=loss_sil.dtype)

        _assert_no_gt("_evaluate_pose_sil_semantic_batch_no_grad", R_gt, T_gt, ref_points_view)
        if R_gt is not None and T_gt is not None:
            # Get mesh vertices
            mesh_verts = mesh.verts_packed()  # (V, 3)
            
            # Compute distance losses
            loss_distance_raw = _compute_distance_loss(R, T, R_gt, T_gt, at_xyz, device)  # (B,)
            
            # Normalize distance loss: divide by typical distance scale (use GT distance as reference)
            with torch.no_grad():
                R_gt_T = R_gt.t()
                cam_center_gt = -torch.matmul(R_gt_T, T_gt)  # (3,)
                dist_gt_ref = torch.norm(cam_center_gt - at_xyz)  # scalar
                # Normalize by GT distance (make loss scale-invariant)
                dist_normalize = torch.clamp(dist_gt_ref, min=1e-6)
            
            loss_distance = loss_distance_raw / dist_normalize  # (B,)
            
            # Compute pointcloud distance loss
            use_cosine_dist = int(os.environ.get("VAL_PCD_USE_COSINE", "1")) != 0
            loss_pointcloud_raw = _compute_pointcloud_distance_loss(
                R, T, R_gt, T_gt, mesh_verts, device, use_cosine=use_cosine_dist
            )  # (B,)
            
            # Normalize pointcloud loss (already normalized in the function if using cosine)
            if not use_cosine_dist:
                # For L2 distance, normalize by mean distance
                with torch.no_grad():
                    R_gt_T = R_gt.t()
                    cam_center_gt = -torch.matmul(R_gt_T, T_gt)  # (3,)
                    dists_gt_all = torch.norm(mesh_verts - cam_center_gt.view(1, 3), dim=-1)  # (V,)
                    mean_dist_gt = torch.mean(dists_gt_all)
                    pcd_normalize = torch.clamp(mean_dist_gt, min=1e-6)
                loss_pointcloud = loss_pointcloud_raw / pcd_normalize
            else:
                # Cosine distance is already in [0, 2], no additional normalization needed
                loss_pointcloud = loss_pointcloud_raw
            
            # 4. Compute 3D pointcloud loss (new loss: current rendered pointcloud vs GT pointcloud)
            if rasterizer is not None and ref_points_view is not None and ref_points_view.shape[0] > 0:
                try:
                    # Extract pointclouds from current poses
                    cams = FoVPerspectiveCameras(device=device, R=R, T=T)
                    fr = rasterizer(meshes_world=mesh_b, cameras=cams)
                    fov_y_deg = float(os.environ.get("VAL_FOVY_DEG", "60.0"))
                    max_points_3d = int(os.environ.get("VAL_CHAMFER_POINTS", "8000"))
                    pcd_method = os.environ.get("VAL_PCD_3D_METHOD", "chamfer").strip().lower()
                    
                    for b in range(B):
                        vis_mask_b = (fr.pix_to_face[..., 0][b] >= 0).float()  # (H, W)
                        depth_b = fr.zbuf[..., 0][b] * vis_mask_b  # (H, W)
                        
                        # Extract pointcloud from current pose
                        points_curr_view = _extract_pointcloud_from_depth(
                            depth_map=depth_b,
                            vis_mask=vis_mask_b,
                            device=device,
                            fov_y_deg=fov_y_deg,
                            max_points=max_points_3d,
                        )
                        
                        if points_curr_view.shape[0] > 0:
                            # Compute 3D pointcloud loss
                            loss_pcd_3d_val = _compute_pointcloud_3d_loss(
                                points_current_view=points_curr_view,
                                points_gt_view=ref_points_view,
                                device=device,
                                method=pcd_method,
                            )
                            loss_pointcloud_3d[b] = loss_pcd_3d_val
                        else:
                            loss_pointcloud_3d[b] = 0.0
                except Exception as e:
                    # If 3D pointcloud extraction fails, use zero loss
                    if B > 0:
                        print(f"[Warning] 3D pointcloud loss calculation failed: {e}, using zero loss")
        
        # 5. Combine all losses with weights
        w_sil = float(os.environ.get("VAL_W_SIL_LOCAL", "1.0"))
        w_semantic = 0.0  # 用户要求：暂时关闭语义损失
        w_distance = float(os.environ.get("VAL_W_DISTANCE", "0.1"))  # Default weight for distance loss
        w_pointcloud = float(os.environ.get("VAL_W_POINTCLOUD", "0.1"))  # Default weight for pointcloud loss
        w_pointcloud_3d = float(os.environ.get("VAL_W_POINTCLOUD_3D", "0.1"))  # Default weight for 3D pointcloud loss
        
        loss_total = w_sil * loss_sil + w_semantic * loss_semantic + w_distance * loss_distance + w_pointcloud * loss_pointcloud + w_pointcloud_3d * loss_pointcloud_3d
        
        return {
            "loss_total": loss_total.detach(),
            "loss_sil": loss_sil.detach(),
            "loss_semantic": loss_semantic.detach(),
            "loss_distance": loss_distance.detach(),
            "loss_pointcloud": loss_pointcloud.detach(),
            "loss_pointcloud_3d": loss_pointcloud_3d.detach(),
        }


def _dominant_axis_from_delta_world(delta_xyz: torch.Tensor) -> str:
    """
    delta_xyz: (3,) in WORLD space.
    Return one of: +x,-x,+y,-y,+z,-z (dominant component sign).
    """
    d = delta_xyz.view(3)
    idx = int(torch.argmax(torch.abs(d)).detach().cpu().item())
    sign = float(d[idx].detach().cpu().item())
    axis = ["x", "y", "z"][idx]
    return f"{'+' if sign >= 0 else '-'}{axis}"


def _batched_eval_candidates_no_grad(
    *,
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    at_xyz: torch.Tensor,
    dist_list: list[float],
    elev_list: list[float],
    azim_list: list[float],
    device: torch.device,
    use_sil_only: bool,
    use_sil_normals_only: bool = False,
) -> dict[str, torch.Tensor]:
    """
    Utility wrapper to evaluate candidate poses in batch under no_grad.
    Returns dict of tensors on device with shape (B,).

    Args:
        use_sil_only: If True, only use silhouette loss (fastest, but less accurate).
        use_sil_normals_only: If True, use silhouette + normals (no depth). Default False.
    """
    if use_sil_only:
        lt = _evaluate_pose_sil_batch_no_grad(
            mesh=mesh,
            silhouette_renderer=silhouette_renderer,
            image_ref_silhouette=image_ref_silhouette,
            at_xyz=at_xyz,
            dist_list=dist_list,
            elev_list=elev_list,
            azim_list=azim_list,
            device=device,
        )["loss_total"]
        B = int(lt.shape[0])
        z = torch.zeros((B,), device=device, dtype=lt.dtype)
        o = torch.ones((B,), device=device, dtype=lt.dtype)
        return {"loss_total": lt, "loss_sil": lt, "loss_depth": z, "loss_normal": z, "valid_ratio": o}
    out = _evaluate_pose_losses_batch_no_grad(
        mesh=mesh,
        silhouette_renderer=silhouette_renderer,
        rasterizer=rasterizer,
        image_ref_silhouette=image_ref_silhouette,
        ref_depth=ref_depth,
        ref_normals=ref_normals,
        at_xyz=at_xyz,
        dist_list=dist_list,
        elev_list=elev_list,
        azim_list=azim_list,
        device=device,
        use_sil_normals_only=use_sil_normals_only,
    )
    return out


def _greedy_neighbor_poses(
    *,
    dist_c: float,
    elev_c: float,
    azim_c: float,
    dist_step: float,
    ang_step: float,
    dist_min: float,
    dist_max: float,
    elev_min: float,
    elev_max: float,
    azim_min: float,
    azim_max: float,
) -> tuple[list[str], list[float], list[float], list[float]]:
    """Six axis-aligned neighbors in (dist, elev, azim) used by greedy_local_search."""

    def _wrap_azim(a: float) -> float:
        return float(((float(a) + 180.0) % 360.0) - 180.0)

    names = ["dist+", "dist-", "elev+", "elev-", "azim+", "azim-"]
    dists = [
        float(np.clip(dist_c * (1.0 + dist_step), dist_min, dist_max)),
        float(np.clip(dist_c / (1.0 + dist_step), dist_min, dist_max)),
        float(dist_c),
        float(dist_c),
        float(dist_c),
        float(dist_c),
    ]
    elevs = [
        float(elev_c),
        float(elev_c),
        float(np.clip(elev_c + ang_step, elev_min, elev_max)),
        float(np.clip(elev_c - ang_step, elev_min, elev_max)),
        float(elev_c),
        float(elev_c),
    ]
    azims = [
        float(azim_c),
        float(azim_c),
        float(azim_c),
        float(azim_c),
        float(np.clip(_wrap_azim(azim_c + ang_step), azim_min, azim_max)),
        float(np.clip(_wrap_azim(azim_c - ang_step), azim_min, azim_max)),
    ]
    return names, dists, elevs, azims


def _batched_best_neighbor_move(
    *,
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    at_xyz: torch.Tensor,
    dist_c: float,
    elev_c: float,
    azim_c: float,
    dist_step: float,
    ang_step: float,
    dist_min: float,
    dist_max: float,
    elev_min: float,
    elev_max: float,
    azim_min: float,
    azim_max: float,
    device: torch.device,
    use_sil_only: bool = False,
) -> tuple[float, str, float, float, float, dict]:
    """
    Evaluate all 6 greedy neighbors in one batched forward pass (faster than 6 serial calls).
    Returns (loss, move_name, dist, elev, azim, ev_dict_for_winner).
    """
    names, dlist, elist, alist = _greedy_neighbor_poses(
        dist_c=dist_c,
        elev_c=elev_c,
        azim_c=azim_c,
        dist_step=dist_step,
        ang_step=ang_step,
        dist_min=dist_min,
        dist_max=dist_max,
        elev_min=elev_min,
        elev_max=elev_max,
        azim_min=azim_min,
        azim_max=azim_max,
    )
    evb = _batched_eval_candidates_no_grad(
        mesh=mesh,
        silhouette_renderer=silhouette_renderer,
        rasterizer=rasterizer,
        image_ref_silhouette=image_ref_silhouette,
        ref_depth=ref_depth,
        ref_normals=ref_normals,
        at_xyz=at_xyz,
        dist_list=dlist,
        elev_list=elist,
        azim_list=alist,
        device=device,
        use_sil_only=use_sil_only,
    )
    lt = evb["loss_total"].detach().cpu().numpy().astype(np.float64)
    best_i = int(np.argmin(lt)) if lt.size > 0 else 0
    d_best, e_best, a_best = float(dlist[best_i]), float(elist[best_i]), float(alist[best_i])
    ev = _evaluate_pose_loss_no_grad(
        mesh,
        silhouette_renderer,
        rasterizer,
        image_ref_silhouette,
        ref_depth,
        ref_normals,
        at_xyz.view(3),
        d_best,
        e_best,
        a_best,
        device,
    )
    return float(lt[best_i]), str(names[best_i]), d_best, e_best, a_best, ev


def _sphere_fd_grad_probe(
    *,
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    at_xyz: torch.Tensor,
    dist_c: float,
    elev_c: float,
    azim_c: float,
    elev_min: float,
    elev_max: float,
    azim_min: float,
    azim_max: float,
    device: torch.device,
    use_sil_only: bool = False,
) -> tuple[float, float] | None:
    """Central FD gradients (g_elev, g_azim) on fixed-dist sphere; 4 batched renders."""
    eps_e = float(os.environ.get("VAL_GREEDY_FD_ELEV_EPS", "1.0"))
    eps_a = float(os.environ.get("VAL_GREEDY_FD_AZIM_EPS", "1.0"))

    def _wrap_azim(a: float) -> float:
        return float(((float(a) + 180.0) % 360.0) - 180.0)

    elist = [
        float(np.clip(elev_c + eps_e, elev_min, elev_max)),
        float(np.clip(elev_c - eps_e, elev_min, elev_max)),
        float(elev_c),
        float(elev_c),
    ]
    alist = [
        float(azim_c),
        float(azim_c),
        float(np.clip(_wrap_azim(azim_c + eps_a), azim_min, azim_max)),
        float(np.clip(_wrap_azim(azim_c - eps_a), azim_min, azim_max)),
    ]
    dlist = [float(dist_c)] * 4
    evb = _batched_eval_candidates_no_grad(
        mesh=mesh,
        silhouette_renderer=silhouette_renderer,
        rasterizer=rasterizer,
        image_ref_silhouette=image_ref_silhouette,
        ref_depth=ref_depth,
        ref_normals=ref_normals,
        at_xyz=at_xyz,
        dist_list=dlist,
        elev_list=elist,
        azim_list=alist,
        device=device,
        use_sil_only=use_sil_only,
    )
    lt = evb["loss_total"].detach().cpu().numpy().astype(np.float64)
    g_elev = float((lt[0] - lt[1]) / (2.0 * max(eps_e, 1e-6)))
    g_azim = float((lt[2] - lt[3]) / (2.0 * max(eps_a, 1e-6)))
    if abs(g_elev) < 1e-8 and abs(g_azim) < 1e-8:
        return None
    return g_elev, g_azim


def _sphere_fd_descent_step(
    *,
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    at_xyz: torch.Tensor,
    dist_c: float,
    elev_c: float,
    azim_c: float,
    elev_min: float,
    elev_max: float,
    azim_min: float,
    azim_max: float,
    device: torch.device,
    use_sil_only: bool = False,
) -> tuple[float, float, float, float, dict] | None:
    """
    One finite-difference step in (elev, azim) on a fixed dist sphere.
    Approximates the steepest local descent direction with 4 extra renders (batched).
    """
    step_scale = float(os.environ.get("VAL_GREEDY_FD_STEP_SCALE", "1.0"))

    def _wrap_azim(a: float) -> float:
        return float(((float(a) + 180.0) % 360.0) - 180.0)

    grad = _sphere_fd_grad_probe(
        mesh=mesh,
        silhouette_renderer=silhouette_renderer,
        rasterizer=rasterizer,
        image_ref_silhouette=image_ref_silhouette,
        ref_depth=ref_depth,
        ref_normals=ref_normals,
        at_xyz=at_xyz,
        dist_c=dist_c,
        elev_c=elev_c,
        azim_c=azim_c,
        elev_min=elev_min,
        elev_max=elev_max,
        azim_min=azim_min,
        azim_max=azim_max,
        device=device,
        use_sil_only=use_sil_only,
    )
    if grad is None:
        return None
    g_elev, g_azim = grad
    step_e = float(-step_scale * g_elev)
    step_a = float(-step_scale * g_azim)
    if abs(step_e) < 1e-8 and abs(step_a) < 1e-8:
        return None
    e_new = float(np.clip(elev_c + step_e, elev_min, elev_max))
    a_new = float(np.clip(_wrap_azim(azim_c + step_a), azim_min, azim_max))
    ev = _evaluate_pose_loss_no_grad(
        mesh,
        silhouette_renderer,
        rasterizer,
        image_ref_silhouette,
        ref_depth,
        ref_normals,
        at_xyz.view(3),
        float(dist_c),
        e_new,
        a_new,
        device,
    )
    return float(dist_c), e_new, a_new, float(ev["loss_total"]), ev


def _mesh_radius_from_verts(verts: torch.Tensor, center_xyz: torch.Tensor) -> float:
    """
    Compute model radius r as max ||v-center|| (world units).
    verts: (V,3), center_xyz: (3,)
    """
    with torch.no_grad():
        v = verts - center_xyz.view(1, 3)
        if v.numel() == 0:
            return 1.0
        r = torch.norm(v, dim=-1).max().clamp(min=1e-6)
        return float(r.detach().cpu().item())


def global_beam_search_on_sphere(
    *,
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    at_xyz: torch.Tensor,
    init_dist: float,
    init_elev: float,
    init_azim: float,
    device: torch.device,
    save_dir: str,
    tag: str,
) -> tuple[float, float, float, float]:
    """
    Beam Search（可选全局搜索，默认不启用；用 VAL_GLOBAL_MODE=beam 打开）：
    - 固定球面半径=init_dist，在球面上做多层扩展搜索。
    - 每层：对 beam 内的每个节点，沿 ±x/±y/±z 做 K 个前向步长的候选（投影回球面），评估 loss，
      取 top-B 作为下一层 beam。
    - 为了速度：支持 sil-only 预筛选 + TopM full-loss 复排（类似 GreedyAxis 的思想）。

    Env（可覆盖）：
    - VAL_BEAM_ENABLE: 1/0
    - VAL_BEAM_WIDTH: B（默认 32）
    - VAL_BEAM_LEVELS: L（默认 8）
    - VAL_BEAM_STEPS_PER_DIR: K（默认 2）
    - VAL_BEAM_STEP_FRAC: 步长比例（默认 0.03，实际步长=step_frac*dist*k）
    - VAL_BEAM_USE_SIL_ONLY: 1/0（默认 1）
    - VAL_BEAM_REFINE_FULL: 1/0（默认 1）
    - VAL_BEAM_TOPM_FULL: 64（默认 64）
    """
    enabled = int(os.environ.get("VAL_BEAM_ENABLE", "1")) != 0
    if not enabled:
        return float(init_dist), float(init_elev), float(init_azim), float("inf")

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{tag}_beam_candidates.csv")
    json_path = os.path.join(save_dir, f"{tag}_beam_best.json")

    beam_width = max(1, int(os.environ.get("VAL_BEAM_WIDTH", "32")))
    levels = max(1, int(os.environ.get("VAL_BEAM_LEVELS", "8")))
    steps_per_dir = max(1, int(os.environ.get("VAL_BEAM_STEPS_PER_DIR", "2")))
    step_frac = float(os.environ.get("VAL_BEAM_STEP_FRAC", "0.03"))
    use_sil_only = int(os.environ.get("VAL_BEAM_USE_SIL_ONLY", "1")) != 0
    refine_full = int(os.environ.get("VAL_BEAM_REFINE_FULL", "1")) != 0
    topm_full = max(1, int(os.environ.get("VAL_BEAM_TOPM_FULL", "64")))
    verbose = int(os.environ.get("VAL_BEAM_VERBOSE", "1"))

    at_xyz = at_xyz.view(3).to(device)
    dist_fixed = float(init_dist)

    # dirs in WORLD
    dirs = [
        ("+x", torch.tensor([1.0, 0.0, 0.0], device=device)),
        ("-x", torch.tensor([-1.0, 0.0, 0.0], device=device)),
        ("+y", torch.tensor([0.0, 1.0, 0.0], device=device)),
        ("-y", torch.tensor([0.0, -1.0, 0.0], device=device)),
        ("+z", torch.tensor([0.0, 0.0, 1.0], device=device)),
        ("-z", torch.tensor([0.0, 0.0, -1.0], device=device)),
    ]

    def _project_to_sphere(xyz: torch.Tensor) -> torch.Tensor:
        v = xyz - at_xyz
        return at_xyz + (v / (torch.norm(v) + 1e-8)) * torch.tensor(dist_fixed, device=device)

    # init node
    cam0 = _camera_position_from_spherical_global(
        torch.tensor(float(dist_fixed), device=device),
        torch.tensor(float(init_elev), device=device),
        torch.tensor(float(init_azim), device=device),
        at_xyz,
    )
    cam0 = _project_to_sphere(cam0)
    d0, e0, a0 = _spherical_from_camera_position_global(cam0, at_xyz)
    cur_eval = _evaluate_pose_sil_no_grad(
        mesh=mesh,
        silhouette_renderer=silhouette_renderer,
        image_ref_silhouette=image_ref_silhouette,
        at_xyz=at_xyz,
        dist_v=float(dist_fixed),
        elev_v=float(e0),
        azim_v=float(a0),
        device=device,
    ) if (use_sil_only and (not refine_full)) else _evaluate_pose_loss_no_grad(
        mesh, silhouette_renderer, rasterizer, image_ref_silhouette, ref_depth, ref_normals,
        at_xyz, float(dist_fixed), float(e0), float(a0), device
    )
    best = {
        "loss": float(cur_eval["loss_total"]),
        "dist": float(dist_fixed),
        "elev": float(e0),
        "azim": float(a0),
        "level": 0,
        "note": "init",
    }

    # beam is list of tuples: (loss, cam_xyz)
    beam = [(float(best["loss"]), cam0.detach().clone())]

    if verbose >= 1:
        print(f"[Beam] {tag}: width={beam_width}, levels={levels}, steps_per_dir={steps_per_dir}, step_frac={step_frac:.4f}, sil_only={int(use_sil_only)}, refine_full={int(refine_full)}")
        print(f"[Beam] {tag}: init loss={best['loss']:.6f} (dist,elev,azim)=({best['dist']:.3f},{best['elev']:.3f},{best['azim']:.3f})")

    rows = []
    rows.append(["level", "parent_rank", "dir", "k", "dist", "elev_deg", "azim_deg", "loss_total"])

    for lvl in range(1, levels + 1):
        step_len = float(step_frac) * float(dist_fixed)
        cand_xyz: list[torch.Tensor] = []
        cand_meta: list[tuple[int, str, int]] = []  # (parent_rank, dir_name, k)
        cand_dist: list[float] = []
        cand_elev: list[float] = []
        cand_azim: list[float] = []

        # expand
        for parent_rank, (_loss_p, xyz_p) in enumerate(beam):
            for dir_name, dv in dirs:
                for k in range(1, steps_per_dir + 1):
                    xyz_c = _project_to_sphere(xyz_p + dv * (step_len * float(k)))
                    _d, _e, _a = _spherical_from_camera_position_global(xyz_c, at_xyz)
                    cand_xyz.append(xyz_c)
                    cand_meta.append((int(parent_rank), str(dir_name), int(k)))
                    cand_dist.append(float(dist_fixed))
                    cand_elev.append(float(_e))
                    cand_azim.append(float(_a))

        if len(cand_xyz) == 0:
            break

        # evaluate (sil-only prefilter optional)
        if use_sil_only:
            evb_sil = _evaluate_pose_sil_batch_no_grad(
                mesh=mesh,
                silhouette_renderer=silhouette_renderer,
                image_ref_silhouette=image_ref_silhouette,
                at_xyz=at_xyz,
                dist_list=cand_dist,
                elev_list=cand_elev,
                azim_list=cand_azim,
                device=device,
            )
            losses_sil = evb_sil["loss_total"].detach().cpu().numpy().astype(np.float64)
            # preselect top-M for full refine (or directly use sil loss)
            m = min(int(topm_full), int(losses_sil.size))
            order = np.argsort(losses_sil)[:m]
            if refine_full and m > 0:
                d2 = [cand_dist[int(j)] for j in order]
                e2 = [cand_elev[int(j)] for j in order]
                a2 = [cand_azim[int(j)] for j in order]
                evb_full = _evaluate_pose_losses_batch_no_grad(
                    mesh, silhouette_renderer, rasterizer,
                    image_ref_silhouette, ref_depth, ref_normals,
                    at_xyz, d2, e2, a2, device
                )
                losses_full = evb_full["loss_total"].detach().cpu().numpy().astype(np.float64)
                # map full losses back to global candidate indices
                scored = [(float(losses_full[ii]), int(order[ii])) for ii in range(int(losses_full.size))]
            else:
                scored = [(float(losses_sil[int(j)]), int(j)) for j in order]
        else:
            evb = _evaluate_pose_losses_batch_no_grad(
                mesh, silhouette_renderer, rasterizer,
                image_ref_silhouette, ref_depth, ref_normals,
                at_xyz, cand_dist, cand_elev, cand_azim, device
            )
            losses = evb["loss_total"].detach().cpu().numpy().astype(np.float64)
            scored = [(float(losses[i]), int(i)) for i in range(int(losses.size))]

        scored.sort(key=lambda x: x[0])

        # record CSV (top few only to reduce I/O)
        try:
            log_top = int(os.environ.get("VAL_BEAM_LOG_TOP", "64"))
            log_top = max(0, min(log_top, len(scored)))
            for rank in range(log_top):
                lv, j = scored[rank]
                pr, dn, kk = cand_meta[j]
                rows.append([int(lvl), int(pr), dn, int(kk), float(dist_fixed), float(cand_elev[j]), float(cand_azim[j]), float(lv)])
        except Exception:
            pass

        # build next beam (top-B unique candidates)
        next_beam = []
        for lv, j in scored[:beam_width]:
            next_beam.append((float(lv), cand_xyz[j].detach().clone()))
            if float(lv) < float(best["loss"]):
                best.update(
                    {
                        "loss": float(lv),
                        "dist": float(dist_fixed),
                        "elev": float(cand_elev[j]),
                        "azim": float(cand_azim[j]),
                        "level": int(lvl),
                        "note": f"parent={cand_meta[j][0]} dir={cand_meta[j][1]} k={cand_meta[j][2]}",
                    }
                )
        beam = next_beam

        if verbose >= 2:
            print(f"[Beam] {tag}: level={lvl} best_loss={best['loss']:.6f} beam0={beam[0][0] if len(beam) else float('inf'):.6f}")
        if len(beam) == 0:
            break

    # write logs
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for row in rows:
                w.writerow(row)
    except Exception as _e:
        print(f"[Warning] Failed to write beam_candidates.csv: {_e}")

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tag": tag,
                    "beam": {
                        "width": int(beam_width),
                        "levels": int(levels),
                        "steps_per_dir": int(steps_per_dir),
                        "step_frac": float(step_frac),
                        "use_sil_only": bool(use_sil_only),
                        "refine_full": bool(refine_full),
                        "topm_full": int(topm_full),
                    },
                    "best": {
                        "loss_total": float(best["loss"]),
                        "dist": float(best["dist"]),
                        "elev_deg": float(best["elev"]),
                        "azim_deg": float(best["azim"]),
                        "level": int(best["level"]),
                        "note": str(best["note"]),
                    },
                    "logs": {"csv": os.path.basename(csv_path)},
                },
                f,
                indent=2,
                ensure_ascii=True,
            )
    except Exception as _e:
        print(f"[Warning] Failed to write beam_best.json: {_e}")

    if verbose >= 1:
        print(
            f"[Beam] {tag}: best_loss={float(best['loss']):.6f} best(dist,elev,azim)=({float(best['dist']):.3f},{float(best['elev']):.3f},{float(best['azim']):.3f}) "
            f"level={best['level']} saved={os.path.basename(csv_path)}"
        )

    return float(best["dist"]), float(best["elev"]), float(best["azim"]), float(best["loss"])


def local_beam_search_on_sphere(
    *,
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    phong_renderer: MeshRenderer | None = None,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    image_ref_rgb: torch.Tensor | None = None,
    at_xyz: torch.Tensor,
    start_dist: float,
    start_elev: float,
    start_azim: float,
    device: torch.device,
    save_dir: str,
    tag: str,
    levels_override: int | None = None,
    R_gt: torch.Tensor | None = None,
    T_gt: torch.Tensor | None = None,
    ref_elev_deg: float | None = None,
) -> tuple[float, float, float, float]:
    """
    局部 Beam Search（用于替换局部贪婪，用户要求）：
    - 固定球面半径= start_dist（相机始终在以 at_xyz 为中心的球面上）
    - 多层扩展：对 beam 中每个节点沿 WORLD 的 ±x/±y/±z 方向探索 K 步（每步投影回球面）
    - 每层保留 top-B（beam width）作为下一层 beam
    - 输出全局最优（dist,elev,azim,loss），并保存日志/可视化（可选）

    设计目标：
    - 相比局部贪婪：更稳（多路径保留），但通常更耗前向评估
    说明：
    - 推荐：`VAL_LOCAL_BEAM_SCORE_MODE=full`，使用 silhouette+depth+normal 与最终 refine 的目标一致。
    - 兼容旧逻辑：`VAL_LOCAL_BEAM_SCORE_MODE=gt_geom`（sil+distance+pointcloud，依赖 GT），仅用于复现实验/对比，不建议作为最终方法。

    Env（可覆盖）：
    - VAL_LOCAL_BEAM_ENABLE: 1/0（默认 1）
    - VAL_LOCAL_BEAM_WIDTH: B（默认 128，宽度）
    - VAL_LOCAL_BEAM_LEVELS: L（默认 128，深度；若传入 levels_override 则优先使用）
    - VAL_LOCAL_BEAM_STEPS_PER_DIR: K（默认 1，每方向探索1步）
    - VAL_LOCAL_BEAM_STEP_FRAC: 步长比例（默认 0.01，实际步长=step_frac*dist*k）
    - VAL_LOCAL_BEAM_USE_SIL_SEMANTIC: 1/0（默认 1，使用轮廓+distance+pointcloud，语义损失已禁用）
    - VAL_W_SIL_LOCAL: 轮廓损失权重（默认 1.0）
    - VAL_W_SEMANTIC: 语义相似度损失权重（已禁用，固定为 0.0）
    - VAL_LOCAL_BEAM_LOG_TOP: 写 CSV 的 top 行数（默认 64）
    """
    enabled = int(os.environ.get("VAL_LOCAL_BEAM_ENABLE", "1")) != 0
    if not enabled:
        return float(start_dist), float(start_elev), float(start_azim), float("inf")

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{tag}_local_beam_candidates.csv")
    json_path = os.path.join(save_dir, f"{tag}_local_beam_best.json")

    beam_width = max(1, int(os.environ.get("VAL_LOCAL_BEAM_WIDTH", "32")))
    if levels_override is None:
        levels = max(1, int(os.environ.get("VAL_LOCAL_BEAM_LEVELS", "32")))
    else:
        levels = max(1, int(levels_override))
    steps_per_dir = max(1, int(os.environ.get("VAL_LOCAL_BEAM_STEPS_PER_DIR", "1")))
    step_frac = float(os.environ.get("VAL_LOCAL_BEAM_STEP_FRAC", "0.01"))
    
    # 自适应步长机制：根据改善情况动态调整
    adaptive_step = int(os.environ.get("VAL_LOCAL_BEAM_ADAPTIVE_STEP", "1")) != 0
    step_shrink_factor = float(os.environ.get("VAL_LOCAL_BEAM_STEP_SHRINK", "0.7"))
    step_grow_factor = float(os.environ.get("VAL_LOCAL_BEAM_STEP_GROW", "1.2"))
    min_step_frac = float(os.environ.get("VAL_LOCAL_BEAM_MIN_STEP_FRAC", "0.002"))
    max_step_frac = float(os.environ.get("VAL_LOCAL_BEAM_MAX_STEP_FRAC", "0.15"))
    initial_step_frac = step_frac
    
    # D. 统一搜索/精炼目标：局部 beam 默认用 full-loss（sil+depth+normal）打分。
    # 可选：sil（更快）或 gt_geom（旧逻辑：sil+distance+pointcloud，依赖 GT；不推荐做最终方法）。
    score_mode = os.environ.get("VAL_LOCAL_BEAM_SCORE_MODE", "full").strip().lower()  # full|sil|gt_geom
    use_sil_semantic = int(os.environ.get("VAL_LOCAL_BEAM_USE_SIL_SEMANTIC", "0")) != 0  # legacy knob (kept for backward compat)
    if score_mode == "":
        score_mode = "full"
    verbose = int(os.environ.get("VAL_LOCAL_BEAM_VERBOSE", "1"))
    log_top = int(os.environ.get("VAL_LOCAL_BEAM_LOG_TOP", "64"))
    log_top = max(0, log_top)
    
    # 早停计数器初始化
    if int(os.environ.get("VAL_LOCAL_BEAM_EARLY_STOP", "0")) != 0:
        import types
        _local_beam_early_stop = types.SimpleNamespace()
        _local_beam_early_stop.no_improve_count = 0
        _local_beam_early_stop.prev_best = float("inf")

    # B. 可选：把 DR 梯度融入局部 beam 排序（top-M 候选做极短 Adam 内循环再打分）。
    dr_inner_enable = int(os.environ.get("VAL_LOCAL_BEAM_DR_INNER_ENABLE", "0")) != 0
    dr_inner_topm = max(1, int(os.environ.get("VAL_LOCAL_BEAM_DR_TOPM", "4")))
    dr_inner_steps = max(0, int(os.environ.get("VAL_LOCAL_BEAM_DR_STEPS", "3")))
    dr_inner_lr = float(os.environ.get("VAL_LOCAL_BEAM_DR_LR", "0.02"))
    # Keep dist fixed on sphere; optimize elev/azim only.
    elev_min = float(os.environ.get("VAL_GT_ELEV_MIN", "5.0"))
    elev_max = float(os.environ.get("VAL_GT_ELEV_MAX", "85.0"))
    azim_min = float(os.environ.get("VAL_GT_AZIM_MIN", "-180.0"))
    azim_max = float(os.environ.get("VAL_GT_AZIM_MAX", "180.0"))
    # Elevation band constraint: if set (not equal to wide defaults), filter beam
    # candidates that fall outside the allowed elevation range.  This prevents
    # high-elevation seeds from sliding into a wrong low-elevation basin.
    # Set VAL_LOCAL_BEAM_ELEV_MIN_CONSTRAIN / MAX_CONSTRAIN before calling this
    # function to enable.  Defaults (-90/90) mean no constraint.
    _elev_constrain_min = float(os.environ.get("VAL_LOCAL_BEAM_ELEV_MIN_CONSTRAIN", "-90.0"))
    _elev_constrain_max = float(os.environ.get("VAL_LOCAL_BEAM_ELEV_MAX_CONSTRAIN", "90.0"))
    _use_elev_constrain = (_elev_constrain_min > -89.0 or _elev_constrain_max < 89.0)
    _azim_constrain_center = os.environ.get("VAL_LOCAL_BEAM_AZIM_CENTER", "").strip()
    _azim_constrain_half = float(os.environ.get("VAL_LOCAL_BEAM_AZIM_HALF_BAND_DEG", "180.0"))
    _use_azim_constrain = (
        _azim_constrain_center != ""
        and _azim_constrain_half < 179.0
        and int(os.environ.get("VAL_LOCAL_BEAM_AZIM_CONSTRAIN", "1")) != 0
    )
    _azim_center_deg = float(_azim_constrain_center) if _use_azim_constrain else 0.0

    # Strict no-GT mode: the GT elevation / pose must never steer the beam.
    if _strict_no_gt():
        if score_mode == "gt_geom":
            raise RuntimeError(
                "[GT-LEAK] local_beam_search_on_sphere: score_mode=gt_geom depends on the "
                "GT pose and is not allowed under VAL_STRICT_NO_GT=1."
            )
        ref_elev_deg = None
        R_gt = None
        T_gt = None

    def _elev_meets_threshold(env_key: str, default: str = "58.0") -> bool:
        """Elevation condition for END/DR/FD triggers.

        Legacy runs compared against the GT elevation (VAL_LOCAL_BEAM_COND_USE_GT_ELEV=1);
        this is a ground-truth leak and is disabled by default. The seed elevation is used.
        """
        thr = float(os.environ.get(env_key, default))
        use_gt = (
            int(os.environ.get("VAL_LOCAL_BEAM_COND_USE_GT_ELEV", "0")) != 0
            and not _strict_no_gt()
        )
        e_chk = float(ref_elev_deg) if (use_gt and ref_elev_deg is not None) else float(start_elev)
        return e_chk >= thr

    at_xyz = at_xyz.view(3).to(device)
    dist_fixed = float(start_dist)
    
    mesh_top_normal = _mesh_top_normal_from_mesh(mesh, device)
    ref_view_dir = _ref_view_direction_from_normals(
        ref_normals, image_ref_silhouette, device, R_gt=R_gt
    )
    try:
        _mesh_r_local = float(
            _mesh_radius_from_verts(mesh.verts_packed(), at_xyz)
        )
    except Exception:
        _mesh_r_local = _mesh_scale_ref_radius()
    ref_dist_hint = _ref_dist_hint_from_depth(ref_depth, image_ref_silhouette, device, mesh_r=_mesh_r_local)
    normal_prior_w = float(os.environ.get("VAL_NORMAL_PRIOR_WEIGHT", "0.35"))
    ref_view_w = float(os.environ.get("VAL_REF_VIEW_PRIOR_WEIGHT", "0.25"))

    dirs = [
        ("+x", torch.tensor([1.0, 0.0, 0.0], device=device)),
        ("-x", torch.tensor([-1.0, 0.0, 0.0], device=device)),
        ("+y", torch.tensor([0.0, 1.0, 0.0], device=device)),
        ("-y", torch.tensor([0.0, -1.0, 0.0], device=device)),
        ("+z", torch.tensor([0.0, 0.0, 1.0], device=device)),
        ("-z", torch.tensor([0.0, 0.0, -1.0], device=device)),
    ]

    def _project_to_sphere(xyz: torch.Tensor) -> torch.Tensor:
        v = xyz - at_xyz
        return at_xyz + (v / (torch.norm(v) + 1e-8)) * torch.tensor(dist_fixed, device=device)

    # init node (ensure on sphere)
    cam0 = _camera_position_from_spherical_global(
        torch.tensor(float(dist_fixed), device=device),
        torch.tensor(float(start_elev), device=device),
        torch.tensor(float(start_azim), device=device),
        at_xyz,
    )
    cam0 = _project_to_sphere(cam0)
    _d0, e0, a0 = _spherical_from_camera_position_global(cam0, at_xyz)

    # 初始化评估（与后续候选评估保持一致）
    if score_mode == "gt_geom" and phong_renderer is not None and image_ref_rgb is not None:
        ref_points_view_curr = None
        if R_gt is not None and T_gt is not None:
            try:
                _, _, _, ref_points_view_curr = render_targets(
                    mesh, R_gt, T_gt, device, int(image_ref_silhouette.shape[0]), int(image_ref_silhouette.shape[1])
                )
            except Exception:
                ref_points_view_curr = None
        cur_eval = _evaluate_pose_sil_semantic_batch_no_grad(
            mesh=mesh,
            silhouette_renderer=silhouette_renderer,
            phong_renderer=phong_renderer,
            rasterizer=rasterizer,
            image_ref_silhouette=image_ref_silhouette,
            image_ref_rgb=image_ref_rgb,
            at_xyz=at_xyz,
            dist_list=[float(dist_fixed)],
            elev_list=[float(e0)],
            azim_list=[float(a0)],
            device=device,
            R_gt=R_gt,
            T_gt=T_gt,
            ref_points_view=ref_points_view_curr,
        )
        cur_loss = float(cur_eval["loss_total"][0].cpu().item())
    elif score_mode == "sil":
        cur_eval = _evaluate_pose_sil_no_grad(
            mesh=mesh,
            silhouette_renderer=silhouette_renderer,
            image_ref_silhouette=image_ref_silhouette,
            at_xyz=at_xyz,
            dist_v=float(dist_fixed),
            elev_v=float(e0),
            azim_v=float(a0),
            device=device,
        )
        cur_loss = float(cur_eval["loss_total"])
    else:
        cur_eval = _evaluate_pose_loss_no_grad(
            mesh=mesh,
            silhouette_renderer=silhouette_renderer,
            rasterizer=rasterizer,
            image_ref_silhouette=image_ref_silhouette,
            ref_depth=ref_depth,
            ref_normals=ref_normals,
            at_xyz=at_xyz,
            dist_v=float(dist_fixed),
            elev_v=float(e0),
            azim_v=float(a0),
            device=device,
        )
        cur_loss = float(cur_eval["loss_total"])
    best = {"loss": cur_loss, "dist": float(dist_fixed), "elev": float(e0), "azim": float(a0), "level": 0, "note": "init"}

    if int(os.environ.get("VAL_LOCAL_BEAM_DIST_REFINE", "1")) != 0:
        _t_da0 = _stage_now()
        try:
            dist_scales = _resolve_local_beam_dist_scales(
                float(dist_fixed), ref_dist_hint, _mesh_r_local
            )
            for s in dist_scales:
                d_try = float(dist_fixed) * float(s)
                if d_try <= 1e-5:
                    continue
                ev_d = _evaluate_pose_loss_no_grad(
                    mesh=mesh,
                    silhouette_renderer=silhouette_renderer,
                    rasterizer=rasterizer,
                    image_ref_silhouette=image_ref_silhouette,
                    ref_depth=ref_depth,
                    ref_normals=ref_normals,
                    at_xyz=at_xyz,
                    dist_v=float(d_try),
                    elev_v=float(e0),
                    azim_v=float(a0),
                    device=device,
                )
                lv = float(ev_d["loss_total"])
                if lv < float(best["loss"]):
                    dist_fixed = float(d_try)
                    cam0 = _camera_position_from_spherical_global(
                        torch.tensor(float(dist_fixed), device=device),
                        torch.tensor(float(e0), device=device),
                        torch.tensor(float(a0), device=device),
                        at_xyz,
                    )
                    cam0 = _project_to_sphere(cam0)
                    best = {
                        "loss": lv,
                        "dist": float(dist_fixed),
                        "elev": float(e0),
                        "azim": float(a0),
                        "level": 0,
                        "note": f"dist_scale={s:.2f}",
                    }
        except Exception:
            pass
        _stage_add("t_dist_align_s", _stage_now() - _t_da0)

    beam = [(float(best["loss"]), cam0.detach().clone())]

    if verbose >= 1:
        print(f"\n{'='*80}")
        print(f"[LocalBeam] {tag} - 局部Beam搜索")
        print(f"{'='*80}")
        print(f"  配置: width={beam_width}, levels={levels}, steps_per_dir={steps_per_dir}, step_frac={step_frac:.4f}")
        print(f"  模式: score_mode={score_mode} | dr_inner={int(dr_inner_enable)} (topm={dr_inner_topm}, steps={dr_inner_steps}, lr={dr_inner_lr:.4g})")
        print(f"\n  [初始化] 初始位姿: dist={best['dist']:.3f}, elev={best['elev']:.3f}, azim={best['azim']:.3f}")
        
        # Print detailed loss breakdown for initial pose
        if score_mode == "gt_geom" and "loss_sil" in cur_eval and "loss_total" in cur_eval:
            loss_sil_val = float(cur_eval["loss_sil"][0].cpu().item()) if hasattr(cur_eval["loss_sil"], "__getitem__") else float(cur_eval["loss_sil"].cpu().item())
            loss_semantic_val = float(cur_eval.get("loss_semantic", torch.tensor(0.0))[0].cpu().item()) if "loss_semantic" in cur_eval and hasattr(cur_eval["loss_semantic"], "__getitem__") else float(cur_eval.get("loss_semantic", torch.tensor(0.0)).cpu().item())
            loss_distance_val = float(cur_eval.get("loss_distance", torch.tensor(0.0))[0].cpu().item()) if "loss_distance" in cur_eval and hasattr(cur_eval["loss_distance"], "__getitem__") else float(cur_eval.get("loss_distance", torch.tensor(0.0)).cpu().item())
            loss_pointcloud_val = float(cur_eval.get("loss_pointcloud", torch.tensor(0.0))[0].cpu().item()) if "loss_pointcloud" in cur_eval and hasattr(cur_eval["loss_pointcloud"], "__getitem__") else float(cur_eval.get("loss_pointcloud", torch.tensor(0.0)).cpu().item())
            w_sil_val = float(os.environ.get("VAL_W_SIL_LOCAL", "1.0"))
            w_semantic_val = float(os.environ.get("VAL_W_SEMANTIC", "1.0"))
            w_distance_val = float(os.environ.get("VAL_W_DISTANCE", "0.1"))
            w_pointcloud_val = float(os.environ.get("VAL_W_POINTCLOUD", "0.1"))
            print(_format_loss_details(
                loss_total=cur_loss,
                loss_sil=loss_sil_val,
                loss_semantic=loss_semantic_val if loss_semantic_val > 0 else None,
                loss_distance=loss_distance_val if loss_distance_val > 0 else None,
                loss_pointcloud=loss_pointcloud_val if loss_pointcloud_val > 0 else None,
                w_sil=w_sil_val,
                w_semantic=w_semantic_val,
                w_distance=w_distance_val,
                w_pointcloud=w_pointcloud_val,
                prefix="  ",
                verbose=(verbose >= 2)
            ))
        else:
            print(f"  初始总损失: {cur_loss:.6f}")

    rows = []
    rows.append(["level", "parent_rank", "dir", "k", "dist", "elev_deg", "azim_deg", "loss_total"])

    # Legacy GT-geometry loss needs GT point cloud; keep optional.
    ref_points_view_batch = None
    if score_mode == "gt_geom" and R_gt is not None and T_gt is not None:
        try:
            _, _, _, ref_points_view_batch = render_targets(
                mesh, R_gt, T_gt, device, int(image_ref_silhouette.shape[0]), int(image_ref_silhouette.shape[1])
            )
        except Exception:
            ref_points_view_batch = None

    def _wrap_azim(a: float) -> float:
        return float(((float(a) + 180.0) % 360.0) - 180.0)

    def _dr_inner_refine_elev_azim(*, elev0: float, azim0: float) -> tuple[float, float, float]:
        """
        B. DR inner-loop refinement for a single candidate on fixed-radius sphere.
        Returns (loss_after, elev_deg_after, azim_deg_after).
        """
        if dr_inner_steps <= 0:
            n_steps = max(0, int(os.environ.get("VAL_LOCAL_BEAM_DR_STEPS", "0")))
        else:
            n_steps = max(0, int(os.environ.get("VAL_LOCAL_BEAM_DR_STEPS", str(dr_inner_steps))))
        if n_steps <= 0:
            return float("inf"), float(elev0), float(azim0)
        lr_local = float(os.environ.get("VAL_LOCAL_BEAM_DR_LR", str(dr_inner_lr)))
        # trainable scalars
        elev_t = torch.tensor(float(elev0), device=device, dtype=torch.float32, requires_grad=True)
        azim_t = torch.tensor(float(azim0), device=device, dtype=torch.float32, requires_grad=True)
        opt = torch.optim.Adam([elev_t, azim_t], lr=float(lr_local))

        # weights: prefer "end" weights when schedule is enabled, to reduce residual error
        use_sched = int(os.environ.get("VAL_LOSS_SCHEDULE", "1")) != 0
        if use_sched:
            w_sil = float(os.environ.get("VAL_W_SIL_END", "0.2"))
            w_depth = float(os.environ.get("VAL_W_DEPTH_END", "2.0"))
            w_normal = float(os.environ.get("VAL_W_NORMAL_END", "1.0"))
        else:
            w_sil = float(os.environ.get("VAL_W_SIL", "0.2"))
            w_depth = float(os.environ.get("VAL_W_DEPTH", "2.0"))
            w_normal = float(os.environ.get("VAL_W_NORMAL", "1.0"))

        ref_sil = torch.clamp(image_ref_silhouette, 0, 1)
        if ref_sil.dim() == 3 and ref_sil.shape[-1] == 4:
            ref_sil = ref_sil[..., 3]
        ref_sil = ref_sil.squeeze()
        ref_d = ref_depth.squeeze()
        ref_n = ref_normals.squeeze()

        for _ in range(int(n_steps)):
            opt.zero_grad(set_to_none=True)
            Rb, Tb = look_at_view_transform(
                dist=torch.tensor([float(dist_fixed)], device=device),
                elev=elev_t.view(1),
                azim=azim_t.view(1),
                at=at_xyz.view(1, 3),
                device=device,
            )
            img = silhouette_renderer(meshes_world=mesh, R=Rb, T=Tb)
            alpha = _postprocess_silhouette_alpha(torch.clamp(img[0, ..., 3], 0, 1))
            # optional: local inner-loop sharpening (usually OFF; main refine handles it)
            if int(os.environ.get("VAL_LOCAL_BEAM_DR_SHARPEN", "0")) != 0:
                tau = float(os.environ.get("VAL_LOCAL_BEAM_DR_SHARP_TAU", "0.05"))
                tau = max(1e-4, tau)
                alpha = torch.sigmoid((alpha - 0.5) / tau)
            loss_mse = torch.mean((alpha - ref_sil) ** 2)
            inter = (alpha * ref_sil).sum()
            union = alpha.sum() + ref_sil.sum() - inter
            loss_iou = 1.0 - (inter / (union + 1e-8))
            w_iou = float(os.environ.get("VAL_W_IOU", "0.1"))
            scale_cfg = os.environ.get("VAL_LOSS_SCALE", "1.0")
            if scale_cfg.strip().lower() == "auto":
                loss_scale = float(alpha.shape[0] * alpha.shape[1])
            else:
                loss_scale = float(scale_cfg)
            loss_sil = (1.0 - w_iou) * (loss_mse * loss_scale) + w_iou * loss_iou

            cams = FoVPerspectiveCameras(device=device, R=Rb, T=Tb)
            fr = rasterizer(meshes_world=mesh, cameras=cams)
            vis_mask = (fr.pix_to_face[..., 0] >= 0).float()[0]
            depth = (fr.zbuf[..., 0][0] * vis_mask)
            valid = (vis_mask > 0.5) & (ref_d > 0)
            if valid.any():
                beta = float(os.environ.get("VAL_DEPTH_BETA", "0.05"))
                depth_norm_eps = float(os.environ.get("VAL_DEPTH_NORM_EPS", "1e-6"))
                ref_scale = ref_d[valid].mean().detach().clamp(min=depth_norm_eps)
                depth_n = depth / ref_scale
                ref_depth_n = ref_d / ref_scale
                loss_depth = torch.nn.functional.smooth_l1_loss(depth_n[valid], ref_depth_n[valid], beta=beta, reduction="mean")

                verts_normals = mesh.verts_normals_packed()
                faces = mesh.faces_packed()
                verts_normals_view = torch.matmul(verts_normals, Rb[0].T)
                faces_verts_normals = verts_normals_view[faces]
                pix_normals = interpolate_face_attributes(fr.pix_to_face, fr.bary_coords, faces_verts_normals)
                normals = pix_normals[..., 0, :][0]
                normals = torch.nn.functional.normalize(normals, dim=-1) * vis_mask.unsqueeze(-1)
                cos = (normals * ref_n).sum(dim=-1).clamp(-1, 1)
                loss_normal = (1.0 - cos)[valid].mean()
            else:
                loss_depth = torch.tensor(0.0, device=device)
                loss_normal = torch.tensor(0.0, device=device)

            loss = w_sil * loss_sil + w_depth * loss_depth + w_normal * loss_normal
            loss.backward()
            torch.nn.utils.clip_grad_norm_([elev_t, azim_t], max_norm=float(os.environ.get("VAL_CLIP_NORM", "10.0")))
            opt.step()

            # project back into feasible ranges
            with torch.no_grad():
                elev_t.clamp_(min=float(elev_min), max=float(elev_max))
                az = _wrap_azim(float(azim_t.item()))
                az = float(np.clip(az, float(azim_min), float(azim_max)))
                azim_t.copy_(torch.tensor(az, device=device, dtype=azim_t.dtype))

        # final no_grad evaluation (same family as refine)
        with torch.no_grad():
            out = _evaluate_pose_loss_no_grad(
                mesh=mesh,
                silhouette_renderer=silhouette_renderer,
                rasterizer=rasterizer,
                image_ref_silhouette=image_ref_silhouette,
                ref_depth=ref_depth,
                ref_normals=ref_normals,
                at_xyz=at_xyz,
                dist_v=float(dist_fixed),
                elev_v=float(elev_t.item()),
                azim_v=float(azim_t.item()),
                device=device,
            )
        return float(out["loss_total"]), float(elev_t.item()), float(azim_t.item())

    for lvl in range(1, levels + 1):
        # 记录本层开始时的 best loss，用于自适应步长判断
        prev_best_at_layer_start = float(best["loss"])
        
        step_len = float(step_frac) * float(dist_fixed)

        cand_xyz: list[torch.Tensor] = []
        cand_meta: list[tuple[int, str, int]] = []
        cand_dist: list[float] = []
        cand_elev: list[float] = []
        cand_azim: list[float] = []

        for parent_rank, (_lp, xyz_p) in enumerate(beam):
            for dn, dv in dirs:
                for k in range(1, steps_per_dir + 1):
                    xyz_c = _project_to_sphere(xyz_p + dv * (step_len * float(k)))
                    _d, _e, _a = _spherical_from_camera_position_global(xyz_c, at_xyz)
                    cand_xyz.append(xyz_c)
                    cand_meta.append((int(parent_rank), str(dn), int(k)))
                    cand_dist.append(float(dist_fixed))
                    cand_elev.append(float(_e))
                    cand_azim.append(float(_a))

        # Elevation band constraint: prevent beam from sliding out of seed's band.
        if _use_elev_constrain and len(cand_elev) > 0:
            _vi = [i for i, e in enumerate(cand_elev)
                   if _elev_constrain_min <= e <= _elev_constrain_max]
            if len(_vi) > 0:
                cand_xyz  = [cand_xyz[i]  for i in _vi]
                cand_meta = [cand_meta[i] for i in _vi]
                cand_dist = [cand_dist[i] for i in _vi]
                cand_elev = [cand_elev[i] for i in _vi]
                cand_azim = [cand_azim[i] for i in _vi]

        if _use_azim_constrain and len(cand_azim) > 0:
            _vi = [
                i for i, a in enumerate(cand_azim)
                if _azim_diff_deg(float(a), _azim_center_deg) <= _azim_constrain_half
            ]
            if len(_vi) > 0:
                cand_xyz  = [cand_xyz[i]  for i in _vi]
                cand_meta = [cand_meta[i] for i in _vi]
                cand_dist = [cand_dist[i] for i in _vi]
                cand_elev = [cand_elev[i] for i in _vi]
                cand_azim = [cand_azim[i] for i in _vi]

        if len(cand_xyz) == 0:
            break

        # 候选评估：按 score_mode 选择（推荐 full）
        if score_mode == "gt_geom" and phong_renderer is not None and image_ref_rgb is not None:
            evb = _evaluate_pose_sil_semantic_batch_no_grad(
                mesh=mesh,
                silhouette_renderer=silhouette_renderer,
                phong_renderer=phong_renderer,
                rasterizer=rasterizer,
                image_ref_silhouette=image_ref_silhouette,
                image_ref_rgb=image_ref_rgb,
                at_xyz=at_xyz,
                dist_list=cand_dist,
                elev_list=cand_elev,
                azim_list=cand_azim,
                device=device,
                R_gt=R_gt,
                T_gt=T_gt,
                ref_points_view=ref_points_view_batch,
            )
            losses = evb["loss_total"].detach().cpu().numpy().astype(np.float64)
            scored = [(float(losses[i]), int(i)) for i in range(int(losses.size))]
        elif score_mode == "sil":
            evb_sil = _evaluate_pose_sil_batch_no_grad(
                mesh=mesh,
                silhouette_renderer=silhouette_renderer,
                image_ref_silhouette=image_ref_silhouette,
                at_xyz=at_xyz,
                dist_list=cand_dist,
                elev_list=cand_elev,
                azim_list=cand_azim,
                device=device,
            )
            losses = evb_sil["loss_total"].detach().cpu().numpy().astype(np.float64)
            scored = [(float(losses[i]), int(i)) for i in range(int(losses.size))]
        else:
            evb_full = _evaluate_pose_losses_batch_no_grad(
                mesh=mesh,
                silhouette_renderer=silhouette_renderer,
                rasterizer=rasterizer,
                image_ref_silhouette=image_ref_silhouette,
                ref_depth=ref_depth,
                ref_normals=ref_normals,
                at_xyz=at_xyz,
                dist_list=cand_dist,
                elev_list=cand_elev,
                azim_list=cand_azim,
                device=device,
                use_sil_normals_only=False,
            )
            losses = evb_full["loss_total"].detach().cpu().numpy().astype(np.float64)
            
            # 算法优化：加入 mesh 法线先验惩罚
            # 对于高俯仰视角，相机方向应该与 mesh 顶面法线大致相反
            normal_prior_losses = np.zeros(len(cand_elev), dtype=np.float64)
            if len(cand_elev) > 0:
                for j in range(len(cand_elev)):
                    geom_prior = _candidate_geometry_prior_score(
                        float(cand_elev[j]),
                        float(cand_azim[j]),
                        mesh_top_normal,
                        ref_view_dir,
                        device,
                    )
                    normal_prior_losses[j] = geom_prior
            
            losses = losses + normal_prior_losses
            
            # 高俯仰种子早期临时提高 depth/normal 权重
            high_elev_boost = False
            if len(cand_elev) > 0:
                avg_elev = np.mean(cand_elev)
                if avg_elev >= 45.0:  # 高俯仰阈值
                    high_elev_boost = True
            
            if high_elev_boost:
                boost_depth = float(os.environ.get("VAL_HIGH_ELEV_BOOST_DEPTH", "2.5"))
                boost_normal = float(os.environ.get("VAL_HIGH_ELEV_BOOST_NORMAL", "1.5"))
                # 重新计算加权损失
                w_sil = float(os.environ.get("VAL_W_SIL_START", "1.0"))
                w_depth = float(os.environ.get("VAL_W_DEPTH_START", "1.5")) * boost_depth
                w_normal = float(os.environ.get("VAL_W_NORMAL_START", "0.8")) * boost_normal
                sil_loss = evb_full["loss_sil"].detach().cpu().numpy().astype(np.float64)
                depth_loss = evb_full["loss_depth"].detach().cpu().numpy().astype(np.float64)
                normal_loss = evb_full["loss_normal"].detach().cpu().numpy().astype(np.float64)
                losses = w_sil * sil_loss + w_depth * depth_loss + w_normal * normal_loss + normal_prior_losses

                # 更彻底的算法改进：在 LocalBeam 内部对高俯仰候选加入 RGB 纹理 loss
                # 这是目前最直接针对“高俯仰滑落”的算法改进
                if phong_renderer is not None and image_ref_rgb is not None:
                    w_rgb_local = float(os.environ.get("VAL_LOCAL_BEAM_RGB_WEIGHT", "6.0"))
                    ref_rgb_t = torch.clamp(image_ref_rgb.to(device), 0, 1)
                    if ref_rgb_t.dim() == 4:
                        ref_rgb_t = ref_rgb_t[0]
                    H_s = silhouette_renderer.rasterizer.raster_settings.image_size[0]
                    W_s = silhouette_renderer.rasterizer.raster_settings.image_size[1]
                    ref_rgb_ds = torch.nn.functional.interpolate(
                        ref_rgb_t.permute(2, 0, 1).unsqueeze(0),
                        size=(int(H_s), int(W_s)),
                        mode="bilinear",
                        align_corners=False,
                    )[0].permute(1, 2, 0)
                    ref_sil_base = torch.clamp(image_ref_silhouette, 0, 1)
                    if ref_sil_base.dim() == 3 and ref_sil_base.shape[-1] == 4:
                        ref_sil_base = ref_sil_base[..., 3]
                    ref_sil_base = ref_sil_base.squeeze().to(device)

                    rgb_losses = []
                    for j in range(len(cand_elev)):
                        try:
                            with torch.no_grad():
                                Rb, Tb = look_at_view_transform(
                                    dist=torch.tensor([float(cand_dist[j])], device=device),
                                    elev=torch.tensor([float(cand_elev[j])], device=device),
                                    azim=torch.tensor([float(cand_azim[j])], device=device),
                                    at=at_xyz.view(1, 3),
                                    device=device,
                                )
                                rgb_r = torch.clamp(phong_renderer(meshes_world=mesh, R=Rb, T=Tb)[0, ..., :3], 0, 1)
                                sil_r = torch.clamp(silhouette_renderer(meshes_world=mesh, R=Rb, T=Tb)[0, ..., 3], 0, 1)
                                mask = ((ref_sil_base > 0.05) | (sil_r > 0.05)).float().unsqueeze(-1)
                                denom = float(mask.sum().item()) * 3.0 + 1e-8
                                rgb_l = float(((rgb_r - ref_rgb_ds) ** 2 * mask).sum().detach().cpu().item()) / denom
                        except Exception:
                            rgb_l = 0.0
                        rgb_losses.append(rgb_l)
                    rgb_losses = np.array(rgb_losses, dtype=np.float64)
                    losses = losses + w_rgb_local * rgb_losses
            
            scored = [(float(losses[i]), int(i)) for i in range(int(losses.size))]

        scored.sort(key=lambda x: x[0])

        # B. DR inner-loop re-ranking on top-M candidates (optional)
        if dr_inner_enable and dr_inner_steps > 0 and len(scored) > 0:
            m = min(int(dr_inner_topm), len(scored))
            refined_map: dict[int, float] = {}
            for rr in range(m):
                _lv, j = scored[rr]
                try:
                    loss_after, e_after, a_after = _dr_inner_refine_elev_azim(elev0=float(cand_elev[j]), azim0=float(cand_azim[j]))
                    refined_map[int(j)] = float(loss_after)
                    cand_elev[j] = float(e_after)
                    cand_azim[j] = float(a_after)
                    # update xyz to refined pose (keep on sphere)
                    cam_ref = _camera_position_from_spherical_global(
                        torch.tensor(float(dist_fixed), device=device),
                        torch.tensor(float(e_after), device=device),
                        torch.tensor(float(a_after), device=device),
                        at_xyz,
                    )
                    cand_xyz[j] = _project_to_sphere(cam_ref).detach()
                except Exception:
                    continue
            if refined_map:
                scored = [(refined_map.get(int(j), float(lv)), int(j)) for (lv, j) in scored]
                scored.sort(key=lambda x: x[0])

        # CSV top
        try:
            t = min(int(log_top), len(scored))
            for r in range(t):
                lv, j = scored[r]
                pr, dn, kk = cand_meta[j]
                rows.append([int(lvl), int(pr), dn, int(kk), float(dist_fixed), float(cand_elev[j]), float(cand_azim[j]), float(lv)])
        except Exception:
            pass

        # Last layer: optional FD polish on top-M beam candidates (no backward; 4 renders each)
        fd_topm = max(0, int(os.environ.get("VAL_LOCAL_BEAM_LAST_LAYER_FD_TOPM", "0")))
        if (
            fd_topm > 0
            and int(lvl) == int(levels)
            and score_mode not in ("sil", "gt_geom")
            and (
                int(os.environ.get("VAL_LOCAL_BEAM_LAST_LAYER_END_COND", "0")) == 0
                or _elev_meets_threshold("VAL_LOCAL_BEAM_LAST_LAYER_END_MIN_ELEV")
            )
        ):
            fd_sil = int(os.environ.get("VAL_LOCAL_BEAM_LAST_LAYER_FD_SIL_ONLY", "0")) != 0
            for _lv_fd, j_fd in scored[:fd_topm]:
                try:
                    fd_out = _sphere_fd_descent_step(
                        mesh=mesh,
                        silhouette_renderer=silhouette_renderer,
                        rasterizer=rasterizer,
                        image_ref_silhouette=image_ref_silhouette,
                        ref_depth=ref_depth,
                        ref_normals=ref_normals,
                        at_xyz=at_xyz,
                        dist_c=float(dist_fixed),
                        elev_c=float(cand_elev[j_fd]),
                        azim_c=float(cand_azim[j_fd]),
                        elev_min=elev_min,
                        elev_max=elev_max,
                        azim_min=azim_min,
                        azim_max=azim_max,
                        device=device,
                        use_sil_only=fd_sil,
                    )
                    if fd_out is not None and float(fd_out[3]) < float(_lv_fd):
                        cand_elev[j_fd] = float(fd_out[1])
                        cand_azim[j_fd] = float(fd_out[2])
                        scored = [
                            (float(fd_out[3]) if int(jj) == int(j_fd) else float(lv), int(jj))
                            for (lv, jj) in scored
                        ]
                        scored.sort(key=lambda x: x[0])
                except Exception:
                    continue

        # next beam
        use_end_layer_score = (
            int(os.environ.get("VAL_LOCAL_BEAM_LAST_LAYER_END_SCORE", "0")) != 0
            and int(lvl) == int(levels)
            and score_mode not in ("sil", "gt_geom")
            and (
                int(os.environ.get("VAL_LOCAL_BEAM_LAST_LAYER_END_COND", "0")) == 0
                or _elev_meets_threshold("VAL_LOCAL_BEAM_LAST_LAYER_END_MIN_ELEV")
            )
        )
        next_beam = []
        for lv, j in scored[:beam_width]:
            lv_use = float(lv)
            if use_end_layer_score:
                try:
                    ev_end = _evaluate_pose_loss_no_grad(
                        mesh=mesh,
                        silhouette_renderer=silhouette_renderer,
                        rasterizer=rasterizer,
                        image_ref_silhouette=image_ref_silhouette,
                        ref_depth=ref_depth,
                        ref_normals=ref_normals,
                        at_xyz=at_xyz,
                        dist_v=float(dist_fixed),
                        elev_v=float(cand_elev[j]),
                        azim_v=float(cand_azim[j]),
                        device=device,
                        progress=1.0,
                    )
                    lv_use = float(ev_end["loss_total"])
                except Exception:
                    lv_use = float(lv)
            next_beam.append((float(lv_use), cand_xyz[j].detach().clone()))
            if float(lv_use) < float(best["loss"]):
                best.update(
                    {
                        "loss": float(lv_use),
                        "dist": float(dist_fixed),
                        "elev": float(cand_elev[j]),
                        "azim": float(cand_azim[j]),
                        "level": int(lvl),
                        "note": (
                            f"parent={cand_meta[j][0]} dir={cand_meta[j][1]} k={cand_meta[j][2]}"
                            + (" end_score=1" if use_end_layer_score else "")
                        ),
                    }
                )
        beam = next_beam

        # 自适应步长调整：如果本层 best 有改善，尝试增大步长加速探索
        if adaptive_step and float(best["loss"]) < prev_best_at_layer_start - 1e-6:
            step_frac = min(step_frac * step_grow_factor, max_step_frac)
        elif adaptive_step:
            step_frac = max(step_frac * step_shrink_factor, min_step_frac)

        # 早停机制：连续多层无改善则停止
        if int(os.environ.get("VAL_LOCAL_BEAM_EARLY_STOP", "0")) != 0:
            early_stop_patience = int(os.environ.get("VAL_LOCAL_BEAM_EARLY_STOP_PATIENCE", "8"))
            
            # 使用局部变量跟踪早停状态
            if "early_stop_state" not in locals():
                early_stop_state = {"no_improve_count": 0, "prev_best": float("inf")}
            
            if float(best["loss"]) < early_stop_state["prev_best"] - 1e-6:
                early_stop_state["no_improve_count"] = 0
                early_stop_state["prev_best"] = float(best["loss"])
            else:
                early_stop_state["no_improve_count"] += 1
            
            if early_stop_state["no_improve_count"] >= early_stop_patience:
                if verbose >= 1:
                    print(f"  [早停] 连续 {early_stop_patience} 层无改善，停止搜索")
                break

        if verbose >= 1 and len(beam) > 0:
            print(f"  [层级 {lvl}/{levels}] 最佳损失: {best['loss']:.6f} | Beam首位: {beam[0][0]:.6f}")
            if verbose >= 2 and use_sil_semantic and len(scored) > 0 and "evb" in locals():
                # Print detailed loss for the best candidate at this level (if available)
                best_idx = scored[0][1]
                if best_idx < len(cand_meta) and "loss_sil" in evb:
                    try:
                        loss_sil_val = float(evb["loss_sil"][best_idx].cpu().item())
                        loss_semantic_val = float(evb.get("loss_semantic", torch.tensor(0.0))[best_idx].cpu().item()) if "loss_semantic" in evb else 0.0
                        loss_distance_val = float(evb.get("loss_distance", torch.tensor(0.0))[best_idx].cpu().item()) if "loss_distance" in evb else 0.0
                        loss_pointcloud_val = float(evb.get("loss_pointcloud", torch.tensor(0.0))[best_idx].cpu().item()) if "loss_pointcloud" in evb else 0.0
                        w_sil_val = float(os.environ.get("VAL_W_SIL_LOCAL", "1.0"))
                        w_semantic_val = float(os.environ.get("VAL_W_SEMANTIC", "1.0"))
                        w_distance_val = float(os.environ.get("VAL_W_DISTANCE", "0.1"))
                        w_pointcloud_val = float(os.environ.get("VAL_W_POINTCLOUD", "0.1"))
                        print(_format_loss_details(
                            loss_total=best['loss'],
                            loss_sil=loss_sil_val,
                            loss_semantic=loss_semantic_val if loss_semantic_val > 0 else None,
                            loss_distance=loss_distance_val if loss_distance_val > 0 else None,
                            loss_pointcloud=loss_pointcloud_val if loss_pointcloud_val > 0 else None,
                            w_sil=w_sil_val,
                            w_semantic=w_semantic_val,
                            w_distance=w_distance_val,
                            w_pointcloud=w_pointcloud_val,
                            prefix="    ",
                            verbose=False
                        ))
                    except Exception:
                        pass
        if len(beam) == 0:
            break

    # v12b: short differentiable elev/azim polish after beam (fixed dist on sphere)
    _dr_cond_ok = (
        int(os.environ.get("VAL_LOCAL_BEAM_FINAL_DR_COND_ENABLE", "0")) == 0
        or _elev_meets_threshold("VAL_LOCAL_BEAM_FINAL_DR_MIN_SEED_ELEV")
    )
    if (
        score_mode not in ("sil", "gt_geom")
        and int(os.environ.get("VAL_LOCAL_BEAM_FINAL_DR_ENABLE", "0")) != 0
        and _dr_cond_ok
    ):
        _prev_dr_steps = os.environ.get("VAL_LOCAL_BEAM_DR_STEPS", "")
        _prev_dr_lr = os.environ.get("VAL_LOCAL_BEAM_DR_LR", "")
        try:
            os.environ["VAL_LOCAL_BEAM_DR_STEPS"] = str(
                max(1, int(os.environ.get("VAL_LOCAL_BEAM_FINAL_DR_STEPS", "4")))
            )
            os.environ["VAL_LOCAL_BEAM_DR_LR"] = str(
                float(os.environ.get("VAL_LOCAL_BEAM_FINAL_DR_LR", "0.015"))
            )
            loss_dr, e_dr, a_dr = _dr_inner_refine_elev_azim(
                elev0=float(best["elev"]), azim0=float(best["azim"])
            )
            if float(loss_dr) < float(best["loss"]):
                best.update(
                    {
                        "loss": float(loss_dr),
                        "elev": float(e_dr),
                        "azim": float(a_dr),
                        "note": str(best.get("note", "")) + " final_dr=1",
                    }
                )
        except Exception as _e_fdr:
            if verbose >= 2:
                print(f"[LocalBeam] {tag}: final DR refine skipped: {_e_fdr}")
        finally:
            if _prev_dr_steps != "":
                os.environ["VAL_LOCAL_BEAM_DR_STEPS"] = _prev_dr_steps
            else:
                os.environ.pop("VAL_LOCAL_BEAM_DR_STEPS", None)
            if _prev_dr_lr != "":
                os.environ["VAL_LOCAL_BEAM_DR_LR"] = _prev_dr_lr
            else:
                os.environ.pop("VAL_LOCAL_BEAM_DR_LR", None)

    # write logs
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for row in rows:
                w.writerow(row)
    except Exception as _e:
        print(f"[Warning] Failed to write local_beam_candidates.csv: {_e}")

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tag": tag,
                    "local_beam": {
                        "width": int(beam_width),
                        "levels": int(levels),
                        "levels_override": None if levels_override is None else int(levels_override),
                        "steps_per_dir": int(steps_per_dir),
                        "step_frac": float(step_frac),
                        "use_sil_semantic": bool(use_sil_semantic),
                    },
                    "best": {
                        "loss_total": float(best["loss"]),
                        "dist": float(best["dist"]),
                        "elev_deg": float(best["elev"]),
                        "azim_deg": float(best["azim"]),
                        "level": int(best["level"]),
                        "note": str(best["note"]),
                    },
                    "logs": {"csv": os.path.basename(csv_path)},
                },
                f,
                indent=2,
                ensure_ascii=True,
            )
    except Exception as _e:
        print(f"[Warning] Failed to write local_beam_best.json: {_e}")

    if verbose >= 1:
        print(f"\n{'='*80}")
        print(f"[LocalBeam] {tag} - 搜索结果")
        print(f"{'='*80}")
        print(f"  最佳位姿: dist={float(best['dist']):.3f}, elev={float(best['elev']):.3f}, azim={float(best['azim']):.3f}")
        print(f"  最佳损失: {float(best['loss']):.6f} (层级 {best['level']})")
        print(f"{'='*80}\n")

    # optional: save best artifacts for quick inspection (if phong_renderer provided)
    try:
        if int(os.environ.get("VAL_LOCAL_BEAM_SAVE_BEST", "1")) != 0:
            _render_and_save_best_artifacts(
                mesh=mesh,
                silhouette_renderer=silhouette_renderer,
                rasterizer=rasterizer,
                phong_renderer=phong_renderer,
                at_xyz=at_xyz,
                dist_v=float(best["dist"]),
                elev_v=float(best["elev"]),
                azim_v=float(best["azim"]),
                device=device,
                save_dir=save_dir,
                prefix="best_local_beam",
            )
    except Exception as _e:
        if verbose >= 2:
            print(f"[Warning] Failed to save best_local_beam artifacts: {_e}")

    return float(best["dist"]), float(best["elev"]), float(best["azim"]), float(best["loss"])


def local_beam_search_from_seeds_on_sphere(
    *,
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    phong_renderer: MeshRenderer | None = None,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    image_ref_rgb: torch.Tensor | None = None,
    at_xyz: torch.Tensor,
    seeds: list[tuple[float, float, float, float]],
    device: torch.device,
    save_dir: str,
    tag: str,
    levels_override: int | None = None,
    R_gt: torch.Tensor | None = None,
    T_gt: torch.Tensor | None = None,
) -> tuple[float, float, float, float]:
    """
    多起点的局部 Beam Search（球面空间）：
    - 输入 seeds: [(dist,elev,azim,loss_hint), ...]
    - 将这些 seed 投影到同一球面半径（用第一个 seed 的 dist 作为固定半径）
    - 先对 seeds 做一次评估（sil-only 或 full-loss，复用 local_beam 的 env），选 top beam_width 作为初始 beam
    - 然后执行与 local_beam_search_on_sphere 相同的 beam 扩展/筛选流程

    用途（对应你的需求）：
    - 对 TOPK seeds 做“深度+广度”探索（例如 100 层）找最优 seed 位姿；
    - 再从该位姿继续 beam（例如 2000 层）进一步精炼。
    """
    if len(seeds) <= 0:
        return float("inf"), 0.0, 0.0, float("inf")

    enabled = int(os.environ.get("VAL_LOCAL_BEAM_ENABLE", "1")) != 0
    if not enabled:
        d0, e0, a0, _ = seeds[0]
        return float(d0), float(e0), float(a0), float("inf")

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{tag}_local_beam_from_seeds.csv")
    json_path = os.path.join(save_dir, f"{tag}_local_beam_from_seeds_best.json")

    # Reuse the same env knobs as local_beam_search_on_sphere.
    beam_width = max(1, int(os.environ.get("VAL_LOCAL_BEAM_WIDTH", "128")))
    if levels_override is None:
        levels = max(1, int(os.environ.get("VAL_LOCAL_BEAM_LEVELS", "128")))
    else:
        levels = max(1, int(levels_override))
    steps_per_dir = max(1, int(os.environ.get("VAL_LOCAL_BEAM_STEPS_PER_DIR", "1")))
    step_frac = float(os.environ.get("VAL_LOCAL_BEAM_STEP_FRAC", "0.01"))
    # Keep consistent with local_beam_search_on_sphere (D): default use full-loss.
    score_mode = os.environ.get("VAL_LOCAL_BEAM_SCORE_MODE", "full").strip().lower()  # full|sil|gt_geom
    use_sil_semantic = int(os.environ.get("VAL_LOCAL_BEAM_USE_SIL_SEMANTIC", "0")) != 0  # legacy
    verbose = int(os.environ.get("VAL_LOCAL_BEAM_VERBOSE", "1"))

    # Strict no-GT mode: GT pose must never enter the beam scoring.
    if _strict_no_gt():
        if score_mode == "gt_geom":
            raise RuntimeError(
                "[GT-LEAK] local_beam_search_from_seeds_on_sphere: score_mode=gt_geom depends "
                "on the GT pose and is not allowed under VAL_STRICT_NO_GT=1."
            )
        R_gt = None
        T_gt = None

    at_xyz = at_xyz.view(3).to(device)
    dist_fixed = float(seeds[0][0])

    dirs = [
        ("+x", torch.tensor([1.0, 0.0, 0.0], device=device)),
        ("-x", torch.tensor([-1.0, 0.0, 0.0], device=device)),
        ("+y", torch.tensor([0.0, 1.0, 0.0], device=device)),
        ("-y", torch.tensor([0.0, -1.0, 0.0], device=device)),
        ("+z", torch.tensor([0.0, 0.0, 1.0], device=device)),
        ("-z", torch.tensor([0.0, 0.0, -1.0], device=device)),
    ]

    def _project_to_sphere(xyz: torch.Tensor) -> torch.Tensor:
        v = xyz - at_xyz
        return at_xyz + (v / (torch.norm(v) + 1e-8)) * torch.tensor(dist_fixed, device=device)

    # 1) Evaluate seeds and build initial beam
    d0 = [float(s[0]) for s in seeds]
    e0 = [float(s[1]) for s in seeds]
    a0 = [float(s[2]) for s in seeds]
    if score_mode == "gt_geom" and phong_renderer is not None and image_ref_rgb is not None:
        ref_points_view_seeds = None
        if R_gt is not None and T_gt is not None:
            try:
                _, _, _, ref_points_view_seeds = render_targets(
                    mesh, R_gt, T_gt, device, int(image_ref_silhouette.shape[0]), int(image_ref_silhouette.shape[1])
                )
            except Exception:
                ref_points_view_seeds = None
        evb = _evaluate_pose_sil_semantic_batch_no_grad(
            mesh=mesh,
            silhouette_renderer=silhouette_renderer,
            phong_renderer=phong_renderer,
            rasterizer=rasterizer,
            image_ref_silhouette=image_ref_silhouette,
            image_ref_rgb=image_ref_rgb,
            at_xyz=at_xyz,
            dist_list=[float(dist_fixed)] * len(d0),
            elev_list=e0,
            azim_list=a0,
            device=device,
            R_gt=R_gt,
            T_gt=T_gt,
            ref_points_view=ref_points_view_seeds,
        )
        losses = evb["loss_total"].detach().cpu().numpy().astype(np.float64)
        order = np.argsort(losses)
    elif score_mode == "sil":
        evb_sil = _evaluate_pose_sil_batch_no_grad(
            mesh=mesh,
            silhouette_renderer=silhouette_renderer,
            image_ref_silhouette=image_ref_silhouette,
            at_xyz=at_xyz,
            dist_list=[float(dist_fixed)] * len(d0),
            elev_list=e0,
            azim_list=a0,
            device=device,
        )
        losses = evb_sil["loss_total"].detach().cpu().numpy().astype(np.float64)
        order = np.argsort(losses)
    else:
        evb_full = _evaluate_pose_losses_batch_no_grad(
            mesh=mesh,
            silhouette_renderer=silhouette_renderer,
            rasterizer=rasterizer,
            image_ref_silhouette=image_ref_silhouette,
            ref_depth=ref_depth,
            ref_normals=ref_normals,
            at_xyz=at_xyz,
            dist_list=[float(dist_fixed)] * len(d0),
            elev_list=e0,
            azim_list=a0,
            device=device,
            use_sil_normals_only=False,
        )
        losses = evb_full["loss_total"].detach().cpu().numpy().astype(np.float64)
        order = np.argsort(losses)
    
    # Select top beam_width seeds
    scored0 = [(float(losses[int(j)]), int(j)) for j in order[:beam_width]]
    scored0.sort(key=lambda x: x[0])

    def _cam_from_seed(idx: int) -> torch.Tensor:
        cam = _camera_position_from_spherical_global(
            torch.tensor(float(dist_fixed), device=device),
            torch.tensor(float(e0[idx]), device=device),
            torch.tensor(float(a0[idx]), device=device),
            at_xyz,
        )
        return _project_to_sphere(cam)

    beam = []
    best = None
    for r, (lv, j) in enumerate(scored0[:beam_width]):
        xyz = _cam_from_seed(int(j)).detach().clone()
        beam.append((float(lv), xyz))
        if best is None or float(lv) < float(best["loss"]):
            best = {
                "loss": float(lv),
                "dist": float(dist_fixed),
                "elev": float(e0[int(j)]),
                "azim": float(a0[int(j)]),
                "level": 0,
                "note": f"seed_rank={r} seed_idx={int(j)}",
            }
    if best is None:
        best = {"loss": float("inf"), "dist": float(dist_fixed), "elev": float(e0[0]), "azim": float(a0[0]), "level": 0, "note": "seed_fallback"}

    if verbose >= 1:
        print(f"[LocalBeamSeeds] {tag}: seeds={len(seeds)} init_beam_width={len(beam)} levels={levels} steps_per_dir={steps_per_dir} step_frac={step_frac:.4f}")
        print(f"[LocalBeamSeeds] {tag}: init best_loss={float(best['loss']):.6f} note={best['note']}")

    rows = []
    rows.append(["level", "parent_rank", "dir", "k", "dist", "elev_deg", "azim_deg", "loss_total"])

    # Generate ref_points_view if R_gt and T_gt are available (reuse in loop)
    ref_points_view_seeds_beam = None
    if use_sil_semantic and R_gt is not None and T_gt is not None:
        try:
            _, _, _, ref_points_view_seeds_beam = render_targets(
                mesh, R_gt, T_gt, device,
                int(image_ref_silhouette.shape[0]), int(image_ref_silhouette.shape[1])
            )
        except Exception:
            ref_points_view_seeds_beam = None

    # 2) Run beam expansion like local_beam_search_on_sphere
    for lvl in range(1, levels + 1):
        step_len = float(step_frac) * float(dist_fixed)

        cand_xyz: list[torch.Tensor] = []
        cand_meta: list[tuple[int, str, int]] = []
        cand_dist: list[float] = []
        cand_elev: list[float] = []
        cand_azim: list[float] = []

        for parent_rank, (_lp, xyz_p) in enumerate(beam):
            for dn, dv in dirs:
                for k in range(1, steps_per_dir + 1):
                    xyz_c = _project_to_sphere(xyz_p + dv * (step_len * float(k)))
                    _d, _e, _a = _spherical_from_camera_position_global(xyz_c, at_xyz)
                    cand_xyz.append(xyz_c)
                    cand_meta.append((int(parent_rank), str(dn), int(k)))
                    cand_dist.append(float(dist_fixed))
                    cand_elev.append(float(_e))
                    cand_azim.append(float(_a))

        if len(cand_xyz) == 0:
            break

        # 候选评估：按 score_mode 选择（推荐 full）
        if score_mode == "gt_geom" and phong_renderer is not None and image_ref_rgb is not None:
            evb = _evaluate_pose_sil_semantic_batch_no_grad(
                mesh=mesh,
                silhouette_renderer=silhouette_renderer,
                phong_renderer=phong_renderer,
                rasterizer=rasterizer,
                image_ref_silhouette=image_ref_silhouette,
                image_ref_rgb=image_ref_rgb,
                at_xyz=at_xyz,
                dist_list=cand_dist,
                elev_list=cand_elev,
                azim_list=cand_azim,
                device=device,
                R_gt=R_gt,
                T_gt=T_gt,
                ref_points_view=ref_points_view_seeds_beam,
            )
            losses = evb["loss_total"].detach().cpu().numpy().astype(np.float64)
            scored = [(float(losses[i]), int(i)) for i in range(int(losses.size))]
        elif score_mode == "sil":
            evb_sil = _evaluate_pose_sil_batch_no_grad(
                mesh=mesh,
                silhouette_renderer=silhouette_renderer,
                image_ref_silhouette=image_ref_silhouette,
                at_xyz=at_xyz,
                dist_list=cand_dist,
                elev_list=cand_elev,
                azim_list=cand_azim,
                device=device,
            )
            losses = evb_sil["loss_total"].detach().cpu().numpy().astype(np.float64)
            scored = [(float(losses[i]), int(i)) for i in range(int(losses.size))]
        else:
            evb_full = _evaluate_pose_losses_batch_no_grad(
                mesh=mesh,
                silhouette_renderer=silhouette_renderer,
                rasterizer=rasterizer,
                image_ref_silhouette=image_ref_silhouette,
                ref_depth=ref_depth,
                ref_normals=ref_normals,
                at_xyz=at_xyz,
                dist_list=cand_dist,
                elev_list=cand_elev,
                azim_list=cand_azim,
                device=device,
                use_sil_normals_only=False,
            )
            losses = evb_full["loss_total"].detach().cpu().numpy().astype(np.float64)
            scored = [(float(losses[i]), int(i)) for i in range(int(losses.size))]
        scored.sort(key=lambda x: x[0])

        # next beam + best update
        next_beam = []
        for lv, j in scored[:beam_width]:
            next_beam.append((float(lv), cand_xyz[j].detach().clone()))
            if float(lv) < float(best["loss"]):
                best.update(
                    {
                        "loss": float(lv),
                        "dist": float(dist_fixed),
                        "elev": float(cand_elev[j]),
                        "azim": float(cand_azim[j]),
                        "level": int(lvl),
                        "note": f"parent={cand_meta[j][0]} dir={cand_meta[j][1]} k={cand_meta[j][2]}",
                    }
                )
        beam = next_beam
        if len(beam) == 0:
            break

    # logs
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for row in rows:
                w.writerow(row)
    except Exception as _e:
        if verbose >= 2:
            print(f"[Warning] Failed to write local_beam_from_seeds.csv: {_e}")

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tag": tag,
                    "local_beam_from_seeds": {
                        "num_seeds": int(len(seeds)),
                        "width": int(beam_width),
                        "levels": int(levels),
                        "levels_override": None if levels_override is None else int(levels_override),
                        "steps_per_dir": int(steps_per_dir),
                        "step_frac": float(step_frac),
                        "use_sil_semantic": bool(use_sil_semantic),
                    },
                    "best": {
                        "loss_total": float(best["loss"]),
                        "dist": float(best["dist"]),
                        "elev_deg": float(best["elev"]),
                        "azim_deg": float(best["azim"]),
                        "level": int(best["level"]),
                        "note": str(best["note"]),
                    },
                    "logs": {"csv": os.path.basename(csv_path)},
                },
                f,
                indent=2,
                ensure_ascii=True,
            )
    except Exception as _e:
        if verbose >= 2:
            print(f"[Warning] Failed to write local_beam_from_seeds_best.json: {_e}")

    # optional artifacts
    try:
        if int(os.environ.get("VAL_LOCAL_BEAM_SAVE_BEST", "1")) != 0:
            _render_and_save_best_artifacts(
                mesh=mesh,
                silhouette_renderer=silhouette_renderer,
                rasterizer=rasterizer,
                phong_renderer=phong_renderer,
                at_xyz=at_xyz,
                dist_v=float(best["dist"]),
                elev_v=float(best["elev"]),
                azim_v=float(best["azim"]),
                device=device,
                save_dir=save_dir,
                prefix="best_local_beam_from_seeds",
            )
    except Exception:
        pass

    return float(best["dist"]), float(best["elev"]), float(best["azim"]), float(best["loss"])


def global_multiradius_axis_greedy_search(
    *,
    mesh: Meshes,
    verts: torch.Tensor,
    mesh_center_xyz: torch.Tensor,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    init_dist: float,
    init_elev: float,
    init_azim: float,
    device: torch.device,
    save_dir: str,
    tag: str,
) -> tuple[float, float, float, float]:
    """
    用户定制的全局方向搜索（贪婪路径）：
    - 已知 mesh，先计算 model radius = r
    - 球体半径序列：
        R0 = r * (4/3)
        R_{k+1} = R_k * (4/3) + r
      共 K=8 次变化
    - 每个半径下，按 WORLD 轴的 6 个方向（±x/±y/±z）做前向探索
      每次探索 16 步，每步“尝试位移”= step_unit * step_idx（默认 2 * (1..16)）
      将 (v + delta) 投影回半径 R_k 的球面，再评估 loss，选择最优方向作为下一步（贪婪）
    - 记录完整候选与选择路径到 CSV/JSON，返回全局最优 (dist,elev,azim,loss)
    """
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{tag}_global_multiradius_path.csv")
    json_path = os.path.join(save_dir, f"{tag}_global_multiradius_best.json")

    # Hyper-params (env overridable)
    stages = int(os.environ.get("VAL_GLOBAL_RADIUS_STAGES", "8"))
    stages = max(1, stages)
    steps_per_stage = int(os.environ.get("VAL_GLOBAL_FORWARD_STEPS", "16"))
    steps_per_stage = max(1, steps_per_stage)
    step_unit = float(os.environ.get("VAL_GLOBAL_STEP_UNIT", "2.0"))
    mult = float(os.environ.get("VAL_GLOBAL_RADIUS_MULT", str(4.0 / 3.0)))
    min_delta = float(os.environ.get("VAL_GLOBAL_MIN_DELTA", "1e-4"))
    use_sil_only = int(os.environ.get("VAL_GLOBAL_USE_SIL_ONLY", "1")) != 0
    verbose = int(os.environ.get("VAL_GLOBAL_VERBOSE", "1"))

    at_xyz = mesh_center_xyz.view(3).to(device)
    r = _mesh_radius_from_verts(verts, at_xyz)

    # radius schedule
    R = float(r * mult)

    # initial direction from init pose (use its world camera position direction)
    cam0 = _camera_position_from_spherical_global(
        torch.tensor(float(init_dist), device=device),
        torch.tensor(float(init_elev), device=device),
        torch.tensor(float(init_azim), device=device),
        at_xyz,
    )
    v_dir = cam0 - at_xyz
    v_dir = v_dir / torch.norm(v_dir).clamp(min=1e-8)

    axes = [
        torch.tensor([1.0, 0.0, 0.0], device=device),
        torch.tensor([-1.0, 0.0, 0.0], device=device),
        torch.tensor([0.0, 1.0, 0.0], device=device),
        torch.tensor([0.0, -1.0, 0.0], device=device),
        torch.tensor([0.0, 0.0, 1.0], device=device),
        torch.tensor([0.0, 0.0, -1.0], device=device),
    ]
    axis_names = ["+x", "-x", "+y", "-y", "+z", "-z"]

    # Global best tracker (across all stages/steps)
    best = {"loss": float("inf"), "dist": None, "elev": None, "azim": None, "stage": None, "step": None, "axis": None}

    # Start at stage0 center on sphere radius R
    cam_center = at_xyz + v_dir * torch.tensor(float(R), device=device)
    cur_dist, cur_elev, cur_azim = _spherical_from_camera_position_global(cam_center, at_xyz)
    cur_ev = _evaluate_pose_sil_no_grad(
        mesh=mesh,
        silhouette_renderer=silhouette_renderer,
        image_ref_silhouette=image_ref_silhouette,
        at_xyz=at_xyz,
        dist_v=float(cur_dist),
        elev_v=float(cur_elev),
        azim_v=float(cur_azim),
        device=device,
    ) if use_sil_only else _evaluate_pose_loss_no_grad(
        mesh, silhouette_renderer, rasterizer, image_ref_silhouette, ref_depth, ref_normals,
        at_xyz, float(cur_dist), float(cur_elev), float(cur_azim), device
    )
    cur_loss = float(cur_ev["loss_total"])
    best.update({"loss": cur_loss, "dist": float(cur_dist), "elev": float(cur_elev), "azim": float(cur_azim), "stage": 0, "step": 0, "axis": "center"})

    if verbose >= 1:
        print(f"[GlobalMR] {tag}: model_radius_r={r:.6f}, mult={mult:.6f}, step_unit={step_unit:.3f}, stages={stages}, steps_per_stage={steps_per_stage}, sil_only={int(use_sil_only)}")
        print(f"[GlobalMR] {tag}: stage=0 radius={R:.6f}, start_loss={cur_loss:.6f}, start(dist,elev,azim)=({cur_dist:.3f},{cur_elev:.3f},{cur_azim:.3f})")

    rows = []
    rows.append([
        "stage", "radius", "step", "candidate", "axis",
        "delta_unit", "dist", "elev_deg", "azim_deg",
        "loss_total", "loss_sil", "loss_depth", "loss_normal", "valid_ratio",
        "picked_in_step"
    ])
    # log center as candidate -1
    rows.append([0, float(R), 0, -1, "center", 0.0, float(cur_dist), float(cur_elev), float(cur_azim), float(cur_loss),
                float(cur_ev["loss_sil"]), float(cur_ev["loss_depth"]), float(cur_ev["loss_normal"]), float(cur_ev["valid_ratio"]), 1])

    path = []
    path.append({"stage": 0, "radius": float(R), "step": 0, "axis": "center", "loss_total": float(cur_loss), "dist": float(cur_dist), "elev_deg": float(cur_elev), "azim_deg": float(cur_azim)})

    # Main loop: stages × steps
    for s in range(stages):
        radius_s = float(R) if s == 0 else float(R)
        # Put current center on this radius (keep direction, reset norm)
        cam_center = at_xyz + (v_dir / torch.norm(v_dir).clamp(min=1e-8)) * torch.tensor(radius_s, device=device)
        v_center = (cam_center - at_xyz)
        v_center = v_center / torch.norm(v_center).clamp(min=1e-8) * torch.tensor(radius_s, device=device)

        if verbose >= 1 and s > 0:
            print(f"[GlobalMR] {tag}: stage={s} radius={radius_s:.6f} reset_center_loss={cur_loss:.6f}")

        for t in range(1, steps_per_stage + 1):
            delta = float(step_unit) * float(t)
            # 6 candidates
            dlist = []
            elist = []
            alist = []
            cams_world = []
            for u in axes:
                v_new = v_center + (delta * u)
                v_new = v_new / torch.norm(v_new).clamp(min=1e-8) * torch.tensor(radius_s, device=device)
                cam_new = at_xyz + v_new
                cams_world.append(cam_new)
                dist_new, elev_new, azim_new = _spherical_from_camera_position_global(cam_new, at_xyz)
                dlist.append(float(dist_new))
                elist.append(float(elev_new))
                alist.append(float(azim_new))

            evb = _batched_eval_candidates_no_grad(
                mesh=mesh,
                silhouette_renderer=silhouette_renderer,
                rasterizer=rasterizer,
                image_ref_silhouette=image_ref_silhouette,
                ref_depth=ref_depth,
                ref_normals=ref_normals,
                at_xyz=at_xyz,
                dist_list=dlist,
                elev_list=elist,
                azim_list=alist,
                device=device,
                use_sil_only=bool(use_sil_only),
            )
            lt = evb["loss_total"].detach().cpu().numpy().astype(np.float64)
            ls = evb["loss_sil"].detach().cpu().numpy().astype(np.float64)
            ld = evb["loss_depth"].detach().cpu().numpy().astype(np.float64)
            ln = evb["loss_normal"].detach().cpu().numpy().astype(np.float64)
            vr = evb["valid_ratio"].detach().cpu().numpy().astype(np.float64)

            best_i = int(np.argmin(lt)) if lt.size > 0 else 0
            picked = [0] * 6
            picked[best_i] = 1

            for i in range(6):
                rows.append([
                    int(s), float(radius_s), int(t), int(i), axis_names[i],
                    float(delta), float(dlist[i]), float(elist[i]), float(alist[i]),
                    float(lt[i]), float(ls[i]), float(ld[i]), float(ln[i]), float(vr[i]),
                    int(picked[i]),
                ])

            cand_loss = float(lt[best_i])
            cand_axis = axis_names[best_i]
            cand_cam = cams_world[best_i]
            cand_dist, cand_elev, cand_azim = dlist[best_i], elist[best_i], alist[best_i]

            # Greedy accept only if improves (with min_delta)
            if cand_loss < (cur_loss - min_delta):
                cur_loss = cand_loss
                # update direction on sphere
                v_center = (cand_cam - at_xyz)
                v_dir = v_center / torch.norm(v_center).clamp(min=1e-8)
                # log path
                path.append({
                    "stage": int(s),
                    "radius": float(radius_s),
                    "step": int(t),
                    "axis": str(cand_axis),
                    "loss_total": float(cur_loss),
                    "dist": float(cand_dist),
                    "elev_deg": float(cand_elev),
                    "azim_deg": float(cand_azim),
                })
                if cand_loss < float(best["loss"]):
                    best.update({"loss": float(cand_loss), "dist": float(cand_dist), "elev": float(cand_elev), "azim": float(cand_azim), "stage": int(s), "step": int(t), "axis": str(cand_axis)})
                if verbose >= 2:
                    print(f"[GlobalMR] {tag}: stage={s} step={t} pick={cand_axis} delta={delta:.3f} loss={cand_loss:.6f}")

        # Update radius for next stage: R = R*(4/3)+r
        R = float(radius_s * mult + r)

    # Write logs
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for row in rows:
                w.writerow(row)
    except Exception as _e:
        print(f"[Warning] Failed to write global_multiradius_path.csv: {_e}")

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tag": tag,
                    "model_radius_r": float(r),
                    "radius_schedule": {"mult": float(mult), "stages": int(stages), "R0": float(r * mult), "rule": "R_{k+1} = R_k * mult + r"},
                    "steps_per_stage": int(steps_per_stage),
                    "step_unit": float(step_unit),
                    "min_delta": float(min_delta),
                    "use_sil_only": bool(use_sil_only),
                    "best": {
                        "loss_total": float(best["loss"]),
                        "dist": float(best["dist"]),
                        "elev_deg": float(best["elev"]),
                        "azim_deg": float(best["azim"]),
                        "stage": int(best["stage"]),
                        "step": int(best["step"]),
                        "axis": best["axis"],
                    },
                    "path": path,
                },
                f,
                indent=2,
                ensure_ascii=True,
            )
    except Exception as _e:
        print(f"[Warning] Failed to write global_multiradius_best.json: {_e}")

    if verbose >= 1:
        print(
            f"[GlobalMR] {tag}: best_loss={float(best['loss']):.6f} best(dist,elev,azim)=({float(best['dist']):.3f},{float(best['elev']):.3f},{float(best['azim']):.3f}) "
            f"at stage={best['stage']} step={best['step']} axis={best['axis']} saved={os.path.basename(csv_path)}"
        )

    return float(best["dist"]), float(best["elev"]), float(best["azim"]), float(best["loss"])


def local_axis_greedy_on_sphere(
    *,
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    at_xyz: torch.Tensor,
    start_dist: float,
    start_elev: float,
    start_azim: float,
    device: torch.device,
    save_dir: str,
    tag: str,
) -> tuple[float, float, float, float]:
    """
    局部贪婪（用户要求）：
    - 固定球面半径=dist（当前相机到中心距离）
    - 每一轮在 ±x/±y/±z 六方向上探索，每个方向“前进 16 步”，步长=2单位（可用 env 覆盖）
    - 选择全局最小 loss 的候选，若改进则接受；否则停止（可选回退/缩步）
    - VAL_LOCAL_AXIS_SMART=1：FD 探梯度后只沿 top-K 世界轴探索（默认 2 轴 ×16 步，约 3× 加速）
    """
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{tag}_local_axis_path.csv")
    json_path = os.path.join(save_dir, f"{tag}_local_axis_best.json")

    steps = int(os.environ.get("VAL_LOCAL_FORWARD_STEPS", "16"))
    steps = max(1, steps)
    step_unit = float(os.environ.get("VAL_LOCAL_STEP_UNIT", "2.0"))
    iters = int(os.environ.get("VAL_LOCAL_ITERS", "64"))
    iters = max(1, iters)
    min_delta = float(os.environ.get("VAL_LOCAL_MIN_DELTA", "1e-5"))
    use_sil_only = int(os.environ.get("VAL_LOCAL_USE_SIL_ONLY", "0")) != 0
    verbose = int(os.environ.get("VAL_LOCAL_VERBOSE", "1"))
    axis_smart = int(os.environ.get("VAL_LOCAL_AXIS_SMART", "0")) != 0
    axis_fd = int(os.environ.get("VAL_LOCAL_AXIS_FD", "1")) != 0
    axis_top = max(1, min(6, int(os.environ.get("VAL_LOCAL_AXIS_TOP_AXES", "2"))))
    elev_min = float(os.environ.get("VAL_GT_ELEV_MIN", "5.0"))
    elev_max = float(os.environ.get("VAL_GT_ELEV_MAX", "85.0"))
    azim_min = float(os.environ.get("VAL_GT_AZIM_MIN", "-180.0"))
    azim_max = float(os.environ.get("VAL_GT_AZIM_MAX", "180.0"))

    at_xyz = at_xyz.view(3).to(device)
    radius = float(start_dist)

    axes = [
        torch.tensor([1.0, 0.0, 0.0], device=device),
        torch.tensor([-1.0, 0.0, 0.0], device=device),
        torch.tensor([0.0, 1.0, 0.0], device=device),
        torch.tensor([0.0, -1.0, 0.0], device=device),
        torch.tensor([0.0, 0.0, 1.0], device=device),
        torch.tensor([0.0, 0.0, -1.0], device=device),
    ]
    axis_names = ["+x", "-x", "+y", "-y", "+z", "-z"]

    # current center vector on sphere
    cam0 = _camera_position_from_spherical_global(
        torch.tensor(float(radius), device=device),
        torch.tensor(float(start_elev), device=device),
        torch.tensor(float(start_azim), device=device),
        at_xyz,
    )
    v_center = cam0 - at_xyz
    v_center = v_center / torch.norm(v_center).clamp(min=1e-8) * torch.tensor(float(radius), device=device)
    cur_dist, cur_elev, cur_azim = _spherical_from_camera_position_global(at_xyz + v_center, at_xyz)

    cur_ev = _evaluate_pose_sil_no_grad(
        mesh=mesh,
        silhouette_renderer=silhouette_renderer,
        image_ref_silhouette=image_ref_silhouette,
        at_xyz=at_xyz,
        dist_v=float(cur_dist),
        elev_v=float(cur_elev),
        azim_v=float(cur_azim),
        device=device,
    ) if use_sil_only else _evaluate_pose_loss_no_grad(
        mesh, silhouette_renderer, rasterizer, image_ref_silhouette, ref_depth, ref_normals,
        at_xyz, float(cur_dist), float(cur_elev), float(cur_azim), device
    )
    cur_loss = float(cur_ev["loss_total"])

    if verbose >= 1:
        print(
            f"[LocalAxis] {tag}: start_loss={cur_loss:.6f} radius={radius:.6f} steps={steps} "
            f"step_unit={step_unit:.3f} iters={iters} sil_only={int(use_sil_only)} "
            f"smart={int(axis_smart)} fd={int(axis_fd)} top_axes={axis_top}"
        )

    rows = []
    rows.append(["iter", "candidate_idx", "axis", "step", "delta_unit", "dist", "elev_deg", "azim_deg", "loss_total", "picked"])
    path = []
    path.append({"iter": 0, "axis": "center", "loss_total": float(cur_loss), "dist": float(cur_dist), "elev_deg": float(cur_elev), "azim_deg": float(cur_azim)})

    best = {"loss": cur_loss, "dist": float(cur_dist), "elev": float(cur_elev), "azim": float(cur_azim)}

    def _axis_indices_from_fd(g_elev: float, g_azim: float, k: int) -> list[int]:
        """Map spherical FD grads to world-axis indices (+y/-y/+z/-z preferred)."""
        scored_axes: list[tuple[float, int]] = []
        if abs(g_elev) >= 1e-8:
            scored_axes.append((abs(g_elev), 2 if g_elev < 0 else 3))  # +y / -y
            scored_axes.append((abs(g_elev) * 0.5, 3 if g_elev < 0 else 2))
        if abs(g_azim) >= 1e-8:
            scored_axes.append((abs(g_azim), 4 if g_azim < 0 else 5))  # +z / -z
            scored_axes.append((abs(g_azim) * 0.5, 5 if g_azim < 0 else 4))
        if not scored_axes:
            return list(range(6))
        scored_axes.sort(key=lambda x: x[0], reverse=True)
        picked: list[int] = []
        for _, idx in scored_axes:
            if idx not in picked:
                picked.append(idx)
            if len(picked) >= k:
                break
        return picked if picked else list(range(min(6, k)))

    for it in range(1, iters + 1):
        if axis_fd:
            fd_out = _sphere_fd_descent_step(
                mesh=mesh,
                silhouette_renderer=silhouette_renderer,
                rasterizer=rasterizer,
                image_ref_silhouette=image_ref_silhouette,
                ref_depth=ref_depth,
                ref_normals=ref_normals,
                at_xyz=at_xyz,
                dist_c=float(cur_dist),
                elev_c=float(cur_elev),
                azim_c=float(cur_azim),
                elev_min=elev_min,
                elev_max=elev_max,
                azim_min=azim_min,
                azim_max=azim_max,
                device=device,
                use_sil_only=use_sil_only,
            )
            if fd_out is not None:
                _d, e_new, a_new, l_new, _ev = fd_out
                if float(l_new) < float(cur_loss) - min_delta:
                    cur_loss = float(l_new)
                    cur_dist, cur_elev, cur_azim = float(_d), float(e_new), float(a_new)
                    cam_fd = _camera_position_from_spherical_global(
                        torch.tensor(float(cur_dist), device=device),
                        torch.tensor(float(cur_elev), device=device),
                        torch.tensor(float(cur_azim), device=device),
                        at_xyz,
                    )
                    v_center = (cam_fd - at_xyz)
                    v_center = v_center / torch.norm(v_center).clamp(min=1e-8) * torch.tensor(float(radius), device=device)
                    if cur_loss < float(best["loss"]):
                        best.update({"loss": cur_loss, "dist": cur_dist, "elev": cur_elev, "azim": cur_azim})

        active_axes = list(range(6))
        if axis_smart:
            grad = _sphere_fd_grad_probe(
                mesh=mesh,
                silhouette_renderer=silhouette_renderer,
                rasterizer=rasterizer,
                image_ref_silhouette=image_ref_silhouette,
                ref_depth=ref_depth,
                ref_normals=ref_normals,
                at_xyz=at_xyz,
                dist_c=float(cur_dist),
                elev_c=float(cur_elev),
                azim_c=float(cur_azim),
                elev_min=elev_min,
                elev_max=elev_max,
                azim_min=azim_min,
                azim_max=azim_max,
                device=device,
                use_sil_only=use_sil_only,
            )
            if grad is not None:
                active_axes = _axis_indices_from_fd(grad[0], grad[1], axis_top)

        # build candidates: active directions × steps
        dlist, elist, alist = [], [], []
        meta = []  # (axis_name, step_idx, delta)
        for ai in active_axes:
            u = axes[ai]
            for st in range(1, steps + 1):
                delta = float(step_unit) * float(st)
                v_new = v_center + (delta * u)
                v_new = v_new / torch.norm(v_new).clamp(min=1e-8) * torch.tensor(float(radius), device=device)
                cam_new = at_xyz + v_new
                dist_new, elev_new, azim_new = _spherical_from_camera_position_global(cam_new, at_xyz)
                dlist.append(float(dist_new))
                elist.append(float(elev_new))
                alist.append(float(azim_new))
                meta.append((axis_names[ai], int(st), float(delta), v_new))

        evb = _batched_eval_candidates_no_grad(
            mesh=mesh,
            silhouette_renderer=silhouette_renderer,
            rasterizer=rasterizer,
            image_ref_silhouette=image_ref_silhouette,
            ref_depth=ref_depth,
            ref_normals=ref_normals,
            at_xyz=at_xyz,
            dist_list=dlist,
            elev_list=elist,
            azim_list=alist,
            device=device,
            use_sil_only=bool(use_sil_only),
        )
        lt = evb["loss_total"].detach().cpu().numpy().astype(np.float64)
        best_i = int(np.argmin(lt)) if lt.size > 0 else 0

        for j in range(len(meta)):
            axis, st, delta, _v = meta[j]
            rows.append([int(it), int(j), axis, int(st), float(delta), float(dlist[j]), float(elist[j]), float(alist[j]), float(lt[j]), 1 if j == best_i else 0])

        cand_loss = float(lt[best_i])
        axis, st, delta, v_new = meta[best_i]
        cand_dist, cand_elev, cand_azim = dlist[best_i], elist[best_i], alist[best_i]

        if cand_loss < (cur_loss - min_delta):
            cur_loss = cand_loss
            v_center = v_new.detach()  # accept move
            path.append({"iter": int(it), "axis": str(axis), "step": int(st), "delta_unit": float(delta), "loss_total": float(cur_loss), "dist": float(cand_dist), "elev_deg": float(cand_elev), "azim_deg": float(cand_azim)})
            if cur_loss < float(best["loss"]):
                best.update({"loss": float(cur_loss), "dist": float(cand_dist), "elev": float(cand_elev), "azim": float(cand_azim)})
            if verbose >= 2:
                print(f"[LocalAxis] {tag}: iter={it} pick={axis} step={st} delta={delta:.3f} loss={cur_loss:.6f}")
        else:
            if verbose >= 1:
                print(f"[LocalAxis] {tag}: stop at iter={it} (no improvement > {min_delta:.2e}); cur_loss={cur_loss:.6f}")
            break

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for row in rows:
                w.writerow(row)
    except Exception as _e:
        print(f"[Warning] Failed to write local_axis_path.csv: {_e}")

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tag": tag,
                    "radius": float(radius),
                    "steps": int(steps),
                    "step_unit": float(step_unit),
                    "iters": int(iters),
                    "min_delta": float(min_delta),
                    "use_sil_only": bool(use_sil_only),
                    "best": {"loss_total": float(best["loss"]), "dist": float(best["dist"]), "elev_deg": float(best["elev"]), "azim_deg": float(best["azim"])},
                    "path": path,
                },
                f,
                indent=2,
                ensure_ascii=True,
            )
    except Exception as _e:
        print(f"[Warning] Failed to write local_axis_best.json: {_e}")

    if verbose >= 1:
        print(f"[LocalAxis] {tag}: best_loss={float(best['loss']):.6f} best(dist,elev,azim)=({float(best['dist']):.3f},{float(best['elev']):.3f},{float(best['azim']):.3f}) saved={os.path.basename(csv_path)}")

    return float(best["dist"]), float(best["elev"]), float(best["azim"]), float(best["loss"])


def _mesh_top_normal_from_mesh(mesh: Meshes, device: torch.device) -> torch.Tensor:
    """Average normal of the top 20% faces (by +Z component) as mesh up prior."""
    try:
        face_normals = mesh.faces_normals_packed()
        z_vals = face_normals[:, 2]
        threshold = torch.quantile(z_vals, 0.8)
        top_face_mask = z_vals >= threshold
        if top_face_mask.sum() > 0:
            return torch.nn.functional.normalize(face_normals[top_face_mask].mean(dim=0), dim=-1)
    except Exception:
        pass
    return torch.tensor([0.0, 0.0, 1.0], device=device)


# ----------------------------------------------------------------------------
# Per-view stage timing (wall clock, CUDA-synchronised). Reset at the start of
# every GT view in main(); written to pose_metrics_summary.csv.
# ----------------------------------------------------------------------------
_STAGE_TIME: dict[str, float] = {}
_STAGE_KEYS = ("t_total_s", "t_global_s", "t_dist_align_s", "t_local_s", "t_select_s", "t_refine_s")


def _stage_reset() -> None:
    for k in _STAGE_KEYS:
        _STAGE_TIME[k] = 0.0


def _stage_now() -> float:
    try:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    return time.perf_counter()


def _stage_add(key: str, dt: float) -> None:
    _STAGE_TIME[key] = float(_STAGE_TIME.get(key, 0.0)) + float(max(0.0, dt))


def _strict_no_gt() -> bool:
    """
    Strict no-ground-truth mode (default ON).

    When enabled, every code path that could inject the ground-truth pose
    (R_gt / T_gt / GT elevation) into search, scoring, constraints or
    selection is disabled, and any attempt to use GT-dependent scoring raises.
    The ground truth is then used ONLY to (a) render the target observations
    and (b) compute the final evaluation metrics.

    Set VAL_STRICT_NO_GT=0 only to reproduce legacy (leak-prone) runs.
    """
    return int(os.environ.get("VAL_STRICT_NO_GT", "1")) != 0


def _assert_no_gt(where: str, *tensors) -> None:
    """Raise if any GT tensor is about to be used while strict mode is on."""
    if _strict_no_gt() and any(t is not None for t in tensors):
        raise RuntimeError(
            f"[GT-LEAK] {where}: ground-truth pose would enter the optimisation "
            f"while VAL_STRICT_NO_GT=1. Pass R_gt/T_gt=None or disable strict mode "
            f"explicitly for legacy debugging runs."
        )


def _ref_view_direction_from_normals(
    ref_normals: torch.Tensor,
    image_ref_silhouette: torch.Tensor,
    device: torch.device,
    R_gt: torch.Tensor | None = None,
) -> torch.Tensor | None:
    """
    Infer reference viewing direction from visible surface normals in the target render.

    IMPORTANT: the target normals are rendered in CAMERA (view) space, so their mean
    carries no information about the world-frame viewing direction. The only way to
    turn it into a world-frame direction is to rotate it by R_gt, which is exactly
    the ground-truth rotation we are trying to estimate. Therefore:
      - in strict mode (default) this function returns None and raises if R_gt is given;
      - legacy mode (VAL_STRICT_NO_GT=0) keeps the historical R_gt-based behaviour.
    """
    if _strict_no_gt():
        _assert_no_gt("_ref_view_direction_from_normals", R_gt)
        return None
    if R_gt is None:
        # Camera-frame mean normal is ~constant; without R_gt it is not a view hint.
        return None
    try:
        ref_n = ref_normals.squeeze().to(device)
        ref_s = torch.clamp(image_ref_silhouette, 0, 1)
        if ref_s.dim() == 3 and ref_s.shape[-1] == 4:
            ref_s = ref_s[..., 3]
        ref_s = ref_s.squeeze().to(device)
        mask = ref_s > 0.05
        if int(mask.sum().item()) < 16:
            return None
        if ref_n.dim() != 3 or ref_n.shape[-1] != 3:
            return None
        vecs = ref_n[mask]
        mean_n = vecs.mean(dim=0)
        if float(torch.norm(mean_n).item()) < 1e-6:
            return None
        if R_gt is not None:
            R = R_gt.view(-1, 3, 3)[0].to(device)
            mean_n = torch.matmul(mean_n, R)
        return torch.nn.functional.normalize(mean_n, dim=-1)
    except Exception:
        return None


def _view_dir_from_spherical_deg(elev_deg: float, azim_deg: float, device: torch.device) -> torch.Tensor:
    """Unit vector from look-at center toward camera."""
    elev_rad = math.radians(float(elev_deg))
    azim_rad = math.radians(float(azim_deg))
    return torch.tensor(
        [
            math.cos(elev_rad) * math.sin(azim_rad),
            math.sin(elev_rad),
            math.cos(elev_rad) * math.cos(azim_rad),
        ],
        device=device,
        dtype=torch.float32,
    )


def _elev_deg_from_view_dir(view_dir: torch.Tensor) -> float:
    """Infer elevation (deg) from a unit view direction (center -> camera)."""
    y = float(torch.clamp(view_dir.view(3)[1], -1.0, 1.0).item())
    return float(math.degrees(math.asin(y)))


def _azim_deg_from_view_dir(view_dir: torch.Tensor) -> float:
    """Infer azimuth (deg) from a unit view direction (center -> camera)."""
    vd = view_dir.view(3)
    return float(math.degrees(math.atan2(float(vd[0].item()), float(vd[2].item()))))


def _wrap_azim_deg(azim_deg: float) -> float:
    """Wrap azimuth to [-180, 180]."""
    a = float(azim_deg)
    while a > 180.0:
        a -= 360.0
    while a < -180.0:
        a += 360.0
    return a


def _azim_diff_deg(a: float, b: float) -> float:
    """Minimal absolute azimuth difference in degrees."""
    return abs(_wrap_azim_deg(float(a) - float(b)))


def _ref_spherical_hint_from_normals(
    ref_normals: torch.Tensor,
    image_ref_silhouette: torch.Tensor,
    device: torch.device,
    R_gt: torch.Tensor | None = None,
) -> tuple[float | None, float | None]:
    """Return (elev_deg, azim_deg) hints inferred from reference visible normals."""
    ref_dir = _ref_view_direction_from_normals(
        ref_normals, image_ref_silhouette, device, R_gt=R_gt
    )
    if ref_dir is None:
        return None, None
    return _elev_deg_from_view_dir(ref_dir), _azim_deg_from_view_dir(ref_dir)


def _silhouette_azim_offset_deg(
    image_ref_silhouette: torch.Tensor,
    device: torch.device,
) -> float | None:
    """Estimate azimuth correction from silhouette horizontal centroid offset."""
    try:
        ref_s = torch.clamp(image_ref_silhouette, 0, 1)
        if ref_s.dim() == 3 and ref_s.shape[-1] == 4:
            ref_s = ref_s[..., 3]
        ref_s = ref_s.squeeze().to(device)
        mask = ref_s > 0.05
        if int(mask.sum().item()) < 16:
            return None
        h, w = ref_s.shape
        xs = torch.where(mask)[1].float()
        cx = (xs.mean() / max(float(w - 1), 1.0) - 0.5) * 2.0
        gain = float(os.environ.get("VAL_SIL_CENTROID_AZIM_GAIN", "28.0"))
        return float(cx.item()) * gain
    except Exception:
        return None


def _silhouette_elev_hint_deg(
    image_ref_silhouette: torch.Tensor,
    device: torch.device,
    ref_elev_normal: float | None = None,
) -> float | None:
    """Estimate elevation from silhouette vertical centroid (complements normal-based hint)."""
    try:
        ref_s = torch.clamp(image_ref_silhouette, 0, 1)
        if ref_s.dim() == 3 and ref_s.shape[-1] == 4:
            ref_s = ref_s[..., 3]
        ref_s = ref_s.squeeze().to(device)
        mask = ref_s > 0.05
        if int(mask.sum().item()) < 16:
            return None
        h, _w = ref_s.shape
        ys = torch.where(mask)[0].float()
        cy = (ys.mean() / max(float(h - 1), 1.0) - 0.5) * 2.0
        gain = float(os.environ.get("VAL_SIL_CENTROID_ELEV_GAIN", "25.0"))
        base = float(ref_elev_normal) if ref_elev_normal is not None else 30.0
        elev_sil = float(base) - float(cy.item()) * gain
        elev_sil = float(max(-10.0, min(89.0, elev_sil)))
        if ref_elev_normal is None:
            return elev_sil
        disagree = abs(float(ref_elev_normal) - elev_sil)
        w_base = float(os.environ.get("VAL_REF_ELEV_CENTROID_BLEND", "0.45"))
        w = w_base if disagree < 12.0 else min(0.85, w_base + 0.035 * disagree)
        return float((1.0 - w) * float(ref_elev_normal) + w * elev_sil)
    except Exception:
        return ref_elev_normal


def _ref_spherical_hint_from_observations(
    ref_normals: torch.Tensor,
    image_ref_silhouette: torch.Tensor,
    device: torch.device,
    R_gt: torch.Tensor | None = None,
) -> tuple[float | None, float | None]:
    """Fuse normal-based and silhouette-centroid azimuth/elevation hints.

    In strict no-GT mode the normal-based hint is unavailable (see
    _ref_view_direction_from_normals) and the silhouette-centroid heuristic is
    disabled unless VAL_SIL_CENTROID_HINTS=1 is set explicitly, because it was
    tuned around the normal-based anchor and is not a reliable stand-alone cue.
    """
    ref_elev_n, ref_azim = _ref_spherical_hint_from_normals(
        ref_normals, image_ref_silhouette, device, R_gt=R_gt
    )
    if ref_elev_n is None and ref_azim is None:
        if int(os.environ.get("VAL_SIL_CENTROID_HINTS", "0")) == 0:
            return None, None
    ref_elev = _silhouette_elev_hint_deg(image_ref_silhouette, device, ref_elev_n)
    azim_off = _silhouette_azim_offset_deg(image_ref_silhouette, device)
    if ref_azim is not None and azim_off is not None:
        w_cent = float(os.environ.get("VAL_REF_AZIM_CENTROID_BLEND", "0.35"))
        ref_azim = _wrap_azim_deg(float(ref_azim) + w_cent * float(azim_off))
    return ref_elev, ref_azim


# Mesh-specific offset (in units of the bounding radius r) between the median visible
# surface depth and the camera-to-centre distance, estimated once per mesh from renders of
# the known mesh at random viewpoints (no observation involved). Set by
# _estimate_depth_centre_offset(); VAL_REF_DIST_OFFSET_FRAC overrides it when given.
_DEPTH_CENTRE_OFFSET_FRAC: float | None = None


def _estimate_depth_centre_offset(
    mesh: MeshData,
    rasterizer: MeshRasterizer,
    at_xyz: torch.Tensor,
    mesh_r: float,
    device: torch.device,
    n_views: int = 32,
    dist_mult: float = 3.0,
) -> float:
    """
    kappa = mean over random views of (d - median visible depth) / r, where d is the
    camera-to-centre distance. The median depth of the visible surface lies in front of the
    object centre; kappa measures by how much for THIS mesh (a property of the known
    geometry, independent of the target observation).
    """
    global _DEPTH_CENTRE_OFFSET_FRAC
    env = os.environ.get("VAL_REF_DIST_OFFSET_FRAC", "").strip()
    if env != "":
        _DEPTH_CENTRE_OFFSET_FRAC = float(env)
        return _DEPTH_CENTRE_OFFSET_FRAC
    g = torch.Generator(device="cpu").manual_seed(2026)
    vals: list[float] = []
    at = at_xyz.view(3).to(device)
    with torch.no_grad():
        for _ in range(int(n_views)):
            v = torch.randn(3, generator=g)
            v = v / v.norm().clamp(min=1e-8)
            elev = math.degrees(math.asin(float(v[1].clamp(-1, 1))))
            azim = math.degrees(math.atan2(float(v[0]), float(v[2])))
            d = float(dist_mult) * float(mesh_r)
            R, T = look_at_view_transform(dist=d, elev=elev, azim=azim, at=at, device=device)
            fr = rasterizer(meshes_world=mesh, R=R, T=T)
            vis = fr.pix_to_face[..., 0][0] >= 0
            if int(vis.sum().item()) < 16:
                continue
            med = float(fr.zbuf[..., 0][0][vis].median().item())
            vals.append((d - med) / max(float(mesh_r), 1e-9))
    kappa = float(np.mean(vals)) if vals else 0.5
    _DEPTH_CENTRE_OFFSET_FRAC = kappa
    print(f"[DepthHint] mesh centre offset kappa={kappa:.3f} r (from {len(vals)} random views)")
    return kappa


def _ref_dist_hint_from_depth(
    ref_depth: torch.Tensor,
    image_ref_silhouette: torch.Tensor,
    device: torch.device,
    mesh_r: float | None = None,
) -> float | None:
    """
    Metric camera-to-centre distance hint from the observed depth:
        d_tilde = median_{x in mask, D(x)>0} D(x) + kappa * r,
    where D is the metric depth channel (mesh units) and kappa*r is the mesh-specific
    offset between the visible surface and the object centre (see
    _estimate_depth_centre_offset; defaults to 0.5 r when not calibrated / r unknown).
    """
    try:
        ref_d = ref_depth.squeeze().to(device)
        ref_s = torch.clamp(image_ref_silhouette, 0, 1)
        if ref_s.dim() == 3 and ref_s.shape[-1] == 4:
            ref_s = ref_s[..., 3]
        ref_s = ref_s.squeeze().to(device)
        mask = (ref_s > 0.05) & (ref_d > 0)
        if int(mask.sum().item()) < 16:
            return None
        med_z = float(ref_d[mask].median().item())
        scale = float(os.environ.get("VAL_REF_DIST_DEPTH_SCALE", "1.0"))
        env = os.environ.get("VAL_REF_DIST_OFFSET_FRAC", "").strip()
        if env != "":
            kappa = float(env)
        elif _DEPTH_CENTRE_OFFSET_FRAC is not None:
            kappa = float(_DEPTH_CENTRE_OFFSET_FRAC)
        else:
            kappa = 0.5
        offset = kappa * float(mesh_r) if (mesh_r is not None and mesh_r > 0) else 0.0
        return max(1e-4, med_z * scale + offset)
    except Exception:
        return None


def _mesh_scale_ref_radius() -> float:
    """Reference mesh bounding-sphere radius (bunny-scale) for scale-adaptive heuristics."""
    return float(os.environ.get("VAL_MESH_SCALE_REF_RADIUS", "0.12"))


def _resolve_radii_mults(mesh_r: float, ref_dist_hint: float | None) -> list[float]:
    """Log-spaced radius multipliers when ``VAL_RAND_RADII_MULTS=auto``; else legacy list/default."""
    cfg = os.environ.get("VAL_RAND_RADII_MULTS", "").strip()
    if cfg != "" and cfg.lower() != "auto":
        try:
            mults = [float(x) for x in cfg.replace(" ", "").split(",") if x != ""]
        except Exception:
            mults = []
        if len(mults) > 0:
            return sorted(list(set(float(x) for x in mults if float(x) > 0)))
    if cfg.lower() != "auto":
        return [round(0.5 + i * 0.1, 1) for i in range(36)]

    min_m = float(os.environ.get("VAL_RAND_MIN_RADIUS_MULT", "0.4"))
    max_m = float(os.environ.get("VAL_RAND_MAX_RADIUS_MULT", "6.0"))
    # Upper cap of the hint-adapted range (in units of r). Real RGB-D scenes (e.g. BOP) place
    # the camera 10-25 r away, so the cap must not clip the hint; 64 r is far beyond any case here.
    hint_cap = float(os.environ.get("VAL_RAND_RADII_HINT_CAP", "64.0"))
    if ref_dist_hint is not None and mesh_r > 1e-6:
        need = float(ref_dist_hint) / float(mesh_r)
        max_m = max(max_m, min(hint_cap, need * 1.4))
        min_m = min(min_m, max(0.3, need * 0.5))
    else:
        r_ref = _mesh_scale_ref_radius()
        if mesh_r > r_ref * 2.0:
            max_m = max(max_m, min(8.0, 4.0 * (mesh_r / r_ref) ** 0.25))
    n_steps = max(12, int(os.environ.get("VAL_RAND_RADIUS_STEPS", "40")))
    import math

    log_min, log_max = math.log(max(1e-3, min_m)), math.log(max(min_m + 1e-3, max_m))
    return [math.exp(log_min + i * (log_max - log_min) / max(1, n_steps - 1)) for i in range(n_steps)]


def _resolve_samples_per_radius(mesh_r: float) -> int:
    cfg = os.environ.get("VAL_RAND_SAMPLES_PER_RADIUS", "1024").strip().lower()
    if cfg != "auto":
        return max(1, int(cfg))
    r_ref = _mesh_scale_ref_radius()
    base = int(os.environ.get("VAL_RAND_SAMPLES_BASE", "1024"))
    scale = (float(mesh_r) / max(r_ref, 1e-6)) ** 0.45
    lo = int(os.environ.get("VAL_RAND_SAMPLES_MIN", "512"))
    hi = int(os.environ.get("VAL_RAND_SAMPLES_MAX", "2048"))
    return max(lo, min(hi, int(round(base * scale))))


def _resolve_local_beam_dist_scales(
    seed_dist: float,
    ref_dist_hint: float | None,
    mesh_r: float,
) -> list[float]:
    """Distance multipliers for pre-beam sweep; ``auto`` bridges seed dist to depth-based hint."""
    cfg = os.environ.get("VAL_LOCAL_BEAM_DIST_SCALES", "").strip()
    if cfg != "" and cfg.lower() != "auto":
        try:
            scales = [float(x) for x in cfg.split(",") if x.strip() != ""]
        except Exception:
            scales = []
        if len(scales) > 0:
            return scales
    if cfg.lower() != "auto":
        return [0.72, 0.86, 1.0, 1.14, 1.28]

    if ref_dist_hint is not None and seed_dist > 1e-5:
        ratio = float(ref_dist_hint) / float(seed_dist)
        ratio = max(0.15, min(6.0, ratio))
        lo = max(0.22, min(0.92, ratio * 0.72))
        hi = max(1.08, min(5.0, ratio * 1.35))
        n = max(7, int(os.environ.get("VAL_LOCAL_BEAM_DIST_SCALE_STEPS", "11")))
        import math

        log_lo, log_hi = math.log(lo), math.log(hi)
        scales = [
            math.exp(log_lo + i * (log_hi - log_lo) / max(1, n - 1)) for i in range(n)
        ]
        scales = sorted(list(set(scales + [1.0, float(ratio)])))
        return scales

    if mesh_r > _mesh_scale_ref_radius() * 2.0:
        return [0.35, 0.5, 0.65, 0.8, 0.92, 1.0, 1.15, 1.35, 1.65, 2.1, 2.8, 3.6]
    return [0.72, 0.86, 1.0, 1.14, 1.28]


def _rgb_weight_for_mesh_radius(mesh_r: float, env_key: str, default: float) -> float:
    """Reduce RGB dominance on large-scale meshes where texture cues are weaker."""
    w = float(os.environ.get(env_key, str(default)))
    if int(os.environ.get("VAL_RGB_WEIGHT_MESH_ADAPT", "1")) == 0:
        return w
    r_ref = _mesh_scale_ref_radius()
    if mesh_r <= r_ref * 1.5:
        return w
    factor = (r_ref / float(mesh_r)) ** 0.35
    return max(0.5, w * factor)


def _candidate_dist_penalty(
    dist_v: float,
    ref_dist_hint: float | None,
) -> float:
    if ref_dist_hint is None:
        return 0.0
    w = float(os.environ.get("VAL_REF_DIST_PENALTY_WEIGHT", "0.35"))
    gap = abs(float(dist_v) - float(ref_dist_hint))
    thr = float(os.environ.get("VAL_REF_DIST_PENALTY_THRESHOLD", "0.04"))
    if int(os.environ.get("VAL_REF_DIST_PENALTY_RELATIVE", "0")) != 0:
        thr = max(thr, 0.06 * float(ref_dist_hint))
    if gap <= thr:
        return 0.0
    pen = w * (gap - thr)
    if int(os.environ.get("VAL_REF_DIST_PENALTY_RELATIVE", "0")) != 0:
        rel = gap / max(float(ref_dist_hint), 1e-3)
        pen *= 1.0 + min(3.0, rel)
    return pen


def _adaptive_elev_bins(ref_elev: float | None, elev_bins: list[float]) -> list[float]:
    """Keep elevation diversity bins near the reference-view hint when available."""
    if ref_elev is None or len(elev_bins) <= 0:
        return elev_bins
    half_span = float(os.environ.get("VAL_RAND_REF_ELEV_BIN_HALF_SPAN", "30.0"))
    margin = float(os.environ.get("VAL_RAND_REF_ELEV_BIN_MARGIN", "15.0"))
    filtered = [b for b in elev_bins if abs(float(b) - float(ref_elev)) <= half_span + margin]
    if len(filtered) >= 3:
        return sorted(list(dict.fromkeys(filtered)))
    center = float(ref_elev)
    return sorted(
        list(
            dict.fromkeys(
                [
                    max(-90.0, center - 20.0),
                    max(-90.0, center - 10.0),
                    center,
                    min(90.0, center + 10.0),
                    min(90.0, center + 20.0),
                ]
            )
        )
    )


def _adaptive_azim_bins(ref_azim: float | None, azim_bins: list[float]) -> list[float]:
    """Keep azimuth diversity bins near the reference-view hint when available."""
    if ref_azim is None or len(azim_bins) <= 0:
        return azim_bins
    half_span = float(os.environ.get("VAL_RAND_REF_AZIM_BIN_HALF_SPAN", "55.0"))
    step = float(os.environ.get("VAL_RAND_REF_AZIM_BIN_STEP", "20.0"))
    step = max(5.0, step)
    center = _wrap_azim_deg(ref_azim)
    generated: list[float] = []
    offset = -half_span
    while offset <= half_span + 1e-6:
        generated.append(_wrap_azim_deg(center + offset))
        offset += step
    merged = sorted(list(dict.fromkeys(generated + azim_bins)))
    filtered = [b for b in merged if _azim_diff_deg(b, center) <= half_span + step]
    if len(filtered) >= 3:
        return sorted(list(dict.fromkeys(filtered)))
    return sorted(
        list(
            dict.fromkeys(
                [
                    _wrap_azim_deg(center - 30.0),
                    _wrap_azim_deg(center - 15.0),
                    center,
                    _wrap_azim_deg(center + 15.0),
                    _wrap_azim_deg(center + 30.0),
                ]
            )
        )
    )


def _adaptive_anchor_azims(ref_azim: float | None, anchor_azims: list[float]) -> list[float]:
    if ref_azim is None:
        return anchor_azims
    half_span = float(os.environ.get("VAL_RAND_REF_AZIM_ANCHOR_HALF_SPAN", "45.0"))
    step = float(os.environ.get("VAL_RAND_REF_AZIM_ANCHOR_STEP", "15.0"))
    step = max(5.0, step)
    center = _wrap_azim_deg(ref_azim)
    anchors: list[float] = []
    offset = -half_span
    while offset <= half_span + 1e-6:
        anchors.append(_wrap_azim_deg(center + offset))
        offset += step
    return sorted(list(dict.fromkeys(anchors)))


def _adaptive_anchor_elevs(ref_elev: float | None, anchor_elevs: list[float]) -> list[float]:
    if ref_elev is None:
        return anchor_elevs
    return _adaptive_elev_bins(ref_elev, anchor_elevs)


def _ref_elev_penalty_deg(elev_deg: float, ref_elev: float) -> float:
    """Elevation deviation penalty in degrees (before lambda/weight). Matches v9 global scoring."""
    diff = float(elev_deg) - float(ref_elev)
    high_thr = float(os.environ.get("VAL_HIGH_ELEV_UNDER_THRESHOLD", "58.0"))
    low_thr = float(os.environ.get("VAL_LOW_ELEV_UNDER_THRESHOLD", "45.0"))
    under_w_high = float(os.environ.get("VAL_HIGH_ELEV_UNDER_PENALTY", "2.0"))
    under_w_low = float(os.environ.get("VAL_LOW_ELEV_UNDER_PENALTY", "2.2"))
    above_w = 0.35
    if float(ref_elev) >= high_thr:
        if diff < 0.0:
            return abs(diff) * under_w_high
        return 0.0
    if float(ref_elev) <= low_thr:
        below = max(0.0, -diff) * under_w_low
        above = max(0.0, diff) * above_w
        return below + above
    return abs(diff)


def _candidate_elevation_bonus(
    elev_deg: float,
    ref_view_dir: torch.Tensor | None,
    elev_pref_default: float,
    elev_lambda: float,
    ref_elev_hint: float | None = None,
) -> float:
    """
    Elevation preference:
    - If ref_elev_hint / ref_view_dir is available, penalize deviation from reference elevation.
    - Otherwise fall back to one-sided preference toward elev_pref_default.
    """
    elev_hint = ref_elev_hint
    if elev_hint is None and ref_view_dir is not None:
        elev_hint = _elev_deg_from_view_dir(ref_view_dir)
    if elev_hint is not None:
        return float(elev_lambda) * _ref_elev_penalty_deg(float(elev_deg), float(elev_hint))
    return float(elev_lambda) * max(0.0, float(elev_pref_default) - float(elev_deg))


def _candidate_geometry_prior_score(
    elev_deg: float,
    azim_deg: float,
    mesh_top_normal: torch.Tensor,
    ref_view_dir: torch.Tensor | None,
    device: torch.device,
) -> float:
    """Mesh-normal + reference-view alignment prior (lower is better)."""
    view_dir = _view_dir_from_spherical_deg(elev_deg, azim_deg, device)
    camera_direction = -view_dir
    normal_w = float(os.environ.get("VAL_NORMAL_PRIOR_WEIGHT", "0.35"))
    ref_view_w = float(os.environ.get("VAL_REF_VIEW_PRIOR_WEIGHT", "0.30"))

    normal_prior = 0.0
    # Mesh top-normal prior is only useful when no reference-view direction is available,
    # or when the reference itself is a high-elevation (top-down) view.
    use_mesh_prior = ref_view_dir is None
    if ref_view_dir is not None:
        elev_hint = _elev_deg_from_view_dir(ref_view_dir)
        use_mesh_prior = float(elev_hint) >= 50.0
    if use_mesh_prior:
        normal_dot = float(torch.dot(camera_direction, mesh_top_normal.view(3)).item())
        normal_prior = (1.0 - (normal_dot + 1.0) / 2.0) * normal_w

    ref_prior = 0.0
    if ref_view_dir is not None:
        ref_dot = float(torch.dot(view_dir, ref_view_dir.view(3)).item())
        ref_prior = (1.0 - (ref_dot + 1.0) / 2.0) * ref_view_w
    return float(normal_prior + ref_prior)


def global_random_sampling_1024(
    *,
    mesh: Meshes,
    verts: torch.Tensor,
    mesh_center_xyz: torch.Tensor,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    phong_renderer: MeshRenderer | None = None,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    image_ref_rgb: torch.Tensor | None = None,
    init_dist: float,
    init_elev: float,
    init_azim: float,
    device: torch.device,
    save_dir: str,
    tag: str,
    R_gt: torch.Tensor | None = None,
    T_gt: torch.Tensor | None = None,
) -> list[tuple[float, float, float, float]]:
    """
    Stage-1 (MultiR 改进版：多半径(0.5R~4.0R共36尺度) + Top8，等比例尺度放大)：
    - 已知 3D mesh，先估计模型半径 r（覆盖整个 mesh 的球半径）
    - 以模型半径 r 为基准，在多个半径 = m*r 的球面上采样（默认 m in [0.5,0.6,...,4.0]，36个尺度，步长0.1）
    - 等比例尺度放大：在第一个半径生成基础方向集合（8个象限均匀分布），后续半径复用这些方向
    - 每个半径固定采样4096个位姿（VAL_RAND_SAMPLES_PER_RADIUS，默认4096），在8个象限中均匀分布
    - 总采样数 = 4096 × 36 = 147456
    - 用完整损失（sil+distance+pointcloud+pointcloud_3d，语义损失已禁用）评估所有候选
    - 在每个半径下找到损失最小的位姿（共36个），然后在这36个最优中选择TOP 8

    损失评估策略：
    - 全局搜索阶段：使用全部损失（silhouette+distance+pointcloud+pointcloud_3d，语义损失已禁用）评估所有候选
    - 局部优化阶段：使用轮廓+distance+pointcloud+pointcloud_3d（语义损失已禁用）进行精确评估

    Outputs：
      - <tag>_rand_octant_samples.csv
      - <tag>_rand_octant_best.json
    """
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{tag}_rand_octant_samples.csv")
    json_path = os.path.join(save_dir, f"{tag}_rand_octant_best.json")

    # Optimized: default changed from 128 to 512 for better GPU utilization (~1.5-3x speedup)
    batch = int(os.environ.get("VAL_RAND_BATCH", "512"))
    batch = max(1, batch)
    verbose = int(os.environ.get("VAL_RAND_VERBOSE", "1"))
    # D. 统一“搜索用的 loss”和“精炼用的 loss”，避免排序目标漂移：
    # full-loss here uses the same observation family as refine (silhouette + depth + normals),
    # and does NOT depend on GT pose.
    use_full_loss_global = int(os.environ.get("VAL_RAND_USE_FULL_LOSS", "1")) != 0
    can_use_full_loss = bool(use_full_loss_global)

    # 用户要求: 每个半径选1个最优，共TOP 8（不再使用topk参数，固定为半径数量）

    at_xyz = mesh_center_xyz.view(3).to(device)
    r = _mesh_radius_from_verts(verts, at_xyz)

    # Strict no-GT mode: the GT pose is never consulted by the global search.
    if _strict_no_gt():
        R_gt = None
        T_gt = None
    
    mesh_top_normal = _mesh_top_normal_from_mesh(mesh, device)
    ref_view_dir = _ref_view_direction_from_normals(
        ref_normals, image_ref_silhouette, device, R_gt=R_gt
    )
    ref_elev_hint, ref_azim_hint = _ref_spherical_hint_from_observations(
        ref_normals, image_ref_silhouette, device, R_gt=R_gt
    )
    ref_dist_hint = _ref_dist_hint_from_depth(ref_depth, image_ref_silhouette, device, mesh_r=float(r))
    
    # Multi-radius sampling: log-spaced mults; ``auto`` widens range on large meshes.
    radii_mults = _resolve_radii_mults(float(r), ref_dist_hint)

    samples_per_radius = _resolve_samples_per_radius(float(r))
    samples_per_octant = int(samples_per_radius // 8)  # 每个象限的采样数量
    # 每个半径固定采样数量
    radius_counts = [samples_per_radius] * int(len(radii_mults))
    n_total_eff = int(samples_per_radius * int(len(radii_mults)))
    
    # 8个象限（由x/y/z的正负组合定义）
    octants: list[tuple[int, int, int]] = [
        (+1, +1, +1), (+1, +1, -1), (+1, -1, +1), (+1, -1, -1),
        (-1, +1, +1), (-1, +1, -1), (-1, -1, +1), (-1, -1, -1),
    ]
    
    def _sample_dirs_in_octant_uniform(n_per_octant: int, sx: int, sy: int, sz: int) -> torch.Tensor:
        """
        在指定象限中均匀采样方向
        
        Args:
            n_per_octant: 每个象限的采样数量
            sx, sy, sz: 象限符号 (+1 或 -1)
            
        Returns:
            dirs: (n_per_octant, 3) 归一化方向向量
        """
        # 在第一个象限（正正正）中生成方向，然后通过符号映射到目标象限
        # 使用Gaussian分布然后归一化，得到球面上均匀分布
        v = torch.randn((int(n_per_octant), 3), device=device, dtype=torch.float32)
        v = torch.abs(v)  # 映射到第一象限
        v = v / torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-8)
        # 应用象限符号
        s = torch.tensor([float(sx), float(sy), float(sz)], device=device, dtype=v.dtype).view(1, 3)
        v = v * s
        v = v / torch.norm(v, dim=-1, keepdim=True).clamp(min=1e-8)
        return v
    
    # 等比例尺度放大：在第一个半径生成基础方向集合，后续半径复用这些方向
    # 在第一个半径（最小半径）时，生成8个象限的基础方向
    base_dirs_per_octant: list[torch.Tensor] = []
    for oi, (sx, sy, sz) in enumerate(octants):
        dirs_oct = _sample_dirs_in_octant_uniform(samples_per_octant, sx, sy, sz)
        base_dirs_per_octant.append(dirs_oct)

    if verbose >= 1:
        loss_mode = "full_loss" if can_use_full_loss else "sil_only"
        print(
            f"[RandMultiR] {tag}: total={n_total_eff} radii={len(radii_mults)} "
            f"samples_per_radius={samples_per_radius} (per_octant={samples_per_octant}) batch={batch} loss_mode={loss_mode} "
            f"r={r:.6f} radii_mults=[{radii_mults[0]:.1f},...,{radii_mults[-1]:.1f}] (scale_amplified) topk=8"
        )

    # Generate candidates across radius groups (等比例尺度放大：复用基础方向)
    cand: list[dict] = []  # each: {r_mult, dist_fixed, dist, elev, azim}
    for ri, rm in enumerate(radii_mults):
        dist_fixed = float(max(1e-3, float(rm) * float(r)))
        
        # 等比例尺度放大：所有半径使用相同的基础方向，只是距离不同
        # 合并8个象限的方向
        dirs_all = torch.cat(base_dirs_per_octant, dim=0)  # (samples_per_radius, 3)
        
        # 计算相机位置：at_xyz + direction * distance
        cams = at_xyz.view(1, 3) + dirs_all * torch.tensor(float(dist_fixed), device=device, dtype=dirs_all.dtype).view(1, 1)
        
        for j in range(len(dirs_all)):
            d, e, a = _spherical_from_camera_position_global(cams[j].view(3), at_xyz.view(3))
            cand.append(
                {
                    "r_mult": float(rm),
                    "dist_fixed": float(dist_fixed),
                    "dist": float(d),
                    "elev": float(e),
                    "azim": float(a),
                }
            )

    # Deterministic anchor grid: keep the old random/multi-radius strategy, but
    # add a sparse set of guaranteed viewpoints so important elevation/azimuth
    # bands are never missed by chance.
    if int(os.environ.get("VAL_RAND_ADD_GRID_ANCHORS", "1")) != 0:
        try:
            anchor_elevs = [float(x) for x in os.environ.get("VAL_RAND_ANCHOR_ELEVS", "5,15,30,45,60,75,85").replace(" ", "").split(",") if x != ""]
        except Exception:
            anchor_elevs = [5.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0]
        try:
            anchor_azims = [float(x) for x in os.environ.get("VAL_RAND_ANCHOR_AZIMS", "-180,-135,-90,-45,0,45,90,135,180").replace(" ", "").split(",") if x != ""]
        except Exception:
            anchor_azims = [-180.0, -135.0, -90.0, -45.0, 0.0, 45.0, 90.0, 135.0, 180.0]
        anchor_elevs = _adaptive_anchor_elevs(ref_elev_hint, anchor_elevs)
        anchor_azims = _adaptive_anchor_azims(ref_azim_hint, anchor_azims)
        anchor_count = 0
        for rm in radii_mults:
            dist_fixed = float(max(1e-3, float(rm) * float(r)))
            for e_anchor in anchor_elevs:
                for a_anchor in anchor_azims:
                    cand.append(
                        {
                            "r_mult": float(rm),
                            "dist_fixed": float(dist_fixed),
                            "dist": float(dist_fixed),
                            "elev": float(e_anchor),
                            "azim": float(a_anchor),
                            "anchor": 1,
                        }
                    )
                    anchor_count += 1
        if verbose >= 1:
            print(f"[RandMultiR] {tag}: added deterministic grid anchors={anchor_count}")

    n_total_eff = int(len(cand))

    # Batched evaluation across all candidates (sampling loss)
    losses_total = np.zeros((len(cand),), dtype=np.float64)
    losses_sil = np.zeros((len(cand),), dtype=np.float64)
    losses_depth = np.zeros((len(cand),), dtype=np.float64)
    losses_normal = np.zeros((len(cand),), dtype=np.float64)
    valid_ratio = np.zeros((len(cand),), dtype=np.float64)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "idx",
                "radius_mult",
                "dist_fixed",
                "dist",
                "elev_deg",
                "azim_deg",
                "loss_total",
                "loss_sil",
                "loss_depth",
                "loss_normal",
                "valid_ratio",
            ]
        )
        for base_i in range(0, len(cand), batch):
            end_i = min(len(cand), base_i + batch)
            dlist = [float(cand[i]["dist"]) for i in range(base_i, end_i)]
            elist = [float(cand[i]["elev"]) for i in range(base_i, end_i)]
            alist = [float(cand[i]["azim"]) for i in range(base_i, end_i)]
            # D. 优先使用与 refine 一致的 full-loss（sil+depth+normal）
            if can_use_full_loss:
                evb = _evaluate_pose_losses_batch_no_grad(
                    mesh=mesh,
                    silhouette_renderer=silhouette_renderer,
                    rasterizer=rasterizer,
                    image_ref_silhouette=image_ref_silhouette,
                    ref_depth=ref_depth,
                    ref_normals=ref_normals,
                    at_xyz=at_xyz,
                    dist_list=dlist,
                    elev_list=elist,
                    azim_list=alist,
                    device=device,
                    use_sil_normals_only=False,
                )
                lt = evb["loss_total"].detach().cpu().numpy().astype(np.float64)
                ls = evb["loss_sil"].detach().cpu().numpy().astype(np.float64)
                ld = evb["loss_depth"].detach().cpu().numpy().astype(np.float64)
                ln = evb["loss_normal"].detach().cpu().numpy().astype(np.float64)
                vr = evb["valid_ratio"].detach().cpu().numpy().astype(np.float64)
            else:
                # 回退到轮廓损失评估
                evb = _batched_eval_candidates_no_grad(
                    mesh=mesh,
                    silhouette_renderer=silhouette_renderer,
                    rasterizer=rasterizer,
                    image_ref_silhouette=image_ref_silhouette,
                    ref_depth=ref_depth,
                    ref_normals=ref_normals,
                    at_xyz=at_xyz,
                    dist_list=dlist,
                    elev_list=elist,
                    azim_list=alist,
                    device=device,
                    use_sil_only=True,
                    use_sil_normals_only=False,
                )
                lt = evb["loss_total"].detach().cpu().numpy().astype(np.float64)
                ls = evb["loss_sil"].detach().cpu().numpy().astype(np.float64)
                ld = evb["loss_depth"].detach().cpu().numpy().astype(np.float64)
                ln = evb["loss_normal"].detach().cpu().numpy().astype(np.float64)
                vr = evb["valid_ratio"].detach().cpu().numpy().astype(np.float64)
            for j in range(end_i - base_i):
                i = base_i + j
                losses_total[i] = float(lt[j])
                losses_sil[i] = float(ls[j])
                losses_depth[i] = float(ld[j])
                losses_normal[i] = float(ln[j])
                valid_ratio[i] = float(vr[j])
                c = cand[i]
                w.writerow(
                    [
                        int(i),
                        float(c.get("r_mult", 0.0)),
                        float(c.get("dist_fixed", 0.0)),
                        float(c["dist"]),
                        float(c["elev"]),
                        float(c["azim"]),
                        float(lt[j]),
                        float(ls[j]),
                        float(ld[j]),
                        float(ln[j]),
                        float(vr[j]),
                    ]
                )

    # 新逻辑：在每个半径下找到损失最小的位姿（每个半径选1个最优），然后在所有半径最优中选TOP 8
    radius_best_indices: list[int] = []
    radius_best_losses: list[float] = []
    
    for ri, rm in enumerate(radii_mults):
        # 找到该半径下的所有候选
        idxs_radius = [i for i, c in enumerate(cand) if abs(float(c.get("r_mult", 0.0)) - float(rm)) < 1e-6]
        if len(idxs_radius) <= 0:
            continue
        
        # 算法核心优化：使用 loss + elevation 偏好 + valid_ratio + RGB 的联合评分挑选候选
        # 避免高俯仰视角（例如 77°）因为几何 loss 稍高而被完全淘汰。
        # score = loss + λ * max(0, elev_pref - elev) + valid_ratio_penalty + rgb_penalty
        elev_pref = float(os.environ.get("VAL_RAND_ELEV_PREFERENCE_DEG", "45.0"))
        elev_lambda = float(os.environ.get("VAL_RAND_ELEV_LAMBDA", "0.008"))
        valid_ratio_weight = float(os.environ.get("VAL_GLOBAL_VALID_RATIO_WEIGHT", "0.8"))
        rgb_weight = float(os.environ.get("VAL_GLOBAL_RGB_WEIGHT", "3.0"))
        
        scores_radius = []
        for i in idxs_radius:
            loss_i = float(losses_total[i])
            elev_i = float(cand[i]["elev"])
            azim_i = float(cand[i]["azim"])
            vr_i = float(valid_ratio[i])
            
            elev_bonus = _candidate_elevation_bonus(
                elev_i, ref_view_dir, elev_pref, elev_lambda, ref_elev_hint=ref_elev_hint
            )
            valid_penalty = valid_ratio_weight * (1.0 - vr_i)
            geom_prior = _candidate_geometry_prior_score(
                elev_i, azim_i, mesh_top_normal, ref_view_dir, device
            )
            
            scores_radius.append(loss_i + elev_bonus + valid_penalty + geom_prior)
        
        # 每 radius 取 score 最低的 1~2 个候选（而不是只取 loss 最低）
        k_per_radius = max(1, int(os.environ.get("VAL_RAND_TOPK_PER_RADIUS", "2")))
        k_per_radius = min(k_per_radius, len(idxs_radius))
        
        sorted_local = np.argsort(scores_radius)
        for j in range(k_per_radius):
            idx = idxs_radius[sorted_local[j]]
            radius_best_indices.append(int(idx))
            radius_best_losses.append(float(losses_total[idx]))
    
    # 如果没有找到任何候选，返回初始位姿
    if len(radius_best_indices) <= 0:
        return [(float(init_dist), float(init_elev), float(init_azim), float("inf"))]
    
    # Select Top-K seeds.
    # Important: selecting only the lowest losses can collapse all seeds into a
    # wrong low-elevation basin.  Keep elevation-stratified seeds first, then
    # fill the remaining slots by global loss.  This preserves visual coverage
    # for high-elevation GT poses.
    topk = int(os.environ.get("VAL_RAND_TOPK", "8"))
    topk = max(1, min(topk, len(cand)))

    # Precompute combined score for every candidate (loss + priors).
    combined_scores = np.zeros(len(cand), dtype=np.float64)
    elev_pref = float(os.environ.get("VAL_RAND_ELEV_PREFERENCE_DEG", "45.0"))
    elev_lambda = float(os.environ.get("VAL_RAND_ELEV_LAMBDA", "0.008"))
    valid_ratio_weight = float(os.environ.get("VAL_GLOBAL_VALID_RATIO_WEIGHT", "0.8"))
    flip_threshold = float(os.environ.get("VAL_REF_ELEV_FLIP_THRESHOLD", "35.0"))
    flip_weight = float(os.environ.get("VAL_REF_ELEV_FLIP_PENALTY_WEIGHT", "0.08"))
    azim_flip_threshold = float(os.environ.get("VAL_REF_AZIM_FLIP_THRESHOLD", "40.0"))
    azim_flip_weight = float(os.environ.get("VAL_REF_AZIM_FLIP_PENALTY_WEIGHT", "0.006"))
    for i in range(len(cand)):
        elev_i = float(cand[i]["elev"])
        azim_i = float(cand[i]["azim"])
        elev_bonus = _candidate_elevation_bonus(
            elev_i, ref_view_dir, elev_pref, elev_lambda, ref_elev_hint=ref_elev_hint
        )
        valid_penalty = valid_ratio_weight * (1.0 - float(valid_ratio[i]))
        geom_prior = _candidate_geometry_prior_score(
            elev_i, azim_i, mesh_top_normal, ref_view_dir, device
        )
        flip_penalty = 0.0
        if ref_elev_hint is not None:
            elev_gap = abs(elev_i - float(ref_elev_hint))
            if elev_gap > flip_threshold:
                flip_penalty += flip_weight * (elev_gap - flip_threshold)
        if ref_azim_hint is not None:
            azim_gap = _azim_diff_deg(azim_i, float(ref_azim_hint))
            if azim_gap > azim_flip_threshold:
                flip_penalty += azim_flip_weight * (azim_gap - azim_flip_threshold)
        dist_penalty = _candidate_dist_penalty(float(cand[i]["dist"]), ref_dist_hint)
        combined_scores[i] = float(losses_total[i]) + elev_bonus + valid_penalty + geom_prior + flip_penalty + dist_penalty

    selected_seed_indices: list[int] = []
    if int(os.environ.get("VAL_RAND_DIVERSE_ELEV_SEEDS", "1")) != 0:
        bins_cfg = os.environ.get("VAL_RAND_ELEV_BINS", "0,15,30,45,60,75,90").strip()
        try:
            elev_bins = [float(x) for x in bins_cfg.replace(" ", "").split(",") if x != ""]
        except Exception:
            elev_bins = [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 90.0]
        elev_bins = sorted(list(dict.fromkeys(elev_bins)))
        elev_bins = _adaptive_elev_bins(ref_elev_hint, elev_bins)
        if len(elev_bins) >= 2:
            bin_best: list[tuple[float, int]] = []
            elev_arr = np.array([float(c["elev"]) for c in cand], dtype=np.float64)
            for bi in range(len(elev_bins) - 1):
                lo = float(elev_bins[bi])
                hi = float(elev_bins[bi + 1])
                if bi == len(elev_bins) - 2:
                    mask = (elev_arr >= lo) & (elev_arr <= hi)
                else:
                    mask = (elev_arr >= lo) & (elev_arr < hi)
                idxs_bin = np.where(mask)[0]
                if idxs_bin.size <= 0:
                    continue
                j = int(idxs_bin[int(np.argmin(combined_scores[idxs_bin]))])
                bin_best.append((float(combined_scores[j]), j))
            # Keep every elevation band represented before filling by loss.
            bin_best.sort(key=lambda x: x[0])
            for _lv, j in bin_best:
                if j not in selected_seed_indices:
                    selected_seed_indices.append(int(j))
                if len(selected_seed_indices) >= topk:
                    break

    if len(selected_seed_indices) < topk and int(os.environ.get("VAL_RAND_DIVERSE_AZIM_SEEDS", "1")) != 0:
        bins_cfg = os.environ.get("VAL_RAND_AZIM_BINS", "-180,-90,0,90,180").strip()
        try:
            azim_bins = [float(x) for x in bins_cfg.replace(" ", "").split(",") if x != ""]
        except Exception:
            azim_bins = [-180.0, -90.0, 0.0, 90.0, 180.0]
        azim_bins = sorted(list(dict.fromkeys(azim_bins)))
        azim_bins = _adaptive_azim_bins(ref_azim_hint, azim_bins)
        if len(azim_bins) >= 2:
            azim_arr = np.array([float(c["azim"]) for c in cand], dtype=np.float64)
            bin_best = []
            for bi in range(len(azim_bins) - 1):
                lo = float(azim_bins[bi])
                hi = float(azim_bins[bi + 1])
                if bi == len(azim_bins) - 2:
                    mask = (azim_arr >= lo) & (azim_arr <= hi)
                else:
                    mask = (azim_arr >= lo) & (azim_arr < hi)
                idxs_bin = np.where(mask)[0]
                if idxs_bin.size <= 0:
                    continue
                j = int(idxs_bin[int(np.argmin(combined_scores[idxs_bin]))])
                bin_best.append((float(combined_scores[j]), j))
            bin_best.sort(key=lambda x: x[0])
            for _lv, j in bin_best:
                if j not in selected_seed_indices:
                    selected_seed_indices.append(int(j))
                if len(selected_seed_indices) >= topk:
                    break

    if len(selected_seed_indices) < topk and int(os.environ.get("VAL_RAND_DIVERSE_DIST_SEEDS", "1")) != 0:
        dist_arr = np.array([float(c["dist"]) for c in cand], dtype=np.float64)
        if dist_arr.size >= 5:
            qs = [20.0, 40.0, 60.0, 80.0]
            for q in qs:
                target = float(np.percentile(dist_arr, q))
                tol = max(
                    0.02 * float(r),
                    0.015 * float(np.median(dist_arr)),
                    0.01,
                )
                idxs_near = np.where(np.abs(dist_arr - target) <= tol)[0]
                if idxs_near.size <= 0:
                    j = int(np.argmin(np.abs(dist_arr - target) + combined_scores))
                else:
                    j = int(idxs_near[int(np.argmin(combined_scores[idxs_near]))])
                if int(j) not in selected_seed_indices:
                    selected_seed_indices.append(int(j))
                if len(selected_seed_indices) >= topk:
                    break

    # Fill the remaining seed slots by best combined scores among radius-best poses first.
    radius_best_combined = [float(combined_scores[int(idx)]) for idx in radius_best_indices]
    sorted_radius = np.argsort(radius_best_combined)
    for idx in sorted_radius:
        best_idx = int(radius_best_indices[int(idx)])
        if best_idx not in selected_seed_indices:
            selected_seed_indices.append(best_idx)
        if len(selected_seed_indices) >= topk:
            break

    # If still not enough, fill from all candidates.
    if len(selected_seed_indices) < topk:
        for best_idx in np.argsort(combined_scores):
            best_idx = int(best_idx)
            if best_idx not in selected_seed_indices:
                selected_seed_indices.append(best_idx)
            if len(selected_seed_indices) >= topk:
                break

    # RGB-assisted global seed reranking.
    # After diversity-based selection, re-sort the chosen seeds by a combined score
    # that adds an RGB texture consistency term.  This prevents a wrong symmetric
    # basin with a low geometric (sil+depth+normal) loss from dominating the seed
    # list when the visually correct view (e.g., high elevation) has a slightly
    # higher geometric loss but much better RGB match.
    if (
        phong_renderer is not None
        and image_ref_rgb is not None
        and len(selected_seed_indices) > 1
        and int(os.environ.get("VAL_GLOBAL_RGB_RERANK", "1")) != 0
    ):
        try:
            w_rgb_global = _rgb_weight_for_mesh_radius(
                float(r), "VAL_GLOBAL_RGB_RERANK_W", 6.0
            )
            ref_rgb_t = torch.clamp(image_ref_rgb.to(device), 0, 1)
            if ref_rgb_t.dim() == 4:
                ref_rgb_t = ref_rgb_t[0]
            H_s = silhouette_renderer.rasterizer.raster_settings.image_size[0]
            W_s = silhouette_renderer.rasterizer.raster_settings.image_size[1]
            ref_rgb_ds = torch.nn.functional.interpolate(
                ref_rgb_t.permute(2, 0, 1).unsqueeze(0),
                size=(int(H_s), int(W_s)),
                mode="bilinear",
                align_corners=False,
            )[0].permute(1, 2, 0)
            ref_sil_base = torch.clamp(image_ref_silhouette, 0, 1)
            if ref_sil_base.dim() == 3 and ref_sil_base.shape[-1] == 4:
                ref_sil_base = ref_sil_base[..., 3]
            ref_sil_base = ref_sil_base.squeeze()
            rgb_combined_scores: list[float] = []
            for best_idx in selected_seed_indices:
                c = cand[best_idx]
                try:
                    with torch.no_grad():
                        Rb, Tb = look_at_view_transform(
                            dist=torch.tensor([float(c["dist"])], device=device),
                            elev=torch.tensor([float(c["elev"])], device=device),
                            azim=torch.tensor([float(c["azim"])], device=device),
                            at=at_xyz.view(1, 3),
                            device=device,
                        )
                        cams = FoVPerspectiveCameras(device=device, R=Rb, T=Tb)
                        rgb_r = torch.clamp(phong_renderer(meshes_world=mesh, cameras=cams)[0, ..., :3], 0, 1)
                        sil_r = torch.clamp(silhouette_renderer(meshes_world=mesh, R=Rb, T=Tb)[0, ..., 3], 0, 1)
                        mask = ((ref_sil_base > 0.05) | (sil_r > 0.05)).float().unsqueeze(-1)
                        denom = float(mask.sum().item()) * 3.0 + 1e-8
                        rgb_loss = float(((rgb_r - ref_rgb_ds) ** 2 * mask).sum().detach().cpu().item()) / denom
                except Exception:
                    rgb_loss = 0.0
                rgb_combined_scores.append(float(combined_scores[best_idx]) + w_rgb_global * rgb_loss)
            paired_rgb = sorted(zip(rgb_combined_scores, selected_seed_indices), key=lambda x: x[0])
            selected_seed_indices = [j for _, j in paired_rgb]
            if verbose >= 1:
                top_c = cand[selected_seed_indices[0]]
                print(
                    f"[RandMultiR] {tag}: RGB-reranked seeds, "
                    f"top=(elev={top_c['elev']:.1f},azim={top_c['azim']:.1f}) "
                    f"combined={paired_rgb[0][0]:.6f}"
                )
        except Exception as _e_rgb:
            if verbose >= 1:
                print(f"[RandMultiR] {tag}: RGB global rerank failed: {_e_rgb}")

    seeds: list[tuple[float, float, float, float]] = []
    for best_idx in selected_seed_indices[:topk]:
        c = cand[best_idx]
        seeds.append((float(c["dist"]), float(c["elev"]), float(c["azim"]), float(losses_total[best_idx])))

    # Best summary: 在所有半径的最优中，找到全局最优（损失最小的那个）
    if len(radius_best_indices) <= 0:
        best_i = 0
        best_loss = float("inf")
        best_c = {"dist": float(init_dist), "elev": float(init_elev), "azim": float(init_azim), "r_mult": 0.0}
    else:
        # 找到所有半径最优中损失最小的
        best_overall_idx = int(np.argmin(radius_best_losses))
        best_i = int(radius_best_indices[best_overall_idx])
        best_c = cand[best_i]
        best_loss = float(losses_total[best_i])

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tag": tag,
                    "sampling": {
                        "total": int(n_total_eff),
                        "radii_mults": [float(x) for x in radii_mults],
                        "samples_per_radius": int(samples_per_radius),
                        "model_radius_r": float(r),
                        "use_full_loss": bool(can_use_full_loss),
                        "topk": int(topk),  # 从36个半径最优中选TOP 8
                    },
                    "best": {
                        "loss_total": float(best_loss),
                        "dist": float(best_c["dist"]),
                        "elev_deg": float(best_c["elev"]),
                        "azim_deg": float(best_c["azim"]),
                        "radius_mult": float(best_c.get("r_mult", 0.0)),
                    },
                    "topk_seeds": [{"dist": d, "elev_deg": e, "azim_deg": a, "loss_total": lt} for (d, e, a, lt) in seeds],
                    "logs": {"samples_csv": os.path.basename(csv_path)},
                    "note": f"topk_seeds: selected TOP {topk} with elevation-diverse seeding first, then radius-best/global-loss fill. Scale-amplified sampling: base directions generated once and reused across all radii. All candidates evaluated with full loss (sil+depth+normals, semantic loss disabled).",
                },
                f,
                indent=2,
                ensure_ascii=True,
            )
    except Exception as _e:
        print(f"[Warning] Failed to write rand_octant_best.json: {_e}")

    if verbose >= 1:
        print(f"\n{'='*80}")
        print(f"[RandMultiR] {tag} - 全局随机采样结果")
        print(f"{'='*80}")
        print(f"  采样总数: {n_total_eff} ({len(radii_mults)}个半径，每个半径固定{samples_per_radius}样本)")
        print(f"  模型半径: {r:.6f}")
        print(f"  最佳候选: dist={best_c['dist']:.3f}, elev={best_c['elev']:.3f}, azim={best_c['azim']:.3f}, radius_mult={best_c.get('r_mult', 0.0):.2f}")
        print(f"  最佳损失: {best_loss:.6f}")
        print(f"  选择TopK种子数: {len(seeds)} (elevation-diverse + radius/global best, TOP {topk})")
        print(f"{'='*80}\n")

    # Save clearly named best artifacts (global best across all radii)
    try:
        _render_and_save_best_artifacts(
            mesh=mesh,
            silhouette_renderer=silhouette_renderer,
            rasterizer=rasterizer,
            phong_renderer=phong_renderer,
            at_xyz=at_xyz,
            dist_v=float(best_c["dist"]),
            elev_v=float(best_c["elev"]),
            azim_v=float(best_c["azim"]),
            device=device,
            save_dir=save_dir,
            prefix="best_rand_octant",
        )
    except Exception as _e:
        print(f"[Warning] Failed to save best_rand_octant artifacts: {_e}")

    return seeds


def greedy_path_on_sample_set(
    *,
    samples: list[tuple[float, float, float, float]],
    at_xyz: torch.Tensor,
    init_dist: float,
    init_elev: float,
    init_azim: float,
    device: torch.device,
    save_dir: str,
    tag: str,
) -> dict:
    """
    Optional: build a greedy descending path using ONLY the sampled candidates.
    Strategy:
    - Represent each sample as camera xyz in WORLD.
    - Start at init camera xyz.
    - Repeatedly jump to a nearby sample with strictly lower loss (greedy), up to K steps.
    Logs:
      - <tag>_rand1024_greedy_path.csv / .json
    """
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{tag}_rand1024_greedy_path.csv")
    json_path = os.path.join(save_dir, f"{tag}_rand1024_greedy_path.json")

    max_steps = int(os.environ.get("VAL_RAND_PATH_STEPS", "32"))
    max_steps = max(1, max_steps)
    knn = int(os.environ.get("VAL_RAND_PATH_KNN", "64"))
    knn = max(8, knn)
    min_delta = float(os.environ.get("VAL_RAND_PATH_MIN_DELTA", "1e-4"))

    at_xyz = at_xyz.view(3).to(device)
    cam_init = _camera_position_from_spherical_global(
        torch.tensor(float(init_dist), device=device),
        torch.tensor(float(init_elev), device=device),
        torch.tensor(float(init_azim), device=device),
        at_xyz,
    )

    # Precompute xyz and losses
    losses = np.array([float(s[0]) for s in samples], dtype=np.float64)
    dlist = [float(s[1]) for s in samples]
    elist = [float(s[2]) for s in samples]
    alist = [float(s[3]) for s in samples]
    xyz = []
    for i in range(len(samples)):
        cam = _camera_position_from_spherical_global(
            torch.tensor(float(dlist[i]), device=device),
            torch.tensor(float(elist[i]), device=device),
            torch.tensor(float(alist[i]), device=device),
            at_xyz,
        )
        xyz.append(cam.detach().cpu().numpy().astype(np.float64))
    xyz = np.stack(xyz, axis=0) if len(xyz) > 0 else np.zeros((0, 3), dtype=np.float64)
    cur_xyz = cam_init.detach().cpu().numpy().astype(np.float64)
    cur_loss = float("inf")
    path = []
    visited = set()

    # Start: pick nearest sample to init as anchor (not necessarily best)
    if xyz.shape[0] > 0:
        dist2 = np.sum((xyz - cur_xyz[None, :]) ** 2, axis=1)
        cur_idx = int(np.argmin(dist2))
        cur_loss = float(losses[cur_idx])
        visited.add(cur_idx)
        path.append({"step": 0, "idx": cur_idx, "loss_total": cur_loss, "dist": float(dlist[cur_idx]), "elev_deg": float(elist[cur_idx]), "azim_deg": float(alist[cur_idx])})
    else:
        cur_idx = -1

    for step in range(1, max_steps + 1):
        if cur_idx < 0:
            break
        # find KNN around current node (euclidean in xyz)
        dist2 = np.sum((xyz - xyz[cur_idx][None, :]) ** 2, axis=1)
        nn = np.argsort(dist2)[: min(knn, xyz.shape[0])]
        best_next = None
        for j in nn:
            j = int(j)
            if j in visited:
                continue
            lj = float(losses[j])
            if lj < (cur_loss - min_delta):
                best_next = j
                break  # nearest-first greedy
        if best_next is None:
            break
        cur_idx = int(best_next)
        cur_loss = float(losses[cur_idx])
        visited.add(cur_idx)
        path.append({"step": int(step), "idx": cur_idx, "loss_total": cur_loss, "dist": float(dlist[cur_idx]), "elev_deg": float(elist[cur_idx]), "azim_deg": float(alist[cur_idx])})

    # direction hint = last - first (xyz)
    if len(path) >= 2:
        a = xyz[int(path[0]["idx"])]
        b = xyz[int(path[-1]["idx"])]
        delta = torch.tensor(b - a, device=device, dtype=torch.float32)
        dom_axis = _dominant_axis_from_delta_world(delta)
        delta_np = (b - a).tolist()
    else:
        dom_axis = None
        delta_np = None

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "idx", "loss_total", "dist", "elev_deg", "azim_deg"])
            for p in path:
                w.writerow([p["step"], p["idx"], p["loss_total"], p["dist"], p["elev_deg"], p["azim_deg"]])
    except Exception as _e:
        print(f"[Warning] Failed to write rand1024_greedy_path.csv: {_e}")

    out = {"path": path, "direction_hint": {"dominant_axis": dom_axis, "delta_cam_xyz_world": delta_np}, "logs": {"csv": os.path.basename(csv_path)}}
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(out, f, indent=2, ensure_ascii=True)
    except Exception as _e:
        print(f"[Warning] Failed to write rand1024_greedy_path.json: {_e}")

    return out


def local_axis_greedy_backtrack_on_sphere(
    *,
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    phong_renderer: MeshRenderer | None = None,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    at_xyz: torch.Tensor,
    start_dist: float,
    start_elev: float,
    start_azim: float,
    device: torch.device,
    save_dir: str,
    tag: str,
) -> tuple[float, float, float, float]:
    """
    局部阶段（你强调的“loss变大就回退/换方向”）：
    - 固定球面半径=当前 dist
    - 每轮对 6 个方向分别做 line-search: 顺序评估 step=1..16，若 loss 开始上升就停止该方向，
      并取该方向上最好的那一步作为候选（这就是“回退”）。
    - 从 6 个方向候选中选全局最好的；若能改进则接受，否则停止。
    - 额外步骤（用户要求）：在完成一轮 6 方向探索后，找到最佳方向，并沿该方向继续贪婪迭代若干次
      （默认 2000 次；若 loss 不再下降则提前停止）。仍然在球面空间上进行。
    """
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{tag}_local_axis_bt.csv")
    json_path = os.path.join(save_dir, f"{tag}_local_axis_bt_best.json")

    # User request: each direction explores 4 steps, then rollback if loss increases.
    # User request: each direction explores 4 steps (default).
    steps = int(os.environ.get("VAL_LOCAL_FORWARD_STEPS", "4"))
    steps = max(1, steps)
    step_unit = float(os.environ.get("VAL_LOCAL_STEP_UNIT", "1.0"))
    iters = int(os.environ.get("VAL_LOCAL_ITERS", "64"))
    iters = max(1, iters)
    min_delta = float(os.environ.get("VAL_LOCAL_MIN_DELTA", "1e-5"))
    best_dir_iters = int(os.environ.get("VAL_LOCAL_BEST_DIR_ITERS", "2000"))  # set 0 to disable
    best_dir_iters = max(0, best_dir_iters)
    best_dir_step_mult = float(os.environ.get("VAL_LOCAL_BEST_DIR_STEP_UNIT_MULT", "1.0"))
    best_dir_step_mult = max(1e-6, best_dir_step_mult)
    best_dir_min_delta = float(os.environ.get("VAL_LOCAL_BEST_DIR_MIN_DELTA", str(min_delta)))
    use_sil_only = int(os.environ.get("VAL_LOCAL_USE_SIL_ONLY", "0")) != 0
    verbose = int(os.environ.get("VAL_LOCAL_VERBOSE", "1"))

    at_xyz = at_xyz.view(3).to(device)
    radius = float(start_dist)

    axes = [
        torch.tensor([1.0, 0.0, 0.0], device=device),
        torch.tensor([-1.0, 0.0, 0.0], device=device),
        torch.tensor([0.0, 1.0, 0.0], device=device),
        torch.tensor([0.0, -1.0, 0.0], device=device),
        torch.tensor([0.0, 0.0, 1.0], device=device),
        torch.tensor([0.0, 0.0, -1.0], device=device),
    ]
    axis_names = ["+x", "-x", "+y", "-y", "+z", "-z"]

    cam0 = _camera_position_from_spherical_global(
        torch.tensor(float(radius), device=device),
        torch.tensor(float(start_elev), device=device),
        torch.tensor(float(start_azim), device=device),
        at_xyz,
    )
    v_center = cam0 - at_xyz
    v_center = v_center / torch.norm(v_center).clamp(min=1e-8) * torch.tensor(float(radius), device=device)
    cur_dist, cur_elev, cur_azim = _spherical_from_camera_position_global(at_xyz + v_center, at_xyz)
    cur_ev = _evaluate_pose_sil_no_grad(
        mesh=mesh,
        silhouette_renderer=silhouette_renderer,
        image_ref_silhouette=image_ref_silhouette,
        at_xyz=at_xyz,
        dist_v=float(cur_dist),
        elev_v=float(cur_elev),
        azim_v=float(cur_azim),
        device=device,
    ) if use_sil_only else _evaluate_pose_loss_no_grad(
        mesh, silhouette_renderer, rasterizer, image_ref_silhouette, ref_depth, ref_normals,
        at_xyz, float(cur_dist), float(cur_elev), float(cur_azim), device
    )
    cur_loss = float(cur_ev["loss_total"])

    if verbose >= 1:
        print(
            f"[LocalAxisBT] {tag}: start_loss={cur_loss:.6f} radius={radius:.6f} steps={steps} step_unit={step_unit:.3f} "
            f"iters={iters} best_dir_iters={best_dir_iters} sil_only={int(use_sil_only)}"
        )

    rows = []
    rows.append(["iter", "axis", "step", "delta_unit", "dist", "elev_deg", "azim_deg", "loss_total", "axis_best_step", "picked_iter"])
    path = [{"iter": 0, "axis": "center", "loss_total": cur_loss, "dist": cur_dist, "elev_deg": cur_elev, "azim_deg": cur_azim}]

    best = {"loss": cur_loss, "dist": float(cur_dist), "elev": float(cur_elev), "azim": float(cur_azim)}
    last_picked_axis_idx: int | None = None

    for it in range(1, iters + 1):
        # SPEED: evaluate ALL 6*steps candidates in ONE batched call, then emulate
        # per-axis "stop when loss starts increasing" logic on the resulting loss curves.
        dlist: list[float] = []
        elist: list[float] = []
        alist: list[float] = []
        meta: list[tuple[int, int, float, torch.Tensor]] = []  # (axis_idx, step_idx, delta, v_new)

        for ai, u in enumerate(axes):
            for st in range(1, steps + 1):
                delta = float(step_unit) * float(st)
                v_new = v_center + (delta * u)
                v_new = v_new / torch.norm(v_new).clamp(min=1e-8) * torch.tensor(float(radius), device=device)
                cam_new = at_xyz + v_new
                dist_new, elev_new, azim_new = _spherical_from_camera_position_global(cam_new, at_xyz)
                dlist.append(float(dist_new))
                elist.append(float(elev_new))
                alist.append(float(azim_new))
                meta.append((int(ai), int(st), float(delta), v_new.detach()))

        evb = _batched_eval_candidates_no_grad(
            mesh=mesh,
            silhouette_renderer=silhouette_renderer,
            rasterizer=rasterizer,
            image_ref_silhouette=image_ref_silhouette,
            ref_depth=ref_depth,
            ref_normals=ref_normals,
            at_xyz=at_xyz,
            dist_list=dlist,
            elev_list=elist,
            azim_list=alist,
            device=device,
            use_sil_only=bool(use_sil_only),
        )
        lt = evb["loss_total"].detach().cpu().numpy().astype(np.float64)

        axis_best = []  # list of dict per axis
        for ai in range(6):
            # Extract this axis's loss curve in step order 1..steps
            idxs = [j for j, (a_idx, _st, _d, _v) in enumerate(meta) if a_idx == ai]
            # idxs should already be in st order due to construction; keep safe sort:
            idxs.sort(key=lambda j: meta[j][1])
            losses = [float(lt[j]) for j in idxs]

            # emulate sequential "if loss increases vs previous by min_delta => stop"
            stop_k = len(losses)  # exclusive
            for k in range(1, len(losses)):
                if losses[k] > losses[k - 1] + float(min_delta):
                    stop_k = k
                    break
            if stop_k <= 0:
                stop_k = 1

            # choose best within explored prefix [0, stop_k)
            best_k = int(np.argmin(np.array(losses[:stop_k], dtype=np.float64)))
            best_global_j = idxs[best_k]
            _ai, best_step, best_delta, best_v = meta[best_global_j]

            axis_best.append(
                {
                    "axis": axis_names[ai],
                    "best_step": int(best_step),
                    "loss_total": float(losses[best_k]),
                    "dist": float(dlist[best_global_j]),
                    "elev_deg": float(elist[best_global_j]),
                    "azim_deg": float(alist[best_global_j]),
                    "v": best_v,
                }
            )

        # Pick best axis candidate
        axis_best.sort(key=lambda x: x["loss_total"])
        picked = axis_best[0]
        for rec in axis_best:
            rows.append([int(it), rec["axis"], int(rec["best_step"]), float(step_unit) * float(rec["best_step"]), float(rec["dist"]), float(rec["elev_deg"]), float(rec["azim_deg"]), float(rec["loss_total"]), int(rec["best_step"]), 1 if rec["axis"] == picked["axis"] else 0])

        if float(picked["loss_total"]) < (cur_loss - min_delta):
            cur_loss = float(picked["loss_total"])
            v_center = picked["v"]
            cur_dist, cur_elev, cur_azim = float(picked["dist"]), float(picked["elev_deg"]), float(picked["azim_deg"])
            path.append({"iter": int(it), "axis": picked["axis"], "step": int(picked["best_step"]), "loss_total": cur_loss, "dist": cur_dist, "elev_deg": cur_elev, "azim_deg": cur_azim})
            # remember best direction for the optional continuation stage
            try:
                last_picked_axis_idx = int(axis_names.index(str(picked["axis"])))
            except Exception:
                last_picked_axis_idx = None
            if cur_loss < float(best["loss"]):
                best.update({"loss": cur_loss, "dist": cur_dist, "elev": cur_elev, "azim": cur_azim})
            if verbose >= 2:
                print(f"[LocalAxisBT] {tag}: iter={it} pick={picked['axis']} step={picked['best_step']} loss={cur_loss:.6f}")
        else:
            if verbose >= 1:
                print(f"[LocalAxisBT] {tag}: stop at iter={it} (no improvement > {min_delta:.2e}); cur_loss={cur_loss:.6f}")
            break

    # Optional: after picking the best direction, keep greedy-walking along that direction for many iterations.
    best_dir_ran = 0
    best_dir_axis = None
    if best_dir_iters > 0 and last_picked_axis_idx is not None:
        u_dir = axes[int(last_picked_axis_idx)]
        best_dir_axis = axis_names[int(last_picked_axis_idx)]
        dir_step_unit = float(step_unit) * float(best_dir_step_mult)
        if verbose >= 1:
            print(
                f"[LocalAxisBT] {tag}: best-direction continue axis={best_dir_axis} "
                f"iters={best_dir_iters} step_unit={dir_step_unit:.6f} min_delta={best_dir_min_delta:.2e}"
            )
        for k in range(1, best_dir_iters + 1):
            v_new = v_center + (float(dir_step_unit) * u_dir)
            v_new = v_new / torch.norm(v_new).clamp(min=1e-8) * torch.tensor(float(radius), device=device)
            cam_new = at_xyz + v_new
            dist_new, elev_new, azim_new = _spherical_from_camera_position_global(cam_new, at_xyz)
            ev_new = _evaluate_pose_sil_no_grad(
                mesh=mesh,
                silhouette_renderer=silhouette_renderer,
                image_ref_silhouette=image_ref_silhouette,
                at_xyz=at_xyz,
                dist_v=float(dist_new),
                elev_v=float(elev_new),
                azim_v=float(azim_new),
                device=device,
            ) if use_sil_only else _evaluate_pose_loss_no_grad(
                mesh,
                silhouette_renderer,
                rasterizer,
                image_ref_silhouette,
                ref_depth,
                ref_normals,
                at_xyz,
                float(dist_new),
                float(elev_new),
                float(azim_new),
                device,
            )
            loss_new = float(ev_new["loss_total"])
            rows.append([int(iters + k), f"bestdir:{best_dir_axis}", 1, float(dir_step_unit), float(dist_new), float(elev_new), float(azim_new), float(loss_new), -1, 1])
            if loss_new < (cur_loss - float(best_dir_min_delta)):
                cur_loss = float(loss_new)
                v_center = v_new
                cur_dist, cur_elev, cur_azim = float(dist_new), float(elev_new), float(azim_new)
                best_dir_ran = int(k)
                path.append({"iter": int(iters + k), "axis": f"bestdir:{best_dir_axis}", "step": 1, "loss_total": cur_loss, "dist": cur_dist, "elev_deg": cur_elev, "azim_deg": cur_azim})
                if cur_loss < float(best["loss"]):
                    best.update({"loss": cur_loss, "dist": cur_dist, "elev": cur_elev, "azim": cur_azim})
                if verbose >= 2 and (k <= 10 or (k % 50 == 0)):
                    print(f"[LocalAxisBT] {tag}: bestdir k={k} axis={best_dir_axis} loss={cur_loss:.6f}")
            else:
                if verbose >= 1:
                    print(f"[LocalAxisBT] {tag}: bestdir stop at k={k} (no improvement > {best_dir_min_delta:.2e}); cur_loss={cur_loss:.6f}")
                break

    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for row in rows:
                w.writerow(row)
    except Exception as _e:
        print(f"[Warning] Failed to write local_axis_bt.csv: {_e}")
    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tag": tag,
                    "radius": float(radius),
                    "steps": int(steps),
                    "step_unit": float(step_unit),
                    "iters": int(iters),
                    "best_dir": {
                        "enabled": bool(best_dir_iters > 0),
                        "axis": best_dir_axis,
                        "iters": int(best_dir_iters),
                        "iters_ran": int(best_dir_ran),
                        "step_unit_mult": float(best_dir_step_mult),
                        "min_delta": float(best_dir_min_delta),
                    },
                    "min_delta": float(min_delta),
                    "use_sil_only": bool(use_sil_only),
                    "best": {"loss_total": float(best["loss"]), "dist": float(best["dist"]), "elev_deg": float(best["elev"]), "azim_deg": float(best["azim"])},
                    "path": path,
                },
                f,
                indent=2,
                ensure_ascii=True,
            )
    except Exception as _e:
        print(f"[Warning] Failed to write local_axis_bt_best.json: {_e}")

    if verbose >= 1:
        print(f"[LocalAxisBT] {tag}: best_loss={float(best['loss']):.6f} best(dist,elev,azim)=({float(best['dist']):.3f},{float(best['elev']):.3f},{float(best['azim']):.3f}) saved={os.path.basename(csv_path)}")

    # Save clearly named best artifacts for quick visual inspection.
    try:
        _render_and_save_best_artifacts(
            mesh=mesh,
            silhouette_renderer=silhouette_renderer,
            rasterizer=rasterizer,
            phong_renderer=phong_renderer,
            at_xyz=at_xyz,
            dist_v=float(best["dist"]),
            elev_v=float(best["elev"]),
            azim_v=float(best["azim"]),
            device=device,
            save_dir=save_dir,
            prefix="best_local_axis_bt",
        )
    except Exception as _e:
        print(f"[Warning] Failed to save best_local_axis_bt_* artifacts: {_e}")

    return float(best["dist"]), float(best["elev"]), float(best["azim"]), float(best["loss"])


def local_octant_hierarchical_search_on_sphere(
    *,
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    phong_renderer: MeshRenderer | None = None,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    at_xyz: torch.Tensor,
    start_dist: float,
    start_elev: float,
    start_azim: float,
    device: torch.device,
    save_dir: str,
    tag: str,
) -> tuple[float, float, float, float]:
    """
    用户要求的两级层次化搜索（球面空间）：
    Level-1: 先在 8 个象限中各探索 K 步（默认 8 步），找到损失下降的最优象限
    Level-2: 在最优象限内，沿 ±x/±y/±z 6 个方向继续细粒度探索
    回退机制: 若损失持续上升，回退到上一层/换象限/换种子

    策略：
    1. 当前位置投影到球面（固定半径 = start_dist）
    2. 对 8 个象限（由 x/y/z 正负组合）各沿其主方向探索 octant_steps 步
    3. 选损失最小的象限作为"当前象限"
    4. 在该象限内，沿 6 个世界轴方向（±x/±y/±z）做 axis_steps 步探索
    5. 若有改进则接受，否则回退；迭代至收敛

    Env（可覆盖）：
    - VAL_LOCAL_OCT_ENABLE: 1/0（默认 1）
    - VAL_LOCAL_OCT_ITERS: 最大迭代数（默认 128）
    - VAL_LOCAL_OCT_OCTANT_STEPS: 象限探索步数（默认 8）
    - VAL_LOCAL_OCT_AXIS_STEPS: 方向探索步数（默认 6）
    - VAL_LOCAL_OCT_STEP_FRAC: 步长比例（默认 0.02，实际步长=step_frac*dist）
    - VAL_LOCAL_OCT_MIN_DELTA: 最小改进阈值（默认 1e-5）
    - VAL_LOCAL_OCT_USE_SIL_ONLY: 1/0（默认 0，用完整损失）
    """
    enabled = int(os.environ.get("VAL_LOCAL_OCT_ENABLE", "1")) != 0
    if not enabled:
        return float(start_dist), float(start_elev), float(start_azim), float("inf")

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{tag}_local_octant_hier.csv")
    json_path = os.path.join(save_dir, f"{tag}_local_octant_hier_best.json")

    iters = max(1, int(os.environ.get("VAL_LOCAL_OCT_ITERS", "128")))
    octant_steps = max(1, int(os.environ.get("VAL_LOCAL_OCT_OCTANT_STEPS", "8")))
    axis_steps = max(1, int(os.environ.get("VAL_LOCAL_OCT_AXIS_STEPS", "6")))
    step_frac = float(os.environ.get("VAL_LOCAL_OCT_STEP_FRAC", "0.02"))
    min_delta = float(os.environ.get("VAL_LOCAL_OCT_MIN_DELTA", "1e-5"))
    use_sil_only = int(os.environ.get("VAL_LOCAL_OCT_USE_SIL_ONLY", "0")) != 0
    verbose = int(os.environ.get("VAL_LOCAL_OCT_VERBOSE", "1"))

    at_xyz = at_xyz.view(3).to(device)
    radius = float(start_dist)

    # 8 octants (sign combinations of x/y/z)
    octants = [
        (+1, +1, +1),
        (+1, +1, -1),
        (+1, -1, +1),
        (+1, -1, -1),
        (-1, +1, +1),
        (-1, +1, -1),
        (-1, -1, +1),
        (-1, -1, -1),
    ]

    # 6 world axes for fine exploration
    axes = [
        ("+x", torch.tensor([1.0, 0.0, 0.0], device=device)),
        ("-x", torch.tensor([-1.0, 0.0, 0.0], device=device)),
        ("+y", torch.tensor([0.0, 1.0, 0.0], device=device)),
        ("-y", torch.tensor([0.0, -1.0, 0.0], device=device)),
        ("+z", torch.tensor([0.0, 0.0, 1.0], device=device)),
        ("-z", torch.tensor([0.0, 0.0, -1.0], device=device)),
    ]

    def _project_to_sphere(xyz: torch.Tensor) -> torch.Tensor:
        v = xyz - at_xyz
        return at_xyz + (v / (torch.norm(v) + 1e-8)) * torch.tensor(radius, device=device)

    # Init position on sphere
    cam0 = _camera_position_from_spherical_global(
        torch.tensor(radius, device=device),
        torch.tensor(float(start_elev), device=device),
        torch.tensor(float(start_azim), device=device),
        at_xyz,
    )
    cam_cur = _project_to_sphere(cam0)
    d_cur, e_cur, a_cur = _spherical_from_camera_position_global(cam_cur, at_xyz)

    # Eval init
    if use_sil_only:
        cur_ev = _evaluate_pose_sil_no_grad(
            mesh, silhouette_renderer, image_ref_silhouette, at_xyz, float(d_cur), float(e_cur), float(a_cur), device
        )
    else:
        cur_ev = _evaluate_pose_loss_no_grad(
            mesh, silhouette_renderer, rasterizer, image_ref_silhouette, ref_depth, ref_normals,
            at_xyz, float(d_cur), float(e_cur), float(a_cur), device
        )
    cur_loss = float(cur_ev["loss_total"])
    best = {"loss": cur_loss, "dist": float(d_cur), "elev": float(e_cur), "azim": float(a_cur), "iter": 0}

    if verbose >= 1:
        print(
            f"[LocalOctHier] {tag}: start_loss={cur_loss:.6f} radius={radius:.6f} "
            f"iters={iters} octant_steps={octant_steps} axis_steps={axis_steps} step_frac={step_frac:.4f} "
            f"use_sil_only={int(use_sil_only)}"
        )

    rows = []
    rows.append(["iter", "level", "octant_or_axis", "step_idx", "loss", "dist", "elev_deg", "azim_deg", "action"])
    rows.append([0, "init", "center", 0, cur_loss, d_cur, e_cur, a_cur, "start"])

    path = []
    path.append({"iter": 0, "level": "init", "loss": cur_loss, "dist": d_cur, "elev_deg": e_cur, "azim_deg": a_cur})

    for it in range(1, iters + 1):
        step_len = float(step_frac) * float(radius)

        # -----------------------
        # Level-1: 8 象限粗探索（每个象限沿其主方向探索 K 步）
        # -----------------------
        oct_cands = []  # list of (loss, cam_xyz, octant_idx, step_k, dist, elev, azim)
        for oi, (sx, sy, sz) in enumerate(octants):
            # 象限主方向：单位向量 (sx, sy, sz) / norm
            oct_dir = torch.tensor([float(sx), float(sy), float(sz)], device=device, dtype=torch.float32)
            oct_dir = oct_dir / torch.norm(oct_dir).clamp(min=1e-8)
            for k in range(1, octant_steps + 1):
                xyz_c = _project_to_sphere(cam_cur + oct_dir * (step_len * float(k)))
                d_c, e_c, a_c = _spherical_from_camera_position_global(xyz_c, at_xyz)
                if use_sil_only:
                    ev_c = _evaluate_pose_sil_no_grad(
                        mesh, silhouette_renderer, image_ref_silhouette, at_xyz, float(d_c), float(e_c), float(a_c), device
                    )
                else:
                    ev_c = _evaluate_pose_loss_no_grad(
                        mesh, silhouette_renderer, rasterizer, image_ref_silhouette, ref_depth, ref_normals,
                        at_xyz, float(d_c), float(e_c), float(a_c), device
                    )
                loss_c = float(ev_c["loss_total"])
                oct_cands.append((loss_c, xyz_c.detach(), int(oi), int(k), float(d_c), float(e_c), float(a_c)))
                rows.append([it, "L1_octant", f"oct{oi}", k, loss_c, d_c, e_c, a_c, "explore"])

        if len(oct_cands) == 0:
            break

        # 选损失最小的象限候选
        oct_cands.sort(key=lambda x: x[0])
        best_oct_loss, best_oct_xyz, best_oct_idx, best_oct_k, best_oct_d, best_oct_e, best_oct_a = oct_cands[0]

        # 若象限探索有改进，进入 Level-2（在该象限内6方向细探索）
        if best_oct_loss < (cur_loss - min_delta):
            if verbose >= 2:
                print(f"[LocalOctHier] iter={it} L1: best_octant={best_oct_idx} step={best_oct_k} loss={best_oct_loss:.6f} -> L2 axis explore")

            # Level-2: 在最优象限内，沿 6 个世界轴方向探索
            axis_cands = []
            for axis_name, axis_vec in axes:
                for k in range(1, axis_steps + 1):
                    xyz_a = _project_to_sphere(best_oct_xyz + axis_vec * (step_len * float(k)))
                    d_a, e_a, a_a = _spherical_from_camera_position_global(xyz_a, at_xyz)
                    if use_sil_only:
                        ev_a = _evaluate_pose_sil_no_grad(
                            mesh, silhouette_renderer, image_ref_silhouette, at_xyz, float(d_a), float(e_a), float(a_a), device
                        )
                    else:
                        ev_a = _evaluate_pose_loss_no_grad(
                            mesh, silhouette_renderer, rasterizer, image_ref_silhouette, ref_depth, ref_normals,
                            at_xyz, float(d_a), float(e_a), float(a_a), device
                        )
                    loss_a = float(ev_a["loss_total"])
                    axis_cands.append((loss_a, xyz_a.detach(), axis_name, int(k), float(d_a), float(e_a), float(a_a)))
                    rows.append([it, "L2_axis", axis_name, k, loss_a, d_a, e_a, a_a, "explore"])

            if len(axis_cands) == 0:
                # L2 为空，直接接受 L1 最优
                cam_cur = best_oct_xyz.detach()
                cur_loss = float(best_oct_loss)
                d_cur, e_cur, a_cur = float(best_oct_d), float(best_oct_e), float(best_oct_a)
                rows.append([it, "accept", f"oct{best_oct_idx}_k{best_oct_k}", 0, cur_loss, d_cur, e_cur, a_cur, "L1_only"])
            else:
                axis_cands.sort(key=lambda x: x[0])
                best_axis_loss, best_axis_xyz, best_axis_name, best_axis_k, best_axis_d, best_axis_e, best_axis_a = axis_cands[0]

                # 比较 L1 最优 vs L2 最优，选全局最优
                if best_axis_loss < best_oct_loss:
                    cam_cur = best_axis_xyz.detach()
                    cur_loss = float(best_axis_loss)
                    d_cur, e_cur, a_cur = float(best_axis_d), float(best_axis_e), float(best_axis_a)
                    rows.append([it, "accept", f"{best_axis_name}_k{best_axis_k}", 0, cur_loss, d_cur, e_cur, a_cur, "L2_best"])
                else:
                    cam_cur = best_oct_xyz.detach()
                    cur_loss = float(best_oct_loss)
                    d_cur, e_cur, a_cur = float(best_oct_d), float(best_oct_e), float(best_oct_a)
                    rows.append([it, "accept", f"oct{best_oct_idx}_k{best_oct_k}", 0, cur_loss, d_cur, e_cur, a_cur, "L1_better_than_L2"])

            path.append({"iter": it, "loss": cur_loss, "dist": d_cur, "elev_deg": e_cur, "azim_deg": a_cur})
            if cur_loss < float(best["loss"]):
                best.update({"loss": cur_loss, "dist": d_cur, "elev": e_cur, "azim": a_cur, "iter": it})
            if verbose >= 2:
                print(f"[LocalOctHier] iter={it} accept: loss={cur_loss:.6f} (dist,elev,azim)=({d_cur:.3f},{e_cur:.3f},{a_cur:.3f})")
        else:
            # 象限探索无改进，停止
            if verbose >= 1:
                print(f"[LocalOctHier] iter={it} stop: no improvement from octant explore (best_oct_loss={best_oct_loss:.6f} vs cur={cur_loss:.6f})")
            rows.append([it, "stop", "no_improve", 0, cur_loss, d_cur, e_cur, a_cur, "octant_no_improve"])
            break

    # Write logs
    try:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            for row in rows:
                w.writerow(row)
    except Exception as _e:
        print(f"[Warning] Failed to write local_octant_hier.csv: {_e}")

    try:
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "tag": tag,
                    "config": {
                        "radius": float(radius),
                        "iters": int(iters),
                        "octant_steps": int(octant_steps),
                        "axis_steps": int(axis_steps),
                        "step_frac": float(step_frac),
                        "min_delta": float(min_delta),
                        "use_sil_only": bool(use_sil_only),
                    },
                    "best": {
                        "loss_total": float(best["loss"]),
                        "dist": float(best["dist"]),
                        "elev_deg": float(best["elev"]),
                        "azim_deg": float(best["azim"]),
                        "iter": int(best["iter"]),
                    },
                    "path": path,
                },
                f,
                indent=2,
                ensure_ascii=True,
            )
    except Exception as _e:
        print(f"[Warning] Failed to write local_octant_hier_best.json: {_e}")

    if verbose >= 1:
        print(
            f"[LocalOctHier] {tag}: best_loss={float(best['loss']):.6f} "
            f"best(dist,elev,azim)=({float(best['dist']):.3f},{float(best['elev']):.3f},{float(best['azim']):.3f}) "
            f"at iter={best['iter']}"
        )

    # Save artifacts
    try:
        if int(os.environ.get("VAL_LOCAL_OCT_SAVE_BEST", "1")) != 0:
            _render_and_save_best_artifacts(
                mesh=mesh,
                silhouette_renderer=silhouette_renderer,
                rasterizer=rasterizer,
                phong_renderer=phong_renderer,
                at_xyz=at_xyz,
                dist_v=float(best["dist"]),
                elev_v=float(best["elev"]),
                azim_v=float(best["azim"]),
                device=device,
                save_dir=save_dir,
                prefix="best_local_oct_hier",
            )
    except Exception as _e:
        if verbose >= 2:
            print(f"[Warning] Failed to save best_local_oct_hier artifacts: {_e}")

    return float(best["dist"]), float(best["elev"]), float(best["azim"]), float(best["loss"])


def _evaluate_pose_losses_batch_no_grad(
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    at_xyz: torch.Tensor,
    dist_list: list[float],
    elev_list: list[float],
    azim_list: list[float],
    device: torch.device,
    use_sil_normals_only: bool = False,
):
    """
    Batched no_grad evaluation for MANY candidate poses (speed critical for greedy search).

    Notes:
    - Uses the same silhouette + depth + normal components as _evaluate_pose_loss_no_grad.
    - Normals here use FACE normals (faces_per_pixel=1) for speed (no barycentric interp).
      This is typically sufficient for "direction selection" in greedy search and much faster.
    - If use_sil_normals_only=True, only computes sil+normals (no depth), which is faster and often sufficient.

    Returns dict:
      - loss_total: (B,) tensor
      - loss_sil, loss_depth, loss_normal: (B,) tensors
      - valid_ratio: (B,) tensors
    """
    assert len(dist_list) == len(elev_list) == len(azim_list)
    B = int(len(dist_list))
    if B <= 0:
        return {
            "loss_total": torch.empty((0,), device=device),
            "loss_sil": torch.empty((0,), device=device),
            "loss_depth": torch.empty((0,), device=device),
            "loss_normal": torch.empty((0,), device=device),
            "valid_ratio": torch.empty((0,), device=device),
        }

    with torch.no_grad():
        dist_t = torch.tensor(dist_list, device=device, dtype=torch.float32)
        elev_t = torch.tensor(elev_list, device=device, dtype=torch.float32)
        azim_t = torch.tensor(azim_list, device=device, dtype=torch.float32)
        at_b = at_xyz.view(1, 3).to(device).expand(B, 3)
        R, T = look_at_view_transform(dist=dist_t, elev=elev_t, azim=azim_t, at=at_b, device=device)

        # replicate mesh for batch size
        mesh_b = mesh.extend(B)

        # silhouette
        img = silhouette_renderer(meshes_world=mesh_b, R=R, T=T)
        alpha = torch.clamp(img[..., 3], 0, 1)  # (B,H,W)
        ref = torch.clamp(image_ref_silhouette, 0, 1)
        if ref.dim() == 3 and ref.shape[-1] == 4:
            ref = ref[..., 3]
        ref = ref.squeeze().to(device)
        ref_b = ref.unsqueeze(0).expand(B, -1, -1)

        loss_mse = torch.mean((alpha - ref_b) ** 2, dim=(1, 2))
        inter = torch.sum(alpha * ref_b, dim=(1, 2))
        union = torch.sum(alpha, dim=(1, 2)) + torch.sum(ref_b, dim=(1, 2)) - inter
        loss_iou = 1.0 - (inter / (union + 1e-8))
        w_iou = float(os.environ.get("VAL_W_IOU", "0.1"))
        scale_cfg = os.environ.get("VAL_LOSS_SCALE", "1.0")
        if scale_cfg.strip().lower() == "auto":
            h = int(ref.shape[0])
            w = int(ref.shape[1])
            loss_scale = float(h * w)
        else:
            loss_scale = float(scale_cfg)
        loss_sil = (1.0 - w_iou) * (loss_mse * loss_scale) + w_iou * loss_iou

        # depth + normals (or normals only if use_sil_normals_only=True)
        cams = FoVPerspectiveCameras(device=device, R=R, T=T)
        fr = rasterizer(meshes_world=mesh_b, cameras=cams)
        face_idx = fr.pix_to_face[..., 0]  # (B,H,W) packed-face indices (global across the packed batch)
        vis_mask = (face_idx >= 0).float()
        if SIL_SCENE_DEPTH is not None:
            _ls = _occlusion_aware_sil(alpha, ref_b, vis_mask, fr.zbuf[..., 0], w_iou, loss_scale, (1, 2))
            if _ls is not None:
                loss_sil = _ls
        depth = (fr.zbuf[..., 0] * vis_mask) if not use_sil_normals_only else None  # (B,H,W)

        ref_d = ref_depth.squeeze().to(device) if not use_sil_normals_only else None  # (H,W)
        ref_n = ref_normals.squeeze().to(device)  # (H,W,3)
        ref_d_b = ref_d.unsqueeze(0).expand(B, -1, -1) if not use_sil_normals_only else None
        ref_n_b = ref_n.unsqueeze(0).expand(B, -1, -1, -1)

        if use_sil_normals_only:
            # Only use normals: valid = vis_mask > 0.5
            valid = (vis_mask > 0.5)
            valid_ratio = valid.float().mean(dim=(1, 2))
            valid_count = valid.float().sum(dim=(1, 2)).clamp(min=1.0)
            loss_depth = torch.zeros((B,), device=device, dtype=loss_sil.dtype)
        else:
            # Full: depth + normals
            valid = (vis_mask > 0.5) & (ref_d_b > 0)
            valid_ratio = valid.float().mean(dim=(1, 2))
            valid_count = valid.float().sum(dim=(1, 2)).clamp(min=1.0)

            beta = float(os.environ.get("VAL_DEPTH_BETA", "0.05"))
            depth_norm_eps = float(os.environ.get("VAL_DEPTH_NORM_EPS", "1e-6"))
            ref_scale = (ref_d_b * valid.float()).sum(dim=(1, 2)) / valid_count
            ref_scale = ref_scale.detach().clamp(min=depth_norm_eps).view(B, 1, 1)
            depth_n = depth / ref_scale
            ref_depth_n = ref_d_b / ref_scale
            # elementwise smooth_l1 then mean over valid pixels
            loss_depth_map = torch.nn.functional.smooth_l1_loss(depth_n, ref_depth_n, beta=beta, reduction="none")
            loss_depth = (loss_depth_map * valid.float()).sum(dim=(1, 2)) / valid_count

        # FAST normals using face normals (no barycentric interpolation)
        # IMPORTANT: pix_to_face indices are packed-face indices across the whole batch (mesh.extend(B)).
        # Convert to per-mesh local face indices by subtracting batch offsets (b * F).
        face_normals_world = mesh.faces_normals_packed()  # (F,3) for the ORIGINAL (single) mesh
        F = int(face_normals_world.shape[0])
        face_normals_view = torch.matmul(face_normals_world.unsqueeze(0).expand(B, -1, -1), R.transpose(1, 2))  # (B,F,3)
        # gather per-pixel face normal
        offsets = (torch.arange(B, device=device, dtype=face_idx.dtype).view(B, 1, 1) * F)
        face_local = face_idx - offsets
        face_local = torch.clamp(face_local, min=0, max=F - 1)
        flat_idx = face_local.view(B, -1)  # (B,HW) local indices in [0, F-1]
        fnv_flat = torch.gather(face_normals_view, 1, flat_idx.unsqueeze(-1).expand(-1, -1, 3))  # (B,HW,3)
        normals = fnv_flat.view(B, ref_n.shape[0], ref_n.shape[1], 3)
        normals = torch.nn.functional.normalize(normals, dim=-1) * vis_mask.unsqueeze(-1)
        cos = (normals * ref_n_b).sum(dim=-1).clamp(-1, 1)
        loss_normal = ((1.0 - cos) * valid.float()).sum(dim=(1, 2)) / valid_count

        w_sil = float(os.environ.get("VAL_W_SIL_START", "2.0"))
        w_depth = float(os.environ.get("VAL_W_DEPTH_START", "0.5")) if not use_sil_normals_only else 0.0
        w_normal = float(os.environ.get("VAL_W_NORMAL_START", "0.5"))
        # Valid-ratio penalty (batch version): same purpose as in single-pose evaluator.
        # Prevents wrong viewpoints with tiny overlap from getting artificially low losses.
        w_vp = float(os.environ.get("VAL_W_VALID_PENALTY", "0.0"))
        valid_penalty_b = w_vp * (1.0 - valid_ratio) if w_vp > 0 else torch.zeros((B,), device=device, dtype=loss_sil.dtype)
        total = w_sil * loss_sil + w_depth * loss_depth + w_normal * loss_normal + valid_penalty_b

        return {
            "loss_total": total.detach(),
            "loss_sil": loss_sil.detach(),
            "loss_depth": loss_depth.detach(),
            "loss_normal": loss_normal.detach(),
            "valid_ratio": valid_ratio.detach(),
        }


def global_sphere_presearch(
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    at_xyz: torch.Tensor,
    init_dist: float,
    init_elev: float,
    init_azim: float,
    device: torch.device,
    save_dir: str,
    tag: str,
):
    """
    Global sphere sampling around the model center (at_xyz) to get a better seed for greedy search.
    - Sample N spherical poses (dist,elev,azim) around at_xyz.
    - Evaluate loss (no_grad) and pick the best as the starting center for greedy_spherical_coarse_search.
    - Save CSV/JSON for offline inspection.
    """
    enabled = int(os.environ.get("VAL_GLOBAL_PRESEARCH", "1")) != 0
    if not enabled:
        return float(init_dist), float(init_elev), float(init_azim)

    verbose = int(os.environ.get("VAL_GLOBAL_VERBOSE", "1"))
    n = int(os.environ.get("VAL_GLOBAL_SAMPLES", "64"))

    # dist sampling mode: fixed (sphere) vs range (shell)
    dist_mode = os.environ.get("VAL_GLOBAL_DIST_MODE", "fixed").strip().lower()  # fixed|range
    dist_fixed = float(os.environ.get("VAL_GLOBAL_DIST_FIXED", str(init_dist)))
    dist_scale_min = float(os.environ.get("VAL_GLOBAL_DIST_SCALE_MIN", "0.60"))
    dist_scale_max = float(os.environ.get("VAL_GLOBAL_DIST_SCALE_MAX", "1.00"))

    elev_min = float(os.environ.get("VAL_GLOBAL_ELEV_MIN", os.environ.get("VAL_GT_ELEV_MIN", "5.0")))
    elev_max = float(os.environ.get("VAL_GLOBAL_ELEV_MAX", os.environ.get("VAL_GT_ELEV_MAX", "85.0")))
    azim_min = float(os.environ.get("VAL_GLOBAL_AZIM_MIN", os.environ.get("VAL_GT_AZIM_MIN", "-180.0")))
    azim_max = float(os.environ.get("VAL_GLOBAL_AZIM_MAX", os.environ.get("VAL_GT_AZIM_MAX", "180.0")))

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{tag}_global_samples.csv")
    json_path = os.path.join(save_dir, f"{tag}_global_best.json")

    at_xyz = at_xyz.view(3).to(device)
    best = None  # (loss_total, dist, elev, azim, details)

    if verbose >= 1:
        print(
            f"[GlobalSearch] {tag}: enabled=1, samples={n}, dist_mode={dist_mode}, "
            f"dist_fixed={dist_fixed:.2f}, elev=[{elev_min:.1f},{elev_max:.1f}], azim=[{azim_min:.1f},{azim_max:.1f}]"
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(
            [
                "idx",
                "dist",
                "elev_deg",
                "azim_deg",
                "loss_total",
                "loss_sil",
                "loss_depth",
                "loss_normal",
                "valid_ratio",
            ]
        )

        for i in range(max(1, n)):
            elev = random.uniform(elev_min, elev_max)
            azim = random.uniform(azim_min, azim_max)
            if dist_mode == "range":
                dist = float(init_dist) * random.uniform(dist_scale_min, dist_scale_max)
            else:
                dist = float(dist_fixed)

            ev = _evaluate_pose_loss_no_grad(
                mesh,
                silhouette_renderer,
                rasterizer,
                image_ref_silhouette,
                ref_depth,
                ref_normals,
                at_xyz,
                float(dist),
                float(elev),
                float(azim),
                device,
            )
            w.writerow(
                [
                    int(i),
                    float(dist),
                    float(elev),
                    float(azim),
                    float(ev["loss_total"]),
                    float(ev["loss_sil"]),
                    float(ev["loss_depth"]),
                    float(ev["loss_normal"]),
                    float(ev["valid_ratio"]),
                ]
            )
            if best is None or ev["loss_total"] < best[0]:
                best = (float(ev["loss_total"]), float(dist), float(elev), float(azim), ev)

    if best is None:
        return float(init_dist), float(init_elev), float(init_azim)

    best_loss, best_dist, best_elev, best_azim, best_ev = best
    try:
        with open(json_path, "w", encoding="utf-8") as fj:
            json.dump(
                {
                    "tag": tag,
                    "best": {
                        "dist": best_dist,
                        "elev_deg": best_elev,
                        "azim_deg": best_azim,
                        "loss_total": best_loss,
                        "loss_sil": best_ev["loss_sil"],
                        "loss_depth": best_ev["loss_depth"],
                        "loss_normal": best_ev["loss_normal"],
                        "valid_ratio": best_ev["valid_ratio"],
                    },
                    "init": {"dist": float(init_dist), "elev_deg": float(init_elev), "azim_deg": float(init_azim)},
                    "config": {
                        "samples": int(n),
                        "dist_mode": dist_mode,
                        "dist_fixed": float(dist_fixed),
                        "dist_scale_min": float(dist_scale_min),
                        "dist_scale_max": float(dist_scale_max),
                        "elev_min": float(elev_min),
                        "elev_max": float(elev_max),
                        "azim_min": float(azim_min),
                        "azim_max": float(azim_max),
                    },
                },
                fj,
                indent=2,
                ensure_ascii=True,
            )
    except Exception as _e:
        print(f"[Warning] Failed to write global_best.json: {_e}")

    if verbose >= 1:
        print(
            f"[GlobalSearch] {tag}: best_loss={best_loss:.6f}, best(dist,elev,azim)=({best_dist:.2f},{best_elev:.2f},{best_azim:.2f}) "
            f"saved: {os.path.basename(csv_path)}, {os.path.basename(json_path)}"
        )

    return float(best_dist), float(best_elev), float(best_azim)


def global_sphere_topk_presearch(
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    at_xyz: torch.Tensor,
    init_dist: float,
    init_elev: float,
    init_azim: float,
    device: torch.device,
    save_dir: str,
    tag: str,
):
    """
    GlobalSearch (Top-K):
    - Sample N poses on a sphere/shell around the model center (at_xyz)
    - Compute loss for each pose (no_grad)
    - Return Top-K best candidates (lowest loss)

    Why Top-K:
    - Single best seed can still fall into a bad local minimum after local greedy search.
    - Running local greedy from Top-K seeds is a practical multi-start strategy for non-convex problems.

    Outputs:
    - <tag>_global_samples.csv (all samples)
    - <tag>_global_topk.csv (top-k subset)
    - <tag>_global_best.json (best summary)
    """
    enabled = int(os.environ.get("VAL_GLOBAL_PRESEARCH", "1")) != 0
    if not enabled:
        return [(float(init_dist), float(init_elev), float(init_azim), float("inf"))]

    verbose = int(os.environ.get("VAL_GLOBAL_VERBOSE", "1"))
    n = int(os.environ.get("VAL_GLOBAL_SAMPLES", "256"))
    topk = int(os.environ.get("VAL_TOPK", "8"))
    topk = max(1, min(topk, max(1, n)))
    use_sil_only = int(os.environ.get("VAL_GLOBAL_USE_SIL_ONLY", "1")) != 0
    batch_size = int(os.environ.get("VAL_GLOBAL_BATCH", "64"))
    batch_size = max(1, batch_size)

    dist_mode = os.environ.get("VAL_GLOBAL_DIST_MODE", "fixed").strip().lower()  # fixed|range
    dist_fixed = float(os.environ.get("VAL_GLOBAL_DIST_FIXED", str(init_dist)))
    dist_scale_min = float(os.environ.get("VAL_GLOBAL_DIST_SCALE_MIN", "0.60"))
    dist_scale_max = float(os.environ.get("VAL_GLOBAL_DIST_SCALE_MAX", "1.00"))

    elev_min = float(os.environ.get("VAL_GLOBAL_ELEV_MIN", os.environ.get("VAL_GT_ELEV_MIN", "5.0")))
    elev_max = float(os.environ.get("VAL_GLOBAL_ELEV_MAX", os.environ.get("VAL_GT_ELEV_MAX", "85.0")))
    azim_min = float(os.environ.get("VAL_GLOBAL_AZIM_MIN", os.environ.get("VAL_GT_AZIM_MIN", "-180.0")))
    azim_max = float(os.environ.get("VAL_GLOBAL_AZIM_MAX", os.environ.get("VAL_GT_AZIM_MAX", "180.0")))

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{tag}_global_samples.csv")
    topk_path = os.path.join(save_dir, f"{tag}_global_topk.csv")
    json_path = os.path.join(save_dir, f"{tag}_global_best.json")

    at_xyz = at_xyz.view(3).to(device)
    samples = []  # list[(loss_total, dist, elev, azim, evdict-lite)]

    if verbose >= 1:
        print(
            f"[GlobalSearch] {tag}: enabled=1, samples={n}, topk={topk}, dist_mode={dist_mode}, "
            f"dist_fixed={dist_fixed:.2f}, elev=[{elev_min:.1f},{elev_max:.1f}], azim=[{azim_min:.1f},{azim_max:.1f}]"
        )

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["idx", "dist", "elev_deg", "azim_deg", "loss_total", "loss_sil", "loss_depth", "loss_normal", "valid_ratio"])
        # Generate candidates first (so we can batch-evaluate and still write full CSV)
        dist_all: list[float] = []
        elev_all: list[float] = []
        azim_all: list[float] = []
        for _i in range(max(1, n)):
            elev_all.append(float(random.uniform(elev_min, elev_max)))
            azim_all.append(float(random.uniform(azim_min, azim_max)))
            if dist_mode == "range":
                dist_all.append(float(init_dist) * float(random.uniform(dist_scale_min, dist_scale_max)))
            else:
                dist_all.append(float(dist_fixed))

        # Chunked batched evaluation to support larger N efficiently.
        for base_i in range(0, len(dist_all), batch_size):
            end_i = min(len(dist_all), base_i + batch_size)
            dlist = dist_all[base_i:end_i]
            elist = elev_all[base_i:end_i]
            alist = azim_all[base_i:end_i]
            evb = _batched_eval_candidates_no_grad(
                mesh=mesh,
                silhouette_renderer=silhouette_renderer,
                rasterizer=rasterizer,
                image_ref_silhouette=image_ref_silhouette,
                ref_depth=ref_depth,
                ref_normals=ref_normals,
                at_xyz=at_xyz,
                dist_list=dlist,
                elev_list=elist,
                azim_list=alist,
                device=device,
                use_sil_only=bool(use_sil_only),
            )
            lt = evb["loss_total"].detach().cpu().numpy().astype(np.float64)
            ls = evb["loss_sil"].detach().cpu().numpy().astype(np.float64)
            ld = evb["loss_depth"].detach().cpu().numpy().astype(np.float64)
            ln = evb["loss_normal"].detach().cpu().numpy().astype(np.float64)
            vr = evb["valid_ratio"].detach().cpu().numpy().astype(np.float64)
            for j in range(end_i - base_i):
                i = base_i + j
                ev_lite = {
                    "loss_total": float(lt[j]),
                    "loss_sil": float(ls[j]),
                    "loss_depth": float(ld[j]),
                    "loss_normal": float(ln[j]),
                    "valid_ratio": float(vr[j]),
                }
                samples.append((float(lt[j]), float(dlist[j]), float(elist[j]), float(alist[j]), ev_lite))
                w.writerow([int(i), float(dlist[j]), float(elist[j]), float(alist[j]), float(lt[j]), float(ls[j]), float(ld[j]), float(ln[j]), float(vr[j])])

    samples.sort(key=lambda x: x[0])
    topk_list = samples[:topk]
    best_loss, best_dist, best_elev, best_azim, best_ev = topk_list[0]

    # Direction hint: dominant axis of delta(camera_xyz) from init -> best (WORLD space)
    try:
        cam_init = _camera_position_from_spherical_global(
            torch.tensor(float(init_dist), device=device),
            torch.tensor(float(init_elev), device=device),
            torch.tensor(float(init_azim), device=device),
            at_xyz,
        )
        cam_best = _camera_position_from_spherical_global(
            torch.tensor(float(best_dist), device=device),
            torch.tensor(float(best_elev), device=device),
            torch.tensor(float(best_azim), device=device),
            at_xyz,
        )
        delta = (cam_best - cam_init).detach()
        dom_axis = _dominant_axis_from_delta_world(delta)
        delta_np = delta.detach().cpu().numpy().astype(np.float64)
    except Exception:
        dom_axis = None
        delta_np = None

    # write topk csv
    with open(topk_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(["rank", "dist", "elev_deg", "azim_deg", "loss_total", "loss_sil", "loss_depth", "loss_normal", "valid_ratio"])
        for r, (lt, d, e, a, ev) in enumerate(topk_list):
            w.writerow([int(r), float(d), float(e), float(a), float(lt), ev["loss_sil"], ev["loss_depth"], ev["loss_normal"], ev["valid_ratio"]])

    try:
        with open(json_path, "w", encoding="utf-8") as fj:
            json.dump(
                {
                    "tag": tag,
                    "best": {
                        "dist": float(best_dist),
                        "elev_deg": float(best_elev),
                        "azim_deg": float(best_azim),
                        "loss_total": float(best_loss),
                        "loss_sil": float(best_ev["loss_sil"]),
                        "loss_depth": float(best_ev["loss_depth"]),
                        "loss_normal": float(best_ev["loss_normal"]),
                        "valid_ratio": float(best_ev["valid_ratio"]),
                        "direction_hint": {
                            "dominant_axis": dom_axis,
                            "delta_cam_xyz_world": (delta_np.tolist() if delta_np is not None else None),
                        },
                    },
                    "init": {"dist": float(init_dist), "elev_deg": float(init_elev), "azim_deg": float(init_azim)},
                    "config": {
                        "samples": int(n),
                        "topk": int(topk),
                        "dist_mode": dist_mode,
                        "dist_fixed": float(dist_fixed),
                        "dist_scale_min": float(dist_scale_min),
                        "dist_scale_max": float(dist_scale_max),
                        "elev_min": float(elev_min),
                        "elev_max": float(elev_max),
                        "azim_min": float(azim_min),
                        "azim_max": float(azim_max),
                    },
                    "logs": {
                        "samples_csv": os.path.basename(csv_path),
                        "topk_csv": os.path.basename(topk_path),
                    },
                },
                fj,
                indent=2,
                ensure_ascii=True,
            )
    except Exception as _e:
        print(f"[Warning] Failed to write global_best.json: {_e}")

    if verbose >= 1:
        print(
            f"[GlobalSearch] {tag}: best_loss={best_loss:.6f}, best(dist,elev,azim)=({best_dist:.2f},{best_elev:.2f},{best_azim:.2f}) "
            f"saved: {os.path.basename(csv_path)}, {os.path.basename(topk_path)}, {os.path.basename(json_path)}"
        )

    # Return list of (dist,elev,azim,loss)
    return [(d, e, a, lt) for (lt, d, e, a, _ev) in topk_list]


def _make_geom_scales(n: int, start: float, growth: float) -> list[float]:
    n = max(1, int(n))
    start = float(start)
    growth = float(growth)
    out: list[float] = []
    x = start
    for _ in range(n):
        out.append(float(x))
        x *= growth
    return out


def dist_sweep_presearch(
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    at_xyz: torch.Tensor,
    init_dist: float,
    init_elev: float,
    init_azim: float,
    device: torch.device,
    save_dir: str,
    tag: str,
):
    """
    Stage-A (T / scale along view axis):
    Keep direction fixed (elev/azim fixed), sweep dist forward/backward with geometric step scales.
    This corresponds to translating the camera along the viewing axis (radial direction).
    """
    enabled = int(os.environ.get("VAL_DIST_SWEEP", "1")) != 0
    if not enabled:
        return float(init_dist), float(init_elev), float(init_azim)

    verbose = int(os.environ.get("VAL_DIST_SWEEP_VERBOSE", "1"))
    check_dir = int(os.environ.get("VAL_DIST_SWEEP_CHECK_DIR", "1")) != 0
    steps = int(os.environ.get("VAL_DIST_SWEEP_STEPS", "8"))
    start = float(os.environ.get("VAL_DIST_SWEEP_START", "0.1"))
    growth = float(os.environ.get("VAL_DIST_SWEEP_GROWTH", "2.0"))
    scales = _make_geom_scales(steps, start, growth)  # e.g., 0.1,0.2,0.4,... (8 steps)

    # safe clamp to avoid degenerate dist->0 which breaks rendering/gradients.
    # Defaults are relative to base dist; can be overridden.
    dist_min_abs = os.environ.get("VAL_DIST_SWEEP_DIST_MIN", os.environ.get("VAL_GT_DIST_MIN", ""))
    dist_max_abs = os.environ.get("VAL_DIST_SWEEP_DIST_MAX", os.environ.get("VAL_GT_DIST_MAX", ""))
    min_frac = float(os.environ.get("VAL_DIST_SWEEP_MIN_FRAC", "0.25"))
    max_frac = float(os.environ.get("VAL_DIST_SWEEP_MAX_FRAC", "4.0"))
    dist_min = float(dist_min_abs) if dist_min_abs != "" else (float(init_dist) * float(min_frac))
    dist_max = float(dist_max_abs) if dist_max_abs != "" else (float(init_dist) * float(max_frac))
    dist_min = max(1e-3, float(dist_min))
    dist_max = max(dist_min, float(dist_max))

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{tag}_dist_sweep.csv")
    json_path = os.path.join(save_dir, f"{tag}_dist_sweep_best.json")

    at_xyz = at_xyz.view(3).to(device)
    base = float(init_dist)
    best = None  # (loss_total, dist, sign, scale, ev)

    # Direction sanity: compute init direction v0_hat = normalize(cam0 - at).
    v0_hat = None
    if check_dir:
        cam0 = _camera_position_from_spherical_global(
            torch.tensor(float(base), device=device),
            torch.tensor(float(init_elev), device=device),
            torch.tensor(float(init_azim), device=device),
            at_xyz,
        )
        v0 = cam0 - at_xyz
        v0_hat = v0 / torch.norm(v0).clamp(min=1e-8)

    # include center
    ev0 = _evaluate_pose_loss_no_grad(
        mesh, silhouette_renderer, rasterizer, image_ref_silhouette, ref_depth, ref_normals,
        at_xyz, float(base), float(init_elev), float(init_azim), device
    )
    best = (float(ev0["loss_total"]), float(base), 0, 0.0, ev0)

    if verbose >= 1:
        print(f"[DistSweep] {tag}: base_dist={base:.2f}, elev/azim=({init_elev:.2f},{init_azim:.2f}), steps={steps}, start={start:.3f}, growth={growth:.2f}")
        if check_dir:
            print("[DistSweep] Direction check enabled: expect dir_dot≈1.0 for all candidates (pure radial motion).")

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(
            [
                "idx",
                "sign",
                "scale",
                "dist",
                "dir_dot_vs_init",
                "cam_x",
                "cam_y",
                "cam_z",
                "loss_total",
                "loss_sil",
                "loss_depth",
                "loss_normal",
                "valid_ratio",
            ]
        )
        # center row
        if check_dir and v0_hat is not None:
            cam0_np = cam0.detach().cpu().numpy().astype(np.float64)
            dir_dot0 = float((v0_hat * v0_hat).sum().detach().cpu().item())
        else:
            cam0_np = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
            dir_dot0 = float("nan")
        w.writerow([0, 0, 0.0, float(base), dir_dot0, float(cam0_np[0]), float(cam0_np[1]), float(cam0_np[2]), best[0], ev0["loss_sil"], ev0["loss_depth"], ev0["loss_normal"], ev0["valid_ratio"]])

        idx = 1
        # Forward/backward along viewing axis, always positive:
        # - forward (farther): dist = base * (1 + sc)
        # - backward (closer): dist = base / (1 + sc)
        for sign in [+1, -1]:
            for sc in scales:
                if sign > 0:
                    dist = base * (1.0 + float(sc))
                else:
                    dist = base / (1.0 + float(sc))
                dist = float(np.clip(float(dist), dist_min, dist_max))
                if check_dir and v0_hat is not None:
                    cam = _camera_position_from_spherical_global(
                        torch.tensor(float(dist), device=device),
                        torch.tensor(float(init_elev), device=device),
                        torch.tensor(float(init_azim), device=device),
                        at_xyz,
                    )
                    v = cam - at_xyz
                    v_hat = v / torch.norm(v).clamp(min=1e-8)
                    dir_dot = float((v_hat * v0_hat).sum().clamp(-1.0, 1.0).detach().cpu().item())
                    cam_np = cam.detach().cpu().numpy().astype(np.float64)
                else:
                    dir_dot = float("nan")
                    cam_np = np.array([np.nan, np.nan, np.nan], dtype=np.float64)
                ev = _evaluate_pose_loss_no_grad(
                    mesh, silhouette_renderer, rasterizer, image_ref_silhouette, ref_depth, ref_normals,
                    at_xyz, float(dist), float(init_elev), float(init_azim), device
                )
                w.writerow(
                    [
                        int(idx),
                        int(sign),
                        float(sc),
                        float(dist),
                        float(dir_dot),
                        float(cam_np[0]),
                        float(cam_np[1]),
                        float(cam_np[2]),
                        ev["loss_total"],
                        ev["loss_sil"],
                        ev["loss_depth"],
                        ev["loss_normal"],
                        ev["valid_ratio"],
                    ]
                )
                if verbose >= 2 and check_dir:
                    print(f"[DistSweep] cand idx={idx} sign={sign:+d} scale={sc:.3f} dist={dist:.2f} dir_dot={dir_dot:.6f}")
                if ev["loss_total"] < best[0]:
                    best = (float(ev["loss_total"]), float(dist), int(sign), float(sc), ev)
                idx += 1

    try:
        with open(json_path, "w", encoding="utf-8") as fj:
            json.dump(
                {
                    "tag": tag,
                    "init": {"dist": float(init_dist), "elev_deg": float(init_elev), "azim_deg": float(init_azim)},
                    "best": {
                        "dist": float(best[1]),
                        "elev_deg": float(init_elev),
                        "azim_deg": float(init_azim),
                        "loss_total": float(best[0]),
                        "sign": int(best[2]),
                        "scale": float(best[3]),
                    },
                    "config": {
                        "steps": int(steps),
                        "start": float(start),
                        "growth": float(growth),
                        "dist_min": float(dist_min),
                        "dist_max": float(dist_max),
                        "dist_min_abs": dist_min_abs,
                        "dist_max_abs": dist_max_abs,
                        "min_frac": float(min_frac),
                        "max_frac": float(max_frac),
                    },
                },
                fj,
                indent=2,
                ensure_ascii=True,
            )
    except Exception as _e:
        print(f"[Warning] Failed to write dist_sweep_best.json: {_e}")

    if verbose >= 1:
        print(f"[DistSweep] {tag}: best_loss={best[0]:.6f}, best_dist={best[1]:.2f} (sign={best[2]}, scale={best[3]}) saved: {os.path.basename(csv_path)}")

    return float(best[1]), float(init_elev), float(init_azim)


def rot_sweep_presearch(
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    at_xyz: torch.Tensor,
    init_dist: float,
    init_elev: float,
    init_azim: float,
    device: torch.device,
    save_dir: str,
    tag: str,
):
    """
    Stage-B (rotation direction search on sphere):
    Keep dist fixed, search elev and azim with discrete +/- steps (8 per direction) using geometric scales.
    This is a coordinate-descent style discrete search in (elev, azim) space.
    """
    enabled = int(os.environ.get("VAL_ROT_SWEEP", "1")) != 0
    if not enabled:
        return float(init_dist), float(init_elev), float(init_azim)

    verbose = int(os.environ.get("VAL_ROT_SWEEP_VERBOSE", "1"))
    steps = int(os.environ.get("VAL_ROT_SWEEP_STEPS", "8"))
    start = float(os.environ.get("VAL_ROT_SWEEP_START", "0.5"))  # degrees multiplier
    growth = float(os.environ.get("VAL_ROT_SWEEP_GROWTH", "2.0"))
    base_deg = float(os.environ.get("VAL_ROT_SWEEP_BASE_DEG", "2.0"))
    rounds = int(os.environ.get("VAL_ROT_SWEEP_ROUNDS", "1"))
    scales = _make_geom_scales(steps, start, growth)

    elev_min = float(os.environ.get("VAL_GT_ELEV_MIN", "5.0"))
    elev_max = float(os.environ.get("VAL_GT_ELEV_MAX", "85.0"))
    azim_min = float(os.environ.get("VAL_GT_AZIM_MIN", "-180.0"))
    azim_max = float(os.environ.get("VAL_GT_AZIM_MAX", "180.0"))

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{tag}_rot_sweep.csv")
    json_path = os.path.join(save_dir, f"{tag}_rot_sweep_best.json")

    at_xyz = at_xyz.view(3).to(device)
    cur_elev = float(init_elev)
    cur_azim = float(init_azim)

    def _wrap_azim(a: float) -> float:
        # keep within [-180, 180] for readability
        x = ((a + 180.0) % 360.0) - 180.0
        return float(x)

    cur_ev = _evaluate_pose_loss_no_grad(
        mesh, silhouette_renderer, rasterizer, image_ref_silhouette, ref_depth, ref_normals,
        at_xyz, float(init_dist), float(cur_elev), float(cur_azim), device
    )
    cur_loss = float(cur_ev["loss_total"])

    if verbose >= 1:
        print(f"[RotSweep] {tag}: dist_fixed={float(init_dist):.2f}, init_elev/azim=({cur_elev:.2f},{cur_azim:.2f}), steps={steps}, base_deg={base_deg:.2f}, rounds={rounds}")

    rows = []
    rows.append(["round", "axis", "idx", "sign", "scale", "delta_deg", "elev_deg", "azim_deg", "loss_total", "loss_sil", "loss_depth", "loss_normal", "valid_ratio"])

    for r in range(max(1, rounds)):
        # 1) sweep elev
        best_local = (cur_loss, cur_elev, cur_azim, "none", 0, 0.0)
        idx = 0
        for sign in [+1, -1]:
            for sc in scales:
                delta = float(sign) * float(sc) * float(base_deg)
                elev = float(np.clip(cur_elev + delta, elev_min, elev_max))
                azim = float(cur_azim)
                ev = _evaluate_pose_loss_no_grad(
                    mesh, silhouette_renderer, rasterizer, image_ref_silhouette, ref_depth, ref_normals,
                    at_xyz, float(init_dist), float(elev), float(azim), device
                )
                rows.append([int(r), "elev", int(idx), int(sign), float(sc), float(delta), float(elev), float(azim), ev["loss_total"], ev["loss_sil"], ev["loss_depth"], ev["loss_normal"], ev["valid_ratio"]])
                if ev["loss_total"] < best_local[0]:
                    best_local = (float(ev["loss_total"]), float(elev), float(azim), "elev", int(sign), float(sc))
                idx += 1
        if best_local[0] < cur_loss:
            cur_loss, cur_elev, cur_azim = best_local[0], best_local[1], best_local[2]
            if verbose >= 1:
                print(f"[RotSweep] {tag}: round={r} axis=elev improved -> loss={cur_loss:.6f}, elev={cur_elev:.2f}")

        # 2) sweep azim
        best_local = (cur_loss, cur_elev, cur_azim, "none", 0, 0.0)
        idx = 0
        for sign in [+1, -1]:
            for sc in scales:
                delta = float(sign) * float(sc) * float(base_deg)
                elev = float(cur_elev)
                azim = _wrap_azim(cur_azim + delta)
                # clamp for feasibility in case user restricts azim range
                if azim < azim_min:
                    azim = float(azim_min)
                if azim > azim_max:
                    azim = float(azim_max)
                ev = _evaluate_pose_loss_no_grad(
                    mesh, silhouette_renderer, rasterizer, image_ref_silhouette, ref_depth, ref_normals,
                    at_xyz, float(init_dist), float(elev), float(azim), device
                )
                rows.append([int(r), "azim", int(idx), int(sign), float(sc), float(delta), float(elev), float(azim), ev["loss_total"], ev["loss_sil"], ev["loss_depth"], ev["loss_normal"], ev["valid_ratio"]])
                if ev["loss_total"] < best_local[0]:
                    best_local = (float(ev["loss_total"]), float(elev), float(azim), "azim", int(sign), float(sc))
                idx += 1
        if best_local[0] < cur_loss:
            cur_loss, cur_elev, cur_azim = best_local[0], best_local[1], best_local[2]
            if verbose >= 1:
                print(f"[RotSweep] {tag}: round={r} axis=azim improved -> loss={cur_loss:.6f}, azim={cur_azim:.2f}")

    # write CSV
    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        for row in rows:
            w.writerow(row)

    try:
        with open(json_path, "w", encoding="utf-8") as fj:
            json.dump(
                {
                    "tag": tag,
                    "init": {"dist": float(init_dist), "elev_deg": float(init_elev), "azim_deg": float(init_azim)},
                    "best": {"dist": float(init_dist), "elev_deg": float(cur_elev), "azim_deg": float(cur_azim), "loss_total": float(cur_loss)},
                    "config": {"steps": steps, "start": start, "growth": growth, "base_deg": base_deg, "rounds": rounds},
                },
                fj,
                indent=2,
                ensure_ascii=True,
            )
    except Exception as _e:
        print(f"[Warning] Failed to write rot_sweep_best.json: {_e}")

    if verbose >= 1:
        print(f"[RotSweep] {tag}: best_loss={cur_loss:.6f}, best(elev,azim)=({cur_elev:.2f},{cur_azim:.2f}) saved: {os.path.basename(csv_path)}")

    return float(init_dist), float(cur_elev), float(cur_azim)


def greedy_spherical_coarse_search(
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    image_ref_silhouette: torch.Tensor,
    ref_depth: torch.Tensor,
    ref_normals: torch.Tensor,
    at_xyz: torch.Tensor,
    init_dist: float,
    init_elev: float,
    init_azim: float,
    device: torch.device,
    save_dir: str,
    tag: str,
):
    """
    Greedy coarse search on the sphere around the initial pose.
    - At each level, sample 6 candidates along +/-x,+/-y,+/-z in WORLD axes, then project back to the SAME sphere.
    - Radius (perturb strength) can auto-grow if no improvement.
    - Returns best (dist,elev,azim) and writes CSV/JSON logs.
    """
    enabled = int(os.environ.get("VAL_PRESEARCH", "1")) != 0
    if not enabled:
        return init_dist, init_elev, init_azim

    verbose = int(os.environ.get("VAL_PRESEARCH_VERBOSE", "1"))
    levels = int(os.environ.get("VAL_PRESEARCH_LEVELS", "8"))
    if levels <= 0:
        return init_dist, init_elev, init_azim

    tries = int(os.environ.get("VAL_PRESEARCH_TRIES_PER_LEVEL", "3"))
    gamma = float(os.environ.get("VAL_PRESEARCH_RADIUS_GROWTH", "2.0"))
    min_delta = float(os.environ.get("VAL_PRESEARCH_MIN_DELTA", "1e-4"))
    r_frac = float(os.environ.get("VAL_PRESEARCH_RADIUS_FRAC", "0.05"))  # relative to dist
    save_images = int(os.environ.get("VAL_PRESEARCH_SAVE_IMAGES", "0")) != 0
    use_sil_only = int(os.environ.get("VAL_PRESEARCH_USE_SIL_ONLY", "0")) != 0
    vis_dir = os.path.join(save_dir, f"{tag}_presearch_vis")
    if save_images:
        os.makedirs(vis_dir, exist_ok=True)

    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, f"{tag}_presearch_candidates.csv")
    json_path = os.path.join(save_dir, f"{tag}_presearch_path.json")

    # initial camera xyz
    at_xyz = at_xyz.view(3).to(device)
    dist0 = float(init_dist)
    cam_xyz = _camera_position_from_spherical_global(
        torch.tensor(dist0, device=device),
        torch.tensor(float(init_elev), device=device),
        torch.tensor(float(init_azim), device=device),
        at_xyz,
    )
    v = cam_xyz - at_xyz
    d = torch.norm(v).clamp(min=1e-6)

    axes = [
        torch.tensor([1.0, 0.0, 0.0], device=device),
        torch.tensor([-1.0, 0.0, 0.0], device=device),
        torch.tensor([0.0, 1.0, 0.0], device=device),
        torch.tensor([0.0, -1.0, 0.0], device=device),
        torch.tensor([0.0, 0.0, 1.0], device=device),
        torch.tensor([0.0, 0.0, -1.0], device=device),
    ]
    axis_names = ["+x", "-x", "+y", "-y", "+z", "-z"]

    # evaluate initial
    cur_sph = _spherical_from_camera_position_global(at_xyz + (v / d) * d, at_xyz)
    cur_dist, cur_elev, cur_azim = cur_sph
    if use_sil_only:
        cur_eval = _evaluate_pose_sil_no_grad(
            mesh=mesh,
            silhouette_renderer=silhouette_renderer,
            image_ref_silhouette=image_ref_silhouette,
            at_xyz=at_xyz,
            dist_v=cur_dist,
            elev_v=cur_elev,
            azim_v=cur_azim,
            device=device,
        )
    else:
        cur_eval = _evaluate_pose_loss_no_grad(
            mesh, silhouette_renderer, rasterizer, image_ref_silhouette, ref_depth, ref_normals,
            at_xyz, cur_dist, cur_elev, cur_azim, device
        )
    cur_loss = cur_eval["loss_total"]
    if verbose >= 1:
        print(
            f"[PreSearch] {tag}: start center(dist,elev,azim)=({cur_dist:.2f},{cur_elev:.2f},{cur_azim:.2f}) "
            f"loss_total={cur_loss:.6f} (levels={levels}, tries_per_level={tries}, r_frac={r_frac:.4f}, growth={gamma:.2f})"
        )
        if use_sil_only:
            print("[PreSearch] Using silhouette-only loss for direction selection (VAL_PRESEARCH_USE_SIL_ONLY=1).")

    path = {
        "tag": tag,
        "init": {"dist": float(init_dist), "elev_deg": float(init_elev), "azim_deg": float(init_azim)},
        "levels": [],
        "final": None,
    }

    with open(csv_path, "w", newline="", encoding="utf-8") as fcsv:
        w = csv.writer(fcsv)
        w.writerow(
            [
                "level",
                "try",
                "candidate",
                "axis",
                "radius",
                "dist",
                "elev_deg",
                "azim_deg",
                "loss_total",
                "loss_sil",
                "loss_depth",
                "loss_normal",
                "valid_ratio",
                "improved_over_center",
            ]
        )

        # Keep a single radius across levels; auto-grow only when no improvement is found.
        # This matches the user's "radius stays consistent for the next sampling" idea.
        r_current = float(r_frac) * float(cur_dist)

        for lvl in range(levels):
            lvl_rec = {"level": int(lvl), "tries": [], "chosen": None, "center": {"dist": cur_dist, "elev_deg": cur_elev, "azim_deg": cur_azim, "loss_total": cur_loss}}

            improved = False
            r = float(r_current)
            best_local = (cur_loss, cur_dist, cur_elev, cur_azim, None)
            if verbose >= 1:
                print(
                    f"[PreSearch] {tag}: level={lvl} center(dist,elev,azim)=({cur_dist:.2f},{cur_elev:.2f},{cur_azim:.2f}) "
                    f"loss={cur_loss:.6f} radius={r:.4f}"
                )

            for t in range(tries):
                try_rec = {"try": int(t), "radius": float(r), "candidates": []}
                if verbose >= 1:
                    print(f"[PreSearch] {tag}: level={lvl} try={t} radius={r:.4f} (sampling 6 directions)")

                # re-compute center vector v on sphere
                cam_center = _camera_position_from_spherical_global(
                    torch.tensor(float(cur_dist), device=device),
                    torch.tensor(float(cur_elev), device=device),
                    torch.tensor(float(cur_azim), device=device),
                    at_xyz,
                )
                v_center = cam_center - at_xyz
                d_center = torch.norm(v_center).clamp(min=1e-6)
                v_center = (v_center / d_center) * d_center  # normalize

                for i, u in enumerate(axes):
                    v_new = v_center + float(r) * u
                    v_new = v_new / torch.norm(v_new).clamp(min=1e-6) * d_center
                    cam_new = at_xyz + v_new
                    dist_new, elev_new, azim_new = _spherical_from_camera_position_global(cam_new, at_xyz)
                    try_rec["candidates"].append(
                        {
                            "axis": axis_names[i],
                            "dist": dist_new,
                            "elev_deg": elev_new,
                            "azim_deg": azim_new,
                        }
                    )

                # Batch-evaluate the 6 candidates (fast path). If save_images is enabled, we will still
                # evaluate per-candidate later to get (R,T) for rendering.
                dlist = [float(c["dist"]) for c in try_rec["candidates"]]
                elist = [float(c["elev_deg"]) for c in try_rec["candidates"]]
                alist = [float(c["azim_deg"]) for c in try_rec["candidates"]]
                evb = _batched_eval_candidates_no_grad(
                    mesh=mesh,
                    silhouette_renderer=silhouette_renderer,
                    rasterizer=rasterizer,
                    image_ref_silhouette=image_ref_silhouette,
                    ref_depth=ref_depth,
                    ref_normals=ref_normals,
                    at_xyz=at_xyz,
                    dist_list=dlist,
                    elev_list=elist,
                    azim_list=alist,
                    device=device,
                    use_sil_only=bool(use_sil_only),
                )
                lt = evb["loss_total"].detach().cpu().numpy().astype(np.float64)
                ls = evb["loss_sil"].detach().cpu().numpy().astype(np.float64)
                ld = evb["loss_depth"].detach().cpu().numpy().astype(np.float64)
                ln = evb["loss_normal"].detach().cpu().numpy().astype(np.float64)
                vr = evb["valid_ratio"].detach().cpu().numpy().astype(np.float64)

                # Fill metrics back + CSV + best_local update.
                for i in range(len(try_rec["candidates"])):
                    c = try_rec["candidates"][i]
                    loss_total = float(lt[i])
                    c["loss_total"] = loss_total
                    c["loss_sil"] = float(ls[i])
                    c["loss_depth"] = float(ld[i])
                    c["loss_normal"] = float(ln[i])
                    c["valid_ratio"] = float(vr[i])
                    if verbose >= 2:
                        print(
                            f"  [PreSearch] cand axis={c['axis']:>2} "
                            f"(dist,elev,azim)=({c['dist']:.2f},{c['elev_deg']:.2f},{c['azim_deg']:.2f}) "
                            f"loss_total={loss_total:.6f} sil={c['loss_sil']:.6f} depth={c['loss_depth']:.6f} normal={c['loss_normal']:.6f} "
                            f"valid={c['valid_ratio']:.3f}"
                        )
                    w.writerow(
                        [
                            int(lvl),
                            int(t),
                            int(i),
                            c["axis"],
                            float(r),
                            float(c["dist"]),
                            float(c["elev_deg"]),
                            float(c["azim_deg"]),
                            float(c["loss_total"]),
                            float(c["loss_sil"]),
                            float(c["loss_depth"]),
                            float(c["loss_normal"]),
                            float(c["valid_ratio"]),
                            1 if (loss_total < cur_loss - min_delta) else 0,
                        ]
                    )
                    if loss_total < best_local[0]:
                        best_local = (loss_total, float(c["dist"]), float(c["elev_deg"]), float(c["azim_deg"]), c["axis"])

                # Optional visualization (slow): re-evaluate chosen/best candidates to obtain R,T.
                if save_images:
                    try:
                        for i in range(len(try_rec["candidates"])):
                            c = try_rec["candidates"][i]
                            ev_img = _evaluate_pose_loss_no_grad(
                                mesh,
                                silhouette_renderer,
                                rasterizer,
                                image_ref_silhouette,
                                ref_depth,
                                ref_normals,
                                at_xyz,
                                float(c["dist"]),
                                float(c["elev_deg"]),
                                float(c["azim_deg"]),
                                device,
                            )
                            rgba = silhouette_renderer(meshes_world=mesh, R=ev_img["R"].unsqueeze(0), T=ev_img["T"].unsqueeze(0))[0]
                            alpha = torch.clamp(rgba[..., 3], 0, 1)
                            ref = torch.clamp(image_ref_silhouette, 0, 1)
                            if ref.dim() == 3 and ref.shape[-1] == 4:
                                ref = ref[..., 3]
                            ref = ref.squeeze()
                            absdiff = torch.abs(alpha - ref)
                            _save_silhouette_pair(
                                vis_dir,
                                f"lvl{lvl:02d}_try{t:02d}_cand{i:02d}_{c['axis'].replace('+','p').replace('-','m')}",
                                alpha,
                            )
                            cv2.imwrite(
                                os.path.join(vis_dir, f"lvl{lvl:02d}_try{t:02d}_cand{i:02d}_{c['axis'].replace('+','p').replace('-','m')}_absdiff.png"),
                                (torch.clamp(absdiff, 0, 1).detach().cpu().numpy() * 255).astype(np.uint8),
                            )
                    except Exception as _e:
                        print(f"[Warning] PreSearch image save failed: {_e}")

                lvl_rec["tries"].append(try_rec)

                # check improvement
                if best_local[0] < cur_loss - min_delta:
                    improved = True
                    if verbose >= 1:
                        print(
                            f"[PreSearch] {tag}: level={lvl} improved at try={t}: "
                            f"best_axis={best_local[4]} best_loss={best_local[0]:.6f} "
                            f"center_loss={cur_loss:.6f} delta={(cur_loss - best_local[0]):.6f}"
                        )
                    break
                if verbose >= 1:
                    print(
                        f"[PreSearch] {tag}: level={lvl} try={t} no improvement (best={best_local[0]:.6f} vs center={cur_loss:.6f}), "
                        f"increasing radius: {r:.4f} -> {r * float(gamma):.4f}"
                    )
                r *= float(gamma)
                r_current = float(r)

            if improved:
                new_loss, new_dist, new_elev, new_azim, best_axis = best_local
                lvl_rec["chosen"] = {"axis": best_axis, "dist": new_dist, "elev_deg": new_elev, "azim_deg": new_azim, "loss_total": new_loss}
                cur_loss = float(new_loss)
                cur_dist, cur_elev, cur_azim = float(new_dist), float(new_elev), float(new_azim)
                # Keep the radius that succeeded for the next level (consistent radius across levels).
                # Do not reset it back to r_frac*dist.
                r_current = float(r_current)
                if verbose >= 1:
                    print(
                        f"[PreSearch] {tag}: level={lvl} choose axis={best_axis}, new_center(dist,elev,azim)=({cur_dist:.2f},{cur_elev:.2f},{cur_azim:.2f}) "
                        f"new_loss={cur_loss:.6f}"
                    )
            else:
                lvl_rec["chosen"] = {"axis": None, "dist": cur_dist, "elev_deg": cur_elev, "azim_deg": cur_azim, "loss_total": cur_loss, "note": "no improvement, stop early"}
                path["levels"].append(lvl_rec)
                if verbose >= 1:
                    print(f"[PreSearch] {tag}: level={lvl} stop early (no improvement).")
                break

            path["levels"].append(lvl_rec)

    path["final"] = {"dist": float(cur_dist), "elev_deg": float(cur_elev), "azim_deg": float(cur_azim), "loss_total": float(cur_loss)}
    try:
        with open(json_path, "w", encoding="utf-8") as fj:
            json.dump(path, fj, indent=2, ensure_ascii=True)
    except Exception as _e:
        print(f"[Warning] Failed to write presearch_path.json: {_e}")

    print(
        f"[PreSearch] {tag}: init_loss={cur_eval['loss_total']:.6f} -> presearch_loss={cur_loss:.6f}, "
        f"init(dist,elev,azim)=({init_dist:.2f},{init_elev:.2f},{init_azim:.2f}) -> ({cur_dist:.2f},{cur_elev:.2f},{cur_azim:.2f})"
    )
    return float(cur_dist), float(cur_elev), float(cur_azim)

# NOTE:
# PyTorch3D-specific imports were moved into the backend-selection block near the top of this file.
# This keeps the script runnable on GPUs where old PyTorch3D wheels can hard-crash (e.g., sm_120)
# (PyTorch3D code path removed; fixed nvdiffrast backend.)


def _pose_rt_to_4x4(R: torch.Tensor, T: torch.Tensor) -> np.ndarray:
    """
    Convert world-to-view (R,T) into a 4x4 pose matrix:
        [R, T]
        [0, 1]
    """
    Rn = R.detach().cpu().numpy()
    Tn = T.detach().cpu().numpy()
    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = Rn
    pose[:3, 3] = Tn
    return pose


def _pose_4x4_to_R_T(pose_4x4: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    pose_4x4: world-to-view 4x4
    Returns (R, T) where:
      - R: (3,3) world-to-view rotation
      - T: (3,)  world-to-view translation
    """
    p = np.asarray(pose_4x4, dtype=np.float64)
    R = p[:3, :3].copy()
    T = p[:3, 3].copy()
    return R, T


def _camera_center_world_from_world2view(R_w2v: np.ndarray, T_w2v: np.ndarray) -> np.ndarray:
    """
    Given world-to-view extrinsics (R,T), camera center in world is:
        C = -R^T * T
    """
    R = np.asarray(R_w2v, dtype=np.float64)
    T = np.asarray(T_w2v, dtype=np.float64).reshape(3)
    return (-R.T @ T).reshape(3)


def _rotation_error_deg(R_gt_w2v: np.ndarray, R_est_w2v: np.ndarray) -> float:
    """
    Rotation error between two world-to-view rotations:
        R_rel = R_est * R_gt^T
        angle = acos((trace(R_rel)-1)/2)
    """
    Rg = np.asarray(R_gt_w2v, dtype=np.float64)
    Re = np.asarray(R_est_w2v, dtype=np.float64)
    R_rel = Re @ Rg.T
    tr = float(np.trace(R_rel))
    c = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
    return float(np.degrees(np.arccos(c)))


def _camera_center_from_RT(R: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
    """
    Compute camera center (position) in world space from R and T.
    Camera center = -R^T @ T
    
    Args:
        R: (3,3) rotation matrix (world-to-view)
        T: (3,) translation vector (world-to-view)
        
    Returns:
        cam_center: (3,) camera position in world space
    """
    R_T = R.transpose(-2, -1) if R.dim() == 3 else R.t()
    if T.dim() == 1:
        cam_center = -torch.matmul(R_T, T)
    else:
        cam_center = -torch.bmm(R_T.unsqueeze(0), T.unsqueeze(-1)).squeeze(-1)
    return cam_center


def _compute_distance_loss(
    R: torch.Tensor,
    T: torch.Tensor,
    R_gt: torch.Tensor,
    T_gt: torch.Tensor,
    mesh_center: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute distance loss: |dist(cam, center) - dist(cam_gt, center)|
    
    Args:
        R: (B, 3, 3) or (3, 3) rotation matrix (current pose)
        T: (B, 3) or (3,) translation vector (current pose)
        R_gt: (3, 3) rotation matrix (GT pose)
        T_gt: (3,) translation vector (GT pose)
        mesh_center: (3,) mesh center in world space
        device: Device
        
    Returns:
        loss_dist: (B,) or scalar distance loss
    """
    # Compute camera centers
    if R.dim() == 2:
        R = R.unsqueeze(0)
        T = T.unsqueeze(0)
    B = R.shape[0]
    
    R_T = R.transpose(-2, -1)  # (B, 3, 3)
    cam_centers = -torch.bmm(R_T, T.unsqueeze(-1)).squeeze(-1)  # (B, 3)
    
    # GT camera center
    R_gt_T = R_gt.t()
    cam_center_gt = -torch.matmul(R_gt_T, T_gt)  # (3,)
    
    # Compute distances to mesh center
    mesh_center_exp = mesh_center.view(1, 3).expand(B, -1)  # (B, 3)
    dists = torch.norm(cam_centers - mesh_center_exp, dim=-1)  # (B,)
    dist_gt = torch.norm(cam_center_gt - mesh_center, dim=-1)  # scalar
    
    # Distance difference loss
    loss_dist = torch.abs(dists - dist_gt)  # (B,)
    
    return loss_dist


def _extract_pointcloud_from_depth(
    depth_map: torch.Tensor,
    vis_mask: torch.Tensor,
    device: torch.device,
    fov_y_deg: float = 60.0,
    max_points: int = 8000,
) -> torch.Tensor:
    """
    从深度图中反投影得到3D点云（在视图空间中）。
    使用与_nv_project相同的投影模型进行反投影。
    
    Args:
        depth_map: (H, W) 深度图（视图空间中的z值，正值表示距离相机的距离）
        vis_mask: (H, W) 可见性掩码
        device: 设备
        fov_y_deg: 垂直视场角（度）
        max_points: 最大采样点数
        
    Returns:
        points_view: (N, 3) 点云（视图空间，N <= max_points）
    """
    H, W = depth_map.shape[0], depth_map.shape[1]
    aspect = float(W) / float(max(1, H))
    
    # 创建像素坐标网格
    y_coords, x_coords = torch.meshgrid(
        torch.arange(H, device=device, dtype=torch.float32),
        torch.arange(W, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # 过滤有效点（深度>0且可见）
    valid = (vis_mask > 0.5) & (depth_map > 0)
    if not valid.any():
        return torch.zeros((1, 3), device=device)
    
    # 获取有效像素坐标和深度
    x_valid = x_coords[valid]
    y_valid = y_coords[valid]
    z_valid = depth_map[valid]  # z就是深度值（视图空间中的距离）
    
    # 转换到NDC空间 [-1, 1]
    x_ndc = 2.0 * (x_valid + 0.5) / float(W) - 1.0
    y_ndc = 2.0 * (y_valid + 0.5) / float(H) - 1.0
    
    # 从NDC和深度反投影到视图空间
    # 使用与_nv_project相同的投影模型：x_clip = f/aspect * x_view / z_view
    # 因此 x_view = x_ndc * z_view * aspect / f
    fov_y_rad = math.radians(fov_y_deg)
    f = 1.0 / max(1e-8, math.tan(0.5 * fov_y_rad))  # 与_nv_project一致
    
    # 视图空间中的x和y坐标
    x_view = x_ndc * z_valid * aspect / f
    y_view = y_ndc * z_valid / f  # y轴方向与_nv_project一致
    z_view = z_valid
    
    # 组合成3D点
    points_view = torch.stack([x_view, y_view, z_view], dim=-1)  # (N, 3)
    
    # 如果点太多，进行下采样
    if points_view.shape[0] > max_points:
        idx = torch.linspace(0, points_view.shape[0] - 1, steps=max_points, device=device).long()
        points_view = points_view.index_select(0, idx)
    
    return points_view


def _compute_pointcloud_3d_loss(
    points_current_view: torch.Tensor,
    points_gt_view: torch.Tensor,
    device: torch.device,
    method: str = "chamfer",
) -> torch.Tensor:
    """
    计算两个3D点云之间的损失（归一化）。
    
    Args:
        points_current_view: (N1, 3) 当前位姿的点云（视图空间）
        points_gt_view: (N2, 3) GT位姿的点云（视图空间）
        device: 设备
        method: 计算方法，可选 "chamfer" 或 "l2_mean"
        
    Returns:
        loss: 归一化后的损失值（标量）
    """
    if points_current_view.shape[0] == 0 or points_gt_view.shape[0] == 0:
        return torch.tensor(0.0, device=device)
    
    N1 = points_current_view.shape[0]
    N2 = points_gt_view.shape[0]
    
    if method == "chamfer":
        # Chamfer distance: 对于每个点，找最近邻，然后平均
        # L1 -> L2: ||p - q||_2
        # Chamfer = mean(min||p_i - q_j||^2) + mean(min||q_i - p_j||^2)
        
        # 计算所有点对之间的距离 (N1, N2)
        dists = torch.cdist(points_current_view, points_gt_view, p=2)  # (N1, N2)
        
        # 对于current中的每个点，找gt中最近的点
        min_dists_curr_to_gt, _ = torch.min(dists, dim=1)  # (N1,)
        loss_curr_to_gt = torch.mean(min_dists_curr_to_gt ** 2)
        
        # 对于gt中的每个点，找current中最近的点
        min_dists_gt_to_curr, _ = torch.min(dists, dim=0)  # (N2,)
        loss_gt_to_curr = torch.mean(min_dists_gt_to_curr ** 2)
        
        # Chamfer distance
        loss_chamfer = loss_curr_to_gt + loss_gt_to_curr
        
        # 归一化：除以点云的平均尺度
        with torch.no_grad():
            # 计算点云的bounding box尺度
            points_all = torch.cat([points_current_view, points_gt_view], dim=0)
            bbox_min = torch.min(points_all, dim=0)[0]
            bbox_max = torch.max(points_all, dim=0)[0]
            bbox_size = torch.norm(bbox_max - bbox_min)
            normalize_factor = torch.clamp(bbox_size, min=1e-6)
        
        loss_normalized = loss_chamfer / (normalize_factor ** 2)
        
    elif method == "l2_mean":
        # 简单方法：如果点数量相同，直接计算L2距离
        if N1 == N2:
            loss_raw = torch.mean((points_current_view - points_gt_view) ** 2)
        else:
            # 如果不相同，使用最近邻匹配
            dists = torch.cdist(points_current_view, points_gt_view, p=2)
            min_dists, _ = torch.min(dists, dim=1)
            loss_raw = torch.mean(min_dists ** 2)
        
        # 归一化
        with torch.no_grad():
            points_all = torch.cat([points_current_view, points_gt_view], dim=0)
            bbox_size = torch.norm(torch.max(points_all, dim=0)[0] - torch.min(points_all, dim=0)[0])
            normalize_factor = torch.clamp(bbox_size, min=1e-6)
        
        loss_normalized = loss_raw / (normalize_factor ** 2)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return loss_normalized


def _compute_pointcloud_distance_loss(
    R: torch.Tensor,
    T: torch.Tensor,
    R_gt: torch.Tensor,
    T_gt: torch.Tensor,
    mesh_verts: torch.Tensor,
    device: torch.device,
    use_cosine: bool = True,
) -> torch.Tensor:
    """
    Compute pointcloud distance loss: similarity between sorted distance vectors.
    
    Args:
        R: (B, 3, 3) or (3, 3) rotation matrix (current pose)
        T: (B, 3) or (3,) translation vector (current pose)
        R_gt: (3, 3) rotation matrix (GT pose)
        T_gt: (3,) translation vector (GT pose)
        mesh_verts: (V, 3) mesh vertices in world space
        device: Device
        use_cosine: If True, use cosine distance; else use L2 distance
        
    Returns:
        loss_pcd: (B,) or scalar pointcloud distance loss
    """
    # Ensure batch dimension
    if R.dim() == 2:
        R = R.unsqueeze(0)
        T = T.unsqueeze(0)
    B = R.shape[0]
    V = mesh_verts.shape[0]
    
    # Compute camera centers
    R_T = R.transpose(-2, -1)  # (B, 3, 3)
    cam_centers = -torch.bmm(R_T, T.unsqueeze(-1)).squeeze(-1)  # (B, 3)
    
    # GT camera center
    R_gt_T = R_gt.t()
    cam_center_gt = -torch.matmul(R_gt_T, T_gt)  # (3,)
    
    # Expand mesh vertices for batch
    mesh_verts_exp = mesh_verts.view(1, V, 3).expand(B, -1, -1)  # (B, V, 3)
    cam_centers_exp = cam_centers.view(B, 1, 3)  # (B, 1, 3)
    
    # Compute distances from camera to each vertex
    dists = torch.norm(mesh_verts_exp - cam_centers_exp, dim=-1)  # (B, V)
    
    # Sort distances (ascending order)
    dists_sorted, _ = torch.sort(dists, dim=-1)  # (B, V)
    
    # Compute GT distances
    cam_center_gt_exp = cam_center_gt.view(1, 3)  # (1, 3)
    mesh_verts_gt = mesh_verts.view(V, 3)  # (V, 3)
    dists_gt = torch.norm(mesh_verts_gt - cam_center_gt_exp, dim=-1)  # (V,)
    dists_gt_sorted, _ = torch.sort(dists_gt, dim=-1)  # (V,)
    
    # Expand GT for batch comparison
    dists_gt_sorted_exp = dists_gt_sorted.view(1, V).expand(B, -1)  # (B, V)
    
    if use_cosine:
        # Cosine distance: 1 - cosine_similarity
        # Normalize vectors
        dists_sorted_norm = torch.nn.functional.normalize(dists_sorted, p=2, dim=-1)  # (B, V)
        dists_gt_sorted_norm = torch.nn.functional.normalize(dists_gt_sorted_exp, p=2, dim=-1)  # (B, V)
        
        # Cosine similarity
        cosine_sim = torch.sum(dists_sorted_norm * dists_gt_sorted_norm, dim=-1)  # (B,)
        loss_pcd = 1.0 - cosine_sim  # (B,) [0, 2], 0 means identical
    else:
        # L2 distance (Euclidean distance between sorted vectors)
        loss_pcd = torch.norm(dists_sorted - dists_gt_sorted_exp, p=2, dim=-1)  # (B,)
        # Normalize by number of points to make it scale-invariant
        loss_pcd = loss_pcd / torch.sqrt(torch.tensor(float(V), device=device))
    
    return loss_pcd


def _format_loss_details(
    loss_total: float,
    loss_sil: float | None = None,
    loss_semantic: float | None = None,
    loss_distance: float | None = None,
    loss_pointcloud: float | None = None,
    w_sil: float = 1.0,
    w_semantic: float = 1.0,
    w_distance: float = 0.1,
    w_pointcloud: float = 0.1,
    prefix: str = "  ",
    verbose: bool = True,
) -> str:
    """
    Format loss details for console output.
    
    Args:
        loss_total: Total combined loss
        loss_sil: Silhouette loss (optional)
        loss_semantic: Semantic similarity loss (optional)
        loss_distance: Distance loss (optional)
        loss_pointcloud: Pointcloud distance loss (optional)
        w_sil, w_semantic, w_distance, w_pointcloud: Loss weights
        prefix: Prefix string for indentation
        verbose: If True, show detailed breakdown; else show summary only
        
    Returns:
        Formatted string for printing
    """
    lines = []
    if verbose:
        lines.append(f"{prefix}━━━ 损失分解 (Loss Breakdown) ━━━")
        if loss_sil is not None:
            lines.append(f"{prefix}  轮廓损失 (Silhouette):     {loss_sil:.6f}  × {w_sil:.2f}  = {loss_sil * w_sil:.6f}")
        if loss_semantic is not None:
            lines.append(f"{prefix}  语义损失 (Semantic):       {loss_semantic:.6f}  × {w_semantic:.2f}  = {loss_semantic * w_semantic:.6f}")
        if loss_distance is not None:
            lines.append(f"{prefix}  距离损失 (Distance):       {loss_distance:.6f}  × {w_distance:.2f}  = {loss_distance * w_distance:.6f}")
        if loss_pointcloud is not None:
            lines.append(f"{prefix}  点云损失 (Pointcloud):     {loss_pointcloud:.6f}  × {w_pointcloud:.2f}  = {loss_pointcloud * w_pointcloud:.6f}")
        lines.append(f"{prefix}  ───────────────────────────────────")
        lines.append(f"{prefix}  总损失 (Total):             {loss_total:.6f}")
        lines.append(f"{prefix}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    else:
        # Compact format
        parts = []
        if loss_sil is not None:
            parts.append(f"Sil={loss_sil:.4f}")
        if loss_semantic is not None:
            parts.append(f"Sem={loss_semantic:.4f}")
        if loss_distance is not None:
            parts.append(f"Dist={loss_distance:.4f}")
        if loss_pointcloud is not None:
            parts.append(f"PCD={loss_pointcloud:.4f}")
        parts.append(f"Total={loss_total:.4f}")
        lines.append(f"{prefix}{' | '.join(parts)}")
    
    return "\n".join(lines)


def _angle_deg_between(v1: np.ndarray, v2: np.ndarray) -> float:
    v1 = np.asarray(v1, dtype=np.float64).reshape(-1)
    v2 = np.asarray(v2, dtype=np.float64).reshape(-1)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-12 or n2 < 1e-12:
        return float("nan")
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def _wrap_deg_180(x: float) -> float:
    return float(((float(x) + 180.0) % 360.0) - 180.0)


def _pose_metrics_text(
    *,
    tag: str,
    pose_gt_w2v: np.ndarray,
    pose_est_w2v: np.ndarray,
    at_xyz: np.ndarray,
) -> tuple[dict, str]:
    """
    Compute pose metrics comparing estimated pose vs GT pose.
    Returns:
      - metrics dict (JSON-serializable)
      - human-readable text (for .txt file)
    """
    Rg, Tg = _pose_4x4_to_R_T(pose_gt_w2v)
    Re, Te = _pose_4x4_to_R_T(pose_est_w2v)
    Cg = _camera_center_world_from_world2view(Rg, Tg)
    Ce = _camera_center_world_from_world2view(Re, Te)

    at = np.asarray(at_xyz, dtype=np.float64).reshape(3)
    vg = at - Cg
    ve = at - Ce

    rot_err_deg = _rotation_error_deg(Rg, Re)
    cam_center_err = float(np.linalg.norm(Ce - Cg))
    view_dir_err_deg = _angle_deg_between(vg, ve)

    # Spherical params around at_xyz (dist,elev,azim)
    # Use existing helper (torch-based) for consistency with the rest of code.
    with torch.no_grad():
        at_t = torch.tensor(at, dtype=torch.float32)
        cg_t = torch.tensor(Cg, dtype=torch.float32)
        ce_t = torch.tensor(Ce, dtype=torch.float32)
        dg, eg, ag = _spherical_from_camera_position_global(cg_t, at_t)
        de, ee, ae = _spherical_from_camera_position_global(ce_t, at_t)

    dist_err = float(abs(de - dg))
    elev_err_deg = float(abs(ee - eg))
    azim_err_deg = float(abs(_wrap_deg_180(ae - ag)))

    metrics = {
        "tag": str(tag),
        "rotation_error_deg": float(rot_err_deg),
        "camera_center_error": float(cam_center_err),
        "view_direction_error_deg": float(view_dir_err_deg),
        "spherical": {
            "gt": {"dist": float(dg), "elev_deg": float(eg), "azim_deg": float(ag)},
            "est": {"dist": float(de), "elev_deg": float(ee), "azim_deg": float(ae)},
            "error": {"dist_abs": float(dist_err), "elev_deg_abs": float(elev_err_deg), "azim_deg_abs": float(azim_err_deg)},
        },
        "camera_center_world": {"gt": Cg.tolist(), "est": Ce.tolist(), "at": at.tolist()},
    }

    txt = []
    txt.append(f"tag: {tag}")
    txt.append("=== Pose Metrics (est vs gt) ===")
    txt.append(f"rotation_error_deg: {rot_err_deg:.6f}")
    txt.append(f"camera_center_error: {cam_center_err:.6f}")
    txt.append(f"view_direction_error_deg: {view_dir_err_deg:.6f}")
    txt.append("")
    txt.append("=== Spherical around at_xyz ===")
    txt.append(f"gt:  dist={dg:.6f}, elev_deg={eg:.6f}, azim_deg={ag:.6f}")
    txt.append(f"est: dist={de:.6f}, elev_deg={ee:.6f}, azim_deg={ae:.6f}")
    txt.append(f"err: dist_abs={dist_err:.6f}, elev_abs_deg={elev_err_deg:.6f}, azim_abs_deg={azim_err_deg:.6f}")
    txt.append("")
    txt.append("=== Camera center (world) ===")
    txt.append(f"at : {at[0]:.6f} {at[1]:.6f} {at[2]:.6f}")
    txt.append(f"gt : {Cg[0]:.6f} {Cg[1]:.6f} {Cg[2]:.6f}")
    txt.append(f"est: {Ce[0]:.6f} {Ce[1]:.6f} {Ce[2]:.6f}")
    txt_str = "\n".join(txt) + "\n"
    return metrics, txt_str


def _to_uint8_rgb(x_rgb: torch.Tensor) -> np.ndarray:
    img = x_rgb.detach().cpu().numpy()
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def _depth_to_vis(depth_map: torch.Tensor) -> np.ndarray:
    d = depth_map.detach().cpu().numpy()
    if d.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    d = d - d.min()
    if d.max() > 0:
        d = d / d.max()
    d = np.clip(d * 255.0, 0, 255).astype(np.uint8)
    return d


def _normal_to_vis(nmap: torch.Tensor) -> np.ndarray:
    n = nmap.detach().cpu().numpy()
    n = np.clip((n + 1.0) * 0.5, 0, 1)
    return (n * 255).astype(np.uint8)


def _save_silhouette_pair(save_dir: str, prefix: str, alpha: torch.Tensor) -> None:
    """
    Save silhouette in 2 forms (NO binarization):
    - soft: raw alpha in [0,1]
    - vis: contrast-stretched visualization (useful if alpha range is tiny)
    """
    a = torch.clamp(alpha.detach().float(), 0, 1)
    a_np = (a.cpu().numpy() * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_dir, f"{prefix}_silhouette_soft.png"), a_np)

    a_min = float(a.min().cpu().item())
    a_max = float(a.max().cpu().item())
    if int(os.environ.get("VAL_DEBUG_SIL", "0")) != 0:
        try:
            a_mean = float(a.mean().cpu().item())
            print(f"[SilDebug] {prefix}: alpha min/max/mean = {a_min:.6f}/{a_max:.6f}/{a_mean:.6f}")
        except Exception:
            pass
    if a_max - a_min < 1e-8:
        # If alpha is constant, "contrast stretch" is undefined.
        # Show the constant itself (all-white when a==1, all-black when a==0).
        a_vis = torch.ones_like(a) * (1.0 if a_max > 0.5 else 0.0)
    else:
        a_vis = (a - a_min) / (a_max - a_min)
    cv2.imwrite(os.path.join(save_dir, f"{prefix}_silhouette_vis.png"), (a_vis.cpu().numpy() * 255).astype(np.uint8))


def _write_o3d_mesh(path: str, mesh: "o3d.geometry.TriangleMesh") -> None:
    """
    Best-effort write of an Open3D TriangleMesh with normals/colors if present.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    try:
        ok = o3d.io.write_triangle_mesh(
            path,
            mesh,
            write_ascii=False,
            compressed=False,
            write_vertex_normals=True,
            write_vertex_colors=True,
            print_progress=False,
        )
        print(f"[MeshSave] write_triangle_mesh ok={bool(ok)} -> {path}")
    except Exception as _e:
        print(f"[Warning] Failed to save mesh to {path}: {_e}")


def _save_npz_tensors(path: str, **tensors: torch.Tensor) -> None:
    """
    Save torch tensors to a compressed .npz for later reuse.
    All tensors are moved to CPU and converted to float32 (unless already int/bool).
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        pass
    try:
        arrs: dict[str, np.ndarray] = {}
        for k, v in tensors.items():
            if v is None:
                continue
            if not isinstance(v, torch.Tensor):
                continue
            x = v.detach()
            # keep ints/bools as is; float -> float32
            if x.is_floating_point():
                x = x.float()
            arrs[str(k)] = x.cpu().numpy()
        np.savez_compressed(path, **arrs)
        print(f"[GTCache] Saved {len(arrs)} tensors -> {path}")
    except Exception as _e:
        print(f"[Warning] Failed to save npz {path}: {_e}")


def _load_npz_tensors(path: str, device: torch.device) -> dict[str, torch.Tensor]:
    """
    Load tensors from a .npz file (created by _save_npz_tensors).
    """
    out: dict[str, torch.Tensor] = {}
    try:
        z = np.load(path)
        for k in z.files:
            out[k] = torch.from_numpy(z[k]).to(device)
    except Exception as _e:
        print(f"[Warning] Failed to load npz {path}: {_e}")
    return out


def _render_and_save_best_artifacts(
    *,
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    phong_renderer: MeshRenderer | None,
    at_xyz: torch.Tensor,
    dist_v: float,
    elev_v: float,
    azim_v: float,
    device: torch.device,
    save_dir: str,
    prefix: str,
) -> None:
    """
    Save a clearly named bundle for a given (dist,elev,azim):
    - <prefix>_pose_4x4.txt/.npy
    - <prefix>_camera_spherical.txt
    - <prefix>_silhouette_soft.png / <prefix>_silhouette_vis.png
    - <prefix>_depth.png
    - <prefix>_normals.png
    - <prefix>_rgb.png (optional, if phong_renderer is not None)
    """
    # Allow skipping heavy full-res saving (useful for debugging GPU kernel hard-crash).
    if int(os.environ.get("VAL_SAVE_BEST_ARTIFACTS", "1")) == 0:
        return
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        Rb, Tb = look_at_view_transform(
            dist=torch.tensor([float(dist_v)], device=device),
            elev=torch.tensor([float(elev_v)], device=device),
            azim=torch.tensor([float(azim_v)], device=device),
            at=at_xyz.view(1, 3),
            device=device,
        )
        R0, T0 = Rb[0], Tb[0]
        pose_4x4 = torch.eye(4, device=device)
        pose_4x4[:3, :3] = R0
        pose_4x4[:3, 3] = T0
        np.savetxt(os.path.join(save_dir, f"{prefix}_pose_4x4.txt"), pose_4x4.cpu().numpy(), fmt="%.8f")
        np.save(os.path.join(save_dir, f"{prefix}_pose_4x4.npy"), pose_4x4.cpu().numpy().astype(np.float32))
        with open(os.path.join(save_dir, f"{prefix}_camera_spherical.txt"), "w", encoding="utf-8") as f:
            f.write(f"dist={float(dist_v):.6f}\n")
            f.write(f"elev_deg={float(elev_v):.6f}\n")
            f.write(f"azim_deg={float(azim_v):.6f}\n")

        rgba = silhouette_renderer(meshes_world=mesh, R=Rb, T=Tb)[0]
        alpha = torch.clamp(rgba[..., 3], 0, 1)
        _save_silhouette_pair(save_dir, prefix, alpha)

        cams = FoVPerspectiveCameras(device=device, R=Rb, T=Tb)
        fr = rasterizer(meshes_world=mesh, cameras=cams)
        vis_mask = (fr.pix_to_face[..., 0] >= 0).float()[0]
        depth = (fr.zbuf[..., 0][0] * vis_mask)
        cv2.imwrite(os.path.join(save_dir, f"{prefix}_depth.png"), _depth_to_vis(depth))

        verts_normals = mesh.verts_normals_packed()
        faces = mesh.faces_packed()
        verts_normals_view = torch.matmul(verts_normals, R0.T)
        faces_verts_normals = verts_normals_view[faces]
        pix_normals = interpolate_face_attributes(fr.pix_to_face, fr.bary_coords, faces_verts_normals)
        normals = pix_normals[..., 0, :][0]
        normals = torch.nn.functional.normalize(normals, dim=-1) * vis_mask.unsqueeze(-1)
        cv2.imwrite(os.path.join(save_dir, f"{prefix}_normals.png"), _normal_to_vis(normals))

        if phong_renderer is not None:
            rgb = phong_renderer(meshes_world=mesh, cameras=cams)[0, ..., :3]
            cv2.imwrite(os.path.join(save_dir, f"{prefix}_rgb.png"), cv2.cvtColor(_to_uint8_rgb(rgb), cv2.COLOR_RGB2BGR))


def _render_and_save_best_artifacts_rt(
    *,
    mesh: Meshes,
    silhouette_renderer: MeshRenderer,
    rasterizer: MeshRasterizer,
    phong_renderer: MeshRenderer | None,
    R_w2v: torch.Tensor,
    T_w2v: torch.Tensor,
    device: torch.device,
    save_dir: str,
    prefix: str,
) -> None:
    """
    Save artifacts for an explicit world-to-view pose (R,T). This is used by SE(3) refinement (E),
    where the result may not satisfy the look-at constraint.
    """
    if int(os.environ.get("VAL_SAVE_BEST_ARTIFACTS", "1")) == 0:
        return
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        Rb = R_w2v.view(1, 3, 3).to(device)
        Tb = T_w2v.view(1, 3).to(device)
        pose_4x4 = torch.eye(4, device=device)
        pose_4x4[:3, :3] = Rb[0]
        pose_4x4[:3, 3] = Tb[0]
        np.savetxt(os.path.join(save_dir, f"{prefix}_pose_4x4.txt"), pose_4x4.cpu().numpy(), fmt="%.8f")
        np.save(os.path.join(save_dir, f"{prefix}_pose_4x4.npy"), pose_4x4.cpu().numpy().astype(np.float32))

        rgba = silhouette_renderer(meshes_world=mesh, R=Rb, T=Tb)[0]
        alpha = torch.clamp(rgba[..., 3], 0, 1)
        _save_silhouette_pair(save_dir, prefix, alpha)

        cams = FoVPerspectiveCameras(device=device, R=Rb, T=Tb)
        fr = rasterizer(meshes_world=mesh, cameras=cams)
        vis_mask = (fr.pix_to_face[..., 0] >= 0).float()[0]
        depth = (fr.zbuf[..., 0][0] * vis_mask)
        cv2.imwrite(os.path.join(save_dir, f"{prefix}_depth.png"), _depth_to_vis(depth))

        verts_normals = mesh.verts_normals_packed()
        faces = mesh.faces_packed()
        verts_normals_view = torch.matmul(verts_normals, Rb[0].T)
        faces_verts_normals = verts_normals_view[faces]
        pix_normals = interpolate_face_attributes(fr.pix_to_face, fr.bary_coords, faces_verts_normals)
        normals = pix_normals[..., 0, :][0]
        normals = torch.nn.functional.normalize(normals, dim=-1) * vis_mask.unsqueeze(-1)
        cv2.imwrite(os.path.join(save_dir, f"{prefix}_normals.png"), _normal_to_vis(normals))

        if phong_renderer is not None:
            rgb = phong_renderer(meshes_world=mesh, cameras=cams)[0, ..., :3]
            cv2.imwrite(os.path.join(save_dir, f"{prefix}_rgb.png"), cv2.cvtColor(_to_uint8_rgb(rgb), cv2.COLOR_RGB2BGR))


def build_renderers(image_height, image_width, cameras, faces_per_pixel_silhouette: int | None = None):
    # Make silhouette optimization more stable/visible by allowing a slightly softer silhouette.
    # Tunable via env vars to match PyTorch3D tutorial behavior.
    sigma = float(os.environ.get("VAL_SIGMA", "5e-3"))
    gamma = float(os.environ.get("VAL_GAMMA", "1e-4"))
    blend_params = BlendParams(sigma=sigma, gamma=gamma)
    # For dense meshes, PyTorch3D's coarse-to-fine rasterization can overflow bins.
    # Setting bin_size=0 switches to naive rasterization (slower but robust).
    bin_size = int(os.environ.get("VAL_BIN_SIZE", "0"))
    max_faces_per_bin = int(os.environ.get("VAL_MAX_FACES_PER_BIN", "200000"))
    if faces_per_pixel_silhouette is None:
        faces_per_pixel_silhouette = int(os.environ.get("VAL_FACES_PER_PIXEL", "50"))
    raster_settings_silhouette = RasterizationSettings(
        image_size=(image_height, image_width),
        blur_radius=np.log(1.0 / 1e-4 - 1.0) * blend_params.sigma,
        faces_per_pixel=int(faces_per_pixel_silhouette),
        bin_size=bin_size,
        max_faces_per_bin=max_faces_per_bin,
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings_silhouette)
    silhouette_renderer = MeshRenderer(rasterizer=rasterizer, shader=SoftSilhouetteShader(blend_params=blend_params))

    # Aux rasterizer (K=1) for depth/normals visualization (cheap & stable)
    raster_settings_aux = RasterizationSettings(
        image_size=(image_height, image_width),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=bin_size,
        max_faces_per_bin=max_faces_per_bin,
    )
    rasterizer_aux = MeshRasterizer(cameras=cameras, raster_settings=raster_settings_aux)

    raster_settings_phong = RasterizationSettings(
        image_size=(image_height, image_width),
        blur_radius=0.0,
        faces_per_pixel=1,
        bin_size=bin_size,
        max_faces_per_bin=max_faces_per_bin,
    )
    lights = PointLights(device=cameras.device, location=((2.0, 2.0, -2.0),))
    phong_renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings_phong),
        shader=HardPhongShader(device=cameras.device, cameras=cameras, lights=lights),
    )
    return silhouette_renderer, rasterizer_aux, phong_renderer


def render_targets(mesh, R_gt, T_gt, device, image_height, image_width):
    cameras_gt = FoVPerspectiveCameras(device=device, R=R_gt.unsqueeze(0), T=T_gt.unsqueeze(0))
    sil_renderer, rast_gt, _ = build_renderers(image_height, image_width, cameras_gt)
    with torch.no_grad():
        sil = sil_renderer(meshes_world=mesh, R=R_gt.unsqueeze(0), T=T_gt.unsqueeze(0))
        alpha = torch.clamp(sil[0, ..., 3], 0, 1)
        fr = rast_gt(meshes_world=mesh, R=R_gt.unsqueeze(0), T=T_gt.unsqueeze(0))
        vis_mask = (fr.pix_to_face[..., 0] >= 0).float()[0]
        depth = (fr.zbuf[..., 0][0] * vis_mask)

        verts_normals = mesh.verts_normals_packed()
        faces = mesh.faces_packed()
        # normals in VIEW SPACE for stable visualization
        verts_normals_view = torch.matmul(verts_normals, R_gt.T)
        faces_verts_normals = verts_normals_view[faces]
        pix_normals = interpolate_face_attributes(fr.pix_to_face, fr.bary_coords, faces_verts_normals)
        normals = pix_normals[..., 0, :][0]
        normals = torch.nn.functional.normalize(normals, dim=-1) * vis_mask.unsqueeze(-1)

        # Stable reference points for any future chamfer usage (deterministic vertex sampling)
        verts_view = torch.matmul(mesh.verts_packed(), R_gt.T) + T_gt  # (V,3) in view
        max_pts = int(os.environ.get("VAL_CHAMFER_POINTS", "8000"))
        if verts_view.shape[0] > max_pts:
            idx = torch.linspace(0, verts_view.shape[0] - 1, steps=max_pts, device=verts_view.device).long()
            ref_points_view = verts_view.index_select(0, idx).detach()
        else:
            ref_points_view = verts_view.detach()
        if ref_points_view.numel() == 0:
            ref_points_view = torch.zeros((1, 3), device=device)

    return alpha.detach(), depth.detach(), normals.detach(), ref_points_view


def _downscaled_hw(h: int, w: int, downscale: float) -> tuple[int, int]:
    downscale = float(downscale)
    downscale = max(0.2, min(1.0, downscale))
    hs = int(max(16, round(h * downscale)))
    ws = int(max(16, round(w * downscale)))
    hs = hs - (hs % 2)
    ws = ws - (ws % 2)
    return int(hs), int(ws)


def optimize_to_target(
    mesh,
    image_ref_silhouette,
    ref_depth,
    ref_normals,
    ref_points_view,
    R_init,
    T_init,
    look_at_at,
    init_dist,
    init_elev,
    init_azim,
    camera_distance,
    device,
    max_iterations=2000,
    lr=0.02,
    loss_type="enhanced",
    save_dir=None,
    target_name="pose0",
):
    """
    按 PyTorch3D 官方示例 camera_position_optimization_with_differentiable_rendering.py 实现：
    - 直接优化 camera_position (x,y,z)
    - 用 look_at_rotation 计算 R，再由公式得到 T
    - 用 silhouette loss 作为主梯度来源（确保相机视点确实变化）
    仍然保存 depth/normal/chamfer 的可视化中间结果，但不让不稳定项主导优化。
    """

    # Optimization method:
    # - adam  : current gradient-based optimization (may oscillate on non-convex losses)
    # - greedy: monotone local search (accept only improvements; faster per-step since no backward)
    # - hybrid: greedy warmup then adam refine
    opt_method = os.environ.get("VAL_OPTIM_METHOD", "greedy_axis").strip().lower()  # adam|greedy|hybrid|greedy_bt|greedy_axis

    # Create renderers/rasterizer (scene setup consistent with main script)
    # For speed control:
    # - Search stage can reduce silhouette faces_per_pixel (VAL_SEARCH_FACES_PER_PIXEL)
    # - Final stage can use the default VAL_FACES_PER_PIXEL
    fpp_sil = int(os.environ.get("VAL_FACES_PER_PIXEL", "50"))
    if int(os.environ.get("VAL_IS_SEARCH_STAGE", "0")) != 0:
        # Optimized: default changed from 10 to 5 for faster rendering (~2-5x speedup)
        fpp_sil = int(os.environ.get("VAL_SEARCH_FACES_PER_PIXEL", "5"))
    cameras_stub = FoVPerspectiveCameras(device=device)
    silhouette_renderer, rasterizer, phong_renderer = build_renderers(
        image_ref_silhouette.shape[0], image_ref_silhouette.shape[1], cameras_stub, faces_per_pixel_silhouette=fpp_sil
    )

    def _sigmoid_to_range(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return lo + (hi - lo) * torch.sigmoid(x)

    def _tanh_to_range(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
        return 0.5 * (hi - lo) * torch.tanh(x) + 0.5 * (hi + lo)

    def _camera_position_from_spherical(dist: torch.Tensor, elev_deg: torch.Tensor, azim_deg: torch.Tensor, at_xyz: torch.Tensor) -> torch.Tensor:
        # dist: scalar, elev/azim in degrees, at_xyz: (3,)
        elev = elev_deg * math.pi / 180.0
        azim = azim_deg * math.pi / 180.0
        x = dist * torch.cos(elev) * torch.sin(azim)
        y = dist * torch.sin(elev)
        z = dist * torch.cos(elev) * torch.cos(azim)
        return torch.stack([x, y, z], dim=0) + at_xyz

    def _save_greedy_state(it: int, dist_v: float, elev_v: float, azim_v: float, loss_v: float, save_prefix: str = "greedy_iter"):
        if save_dir is None:
            return
        try:
            os.makedirs(save_dir, exist_ok=True)
            with torch.no_grad():
                Rb, Tb = look_at_view_transform(
                    dist=torch.tensor([float(dist_v)], device=device),
                    elev=torch.tensor([float(elev_v)], device=device),
                    azim=torch.tensor([float(azim_v)], device=device),
                    at=look_at_at.view(1, 3),
                    device=device,
                )
                R0, T0 = Rb[0], Tb[0]
                cameras_cur = FoVPerspectiveCameras(device=device, R=Rb, T=Tb)

                rgba = silhouette_renderer(meshes_world=mesh, R=Rb, T=Tb)[0]
                alpha = torch.clamp(rgba[..., 3], 0, 1)
                _save_silhouette_pair(save_dir, f"{save_prefix}_{it:04d}", alpha)

                diff = torch.abs(alpha - torch.clamp(image_ref_silhouette.squeeze(), 0, 1))
                cv2.imwrite(
                    os.path.join(save_dir, f"{save_prefix}_{it:04d}_silhouette_absdiff.png"),
                    (torch.clamp(diff / (diff.max() + 1e-8), 0, 1).cpu().numpy() * 255).astype(np.uint8),
                )

                fr = rasterizer(meshes_world=mesh, cameras=cameras_cur)
                vis_mask = (fr.pix_to_face[..., 0] >= 0).float()[0]
                depth = (fr.zbuf[..., 0][0] * vis_mask)
                cv2.imwrite(os.path.join(save_dir, f"{save_prefix}_{it:04d}_depth.png"), _depth_to_vis(depth))

                verts_normals = mesh.verts_normals_packed()
                faces = mesh.faces_packed()
                verts_normals_view = torch.matmul(verts_normals, R0.T)
                faces_verts_normals = verts_normals_view[faces]
                pix_normals = interpolate_face_attributes(fr.pix_to_face, fr.bary_coords, faces_verts_normals)
                normals = pix_normals[..., 0, :][0]
                normals = torch.nn.functional.normalize(normals, dim=-1) * vis_mask.unsqueeze(-1)
                cv2.imwrite(os.path.join(save_dir, f"{save_prefix}_{it:04d}_normals.png"), _normal_to_vis(normals))

                rgb = phong_renderer(meshes_world=mesh, cameras=cameras_cur)[0, ..., :3]
                cv2.imwrite(os.path.join(save_dir, f"{save_prefix}_{it:04d}_rgb.png"), cv2.cvtColor(_to_uint8_rgb(rgb), cv2.COLOR_RGB2BGR))

                pose_4x4 = torch.eye(4, device=device)
                pose_4x4[:3, :3] = R0
                pose_4x4[:3, 3] = T0
                np.savetxt(os.path.join(save_dir, f"{save_prefix}_{it:04d}_pose_4x4.txt"), pose_4x4.cpu().numpy(), fmt="%.8f")
                with open(os.path.join(save_dir, f"{save_prefix}_{it:04d}_camera_spherical.txt"), "w", encoding="utf-8") as f:
                    f.write(f"dist={float(dist_v):.6f}\n")
                    f.write(f"elev_deg={float(elev_v):.6f}\n")
                    f.write(f"azim_deg={float(azim_v):.6f}\n")
                    f.write(f"loss={float(loss_v):.10f}\n")
        except Exception as _e:
            print(f"[Warning] Failed to save greedy state at iter {it}: {_e}")

    def greedy_local_search(init_dist_g: float, init_elev_g: float, init_azim_g: float, greedy_iters: int):
        """
        Monotone greedy local search in (dist,elev,azim).
        Each iter evaluates 6 neighbors:
          dist+/dist- (multiplicative), elev+/elev-, azim+/azim-
        Accepts ONLY if loss improves by min_delta, otherwise shrinks steps.
        """
        save_interval = int(os.environ.get("VAL_SAVE_INTERVAL", "50"))
        log_interval = int(os.environ.get("VAL_LOG_INTERVAL", "50"))
        history_interval = int(os.environ.get("VAL_HISTORY_INTERVAL", str(log_interval)))

        # steps
        dist_step = float(os.environ.get("VAL_GREEDY_DIST_STEP", "0.10"))  # multiplicative (1+step)
        ang_step = float(os.environ.get("VAL_GREEDY_ANG_STEP_DEG", "2.0"))  # degrees
        shrink = float(os.environ.get("VAL_GREEDY_SHRINK", "0.5"))
        min_dist_step = float(os.environ.get("VAL_GREEDY_MIN_DIST_STEP", "0.01"))
        min_ang_step = float(os.environ.get("VAL_GREEDY_MIN_ANG_STEP_DEG", "0.1"))
        min_delta = float(os.environ.get("VAL_GREEDY_MIN_DELTA", "1e-5"))

        # bounds (match existing ranges)
        elev_min = float(os.environ.get("VAL_GT_ELEV_MIN", "5.0"))
        elev_max = float(os.environ.get("VAL_GT_ELEV_MAX", "85.0"))
        azim_min = float(os.environ.get("VAL_GT_AZIM_MIN", "-180.0"))
        azim_max = float(os.environ.get("VAL_GT_AZIM_MAX", "180.0"))

        dist_base_mode = os.environ.get("VAL_DIST_BASE_MODE", "init").strip().lower()
        base_for_bounds = float(camera_distance) if dist_base_mode == "camera" else float(init_dist_g)
        dist_min = float(os.environ.get("VAL_GT_DIST_SCALE_MIN", "0.15")) * base_for_bounds
        dist_max = float(os.environ.get("VAL_GT_DIST_SCALE_MAX", "0.85")) * base_for_bounds
        if os.environ.get("VAL_GT_DIST_MIN", "") != "":
            dist_min = max(dist_min, float(os.environ.get("VAL_GT_DIST_MIN")))
        if os.environ.get("VAL_GT_DIST_MAX", "") != "":
            dist_max = min(dist_max, float(os.environ.get("VAL_GT_DIST_MAX")))
        dist_min = max(1e-3, float(dist_min))
        dist_max = max(dist_min, float(dist_max))

        def _wrap_azim(a: float) -> float:
            return float(((a + 180.0) % 360.0) - 180.0)

        # init
        dist_c = float(np.clip(init_dist_g, dist_min, dist_max))
        elev_c = float(np.clip(init_elev_g, elev_min, elev_max))
        azim_c = float(_wrap_azim(init_azim_g))
        azim_c = float(np.clip(azim_c, azim_min, azim_max))

        # history file (greedy uses a different schema)
        hist_fp = None
        hist_writer = None
        if save_dir is not None:
            try:
                os.makedirs(save_dir, exist_ok=True)
                hist_fp = open(os.path.join(save_dir, f"{target_name}_pose_history.csv"), "w", newline="", encoding="utf-8")
                hist_writer = csv.writer(hist_fp)
                hist_writer.writerow(["iter", "method", "loss_total", "dist", "elev_deg", "azim_deg", "dist_step", "ang_step_deg"])
            except Exception as _e:
                print(f"[Warning] Failed to create greedy pose_history.csv: {_e}")
                hist_fp = None
                hist_writer = None

        cur_ev = _evaluate_pose_loss_no_grad(
            mesh, silhouette_renderer, rasterizer, image_ref_silhouette, ref_depth, ref_normals,
            look_at_at.view(3), dist_c, elev_c, azim_c, device
        )
        cur_loss = float(cur_ev["loss_total"])
        best_loss = float(cur_loss)
        best_dist, best_elev, best_azim = float(dist_c), float(elev_c), float(azim_c)
        best_R, best_T = cur_ev["R"], cur_ev["T"]

        loss_hist: list[float] = []
        if save_dir is not None and save_interval > 0:
            _save_greedy_state(0, dist_c, elev_c, azim_c, cur_loss)

        greedy_sil_only = int(os.environ.get("VAL_GREEDY_USE_SIL_ONLY", "0")) != 0
        greedy_batch = int(os.environ.get("VAL_GREEDY_BATCH_NEIGHBORS", "1")) != 0
        greedy_fd = int(os.environ.get("VAL_GREEDY_FD_STEP", "0")) != 0

        for it in range(int(greedy_iters)):
            loss_hist.append(float(cur_loss))
            if hist_writer is not None and (it % history_interval == 0 or it == int(greedy_iters) - 1):
                try:
                    hist_writer.writerow([int(it), "greedy", float(cur_loss), float(dist_c), float(elev_c), float(azim_c), float(dist_step), float(ang_step)])
                    if hist_fp is not None:
                        hist_fp.flush()
                except Exception:
                    pass

            if it % log_interval == 0:
                print(
                    f"[Greedy] {target_name} iter {it}/{greedy_iters} loss={cur_loss:.6f} "
                    f"(dist,elev,azim)=({dist_c:.2f},{elev_c:.2f},{azim_c:.2f}) steps(dist={dist_step:.4f}, ang={ang_step:.3f})"
                )

            if greedy_fd:
                fd_out = _sphere_fd_descent_step(
                    mesh=mesh,
                    silhouette_renderer=silhouette_renderer,
                    rasterizer=rasterizer,
                    image_ref_silhouette=image_ref_silhouette,
                    ref_depth=ref_depth,
                    ref_normals=ref_normals,
                    at_xyz=look_at_at.view(3),
                    dist_c=dist_c,
                    elev_c=elev_c,
                    azim_c=azim_c,
                    elev_min=elev_min,
                    elev_max=elev_max,
                    azim_min=azim_min,
                    azim_max=azim_max,
                    device=device,
                    use_sil_only=greedy_sil_only,
                )
                if fd_out is not None:
                    _d, _e, _a, _l, _ev = fd_out
                    if float(_l) < float(cur_loss) - min_delta:
                        dist_c, elev_c, azim_c = float(_d), float(_e), float(_a)
                        cur_loss, cur_ev = float(_l), _ev
                        if cur_loss < best_loss:
                            best_loss, best_dist, best_elev, best_azim = cur_loss, dist_c, elev_c, azim_c
                            best_R, best_T = cur_ev["R"], cur_ev["T"]

            if greedy_batch:
                new_loss, move_name, d_new, e_new, a_new, ev_new = _batched_best_neighbor_move(
                    mesh=mesh,
                    silhouette_renderer=silhouette_renderer,
                    rasterizer=rasterizer,
                    image_ref_silhouette=image_ref_silhouette,
                    ref_depth=ref_depth,
                    ref_normals=ref_normals,
                    at_xyz=look_at_at.view(3),
                    dist_c=dist_c,
                    elev_c=elev_c,
                    azim_c=azim_c,
                    dist_step=dist_step,
                    ang_step=ang_step,
                    dist_min=dist_min,
                    dist_max=dist_max,
                    elev_min=elev_min,
                    elev_max=elev_max,
                    azim_min=azim_min,
                    azim_max=azim_max,
                    device=device,
                    use_sil_only=greedy_sil_only,
                )
            else:
                cands = list(
                    zip(
                        *_greedy_neighbor_poses(
                            dist_c=dist_c,
                            elev_c=elev_c,
                            azim_c=azim_c,
                            dist_step=dist_step,
                            ang_step=ang_step,
                            dist_min=dist_min,
                            dist_max=dist_max,
                            elev_min=elev_min,
                            elev_max=elev_max,
                            azim_min=azim_min,
                            azim_max=azim_max,
                        )
                    )
                )
                best_move = None
                for name, d1, e1, a1 in cands:
                    ev = _evaluate_pose_loss_no_grad(
                        mesh,
                        silhouette_renderer,
                        rasterizer,
                        image_ref_silhouette,
                        ref_depth,
                        ref_normals,
                        look_at_at.view(3),
                        float(d1),
                        float(e1),
                        float(a1),
                        device,
                    )
                    lv = float(ev["loss_total"])
                    if best_move is None or lv < best_move[0]:
                        best_move = (lv, name, float(d1), float(e1), float(a1), ev)
                assert best_move is not None
                new_loss, move_name, d_new, e_new, a_new, ev_new = best_move

            if new_loss < (cur_loss - min_delta):
                dist_c, elev_c, azim_c = d_new, e_new, a_new
                cur_loss = float(new_loss)
                cur_ev = ev_new
                if cur_loss < best_loss:
                    best_loss = cur_loss
                    best_dist, best_elev, best_azim = dist_c, elev_c, azim_c
                    best_R, best_T = cur_ev["R"], cur_ev["T"]
                if it % log_interval == 0:
                    print(f"[Greedy]   accept move={move_name} -> loss={cur_loss:.6f}")
            else:
                # shrink step sizes
                dist_step = max(min_dist_step, dist_step * shrink)
                ang_step = max(min_ang_step, ang_step * shrink)
                if it % log_interval == 0:
                    print(f"[Greedy]   no improve; shrink -> dist_step={dist_step:.4f}, ang_step={ang_step:.3f}")

            if save_dir is not None and save_interval > 0 and (it % save_interval == 0 or it == int(greedy_iters) - 1):
                _save_greedy_state(it, dist_c, elev_c, azim_c, cur_loss)

            # stop if steps become too small
            if dist_step <= min_dist_step and ang_step <= min_ang_step:
                if it % log_interval == 0:
                    print("[Greedy] stop: step sizes are small.")
                break

        if hist_fp is not None:
            try:
                hist_fp.close()
            except Exception:
                pass

        pose_best = np.eye(4, dtype=np.float32)
        pose_best[:3, :3] = best_R.detach().cpu().numpy().astype(np.float32)
        pose_best[:3, 3] = best_T.detach().cpu().numpy().astype(np.float32)
        return pose_best, loss_hist, (best_dist, best_elev, best_azim, best_loss)

    def greedy_backtracking_search(init_dist_g: float, init_elev_g: float, init_azim_g: float, budget_iters: int):
        """
        Greedy with BACKTRACKING (user request):
        - Forward explore up to N forward steps (VAL_BT_FORWARD_STEPS)
        - Backtrack up to N back steps (VAL_BT_BACKTRACK_STEPS)
        - At each node, evaluate 6 neighbors and explore them in ascending loss order.
        - Allows limited uphill exploration within a tolerance (VAL_BT_UPHILL_TOL) to escape local minima.
        This is still a local search on a discrete neighborhood, not a global-opt guarantee.
        """
        save_interval = int(os.environ.get("VAL_SAVE_INTERVAL", "50"))
        log_interval = int(os.environ.get("VAL_LOG_INTERVAL", "50"))
        forward_max = int(os.environ.get("VAL_BT_FORWARD_STEPS", "64"))
        back_max = int(os.environ.get("VAL_BT_BACKTRACK_STEPS", "64"))
        uphill_tol = float(os.environ.get("VAL_BT_UPHILL_TOL", "0.05"))  # absolute loss tolerance above best
        min_delta = float(os.environ.get("VAL_BT_MIN_DELTA", "1e-6"))

        # base step sizes (will shrink on dead-ends)
        dist_step0 = float(os.environ.get("VAL_BT_DIST_STEP", "0.10"))
        ang_step0 = float(os.environ.get("VAL_BT_ANG_STEP_DEG", "2.0"))
        shrink = float(os.environ.get("VAL_BT_SHRINK", "0.5"))
        min_dist_step = float(os.environ.get("VAL_BT_MIN_DIST_STEP", "0.01"))
        min_ang_step = float(os.environ.get("VAL_BT_MIN_ANG_STEP_DEG", "0.1"))

        elev_min = float(os.environ.get("VAL_GT_ELEV_MIN", "5.0"))
        elev_max = float(os.environ.get("VAL_GT_ELEV_MAX", "85.0"))
        azim_min = float(os.environ.get("VAL_GT_AZIM_MIN", "-180.0"))
        azim_max = float(os.environ.get("VAL_GT_AZIM_MAX", "180.0"))

        dist_base_mode = os.environ.get("VAL_DIST_BASE_MODE", "init").strip().lower()
        base_for_bounds = float(camera_distance) if dist_base_mode == "camera" else float(init_dist_g)
        dist_min = float(os.environ.get("VAL_GT_DIST_SCALE_MIN", "0.15")) * base_for_bounds
        dist_max = float(os.environ.get("VAL_GT_DIST_SCALE_MAX", "0.85")) * base_for_bounds
        if os.environ.get("VAL_GT_DIST_MIN", "") != "":
            dist_min = max(dist_min, float(os.environ.get("VAL_GT_DIST_MIN")))
        if os.environ.get("VAL_GT_DIST_MAX", "") != "":
            dist_max = min(dist_max, float(os.environ.get("VAL_GT_DIST_MAX")))
        dist_min = max(1e-3, float(dist_min))
        dist_max = max(dist_min, float(dist_max))

        def _wrap_azim(a: float) -> float:
            return float(((a + 180.0) % 360.0) - 180.0)

        # trace logs
        trace_csv = None
        trace_writer = None
        if save_dir is not None:
            try:
                trace_csv = open(os.path.join(save_dir, f"{target_name}_greedy_bt_trace.csv"), "w", newline="", encoding="utf-8")
                trace_writer = csv.writer(trace_csv)
                trace_writer.writerow(["step", "depth", "action", "move", "loss_total", "dist", "elev_deg", "azim_deg", "dist_step", "ang_step_deg"])
            except Exception as _e:
                print(f"[Warning] Failed to create greedy_bt_trace.csv: {_e}")
                trace_csv = None
                trace_writer = None

        # node structure: dict with state + neighbor list + cursor
        def _eval_state(dist_v: float, elev_v: float, azim_v: float):
            ev = _evaluate_pose_loss_no_grad(
                mesh, silhouette_renderer, rasterizer, image_ref_silhouette, ref_depth, ref_normals,
                look_at_at.view(3), float(dist_v), float(elev_v), float(azim_v), device
            )
            return float(ev["loss_total"]), ev

        dist0 = float(np.clip(init_dist_g, dist_min, dist_max))
        elev0 = float(np.clip(init_elev_g, elev_min, elev_max))
        azim0 = float(np.clip(_wrap_azim(init_azim_g), azim_min, azim_max))
        loss0, ev0 = _eval_state(dist0, elev0, azim0)

        best_loss = float(loss0)
        best_state = {"dist": dist0, "elev": elev0, "azim": azim0, "loss": best_loss, "ev": ev0}

        # stack root
        stack = [
            {
                "dist": dist0,
                "elev": elev0,
                "azim": azim0,
                "loss": loss0,
                "ev": ev0,
                "dist_step": float(dist_step0),
                "ang_step": float(ang_step0),
                "neighbors": None,
                "cursor": 0,
            }
        ]
        backtracks = 0
        loss_hist = []

        def _build_neighbors(node):
            d = float(node["dist"])
            e = float(node["elev"])
            a = float(node["azim"])
            ds = float(node["dist_step"])
            ang = float(node["ang_step"])
            cands = [
                ("dist+", float(np.clip(d * (1.0 + ds), dist_min, dist_max)), e, a),
                ("dist-", float(np.clip(d / (1.0 + ds), dist_min, dist_max)), e, a),
                ("elev+", d, float(np.clip(e + ang, elev_min, elev_max)), a),
                ("elev-", d, float(np.clip(e - ang, elev_min, elev_max)), a),
                ("azim+", d, e, float(np.clip(_wrap_azim(a + ang), azim_min, azim_max))),
                ("azim-", d, e, float(np.clip(_wrap_azim(a - ang), azim_min, azim_max))),
            ]
            neigh = []
            for mv, d1, e1, a1 in cands:
                lv, ev = _eval_state(d1, e1, a1)
                neigh.append((lv, mv, float(d1), float(e1), float(a1), ev))
            neigh.sort(key=lambda x: x[0])
            node["neighbors"] = neigh
            node["cursor"] = 0

        if save_dir is not None and save_interval > 0:
            _save_greedy_state(0, dist0, elev0, azim0, loss0, save_prefix="greedy_bt_iter")

        for step in range(int(budget_iters)):
            cur = stack[-1]
            loss_hist.append(float(best_loss))
            depth = len(stack) - 1
            if trace_writer is not None:
                trace_writer.writerow([int(step), int(depth), "at", "", float(cur["loss"]), float(cur["dist"]), float(cur["elev"]), float(cur["azim"]), float(cur["dist_step"]), float(cur["ang_step"])])
                trace_csv.flush()

            if step % log_interval == 0:
                print(
                    f"[GreedyBT] {target_name} step {step}/{budget_iters} depth={depth} "
                    f"cur_loss={float(cur['loss']):.6f} best_loss={best_loss:.6f} "
                    f"(dist,elev,azim)=({cur['dist']:.2f},{cur['elev']:.2f},{cur['azim']:.2f})"
                )

            # depth limit: force backtrack
            if depth >= forward_max:
                stack.pop()
                backtracks += 1
                if trace_writer is not None:
                    trace_writer.writerow([int(step), int(depth), "backtrack", "depth_limit", float(best_loss), float(best_state["dist"]), float(best_state["elev"]), float(best_state["azim"]), float(cur["dist_step"]), float(cur["ang_step"])])
                    trace_csv.flush()
                if backtracks >= back_max or len(stack) == 0:
                    break
                continue

            if cur["neighbors"] is None:
                _build_neighbors(cur)

            neigh = cur["neighbors"]
            moved = False
            while cur["cursor"] < len(neigh):
                lv, mv, d1, e1, a1, ev = neigh[cur["cursor"]]
                cur["cursor"] += 1

                # prune too-uphill candidates (but allow limited uphill exploration)
                if lv > (best_loss + uphill_tol):
                    continue

                # accept this branch (even if uphill) for exploration
                child = {
                    "dist": d1,
                    "elev": e1,
                    "azim": a1,
                    "loss": float(lv),
                    "ev": ev,
                    "dist_step": float(cur["dist_step"]),
                    "ang_step": float(cur["ang_step"]),
                    "neighbors": None,
                    "cursor": 0,
                }
                stack.append(child)
                moved = True
                if trace_writer is not None:
                    trace_writer.writerow([int(step), int(depth), "forward", mv, float(lv), float(d1), float(e1), float(a1), float(child["dist_step"]), float(child["ang_step"])])
                    trace_csv.flush()
                # update best if improved
                if lv < (best_loss - min_delta):
                    best_loss = float(lv)
                    best_state = {"dist": d1, "elev": e1, "azim": a1, "loss": best_loss, "ev": ev}
                break

            if moved:
                if save_dir is not None and save_interval > 0 and (step % save_interval == 0):
                    _save_greedy_state(step, float(best_state["dist"]), float(best_state["elev"]), float(best_state["azim"]), float(best_loss), save_prefix="greedy_bt_best")
                continue

            # no move => try shrink step sizes once, else backtrack
            ds = float(cur["dist_step"])
            ang = float(cur["ang_step"])
            ds2 = max(min_dist_step, ds * shrink)
            ang2 = max(min_ang_step, ang * shrink)
            if (ds2 < ds - 1e-12) or (ang2 < ang - 1e-12):
                cur["dist_step"] = ds2
                cur["ang_step"] = ang2
                cur["neighbors"] = None
                cur["cursor"] = 0
                if trace_writer is not None:
                    trace_writer.writerow([int(step), int(depth), "shrink", "", float(cur["loss"]), float(cur["dist"]), float(cur["elev"]), float(cur["azim"]), float(ds2), float(ang2)])
                    trace_csv.flush()
                continue

            # backtrack
            stack.pop()
            backtracks += 1
            if trace_writer is not None:
                trace_writer.writerow([int(step), int(depth), "backtrack", "dead_end", float(best_loss), float(best_state["dist"]), float(best_state["elev"]), float(best_state["azim"]), float(ds), float(ang)])
                trace_csv.flush()
            if backtracks >= back_max or len(stack) == 0:
                break

        if trace_csv is not None:
            try:
                trace_csv.close()
            except Exception:
                pass

        # build best pose 4x4
        Rb = best_state["ev"]["R"]
        Tb = best_state["ev"]["T"]
        pose_best = np.eye(4, dtype=np.float32)
        pose_best[:3, :3] = Rb.detach().cpu().numpy().astype(np.float32)
        pose_best[:3, 3] = Tb.detach().cpu().numpy().astype(np.float32)
        return pose_best, loss_hist, (float(best_state["dist"]), float(best_state["elev"]), float(best_state["azim"]), float(best_loss))

    def greedy_axis_sphere_search(init_dist_g: float, init_elev_g: float, init_azim_g: float, budget_iters: int):
        """
        Classic greedy "probe multiple steps per axis then choose direction" ON THE SPHERE:
        - Keep radius(dist) fixed (camera stays on sphere around at_xyz).
        - For each iteration:
            * Probe along ±X/±Y/±Z directions on the sphere for K forward steps.
            * Each probe is done by shifting camera position in world xyz then projecting back to the same sphere.
            * Compare the best loss among all directions, choose the best direction+step, and move there.
            * If no improvement: shrink step size; optionally backtrack.

        This matches your description:
        "explore a few steps along x, y, z; compare loss changes; decide next direction; forward; backtrack if needed."
        """
        save_interval = int(os.environ.get("VAL_SAVE_INTERVAL", "50"))
        log_interval = int(os.environ.get("VAL_LOG_INTERVAL", "50"))
        steps_per_axis = int(os.environ.get("VAL_AXIS_STEPS_PER_DIR", "4"))  # K (reduced for speed)
        # Speed/quality tradeoff:
        # - Evaluate all candidates with silhouette-only first, then (optionally) refine Top-M by FULL loss.
        # This avoids running the expensive rasterizer(depth/normals) for every candidate.
        axis_use_sil_only = int(os.environ.get("VAL_AXIS_USE_SIL_ONLY", "0")) != 0
        axis_refine_full = int(os.environ.get("VAL_AXIS_REFINE_FULL", "1")) != 0
        axis_topm_full = int(os.environ.get("VAL_AXIS_TOPM_FULL", "4"))
        axis_topm_full = max(1, axis_topm_full)
        step_frac = float(os.environ.get("VAL_AXIS_STEP_FRAC", "0.03"))  # step length = step_frac * dist
        shrink = float(os.environ.get("VAL_AXIS_SHRINK", "0.5"))
        min_step_frac = float(os.environ.get("VAL_AXIS_MIN_STEP_FRAC", "0.003"))
        min_delta = float(os.environ.get("VAL_AXIS_MIN_DELTA", "1e-6"))
        no_improve_patience = int(os.environ.get("VAL_AXIS_NO_IMPROVE_PATIENCE", "80"))
        allow_uphill = int(os.environ.get("VAL_AXIS_ALLOW_UPHILL", "0")) != 0
        uphill_tol = float(os.environ.get("VAL_AXIS_UPHILL_TOL", "0.05"))
        forward_max = int(os.environ.get("VAL_AXIS_FORWARD_STEPS", "200"))
        back_max = int(os.environ.get("VAL_AXIS_BACKTRACK_STEPS", "200"))

        at_xyz = look_at_at.view(3).to(device)
        dist_fixed = float(init_dist_g)  # sphere radius

        # clamp the radius into allowed range (keep consistent with constraints)
        dist_base_mode = os.environ.get("VAL_DIST_BASE_MODE", "init").strip().lower()
        base_for_bounds = float(camera_distance) if dist_base_mode == "camera" else float(init_dist_g)
        dist_min = float(os.environ.get("VAL_GT_DIST_SCALE_MIN", "0.15")) * base_for_bounds
        dist_max = float(os.environ.get("VAL_GT_DIST_SCALE_MAX", "0.85")) * base_for_bounds
        if os.environ.get("VAL_GT_DIST_MIN", "") != "":
            dist_min = max(dist_min, float(os.environ.get("VAL_GT_DIST_MIN")))
        if os.environ.get("VAL_GT_DIST_MAX", "") != "":
            dist_max = min(dist_max, float(os.environ.get("VAL_GT_DIST_MAX")))
        dist_fixed = float(np.clip(dist_fixed, max(1e-3, dist_min), max(dist_min, dist_max)))

        # init camera position on the sphere
        cam_xyz = _camera_position_from_spherical_global(
            torch.tensor(dist_fixed, device=device),
            torch.tensor(float(init_elev_g), device=device),
            torch.tensor(float(init_azim_g), device=device),
            at_xyz,
        )
        # project exactly to sphere
        v0 = cam_xyz - at_xyz
        cam_xyz = at_xyz + (v0 / (torch.norm(v0) + 1e-8)) * torch.tensor(dist_fixed, device=device)

        def _project_to_sphere(xyz: torch.Tensor) -> torch.Tensor:
            v = xyz - at_xyz
            return at_xyz + (v / (torch.norm(v) + 1e-8)) * torch.tensor(dist_fixed, device=device)

        def _wrap_azim(a: float) -> float:
            return float(((a + 180.0) % 360.0) - 180.0)

        # trace
        trace_fp = None
        trace_writer = None
        if save_dir is not None:
            try:
                trace_fp = open(os.path.join(save_dir, f"{target_name}_greedy_axis_trace.csv"), "w", newline="", encoding="utf-8")
                trace_writer = csv.writer(trace_fp)
                trace_writer.writerow(["iter", "step_frac", "cur_loss", "best_loss", "chosen_dir", "chosen_k", "cand_loss", "dist", "elev_deg", "azim_deg"])
            except Exception as _e:
                print(f"[Warning] Failed to create greedy_axis_trace.csv: {_e}")
                trace_fp = None
                trace_writer = None

        # current eval
        dist0, elev0, azim0 = _spherical_from_camera_position_global(cam_xyz, at_xyz)
        azim0 = _wrap_azim(azim0)
        # Keep acceptance comparisons consistent:
        # - If we refine candidates by full loss, we keep cur_loss in FULL loss too.
        cur_ev = _evaluate_pose_sil_no_grad(
            mesh=mesh,
            silhouette_renderer=silhouette_renderer,
            image_ref_silhouette=image_ref_silhouette,
            at_xyz=at_xyz,
            dist_v=float(dist_fixed),
            elev_v=float(elev0),
            azim_v=float(azim0),
            device=device,
        ) if (axis_use_sil_only and (not axis_refine_full)) else _evaluate_pose_loss_no_grad(
            mesh, silhouette_renderer, rasterizer, image_ref_silhouette, ref_depth, ref_normals,
            at_xyz, float(dist_fixed), float(elev0), float(azim0), device
        )
        cur_loss = float(cur_ev["loss_total"])
        best_loss = float(cur_loss)
        best_xyz = cam_xyz.detach().clone()
        best_sph = (float(dist_fixed), float(elev0), float(azim0))
        loss_hist = []
        last_improve_it = 0

        # stack for optional backtracking
        stack = [(cam_xyz.detach().clone(), float(cur_loss), float(step_frac))]
        backtracks = 0

        dirs = [
            ("+x", torch.tensor([1.0, 0.0, 0.0], device=device)),
            ("-x", torch.tensor([-1.0, 0.0, 0.0], device=device)),
            ("+y", torch.tensor([0.0, 1.0, 0.0], device=device)),
            ("-y", torch.tensor([0.0, -1.0, 0.0], device=device)),
            ("+z", torch.tensor([0.0, 0.0, 1.0], device=device)),
            ("-z", torch.tensor([0.0, 0.0, -1.0], device=device)),
        ]

        for it in range(int(budget_iters)):
            loss_hist.append(float(best_loss))
            depth = len(stack) - 1
            if it % log_interval == 0:
                dcur, ecur, acur = _spherical_from_camera_position_global(cam_xyz, at_xyz)
                print(
                    f"[GreedyAxis] {target_name} iter {it}/{budget_iters} depth={depth} "
                    f"cur_loss={cur_loss:.6f} best_loss={best_loss:.6f} "
                    f"(dist,elev,azim)=({dist_fixed:.2f},{ecur:.2f},{acur:.2f}) step_frac={step_frac:.4f}"
                )

            # early stop if no improvement for a long time
            if no_improve_patience > 0 and (it - last_improve_it) >= no_improve_patience:
                if it % log_interval == 0:
                    print(f"[GreedyAxis] stop: no improvement for {no_improve_patience} iters.")
                break
            # depth limit -> backtrack
            if depth >= forward_max:
                if len(stack) > 1:
                    stack.pop()
                    cam_xyz, cur_loss, step_frac = stack[-1][0].clone(), float(stack[-1][1]), float(stack[-1][2])
                    backtracks += 1
                if backtracks >= back_max:
                    break
                continue

            step_len = float(step_frac) * float(dist_fixed)
            # build candidates: 6 directions * K steps
            cand_dist: list[float] = []
            cand_elev: list[float] = []
            cand_azim: list[float] = []
            cand_meta: list[tuple[str, int, torch.Tensor]] = []
            for name, dv in dirs:
                for k in range(1, max(1, steps_per_axis) + 1):
                    cand_xyz = _project_to_sphere(cam_xyz + dv * (step_len * float(k)))
                    _d, _e, _a = _spherical_from_camera_position_global(cand_xyz, at_xyz)
                    cand_dist.append(float(dist_fixed))
                    cand_elev.append(float(_e))
                    cand_azim.append(float(_wrap_azim(_a)))
                    cand_meta.append((name, int(k), cand_xyz))

            if axis_use_sil_only:
                evb_sil = _evaluate_pose_sil_batch_no_grad(
                    mesh=mesh,
                    silhouette_renderer=silhouette_renderer,
                    image_ref_silhouette=image_ref_silhouette,
                    at_xyz=at_xyz,
                    dist_list=cand_dist,
                    elev_list=cand_elev,
                    azim_list=cand_azim,
                    device=device,
                )
                losses_sil = evb_sil["loss_total"].detach().cpu().numpy().astype(np.float64)
                # preselect top-M
                m = min(int(axis_topm_full), int(losses_sil.size))
                order = np.argsort(losses_sil)[:m]
                if axis_refine_full and m > 0:
                    d2 = [cand_dist[int(j)] for j in order]
                    e2 = [cand_elev[int(j)] for j in order]
                    a2 = [cand_azim[int(j)] for j in order]
                    evb_full = _evaluate_pose_losses_batch_no_grad(
                        mesh, silhouette_renderer, rasterizer,
                        image_ref_silhouette, ref_depth, ref_normals,
                        at_xyz, d2, e2, a2, device
                    )
                    losses_full = evb_full["loss_total"].detach().cpu().numpy().astype(np.float64)
                    jj = int(np.argmin(losses_full)) if losses_full.size > 0 else 0
                    j_best = int(order[jj])
                    cand_best_loss = float(losses_full[jj])
                else:
                    j_best = int(order[0]) if m > 0 else 0
                    cand_best_loss = float(losses_sil[j_best]) if losses_sil.size > 0 else float("inf")
                chosen_dir, chosen_k, chosen_xyz = cand_meta[j_best]
            else:
                evb = _evaluate_pose_losses_batch_no_grad(
                    mesh, silhouette_renderer, rasterizer,
                    image_ref_silhouette, ref_depth, ref_normals,
                    at_xyz, cand_dist, cand_elev, cand_azim, device
                )
                losses = evb["loss_total"].detach().cpu().numpy().astype(np.float64)

                # pick best candidate overall
                j_best = int(np.argmin(losses))
                cand_best_loss = float(losses[j_best])
                chosen_dir, chosen_k, chosen_xyz = cand_meta[j_best]

            # decide accept
            improved = cand_best_loss < (cur_loss - min_delta)
            can_explore_uphill = allow_uphill and (cand_best_loss <= (best_loss + uphill_tol))
            if improved or can_explore_uphill:
                cam_xyz = chosen_xyz.detach().clone()
                cur_loss = float(cand_best_loss)
                stack.append((cam_xyz.detach().clone(), float(cur_loss), float(step_frac)))
                if cur_loss < (best_loss - min_delta):
                    best_loss = float(cur_loss)
                    best_xyz = cam_xyz.detach().clone()
                    bd, be, ba = _spherical_from_camera_position_global(best_xyz, at_xyz)
                    best_sph = (float(dist_fixed), float(be), float(_wrap_azim(ba)))
                    last_improve_it = int(it)

                if trace_writer is not None:
                    trace_writer.writerow([int(it), float(step_frac), float(cur_loss), float(best_loss), chosen_dir, int(chosen_k), float(cand_best_loss), float(dist_fixed), float(best_sph[1]), float(best_sph[2])])
                    trace_fp.flush()

                # Save strategy: save on improvement (default) and/or periodic.
                save_on_improve = int(os.environ.get("VAL_SAVE_ON_IMPROVE", "1")) != 0
                if save_dir is not None and save_interval > 0:
                    if (save_on_improve and (last_improve_it == it)) or (it % save_interval == 0):
                        _save_greedy_state(it, float(dist_fixed), float(best_sph[1]), float(best_sph[2]), float(best_loss), save_prefix="greedy_axis_best")
                continue

            # no accept: shrink step, if too small -> try backtrack else stop
            step_frac = float(step_frac) * float(shrink)
            if trace_writer is not None:
                trace_writer.writerow([int(it), float(step_frac), float(cur_loss), float(best_loss), "shrink", 0, float(cand_best_loss), float(dist_fixed), float(best_sph[1]), float(best_sph[2])])
                trace_fp.flush()

            if step_frac <= min_step_frac:
                # optional backtrack
                if len(stack) > 1:
                    stack.pop()
                    cam_xyz, cur_loss, step_frac = stack[-1][0].clone(), float(stack[-1][1]), float(stack[-1][2] * shrink)
                    backtracks += 1
                    if backtracks >= back_max:
                        break
                    continue
                break

        if trace_fp is not None:
            try:
                trace_fp.close()
            except Exception:
                pass

        # best pose from best_sph
        Rb, Tb = look_at_view_transform(
            dist=torch.tensor([float(dist_fixed)], device=device),
            elev=torch.tensor([float(best_sph[1])], device=device),
            azim=torch.tensor([float(best_sph[2])], device=device),
            at=look_at_at.view(1, 3),
            device=device,
        )
        pose_best = np.eye(4, dtype=np.float32)
        pose_best[:3, :3] = Rb[0].detach().cpu().numpy().astype(np.float32)
        pose_best[:3, 3] = Tb[0].detach().cpu().numpy().astype(np.float32)
        return pose_best, loss_hist, (float(dist_fixed), float(best_sph[1]), float(best_sph[2]), float(best_loss))

    # If user chooses greedy only, run it and exit early.
    if opt_method == "greedy":
        greedy_iters = int(os.environ.get("VAL_GREEDY_MAX_ITERS", str(max_iterations)))
        print(f"[Opt] Using GREEDY optimizer (VAL_OPTIM_METHOD=greedy), iters={greedy_iters}")
        best_pose, loss_hist, (bd, be, ba, bl) = greedy_local_search(float(init_dist), float(init_elev), float(init_azim), greedy_iters)
        if save_dir is not None:
            try:
                np.savetxt(os.path.join(save_dir, "best_pose_4x4.txt"), best_pose, fmt="%.8f")
                np.save(os.path.join(save_dir, "best_pose_4x4.npy"), best_pose.astype(np.float32))
                # Also save best_* renders (sil/depth/normals/rgb) for consistency with other methods.
                _render_and_save_best_artifacts(
                    mesh=mesh,
                    silhouette_renderer=silhouette_renderer,
                    rasterizer=rasterizer,
                    phong_renderer=phong_renderer,
                    at_xyz=look_at_at.view(3),
                    dist_v=float(bd),
                    elev_v=float(be),
                    azim_v=float(ba),
                    device=device,
                    save_dir=save_dir,
                    prefix="best",
                )
                # Compatibility alias
                best_soft = os.path.join(save_dir, "best_silhouette_soft.png")
                best_png = os.path.join(save_dir, "best_silhouette.png")
                if os.path.exists(best_soft) and (not os.path.exists(best_png)):
                    shutil.copy2(best_soft, best_png)
                with open(os.path.join(save_dir, "best_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "method": "greedy",
                            "best_loss": float(bl),
                            "best_spherical": {"dist": float(bd), "elev_deg": float(be), "azim_deg": float(ba)},
                            "note": "Greedy accepts only improvements, so accepted loss is monotone non-increasing. Still non-convex => no global-opt guarantee.",
                        },
                        f,
                        indent=2,
                    )
            except Exception as _e:
                print(f"[Warning] Failed to save greedy best_*: {_e}")
        return best_pose, loss_hist

    if opt_method == "greedy_bt":
        budget = int(os.environ.get("VAL_BT_MAX_ITERS", str(max_iterations)))
        print(f"[Opt] Using GREEDY_BACKTRACK optimizer (VAL_OPTIM_METHOD=greedy_bt), iters={budget}")
        best_pose, loss_hist, (bd, be, ba, bl) = greedy_backtracking_search(float(init_dist), float(init_elev), float(init_azim), budget)
        if save_dir is not None:
            try:
                np.savetxt(os.path.join(save_dir, "best_pose_4x4.txt"), best_pose, fmt="%.8f")
                np.save(os.path.join(save_dir, "best_pose_4x4.npy"), best_pose.astype(np.float32))
                _render_and_save_best_artifacts(
                    mesh=mesh,
                    silhouette_renderer=silhouette_renderer,
                    rasterizer=rasterizer,
                    phong_renderer=phong_renderer,
                    at_xyz=look_at_at.view(3),
                    dist_v=float(bd),
                    elev_v=float(be),
                    azim_v=float(ba),
                    device=device,
                    save_dir=save_dir,
                    prefix="best",
                )
                best_soft = os.path.join(save_dir, "best_silhouette_soft.png")
                best_png = os.path.join(save_dir, "best_silhouette.png")
                if os.path.exists(best_soft) and (not os.path.exists(best_png)):
                    shutil.copy2(best_soft, best_png)
                with open(os.path.join(save_dir, "best_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "method": "greedy_bt",
                            "best_loss": float(bl),
                            "best_spherical": {"dist": float(bd), "elev_deg": float(be), "azim_deg": float(ba)},
                            "bt": {
                                "forward_steps": int(os.environ.get("VAL_BT_FORWARD_STEPS", "64")),
                                "backtrack_steps": int(os.environ.get("VAL_BT_BACKTRACK_STEPS", "64")),
                                "uphill_tol": float(os.environ.get("VAL_BT_UPHILL_TOL", "0.05")),
                            },
                            "note": "GreedyBT explores with backtracking; not a global-opt guarantee. Use Top-K seeds for better global coverage.",
                        },
                        f,
                        indent=2,
                    )
            except Exception as _e:
                print(f"[Warning] Failed to save greedy_bt best_*: {_e}")
        return best_pose, loss_hist

    if opt_method == "greedy_axis":
        budget = int(os.environ.get("VAL_AXIS_MAX_ITERS", str(max_iterations)))
        print(f"[Opt] Using GREEDY_AXIS optimizer (VAL_OPTIM_METHOD=greedy_axis), iters={budget}")
        best_pose, loss_hist, (bd, be, ba, bl) = greedy_axis_sphere_search(float(init_dist), float(init_elev), float(init_azim), budget)
        if save_dir is not None:
            try:
                np.savetxt(os.path.join(save_dir, "best_pose_4x4.txt"), best_pose, fmt="%.8f")
                np.save(os.path.join(save_dir, "best_pose_4x4.npy"), best_pose.astype(np.float32))
                _render_and_save_best_artifacts(
                    mesh=mesh,
                    silhouette_renderer=silhouette_renderer,
                    rasterizer=rasterizer,
                    phong_renderer=phong_renderer,
                    at_xyz=look_at_at.view(3),
                    dist_v=float(bd),
                    elev_v=float(be),
                    azim_v=float(ba),
                    device=device,
                    save_dir=save_dir,
                    prefix="best",
                )
                best_soft = os.path.join(save_dir, "best_silhouette_soft.png")
                best_png = os.path.join(save_dir, "best_silhouette.png")
                if os.path.exists(best_soft) and (not os.path.exists(best_png)):
                    shutil.copy2(best_soft, best_png)
                with open(os.path.join(save_dir, "best_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "method": "greedy_axis",
                            "best_loss": float(bl),
                            "best_spherical": {"dist": float(bd), "elev_deg": float(be), "azim_deg": float(ba)},
                            "axis": {
                                "steps_per_dir": int(os.environ.get("VAL_AXIS_STEPS_PER_DIR", "6")),
                                "step_frac": float(os.environ.get("VAL_AXIS_STEP_FRAC", "0.03")),
                                "shrink": float(os.environ.get("VAL_AXIS_SHRINK", "0.5")),
                                "min_step_frac": float(os.environ.get("VAL_AXIS_MIN_STEP_FRAC", "0.003")),
                                "forward_steps": int(os.environ.get("VAL_AXIS_FORWARD_STEPS", "200")),
                                "backtrack_steps": int(os.environ.get("VAL_AXIS_BACKTRACK_STEPS", "200")),
                                "allow_uphill": bool(int(os.environ.get("VAL_AXIS_ALLOW_UPHILL", "0")) != 0),
                                "uphill_tol": float(os.environ.get("VAL_AXIS_UPHILL_TOL", "0.05")),
                            },
                            "note": "GreedyAxis probes multi-step along ±x/±y/±z on a fixed-radius sphere, then chooses the best direction.",
                        },
                        f,
                        indent=2,
                    )
            except Exception as _e:
                print(f"[Warning] Failed to save greedy_axis best_*: {_e}")
        return best_pose, loss_hist

    # Hybrid: greedy warmup, then continue with Adam using the improved init (dist,elev,azim).
    if opt_method == "hybrid":
        greedy_warm = int(os.environ.get("VAL_GREEDY_WARMUP_ITERS", "200"))
        print(f"[Opt] Using HYBRID optimizer: greedy warmup {greedy_warm} iters then Adam (VAL_OPTIM_METHOD=hybrid)")
        _pose_tmp, _hist_tmp, (bd, be, ba, _bl) = greedy_local_search(float(init_dist), float(init_elev), float(init_azim), greedy_warm)
        init_dist, init_elev, init_azim = float(bd), float(be), float(ba)

    class SphericalCameraModel(torch.nn.Module):
        def __init__(
            self,
            meshes,
            renderer_sil,
            rasterizer_aux,
            image_ref_silhouette,
            ref_depth,
            ref_normals,
            at,
            init_dist_v,
            init_elev_v,
            init_azim_v,
            cam_dist_base,
        ):
            super().__init__()
            self.meshes = meshes
            if hasattr(meshes, "device"):
                self.device = meshes.device
            elif hasattr(meshes, "verts_packed"):
                self.device = meshes.verts_packed().device
            else:
                self.device = image_ref_silhouette.device
            self.renderer = renderer_sil
            self.rasterizer = rasterizer_aux
            ref = image_ref_silhouette
            if ref.dim() == 3 and ref.shape[-1] == 4:
                ref = ref[..., 3]
            self.register_buffer("image_ref", ref.squeeze())
            self.register_buffer("ref_depth", ref_depth.squeeze())
            self.register_buffer("ref_normals", ref_normals.squeeze())
            self.register_buffer("at", at.squeeze().to(self.device))
            # progress in [0,1] used for optional loss-weight scheduling
            self.register_buffer("progress", torch.tensor(0.0, dtype=torch.float32, device=self.device))
            # Reference init distance (as computed from first-frame setup; may be outside the constrained range)
            self.register_buffer("init_dist_ref", torch.tensor(float(init_dist_v), dtype=torch.float32, device=self.device))
            self.register_buffer("init_elev", torch.tensor(float(init_elev_v), dtype=torch.float32, device=self.device))
            self.register_buffer("init_azim", torch.tensor(float(init_azim_v), dtype=torch.float32, device=self.device))
            # NOTE:
            # - cam_dist_base is used for GT sampling (based on first-frame camera_distance).
            # - For optimization stability, we constrain *optimized dist* around init_dist_v by default,
            #   so init starts inside the feasible range (avoids sigmoid saturation where dist won't move).
            self.register_buffer("cam_dist_base", torch.tensor(float(cam_dist_base), dtype=torch.float32, device=self.device))
            self.register_buffer("dist_base_opt", torch.tensor(float(init_dist_v), dtype=torch.float32, device=self.device))

            # dist constrained to [scale_min, scale_max] * dist_base_opt (DEFAULT: init_dist)
            self.dist_scale_min = float(os.environ.get("VAL_GT_DIST_SCALE_MIN", "0.15"))
            self.dist_scale_max = float(os.environ.get("VAL_GT_DIST_SCALE_MAX", "0.85"))
            dist_base_mode = os.environ.get("VAL_DIST_BASE_MODE", "init").strip().lower()  # init|camera
            if dist_base_mode == "camera":
                base = float(cam_dist_base)
            else:
                base = float(init_dist_v)
            self.dist_min = float(self.dist_scale_min) * base
            self.dist_max = float(self.dist_scale_max) * base

            # optional hard clamp overrides
            dist_min_abs = os.environ.get("VAL_GT_DIST_MIN", "")
            dist_max_abs = os.environ.get("VAL_GT_DIST_MAX", "")
            if dist_min_abs != "":
                self.dist_min = max(self.dist_min, float(dist_min_abs))
            if dist_max_abs != "":
                self.dist_max = min(self.dist_max, float(dist_max_abs))

            # Ensure init dist is inside (dist_min, dist_max) with a safety margin.
            # Otherwise sigmoid can saturate at the boundary and dist appears "stuck".
            margin_ratio = float(os.environ.get("VAL_DIST_INIT_MARGIN_RATIO", "0.05"))
            margin = max(1e-3, margin_ratio * max(1e-6, (self.dist_max - self.dist_min)))
            init_dist_interior = float(np.clip(float(init_dist_v), self.dist_min + margin, self.dist_max - margin))
            # This is the optimization-start dist (used for delta logging and initialization).
            self.register_buffer("init_dist", torch.tensor(init_dist_interior, dtype=torch.float32, device=self.device))

            # elev/azim constrained to the same ranges used for GT sampling (defaults are wide)
            self.elev_min = float(os.environ.get("VAL_GT_ELEV_MIN", "5.0"))
            self.elev_max = float(os.environ.get("VAL_GT_ELEV_MAX", "85.0"))
            self.azim_min = float(os.environ.get("VAL_GT_AZIM_MIN", "-180.0"))
            self.azim_max = float(os.environ.get("VAL_GT_AZIM_MAX", "180.0"))

            # Precompute init view direction for hemisphere constraint (helps avoid mirrored/opposite-side minima).
            # Direction is in WORLD space: normalize(camera_xyz - at).
            init_cam_xyz = _camera_position_from_spherical(self.init_dist, self.init_elev, self.init_azim, self.at)
            init_dir = init_cam_xyz - self.at
            init_dir = init_dir / torch.norm(init_dir).clamp(min=1e-8)
            self.register_buffer("init_view_dir", init_dir)

            # Initialize raw parameters so that mapped values start near init (stable)
            # dist: invert sigmoid approximately
            init_dist_clamped = float(np.clip(float(init_dist_interior), self.dist_min + 1e-6, self.dist_max - 1e-6))
            p = (init_dist_clamped - self.dist_min) / (self.dist_max - self.dist_min + 1e-12)
            raw_dist0 = float(np.log(p / (1.0 - p + 1e-12) + 1e-12))
            raw_dist0 = float(np.log(p / (1.0 - p + 1e-12) + 1e-12))
            self.raw_dist = torch.nn.Parameter(torch.tensor(raw_dist0, dtype=torch.float32, device=self.device))
            # elev/azim: start as 0 so tanh maps to mid; we instead bias to init via inverse tanh
            def inv_tanh_to_range(v, lo, hi):
                t = (2.0 * (v - 0.5 * (hi + lo)) / (hi - lo + 1e-12))
                t = float(np.clip(t, -0.999, 0.999))
                return float(0.5 * np.log((1 + t) / (1 - t)))

            self.raw_elev = torch.nn.Parameter(torch.tensor(inv_tanh_to_range(float(init_elev_v), self.elev_min, self.elev_max), dtype=torch.float32, device=self.device))
            self.raw_azim = torch.nn.Parameter(torch.tensor(inv_tanh_to_range(float(init_azim_v), self.azim_min, self.azim_max), dtype=torch.float32, device=self.device))

        def set_progress(self, it: int, iters: int):
            try:
                denom = max(1, int(iters) - 1)
                p = float(it) / float(denom)
                self.progress.fill_(float(np.clip(p, 0.0, 1.0)))
            except Exception:
                pass

        def forward(self):
            dist = _sigmoid_to_range(self.raw_dist, self.dist_min, self.dist_max)
            elev = _tanh_to_range(self.raw_elev, self.elev_min, self.elev_max)
            azim = _tanh_to_range(self.raw_azim, self.azim_min, self.azim_max)
            R, T = look_at_view_transform(dist=dist[None], elev=elev[None], azim=azim[None], at=self.at[None, :], device=self.device)
            image = self.renderer(meshes_world=self.meshes, R=R, T=T)
            alpha = _postprocess_silhouette_alpha(torch.clamp(image[0, ..., 3], 0, 1))
            # Optional: make silhouette "sharper" towards the end of optimization.
            # This is a practical way to reduce the residual caused by soft edge representation.
            # Note: for nvdiffrast, edge gradients mainly come from antialias; sharpening helps lock-in late.
            if int(os.environ.get("VAL_SIL_SHARPEN", "1")) != 0:
                tau0 = float(os.environ.get("VAL_SIL_SHARP_TAU_START", "0.08"))
                tau1 = float(os.environ.get("VAL_SIL_SHARP_TAU_END", "0.02"))
                try:
                    p = float(self.progress.detach().cpu().item())
                except Exception:
                    p = 0.0
                tau = (1.0 - p) * float(tau0) + p * float(tau1)
                tau = max(1e-4, float(tau))
                alpha = torch.sigmoid((alpha - 0.5) / tau)
            ref = torch.clamp(self.image_ref, 0, 1)
            # Use mean to avoid huge scale and make "no change" easier to spot numerically.
            loss_mse = torch.mean((alpha - ref) ** 2)
            # Small IoU term helps when silhouettes are sparse.
            inter = (alpha * ref).sum()
            union = alpha.sum() + ref.sum() - inter
            loss_iou = 1.0 - (inter / (union + 1e-8))
            w_iou = float(os.environ.get("VAL_W_IOU", "0.1"))
            # IMPORTANT (Option A):
            # Use *mean-scale* silhouette by default so it doesn't dominate depth/normals by 1e2~1e4.
            # If you really want tutorial-like magnitude, set VAL_LOSS_SCALE=auto (== H*W).
            scale_cfg = os.environ.get("VAL_LOSS_SCALE", "1.0")
            if scale_cfg.strip().lower() == "auto":
                h = int(ref.shape[0])
                w = int(ref.shape[1])
                loss_scale = float(h * w)
            else:
                loss_scale = float(scale_cfg)
            loss_sil = (1.0 - w_iou) * (loss_mse * loss_scale) + w_iou * loss_iou

            # Depth + normals losses (to disambiguate pose and converge back to GT)
            cameras_cur = FoVPerspectiveCameras(device=self.device, R=R, T=T)
            fr = self.rasterizer(meshes_world=self.meshes, cameras=cameras_cur)
            vis_mask = (fr.pix_to_face[..., 0] >= 0).float()[0]  # (H,W)
            depth = (fr.zbuf[..., 0][0] * vis_mask)  # background=0

            ref_depth = self.ref_depth
            ref_normals = self.ref_normals
            valid = (vis_mask > 0.5) & (ref_depth > 0)

            if valid.any():
                beta = float(os.environ.get("VAL_DEPTH_BETA", "0.05"))
                # Normalize depth by reference mean depth (Option A: scale alignment with silhouette).
                # This makes depth loss roughly O(1) instead of O(1e2~1e3) when units are pixels/world.
                depth_norm_eps = float(os.environ.get("VAL_DEPTH_NORM_EPS", "1e-6"))
                ref_scale = ref_depth[valid].mean().detach().clamp(min=depth_norm_eps)
                depth_n = depth / ref_scale
                ref_depth_n = ref_depth / ref_scale
                loss_depth = torch.nn.functional.smooth_l1_loss(depth_n[valid], ref_depth_n[valid], beta=beta, reduction="mean")

                verts_normals = self.meshes.verts_normals_packed()
                faces = self.meshes.faces_packed()
                verts_normals_view = torch.matmul(verts_normals, R[0].T)
                faces_verts_normals = verts_normals_view[faces]
                pix_normals = interpolate_face_attributes(fr.pix_to_face, fr.bary_coords, faces_verts_normals)
                normals = pix_normals[..., 0, :][0]
                normals = torch.nn.functional.normalize(normals, dim=-1) * vis_mask.unsqueeze(-1)

                cos = (normals * ref_normals).sum(dim=-1).clamp(-1, 1)
                loss_normal = (1.0 - cos)[valid].mean()
            else:
                loss_depth = torch.tensor(0.0, device=self.device)
                loss_normal = torch.tensor(0.0, device=self.device)

            # -------------------------
            # Anti-mirror / anti-opposite constraints (soft priors)
            # -------------------------
            # 1) Hemisphere constraint: keep camera direction in the same hemisphere as the init view direction.
            #    This reduces "flipping to the opposite side" solutions which can look like mirrored poses.
            #    penalty = ReLU(-dot(dir, init_dir))^2
            w_hemi = float(os.environ.get("VAL_W_HEMISPHERE", "0.0"))
            hemi_pen = torch.tensor(0.0, device=self.device)
            hemi_dot = torch.tensor(1.0, device=self.device)
            if w_hemi > 0:
                cam_xyz = _camera_position_from_spherical(dist, elev, azim, self.at)
                v = cam_xyz - self.at
                v = v / torch.norm(v).clamp(min=1e-8)
                hemi_dot = (v * self.init_view_dir).sum().clamp(-1.0, 1.0)
                hemi_pen = torch.relu(-hemi_dot) ** 2

            # 2) Prior to stay near init (optional): helps avoid ambiguous minima when silhouettes are similar.
            #    Use circular distance for azim to handle wrap-around.
            w_prior = float(os.environ.get("VAL_W_SPHERE_PRIOR", "0.0"))
            prior_pen = torch.tensor(0.0, device=self.device)
            if w_prior > 0:
                # Sigmas (in degrees for angles; relative fraction for dist)
                sigma_elev = float(os.environ.get("VAL_PRIOR_SIGMA_ELEV_DEG", "15.0"))
                sigma_azim = float(os.environ.get("VAL_PRIOR_SIGMA_AZIM_DEG", "25.0"))
                sigma_dist_frac = float(os.environ.get("VAL_PRIOR_SIGMA_DIST_FRAC", "0.15"))
                sigma_dist = torch.tensor(float(self.init_dist.item()) * sigma_dist_frac, device=self.device).clamp(min=1e-6)

                de = (elev - self.init_elev) / max(1e-6, sigma_elev)

                # circular azim diff in radians
                a = azim * math.pi / 180.0
                a0 = self.init_azim * math.pi / 180.0
                da = torch.atan2(torch.sin(a - a0), torch.cos(a - a0))  # [-pi, pi]
                da_deg = da * 180.0 / math.pi
                da_n = da_deg / max(1e-6, sigma_azim)

                dd = (dist - self.init_dist) / sigma_dist
                prior_pen = dd.pow(2) + de.pow(2) + da_n.pow(2)

            # Loss weights
            # - Without schedule: Option A defaults (depth/normals emphasized, silhouette kept but not dominating)
            # - With schedule: start with stronger silhouette to create overlap, then shift to depth/normals to lock pose.
            use_sched = int(os.environ.get("VAL_LOSS_SCHEDULE", "1")) != 0
            if use_sched:
                p = float(self.progress.detach().cpu().item())
                w_sil0 = float(os.environ.get("VAL_W_SIL_START", "2.0"))
                w_sil1 = float(os.environ.get("VAL_W_SIL_END", "0.2"))
                w_depth0 = float(os.environ.get("VAL_W_DEPTH_START", "0.5"))
                w_depth1 = float(os.environ.get("VAL_W_DEPTH_END", "2.0"))
                w_norm0 = float(os.environ.get("VAL_W_NORMAL_START", "0.5"))
                w_norm1 = float(os.environ.get("VAL_W_NORMAL_END", "1.0"))
                w_sil = (1.0 - p) * w_sil0 + p * w_sil1
                w_depth = (1.0 - p) * w_depth0 + p * w_depth1
                w_normal = (1.0 - p) * w_norm0 + p * w_norm1
            else:
                w_sil = float(os.environ.get("VAL_W_SIL", "0.2"))
                w_depth = float(os.environ.get("VAL_W_DEPTH", "2.0"))
                w_normal = float(os.environ.get("VAL_W_NORMAL", "1.0"))
            if int(os.environ.get("VAL_REFINE_SIL_ONLY", "0")) != 0:
                w_depth = 0.0
                w_normal = 0.0
            loss = w_sil * loss_sil + w_depth * loss_depth + w_normal * loss_normal + w_hemi * hemi_pen + w_prior * prior_pen
            return (
                loss,
                image,
                R[0],
                T[0],
                dist,
                elev,
                azim,
                loss_sil,
                loss_depth,
                loss_normal,
                hemi_dot,
                hemi_pen,
                prior_pen,
            )

    model = SphericalCameraModel(
        mesh,
        silhouette_renderer,
        rasterizer,
        image_ref_silhouette,
        ref_depth,
        ref_normals,
        at=look_at_at,
        init_dist_v=float(init_dist),
        init_elev_v=float(init_elev),
        init_azim_v=float(init_azim),
        cam_dist_base=float(camera_distance),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = None
    sched_name = os.environ.get("VAL_SCHED", "cosine").strip().lower()
    if sched_name == "cosine":
        try:
            from torch.optim.lr_scheduler import CosineAnnealingLR
            sched = CosineAnnealingLR(opt, T_max=max_iterations, eta_min=lr * 0.1)
        except Exception:
            sched = None
    elif sched_name == "step":
        try:
            from torch.optim.lr_scheduler import StepLR
            sched = StepLR(opt, step_size=max(1, max_iterations // 3), gamma=0.5)
        except Exception:
            sched = None

    loss_history = []
    best_loss = float("inf")
    best_pose = None
    best_iter = -1
    save_interval = int(os.environ.get("VAL_SAVE_INTERVAL", "50"))
    log_interval = int(os.environ.get("VAL_LOG_INTERVAL", "50"))
    history_interval = int(os.environ.get("VAL_HISTORY_INTERVAL", str(log_interval)))
    # Early stop to save time on "easy" GT while allowing hard cases to run longer.
    early_stop_enabled = int(os.environ.get("VAL_EARLY_STOP", "1")) != 0
    early_stop_warmup = int(os.environ.get("VAL_EARLY_STOP_WARMUP", "200"))
    early_stop_patience = int(os.environ.get("VAL_EARLY_STOP_PATIENCE", "400"))
    early_stop_min_delta = float(os.environ.get("VAL_EARLY_STOP_MIN_DELTA", "1e-5"))
    early_stop_triggered = False
    early_stop_iter = -1

    # Structured pose/loss trace for later analysis.
    hist_fp = None
    hist_writer = None
    if save_dir is not None:
        try:
            os.makedirs(save_dir, exist_ok=True)
            hist_fp = open(os.path.join(save_dir, f"{target_name}_pose_history.csv"), "w", newline="", encoding="utf-8")
            hist_writer = csv.writer(hist_fp)
            hist_writer.writerow(
                [
                    "iter",
                    "lr",
                    "loss_total_pre",
                    "loss_total_post",
                    "loss_sil_post",
                    "loss_depth_post",
                    "loss_normal_post",
                    "dist",
                    "elev_deg",
                    "azim_deg",
                ]
            )
        except Exception as _e:
            print(f"[Warning] Failed to create pose_history.csv: {_e}")
            hist_fp = None
            hist_writer = None

    # Sanity check: ensure we are optimizing camera position (hence camera RT)
    trainable = [(n, p.shape) for n, p in model.named_parameters() if p.requires_grad]
    print(f"[Opt] Trainable parameters: {trainable}")
    if len(trainable) == 0:
        print("[Warning] No trainable parameters found. Optimization will NOT update camera pose.")

    for it in range(max_iterations):
        # update schedule progress (if enabled)
        model.set_progress(it, max_iterations)
        opt.zero_grad()
        (
            loss,
            image_rgba,
            R_cur,
            T_cur,
            dist_cur,
            elev_cur,
            azim_cur,
            loss_sil,
            loss_depth,
            loss_normal,
            hemi_dot,
            hemi_pen,
            prior_pen,
        ) = model()
        loss.backward()

        # Gradient sanity checks
        if it % log_interval == 0:
            try:
                if model.raw_dist.grad is not None:
                    gnorm = float(torch.sqrt(model.raw_dist.grad.pow(2) + model.raw_elev.grad.pow(2) + model.raw_azim.grad.pow(2)).detach().cpu().item())
                    print(f"[Opt] iter {it}: ||grad(raw_params)||={gnorm:.6e}")
                print(f"[Opt] iter {it}: dist/elev/azim = {float(dist_cur.item()):.3f}, {float(elev_cur.item()):.3f}, {float(azim_cur.item()):.3f}")
                d_dist = float((dist_cur - model.init_dist).abs().detach().cpu().item())
                d_elev = float((elev_cur - model.init_elev).abs().detach().cpu().item())
                d_azim = float((azim_cur - model.init_azim).abs().detach().cpu().item())
                print(f"[Opt] iter {it}: |delta(dist,elev,azim)| = {d_dist:.3f}, {d_elev:.3f}, {d_azim:.3f}")
                w_hemi = float(os.environ.get("VAL_W_HEMISPHERE", "0.0"))
                w_prior = float(os.environ.get("VAL_W_SPHERE_PRIOR", "0.0"))
                print(
                    f"[Opt] iter {it}: loss_sil={float(loss_sil.detach().cpu().item()):.6f} "
                    f"loss_depth={float(loss_depth.detach().cpu().item()):.6f} "
                    f"loss_normal={float(loss_normal.detach().cpu().item()):.6f} "
                    f"hemi_dot={float(hemi_dot.detach().cpu().item()):.4f} hemi_pen={float(hemi_pen.detach().cpu().item()):.6f} (w={w_hemi:.3f}) "
                    f"prior_pen={float(prior_pen.detach().cpu().item()):.6f} (w={w_prior:.3f})"
                )
            except Exception as _e:
                print(f"[Warning] Failed to log grads at iter {it}: {_e}")

        # Guard against inf/nan gradients; if detected, skip the step and reduce LR.
        bad_grad = False
        for p in [model.raw_dist, model.raw_elev, model.raw_azim]:
            if p.grad is not None and (not torch.isfinite(p.grad).all()):
                bad_grad = True
        if bad_grad or (not torch.isfinite(loss).all()):
            if it % log_interval == 0:
                print("[Warning] Non-finite loss/grad detected. Skipping step and reducing lr by 0.5.")
            for pg in opt.param_groups:
                pg["lr"] = pg["lr"] * 0.5
            opt.zero_grad(set_to_none=True)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(os.environ.get("VAL_CLIP_NORM", "10.0")))
            opt.step()
            if sched is not None:
                sched.step()

        # NOTE: loss before step (can go up/down). We'll also log loss AFTER step to avoid confusion.
        lv_pre = float(loss.detach().item())
        loss_history.append(lv_pre)
        if lv_pre < (best_loss - early_stop_min_delta):
            best_loss = lv_pre
            # best pose = 4x4 world-to-view from current R,T
            pose_best = torch.eye(4, device=device)
            pose_best[:3, :3] = R_cur
            pose_best[:3, 3] = T_cur
            best_pose = pose_best.detach().cpu().numpy()
            best_iter = it

        # Early stopping: after warmup, if no improvement for patience steps, stop.
        if (
            early_stop_enabled
            and it >= early_stop_warmup
            and best_iter >= 0
            and (it - best_iter) >= early_stop_patience
        ):
            early_stop_triggered = True
            early_stop_iter = it
            print(
                f"[EarlyStop] {target_name}: no improvement > {early_stop_min_delta:.2e} "
                f"for {early_stop_patience} iters (best_iter={best_iter}, best_loss={best_loss:.6f}). Stop at iter {it}."
            )
            break

        # Save intermediate results every N iterations.
        # IMPORTANT: re-render AFTER opt.step so saved silhouettes reflect updated camera pose.
        if save_dir is not None and save_interval > 0 and (it % save_interval == 0 or it == max_iterations - 1):
            try:
                os.makedirs(save_dir, exist_ok=True)
                with torch.no_grad():
                    (
                        loss_s,
                        image_rgba_s,
                        R_s,
                        T_s,
                        dist_s,
                        elev_s,
                        azim_s,
                        loss_sil_s,
                        loss_depth_s,
                        loss_normal_s,
                        hemi_dot_s,
                        hemi_pen_s,
                        prior_pen_s,
                    ) = model()
                    pose_4x4 = torch.eye(4, device=device)
                    pose_4x4[:3, :3] = R_s
                    pose_4x4[:3, 3] = T_s
                    cameras_cur = FoVPerspectiveCameras(device=device, R=R_s.unsqueeze(0), T=T_s.unsqueeze(0))

                    # silhouette
                    alpha = torch.clamp(image_rgba_s[0, ..., 3], 0, 1)
                    _save_silhouette_pair(save_dir, f"iter_{it:04d}", alpha)
                    # diff heatmap (makes small changes visible)
                    diff = torch.abs(alpha - model.image_ref)
                    cv2.imwrite(
                        os.path.join(save_dir, f"iter_{it:04d}_silhouette_absdiff.png"),
                        (torch.clamp(diff / (diff.max() + 1e-8), 0, 1).cpu().numpy() * 255).astype(np.uint8),
                    )
                    # side-by-side: target vs current vs absdiff (very explicit)
                    tgt_u8 = (torch.clamp(model.image_ref, 0, 1).cpu().numpy() * 255).astype(np.uint8)
                    cur_u8 = (alpha.cpu().numpy() * 255).astype(np.uint8)
                    diff_u8 = (torch.clamp(diff / (diff.max() + 1e-8), 0, 1).cpu().numpy() * 255).astype(np.uint8)
                    diff_color = cv2.applyColorMap(diff_u8, cv2.COLORMAP_JET)
                    side = np.concatenate(
                        [cv2.cvtColor(tgt_u8, cv2.COLOR_GRAY2BGR), cv2.cvtColor(cur_u8, cv2.COLOR_GRAY2BGR), diff_color],
                        axis=1,
                    )
                    cv2.imwrite(os.path.join(save_dir, f"iter_{it:04d}_target_curr_absdiff.png"), side)

                    # depth
                    fr = rasterizer(meshes_world=mesh, cameras=cameras_cur)
                    vis_mask = (fr.pix_to_face[..., 0] >= 0).float()[0]
                    depth = (fr.zbuf[..., 0][0] * vis_mask)
                    cv2.imwrite(os.path.join(save_dir, f"iter_{it:04d}_depth.png"), _depth_to_vis(depth))

                    # normals
                    verts_normals = mesh.verts_normals_packed()
                    faces = mesh.faces_packed()
                    verts_normals_view = torch.matmul(verts_normals, R_s.T)
                    faces_verts_normals = verts_normals_view[faces]
                    pix_normals = interpolate_face_attributes(fr.pix_to_face, fr.bary_coords, faces_verts_normals)
                    normals = pix_normals[..., 0, :][0]
                    normals = torch.nn.functional.normalize(normals, dim=-1) * vis_mask.unsqueeze(-1)
                    cv2.imwrite(os.path.join(save_dir, f"iter_{it:04d}_normals.png"), _normal_to_vis(normals))

                    # rgb (phong) for visualization
                    rgb = phong_renderer(meshes_world=mesh, cameras=cameras_cur)[0, ..., :3]
                    cv2.imwrite(os.path.join(save_dir, f"iter_{it:04d}_rgb.png"), cv2.cvtColor(_to_uint8_rgb(rgb), cv2.COLOR_RGB2BGR))

                    # pose
                    np.savetxt(
                        os.path.join(save_dir, f"iter_{it:04d}_pose_4x4.txt"),
                        pose_4x4.cpu().numpy(),
                        fmt="%.8f",
                    )
                    # Save spherical parameters (directly optimized)
                    with open(os.path.join(save_dir, f"iter_{it:04d}_camera_spherical.txt"), "w", encoding="utf-8") as f:
                        f.write(f"dist={float(dist_s.item()):.6f}\n")
                        f.write(f"elev_deg={float(elev_s.item()):.6f}\n")
                        f.write(f"azim_deg={float(azim_s.item()):.6f}\n")
                        f.write(f"dist_range=[{model.dist_min:.6f}, {model.dist_max:.6f}]\n")

                    # Derived camera position (xyz) for convenience/debug
                    cam_xyz = _camera_position_from_spherical(dist_s, elev_s, azim_s, model.at)
                    cam_pos_np = cam_xyz.detach().cpu().numpy().astype(np.float64)
                    np.savetxt(os.path.join(save_dir, f"iter_{it:04d}_camera_position_xyz.txt"), cam_pos_np[None, :], fmt="%.8f")

                    # delta magnitude (in spherical space)
                    d_dist = float((dist_s - model.init_dist).abs().cpu().item())
                    d_elev = float((elev_s - model.init_elev).abs().cpu().item())
                    d_azim = float((azim_s - model.init_azim).abs().cpu().item())
                    # quick numeric sanity for "no change" complaints
                    diff_mean = float(diff.mean().cpu().item())
                    print(
                        f"[Save] iter {it}: loss={float(loss_s.item()):.6f}, absdiff_mean={diff_mean:.6e}, "
                        f"|d(dist,elev,azim)|=({d_dist:.3f},{d_elev:.3f},{d_azim:.3f}), "
                        f"loss_sil={float(loss_sil_s.item()):.6f}, loss_depth={float(loss_depth_s.item()):.6f}, loss_normal={float(loss_normal_s.item()):.6f}, "
                        f"hemi_dot={float(hemi_dot_s.item()):.4f}, hemi_pen={float(hemi_pen_s.item()):.6f}, prior_pen={float(prior_pen_s.item()):.6f}"
                    )
            except Exception as _e:
                print(f"[Warning] Failed to save intermediate results at iter {it}: {_e}")
        if it % 20 == 0 or it == max_iterations - 1:
            # Log loss AFTER step to avoid misleading "loss got bigger" impression.
            with torch.no_grad():
                loss_post, _, _, _, dist_p, elev_p, azim_p, loss_sil_p, loss_depth_p, loss_normal_p, hemi_dot_p, hemi_pen_p, prior_pen_p = model()
            lr_now = float(opt.param_groups[0]["lr"])
            print(
                f"[Opt] {target_name} iter {it}/{max_iterations} "
                f"loss_pre={lv_pre:.6f} loss_post={float(loss_post.item()):.6f} "
                f"lr={lr_now:.6f} dist/elev/azim=({float(dist_p.item()):.2f},{float(elev_p.item()):.2f},{float(azim_p.item()):.2f}) "
                f"loss_sil={float(loss_sil_p.item()):.3f} loss_depth={float(loss_depth_p.item()):.3f} loss_normal={float(loss_normal_p.item()):.3f} "
                f"hemi_dot={float(hemi_dot_p.item()):.3f}"
            )

        # Write pose history periodically (post-step values; easier to compare).
        if hist_writer is not None and history_interval > 0 and (it % history_interval == 0 or it == max_iterations - 1):
            try:
                with torch.no_grad():
                    loss_post_h, _, _, _, dist_h, elev_h, azim_h, ls_h, ld_h, ln_h, _, _, _ = model()
                lr_now_h = float(opt.param_groups[0]["lr"])
                hist_writer.writerow(
                    [
                        int(it),
                        lr_now_h,
                        float(lv_pre),
                        float(loss_post_h.item()),
                        float(ls_h.item()),
                        float(ld_h.item()),
                        float(ln_h.item()),
                        float(dist_h.item()),
                        float(elev_h.item()),
                        float(azim_h.item()),
                    ]
                )
                if hist_fp is not None:
                    hist_fp.flush()
            except Exception as _e:
                print(f"[Warning] Failed to write pose history at iter {it}: {_e}")

    # Save results
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, f"{target_name}_best_pose.npy"), best_pose)
        with open(os.path.join(save_dir, f"{target_name}_loss.json"), "w", encoding="utf-8") as f:
            json.dump({"loss_history": loss_history, "best_loss": best_loss}, f, indent=2)

        # Save BEST results with clear naming: best_*
        try:
            if best_pose is not None:
                np.savetxt(os.path.join(save_dir, "best_pose_4x4.txt"), best_pose, fmt="%.8f")
                np.save(os.path.join(save_dir, "best_pose_4x4.npy"), best_pose.astype(np.float32))
                with open(os.path.join(save_dir, "best_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "best_iter": int(best_iter),
                            "best_loss": float(best_loss),
                            "early_stop_enabled": bool(early_stop_enabled),
                            "early_stop_triggered": bool(early_stop_triggered),
                            "early_stop_iter": int(early_stop_iter),
                            "note": "best_pose_* is the minimum-loss camera pose during optimization (world-to-view).",
                        },
                        f,
                        indent=2,
                    )

                # Render and save best silhouette/depth/normals
                with torch.no_grad():
                    pose_4x4 = torch.tensor(best_pose, dtype=torch.float32, device=device)
                    R_best = pose_4x4[:3, :3].unsqueeze(0)
                    T_best = pose_4x4[:3, 3].unsqueeze(0)
                    cameras_best = FoVPerspectiveCameras(device=device, R=R_best, T=T_best)

                    img_sil = silhouette_renderer(meshes_world=mesh, cameras=cameras_best)
                    alpha = torch.clamp(img_sil[0, ..., 3], 0, 1)
                    cv2.imwrite(os.path.join(save_dir, "best_silhouette.png"), (alpha.cpu().numpy() * 255).astype(np.uint8))

                    fr = rasterizer(meshes_world=mesh, cameras=cameras_best)
                    vis_mask = (fr.pix_to_face[..., 0] >= 0).float()[0]
                    depth = (fr.zbuf[..., 0][0] * vis_mask)
                    cv2.imwrite(os.path.join(save_dir, "best_depth.png"), _depth_to_vis(depth))

                    verts_normals = mesh.verts_normals_packed()
                    faces = mesh.faces_packed()
                    # normals in VIEW SPACE (consistent with reference/optimization)
                    verts_normals_view = torch.matmul(verts_normals, R_best[0].T)
                    faces_verts_normals = verts_normals_view[faces]
                    pix_normals = interpolate_face_attributes(fr.pix_to_face, fr.bary_coords, faces_verts_normals)
                    normals = pix_normals[..., 0, :][0]
                    normals = torch.nn.functional.normalize(normals, dim=-1) * vis_mask.unsqueeze(-1)
                    cv2.imwrite(os.path.join(save_dir, "best_normals.png"), _normal_to_vis(normals))

                    rgb = phong_renderer(meshes_world=mesh, cameras=cameras_best)[0, ..., :3]
                    cv2.imwrite(os.path.join(save_dir, "best_rgb.png"), cv2.cvtColor(_to_uint8_rgb(rgb), cv2.COLOR_RGB2BGR))
        except Exception as _e:
            print(f"[Warning] Failed to save best_* results: {_e}")
    if hist_fp is not None:
        try:
            hist_fp.close()
        except Exception:
            pass
    return best_pose, loss_history


def _resolve_run_params() -> dict[str, str]:
    """Merge CODE_RUN_PARAMS with RUN_PROFILE_PRESETS[VAL_RUN_PROFILE]."""
    # Env VAL_RUN_PROFILE wins over CODE_RUN_PARAMS so scripts can run v12d without editing the file.
    profile = str(
        os.environ.get("VAL_RUN_PROFILE", CODE_RUN_PARAMS.get("VAL_RUN_PROFILE", "v9"))
    ).strip().lower()
    merged = dict(CODE_RUN_PARAMS)
    preset = RUN_PROFILE_PRESETS.get(profile, {})
    if preset:
        merged.update(preset)
    merged["VAL_RUN_PROFILE"] = profile
    return merged


def main():
    # nvdiffrast-only: requires CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("This experiment uses nvdiffrast and requires CUDA, but torch.cuda.is_available() is False.")
    device = torch.device("cuda")
    if int(os.environ.get("FORCE_CODE_RUN_PARAMS", "1" if FORCE_CODE_RUN_PARAMS else "0")) != 0:
        run_params = _resolve_run_params()
        preserve_env = int(os.environ.get("BENCHMARK_PRESERVE_ENV", "0")) != 0
        for k, v in run_params.items():
            if preserve_env and str(k) in os.environ:
                continue
            os.environ[str(k)] = str(v)
        print(
            f"[Config] FORCE_CODE_RUN_PARAMS=1: profile={run_params.get('VAL_RUN_PROFILE', 'v9')} "
            f"-> {run_params.get('OUTPUT_DIR', DEFAULT_OUTPUT_ROOT)}"
        )
    else:
        print("[Config] FORCE_CODE_RUN_PARAMS=0: using environment / batch parameters.")
    seed_cfg = int(os.environ.get("VAL_RANDOM_SEED", "12291"))
    random.seed(seed_cfg)
    np.random.seed(seed_cfg)
    torch.manual_seed(seed_cfg)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_cfg)
    if int(os.environ.get("VAL_DETERMINISTIC", "0")) != 0:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass
        print("[Config] VAL_DETERMINISTIC=1 (cudnn deterministic, benchmark off)")
    print(f"[Config] VAL_RANDOM_SEED={seed_cfg}")
    input_image_path = os.environ.get("INPUT_IMAGE", "data/real_frames")
    output_root = os.path.abspath(os.environ.get("OUTPUT_DIR", DEFAULT_OUTPUT_ROOT))
    os.makedirs(output_root, exist_ok=True)

    print(f"[Info] Device: {device}")
    print(f"[Info] Input image: {input_image_path}")
    print(f"[Info] Output dir : {output_root}")

    # -----------------------
    # Stage0: config/speed defaults
    # -----------------------

    # ------------------------------------------------------------
    # Fast mode: set a group of speed-oriented defaults WITHOUT overriding user-provided envs.
    # Enable with: VAL_FAST_MODE=1
    # ------------------------------------------------------------
    if int(os.environ.get("VAL_FAST_MODE", "0")) != 0:
        fast_defaults = {
            # Skip full-res final refinement unless explicitly enabled.
            "VAL_FINAL_ENABLE": "0",
            # Reduce per-GT optimization work.
            "VAL_SEARCH_MAX_ITERS": "400",
            "VAL_FINAL_MAX_ITERS": "200",
            # Make GreedyAxis cheaper per-iter (sil-only prefilter + small topM full).
            "VAL_AXIS_USE_SIL_ONLY": "1",
            "VAL_AXIS_REFINE_FULL": "1",
            "VAL_AXIS_TOPM_FULL": "4",
            "VAL_AXIS_STEPS_PER_DIR": "2",
            "VAL_AXIS_NO_IMPROVE_PATIENCE": "40",
            "VAL_AXIS_MIN_STEP_FRAC": "0.003",
            # Reduce I/O.
            "VAL_SAVE_INTERVAL": "200",
            "VAL_SAVE_ON_IMPROVE": "0",
            # Speed up beam search: reduce width and levels
            "VAL_LOCAL_BEAM_WIDTH": "16",
            "VAL_LOCAL_BEAM_LEVELS": "16",
            # Global search samples (keep original for better coverage)
            "VAL_RAND_SAMPLES": "16384",  # Original value for better global search coverage
            "VAL_RAND_TOPK": "8",  # Set to 8
            "VAL_RAND_BATCH": "512",  # Reduce batch size to avoid CUDA OOM
            # Disable semantic similarity to speed up (skip model loading)
            "VAL_LOCAL_BEAM_USE_SIL_SEMANTIC": "0",  # Disable semantic similarity
            "VAL_DISABLE_SEMANTIC": "1",  # Global flag to disable semantic model initialization
        }
        for k, v in fast_defaults.items():
            os.environ.setdefault(k, v)
        print("[Speed] VAL_FAST_MODE=1 enabled: applied speed defaults (non-overriding).")
        print(f"[Speed] Beam search: width={fast_defaults['VAL_LOCAL_BEAM_WIDTH']}, levels={fast_defaults['VAL_LOCAL_BEAM_LEVELS']}")
        print(f"[Speed] Global search: samples={fast_defaults['VAL_RAND_SAMPLES']}, topk={fast_defaults['VAL_RAND_TOPK']}")
        print(f"[Speed] Semantic similarity: DISABLED (to speed up, skip model loading)")

    # ------------------------------------------------------------
    # Accuracy-first final refinement defaults.
    # These defaults target the common failure mode where the discrete/global
    # search reaches the correct basin but cannot remove the last pose error.
    # Users can still override every item from the environment.
    # ------------------------------------------------------------
    accuracy_defaults = {
        "VAL_REFINE_RES_MODE": "full",
        "VAL_REFINE_MAX_ITERS": "1200",
        "VAL_REFINE_LR": "0.01",
        "VAL_SE3_REFINE_ENABLE": "1",
        "VAL_SE3_REFINE_ITERS": "800",
        "VAL_SE3_REFINE_LR": "0.003",
        "VAL_SE3_ACCEPT_MIN_DELTA": "0.0",
        "VAL_SE3_W_REG": "1e-5",
        "VAL_SE3_T_REG": "1e-5",
        "VAL_SE3_USE_SCHEDULER": "1",
        "VAL_SE3_EARLY_STOP_PATIENCE": "160",
        "VAL_SE3_NORMAL_ABS": "1",
        "VAL_LOSS_SCHEDULE": "1",
        "VAL_W_SIL_END": "0.2",
        "VAL_W_DEPTH_END": "2.0",
        "VAL_W_NORMAL_END": "1.0",
        "VAL_RAND_TOPK": "8",
        "VAL_MS_REFINE_ENABLE": "1",
        "VAL_MS_REFINE_TOPK": "8",
        "VAL_MS_LONG_FORCE_FULL": "1",
    }
    for k, v in accuracy_defaults.items():
        os.environ.setdefault(k, v)
    print("[Accuracy] Applied final full-res + SE(3) refinement defaults (non-overriding).")
    print(
        f"[Accuracy] SE3 enable={os.environ.get('VAL_SE3_REFINE_ENABLE')} "
        f"iters={os.environ.get('VAL_SE3_REFINE_ITERS')} lr={os.environ.get('VAL_SE3_REFINE_LR')}"
    )

    # 1) 直接读取3D模型文件
    input_model_path = os.environ.get("INPUT_MODEL_PATH", os.path.join(os.environ.get("STANFORD_ROOT", "data/stanford"), "bun_zipper.ply"))
    if not os.path.exists(input_model_path):
        print(f"[Error] Input model file not found: {input_model_path}")
        return
    
    try:
        mesh_o3d = o3d.io.read_triangle_mesh(input_model_path)
        if not mesh_o3d.has_vertices() or not mesh_o3d.has_triangles():
            print(f"[Error] Loaded mesh from {input_model_path} is empty or invalid.")
            return
        print(f"[Mesh] Loaded 3D model from: {input_model_path}")
        print(f"[Mesh] mesh_o3d verts={len(mesh_o3d.vertices)}, tris={len(mesh_o3d.triangles)}")
        # Ensure normals are computed for rendering
        if not mesh_o3d.has_vertex_normals():
            mesh_o3d.compute_vertex_normals()
    except Exception as e:
        print(f"[Error] Failed to load 3D model from {input_model_path}: {e}")
        return

    # Optional: simplify mesh to reduce triangle count for faster rendering.
    # This can dramatically speed up rasterization and all pose evaluations.
    # Enable with VAL_MESH_SIMPLIFY=1.
    if int(os.environ.get("VAL_MESH_SIMPLIFY", "0")) != 0:
        try:
            tris0 = int(len(mesh_o3d.triangles))
            verts0 = int(len(mesh_o3d.vertices))
            # Strategy:
            # 1) If VAL_MESH_CLUSTER_VOXEL > 0: use vertex clustering (fast, coarse)
            # 2) Else: use quadric decimation towards target tris
            voxel = float(os.environ.get("VAL_MESH_CLUSTER_VOXEL", "0"))
            if voxel > 0:
                mesh_o3d = mesh_o3d.simplify_vertex_clustering(
                    voxel_size=float(voxel),
                    contraction=o3d.geometry.SimplificationContraction.Average,
                )
            else:
                target_tris_env = os.environ.get("VAL_MESH_TARGET_TRIS", "")
                ratio = float(os.environ.get("VAL_MESH_SIMPLIFY_RATIO", "0.25"))
                ratio = max(0.01, min(1.0, ratio))
                if target_tris_env != "":
                    target_tris = int(float(target_tris_env))
                else:
                    target_tris = int(max(50, round(tris0 * ratio)))
                # Clamp to valid range
                target_tris = int(max(50, min(target_tris, tris0)))
                if target_tris < tris0:
                    mesh_o3d = mesh_o3d.simplify_quadric_decimation(target_number_of_triangles=int(target_tris))

            # clean up (helps avoid renderer issues)
            try:
                mesh_o3d.remove_degenerate_triangles()
                mesh_o3d.remove_duplicated_triangles()
                mesh_o3d.remove_duplicated_vertices()
                mesh_o3d.remove_non_manifold_edges()
            except Exception:
                pass
            try:
                mesh_o3d.compute_vertex_normals()
            except Exception:
                pass

            tris1 = int(len(mesh_o3d.triangles))
            verts1 = int(len(mesh_o3d.vertices))
            print(f"[Mesh] Simplified mesh: verts {verts0} -> {verts1}, tris {tris0} -> {tris1}")
        except Exception as _e:
            print(f"[Warning] Mesh simplification failed, using original mesh. err={_e}")

    # Save the raw Open3D mesh early (useful for debugging / external visualization).
    # You can disable with VAL_SAVE_MESH=0
    if int(os.environ.get("VAL_SAVE_MESH", "1")) != 0:
        try:
            mesh_dir = os.path.join(output_root, "mesh_exports")
            _write_o3d_mesh(os.path.join(mesh_dir, "mesh_raw_o3d.ply"), mesh_o3d)
        except Exception as _e:
            print(f"[Warning] Failed to save raw mesh_o3d: {_e}")

    # 2) 设置渲染参数 (ImageProcessor 不再使用，手动设置)
    # 图像分辨率通常由渲染任务决定，这里设置为常用值，或从环境变量获取
    image_height = int(os.environ.get("VAL_RENDER_H", "480"))
    image_width = int(os.environ.get("VAL_RENDER_W", "640"))
    print(f"[Info] Render image size: {image_width}x{image_height}")
    
    # 计算mesh的中心点作为 look-at 目标
    vertices = np.asarray(mesh_o3d.vertices)
    mesh_min = vertices.min(axis=0)
    mesh_max = vertices.max(axis=0)
    mesh_center_np = vertices.mean(axis=0)
    mesh_size = mesh_max - mesh_min
    mesh_diagonal_length = np.linalg.norm(mesh_size)
    mesh_center = torch.tensor(mesh_center_np, dtype=torch.float32, device=device).view(1, 3)
    
    # 初始相机位姿 (dist, elev, azim) - 可以根据模型大小和初始视角进行调整
    init_dist = float(os.environ.get("VAL_INIT_DIST", str(mesh_diagonal_length * 2.0)))  # 通常设置为模型对角线长度的1.5到3倍
    init_elev = float(os.environ.get("VAL_INIT_ELEV", "30.0"))  # 初始仰角
    init_azim = float(os.environ.get("VAL_INIT_AZIM", "0.0"))  # 初始方位角
    
    # 计算相机内参（基于图像尺寸）
    fx = image_width * 0.8
    fy = image_height * 0.8
    cx = image_width / 2.0
    cy = image_height / 2.0
    
    # Render size
    render_w = image_width
    render_h = image_height
    # Search stage downscale (speed): search uses smaller image size, final uses render_h/render_w.
    # Optimized: default changed from 0.5 to 0.3 for faster search (~4x speedup)
    search_downscale = float(os.environ.get("VAL_SEARCH_DOWNSCALE", "0.3"))
    search_h, search_w = _downscaled_hw(render_h, render_w, search_downscale)
    print(f"[Speed] Search stage downscale={search_downscale:.3f} -> search_size=({search_w}x{search_h}), final_size=({render_w}x{render_h})")
    print(f"[Mesh] mesh_center={mesh_center_np}, mesh_diagonal={mesh_diagonal_length:.2f}")
    print(f"[Camera] init_dist={init_dist:.2f}, init_elev={init_elev:.1f}°, init_azim={init_azim:.1f}°")
    print(f"[Camera] fx={fx:.1f}, fy={fy:.1f}, cx={cx:.1f}, cy={cy:.1f}")
    
    # 计算初始相机位姿
    R_init_b, T_init_b = look_at_view_transform(dist=init_dist, elev=init_elev, azim=init_azim, at=mesh_center, device=device)
    R_init = R_init_b[0]
    T_init = T_init_b[0]

    # Build reusable renderers/rasterizers once (avoid rebuilding per-GT).
    cameras_stub_full = FoVPerspectiveCameras(device=device)
    sil_full, rast_full, phong_full = build_renderers(render_h, render_w, cameras_stub_full)
    cameras_stub_search = FoVPerspectiveCameras(device=device)
    # Optimized: default changed from 10 to 5 for faster rendering (~2-5x speedup)
    search_fpp = int(os.environ.get("VAL_SEARCH_FACES_PER_PIXEL", "5"))
    sil_eval, rast_eval, phong_eval = build_renderers(search_h, search_w, cameras_stub_search, faces_per_pixel_silhouette=search_fpp)

    # 3) 构建 Mesh 数据（nvdiffrast-only）
    verts = torch.tensor(np.asarray(mesh_o3d.vertices), dtype=torch.float32, device=device)
    faces = torch.tensor(np.asarray(mesh_o3d.triangles), dtype=torch.long, device=device)
    def _sanitize_faces(f: torch.Tensor, num_verts: int) -> torch.Tensor:
        """
        Filter out invalid/degenerate faces to avoid CUDA gather OOB asserts.
        - invalid: any index <0 or >= num_verts
        - degenerate: duplicated vertex indices in a triangle
        """
        if f.numel() == 0:
            return f
        f = f.long()
        # validate range (do on device, but cheap)
        valid = (f >= 0) & (f < int(num_verts))
        valid = valid.all(dim=1)
        # remove degenerates
        deg = (f[:, 0] == f[:, 1]) | (f[:, 0] == f[:, 2]) | (f[:, 1] == f[:, 2])
        valid = valid & (~deg)
        if bool((~valid).any().detach().cpu().item()):
            bad = int((~valid).sum().detach().cpu().item())
            print(f"[Warning] Filtering {bad} invalid/degenerate faces out of {int(f.shape[0])}.")
        f2 = f[valid]
        if f2.numel() == 0:
            raise ValueError("All mesh faces are invalid/degenerate after sanitization; cannot render.")
        return f2

    # Preflight check (CPU) + sanitize to avoid GPU-side asserts later
    try:
        V = int(verts.shape[0])
        f_cpu = faces.detach().cpu()
        fmin = int(f_cpu.min().item()) if f_cpu.numel() else 0
        fmax = int(f_cpu.max().item()) if f_cpu.numel() else -1
        if fmin < 0 or fmax >= V:
            print(f"[Warning] Face index out of range detected: min={fmin}, max={fmax}, num_verts={V}. Will sanitize.")
    except Exception:
        pass
    faces = _sanitize_faces(faces, int(verts.shape[0]))
    def _vertex_colors_checkerboard_xy(v: torch.Tensor, grid_size: float) -> torch.Tensor:
        """
        True checkerboard (black/white) based on XY coordinates in world units.
        Returns RGB in [0,1].
        """
        gs = float(max(1e-6, grid_size))
        xi = torch.floor(v[:, 0] / gs).to(torch.int64)
        yi = torch.floor(v[:, 1] / gs).to(torch.int64)
        parity = ((xi + yi) & 1).to(torch.float32).unsqueeze(1)  # (N,1) 0/1
        c0 = torch.tensor([0.95, 0.95, 0.95], device=v.device, dtype=torch.float32).view(1, 3)
        c1 = torch.tensor([0.15, 0.15, 0.15], device=v.device, dtype=torch.float32).view(1, 3)
        return c0 * (1.0 - parity) + c1 * parity

    tex_grid = float(os.environ.get("TEXTURE_GRID_SIZE", "7.0"))
    tex_mode = os.environ.get("VAL_TEX_MODE", "checkerboard").strip().lower()  # checkerboard|pink(debug)
    print(f"[Texture] VAL_TEX_MODE={tex_mode}, TEXTURE_GRID_SIZE={tex_grid}")
    if tex_mode == "pink":
        # Debug mode: generate pink texture (simple solid pink color)
        pink_color = torch.tensor([1.0, 0.75, 0.8], dtype=torch.float32, device=device)
        colors = pink_color.unsqueeze(0).repeat(len(verts), 1)
        try:
            # If it's nearly constant (common "missing texture" placeholder), fall back to a visible checkerboard.
            if (
                isinstance(colors, torch.Tensor)
                and colors.dim() == 2
                and colors.shape[1] == 3
                and torch.all((colors.max(dim=0).values - colors.min(dim=0).values) < 1e-6)
            ):
                print(
                    "[Warning] VAL_TEX_MODE=pink produced a near-constant color (often solid magenta/pink). "
                    "Falling back to local checkerboard vertex colors. "
                    "To force solid pink for debugging, set VAL_FORCE_SOLID_PINK=1."
                )
                if int(os.environ.get("VAL_FORCE_SOLID_PINK", "0")) == 0:
                    colors = _vertex_colors_checkerboard_xy(verts, grid_size=tex_grid)
        except Exception:
            # If any check fails, still keep something usable.
            colors = _vertex_colors_checkerboard_xy(verts, grid_size=tex_grid)
    else:
        colors = _vertex_colors_checkerboard_xy(verts, grid_size=tex_grid)

    # Safety: clamp to [0,1] to avoid weird OpenCV/renderer artifacts from out-of-range values.
    if isinstance(colors, torch.Tensor):
        colors = torch.clamp(colors.float(), 0.0, 1.0)

    # (mesh exports moved below, after verts_normals is computed)
    # vertex normals (preferred: from Open3D). Fallback to zeros if missing.
    try:
        vn = np.asarray(mesh_o3d.vertex_normals)
        if vn is None or len(vn) == 0:
            raise ValueError("open3d vertex_normals empty")
        verts_normals = torch.tensor(vn, dtype=torch.float32, device=device)
    except Exception:
        verts_normals = torch.zeros_like(verts)

    # Save colored meshes (Open3D + sanitized faces) for inspection in MeshLab/Blender.
    if int(os.environ.get("VAL_SAVE_MESH", "1")) != 0:
        try:
            mesh_dir = os.path.join(output_root, "mesh_exports")
            # Colored Open3D mesh (keep original topology)
            try:
                m_col = copy.deepcopy(mesh_o3d)
                c_np = colors.detach().cpu().numpy().astype(np.float64)
                if c_np.ndim == 2 and c_np.shape[1] == 3 and len(m_col.vertices) == c_np.shape[0]:
                    m_col.vertex_colors = o3d.utility.Vector3dVector(c_np)
                _write_o3d_mesh(os.path.join(mesh_dir, "mesh_colored_o3d.ply"), m_col)
            except Exception as _e2:
                print(f"[Warning] Failed to build/save mesh_colored_o3d.ply: {_e2}")

            # Sanitized topology mesh (vertices + sanitized faces + colors/normals from tensors)
            try:
                m_san = o3d.geometry.TriangleMesh()
                m_san.vertices = o3d.utility.Vector3dVector(verts.detach().cpu().numpy().astype(np.float64))
                m_san.triangles = o3d.utility.Vector3iVector(faces.detach().cpu().numpy().astype(np.int32))
                if isinstance(verts_normals, torch.Tensor) and verts_normals.numel() > 0 and verts_normals.shape == verts.shape:
                    m_san.vertex_normals = o3d.utility.Vector3dVector(verts_normals.detach().cpu().numpy().astype(np.float64))
                c_np = colors.detach().cpu().numpy().astype(np.float64)
                if c_np.ndim == 2 and c_np.shape[1] == 3 and c_np.shape[0] == len(m_san.vertices):
                    m_san.vertex_colors = o3d.utility.Vector3dVector(c_np)
                _write_o3d_mesh(os.path.join(mesh_dir, "mesh_sanitized_colored.ply"), m_san)
            except Exception as _e3:
                print(f"[Warning] Failed to build/save mesh_sanitized_colored.ply: {_e3}")
        except Exception as _e:
            print(f"[Warning] Failed to save colored meshes: {_e}")

    mesh = MeshData(verts=verts, faces=faces, verts_normals=verts_normals, verts_colors=colors)

    # ------------------------------------------------------------
    # Init pose visibility check:
    # If the "init" camera pose cannot see the mesh (silhouette all black),
    # automatically push the camera outward by increasing init_dist.
    # This avoids confusing debug artifacts like init_silhouette_soft being all zeros.
    # Disable with VAL_AUTO_FIX_INIT_POSE=0
    # ------------------------------------------------------------
    if int(os.environ.get("VAL_AUTO_FIX_INIT_POSE", "1")) != 0:
        try:
            def _vis_ratio_for_pose(R_: torch.Tensor, T_: torch.Tensor) -> float:
                fr0 = rast_full(meshes_world=mesh, R=R_.unsqueeze(0), T=T_.unsqueeze(0))
                return float((fr0.pix_to_face[..., 0] >= 0).float().mean().detach().cpu().item())

            with torch.no_grad():
                vr0 = _vis_ratio_for_pose(R_init, T_init)
            if vr0 <= float(os.environ.get("VAL_INIT_MIN_VIS_RATIO", "1e-4")):
                print(f"[Warning] Init pose sees nothing (vis_ratio={vr0:.6f}). Auto-fixing init pose (zfar + dist scan).")
                init_dist_orig = float(init_dist)

                # 1) Make sure zfar is not the reason (far clipping).
                try:
                    zfar_cur = float(os.environ.get("VAL_ZFAR", "1000.0"))
                    # push zfar to comfortably include the look-at target and mesh extent
                    zfar_new = max(zfar_cur, init_dist_orig * 10.0, 10000.0)
                    if zfar_new > zfar_cur + 1e-6:
                        os.environ["VAL_ZFAR"] = f"{zfar_new:.6f}"
                        print(f"[InitFix] VAL_ZFAR {zfar_cur:.3f} -> {zfar_new:.3f}")
                except Exception:
                    pass

                # 2) Try both nearer and farther distances; pick the best visibility ratio.
                #    If your init_dist is way off (too near/too far), one of these usually fixes it.
                scales = os.environ.get("VAL_INIT_DIST_SCALES", "0.25,0.4,0.6,0.8,1.0,1.3,1.7,2.2,3.0").strip()
                try:
                    scale_list = [float(x) for x in scales.split(",") if x.strip() != ""]
                    if not scale_list:
                        scale_list = [1.0]
                except Exception:
                    scale_list = [0.25, 0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2, 3.0]

                best = (vr0, float(init_dist_orig), R_init, T_init)
                for s in scale_list:
                    d_try = float(init_dist_orig) * float(s)
                    # avoid non-positive distances
                    if not (d_try > 1e-6):
                        continue
                    Rb, Tb = look_at_view_transform(dist=float(d_try), elev=float(init_elev), azim=float(init_azim), at=mesh_center, device=device)
                    R_try, T_try = Rb[0], Tb[0]
                    with torch.no_grad():
                        vr = _vis_ratio_for_pose(R_try, T_try)
                    if vr > best[0]:
                        best = (vr, float(d_try), R_try, T_try)

                vr_best, dist_best, R_best, T_best = best
                R_init, T_init = R_best, T_best
                init_dist = float(dist_best)
                print(f"[InitFix] init_dist {init_dist_orig:.3f} -> {init_dist:.3f}, vis_ratio {vr0:.6f} -> {vr_best:.6f}")
        except Exception as _e:
            print(f"[Warning] Init pose auto-fix failed (kept original init pose): {_e}")

    # 4) 生成目标相机位姿（GT，随机球面采样，但要求相邻 pose 变化要大）
    # Default: generate 100 GT poses (pose00..pose99).
    num_targets = int(os.environ.get("NUM_GT_POSES", "100"))
    camera_distance = init_dist
    # Optional absolute distance clamp to keep the camera close but avoid "inside mesh" cases.
    # If not set, it will use relative sampling only.
    # Enforce "big change" between consecutive GT poses (user request).
    # Criteria (env overridable):
    # - minimum camera position displacement in WORLD: VAL_GT_MIN_CAM_DIST (default: 0.6 * model_radius_r)
    # - and/or minimum angular changes: VAL_GT_MIN_DAZIM_DEG / VAL_GT_MIN_DELEV_DEG
    gt_tries = int(os.environ.get("VAL_GT_MAX_TRIES_PER_POSE", "500"))
    gt_tries = max(10, gt_tries)
    gt_min_dazim = float(os.environ.get("VAL_GT_MIN_DAZIM_DEG", "30.0"))
    gt_min_delev = float(os.environ.get("VAL_GT_MIN_DELEV_DEG", "15.0"))
    gt_min_cam_dist_env = os.environ.get("VAL_GT_MIN_CAM_DIST", "")
    # model radius r (world units) from verts/mesh_center
    try:
        r_model = _mesh_radius_from_verts(verts, mesh_center[0])
    except Exception:
        r_model = float(mesh_diagonal_length) * 0.25
    gt_min_cam_dist = float(gt_min_cam_dist_env) if gt_min_cam_dist_env != "" else (0.60 * float(r_model))

    # Mesh-specific depth->centre offset used by the metric distance hint (Stage II).
    # Uses only the known mesh (random viewpoints), never the target observation.
    try:
        _estimate_depth_centre_offset(mesh, rast_eval, mesh_center[0], float(r_model), device)
    except Exception as _e_k:
        print(f"[DepthHint] kappa calibration failed ({_e_k}); using 0.5 r")

    elev_lo = float(os.environ.get("VAL_GT_ELEV_MIN", "5.0"))
    elev_hi = float(os.environ.get("VAL_GT_ELEV_MAX", "85.0"))
    azim_lo = float(os.environ.get("VAL_GT_AZIM_MIN", "-180.0"))
    azim_hi = float(os.environ.get("VAL_GT_AZIM_MAX", "180.0"))
    dist_scale_min = float(os.environ.get("VAL_GT_DIST_SCALE_MIN", "0.15"))
    dist_scale_max = float(os.environ.get("VAL_GT_DIST_SCALE_MAX", "0.85"))
    dist_min_abs = os.environ.get("VAL_GT_DIST_MIN", "")
    dist_max_abs = os.environ.get("VAL_GT_DIST_MAX", "")

    def _wrap_deg(x: float) -> float:
        return float(((x + 180.0) % 360.0) - 180.0)

    def _ang_dist_deg(a: float, b: float) -> float:
        return abs(_wrap_deg(float(a) - float(b)))

    target_poses = []
    prev = None  # (dist,elev,azim,cam_xyz_np)
    # GT-pose RNG. By default the global `random` module (seeded by VAL_RANDOM_SEED) is used,
    # which reproduces the historical GT sets. VAL_GT_POSE_SEED decouples the GT labels from the
    # search RNG; it is used by tools/test_no_gt_leak.py to swap GT labels while keeping the
    # cached observations and the sampler state fixed.
    _gt_seed_env = os.environ.get("VAL_GT_POSE_SEED", "").strip()
    _gt_rng = random.Random(int(_gt_seed_env)) if _gt_seed_env != "" else random
    for i in range(num_targets):
        best_fallback = None  # (score, dist,elev,azim, cam_xyz_np)
        for t in range(gt_tries):
            elev = _gt_rng.uniform(elev_lo, elev_hi)
            azim = _gt_rng.uniform(azim_lo, azim_hi)
            dist = float(camera_distance) * _gt_rng.uniform(dist_scale_min, dist_scale_max)
            if dist_min_abs != "":
                dist = max(dist, float(dist_min_abs))
            if dist_max_abs != "":
                dist = min(dist, float(dist_max_abs))

            cam_xyz = _camera_position_from_spherical_global(
                torch.tensor(float(dist), device=device),
                torch.tensor(float(elev), device=device),
                torch.tensor(float(azim), device=device),
                mesh_center[0].view(3),
            )
            cam_np = cam_xyz.detach().cpu().numpy().astype(np.float64)

            if prev is None:
                ok = True
                score = 1e9
            else:
                prev_dist, prev_elev, prev_azim, prev_np = prev
                d_cam = float(np.linalg.norm(cam_np - prev_np))
                d_az = _ang_dist_deg(azim, prev_azim)
                d_el = abs(float(elev) - float(prev_elev))
                # Prefer large camera displacement; also require angle separation for diversity.
                ok = (d_cam >= gt_min_cam_dist) and (d_az >= gt_min_dazim or d_el >= gt_min_delev)
                score = d_cam + 0.5 * d_az + 0.3 * d_el

            # keep best fallback (max score) in case we can't satisfy constraints
            if best_fallback is None or score > best_fallback[0]:
                best_fallback = (score, float(dist), float(elev), float(azim), cam_np)
            if ok:
                chosen = (float(dist), float(elev), float(azim), cam_np)
                break
        else:
            # could not find a strict-valid sample; take the farthest (by score) as fallback
            chosen = (best_fallback[1], best_fallback[2], best_fallback[3], best_fallback[4]) if best_fallback is not None else (float(camera_distance), 50.0, 0.0, None)
            print(f"[Warning] GT pose {i}: failed to meet min-sep after {gt_tries} tries; using best fallback (relaxed).")

        dist, elev, azim, cam_np = chosen
        R_gt, T_gt = look_at_view_transform(dist=dist, elev=elev, azim=azim, at=mesh_center, device=device)
        target_poses.append((R_gt[0], T_gt[0], float(dist), float(elev), float(azim)))
        prev = (float(dist), float(elev), float(azim), cam_np if cam_np is not None else np.array([np.nan, np.nan, np.nan], dtype=np.float64))
        if i % 10 == 0 or i == num_targets - 1:
            print(f"  [GT] 生成进度: {i+1}/{num_targets} | 当前: pose {i:02d}: dist={dist:.2f}, elev={elev:.2f}, azim={azim:.2f}")

    # ------------------------------------------------------------
    # 5-A) 先采集/记录 NUM_GT_POSES 个位姿，并保存每个 GT 的渲染产物（sil/depth/normals/rgb + ref npz）
    # 然后再执行后续搜索/优化逻辑。
    # 控制开关：VAL_PRECOMPUTE_GT=1/0（默认 1）
    # ------------------------------------------------------------
    gt_csv = os.path.join(output_root, "gt_poses.csv")
    gt_pose_npy = os.path.join(output_root, "gt_poses_pose4x4.npy")
    try:
        poses_4x4 = []
        with open(gt_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["idx", "dist", "elev_deg", "azim_deg"] + [f"m{i}{j}" for i in range(4) for j in range(4)])
            for idx, (R_gt, T_gt, dist_gt, elev, azim) in enumerate(target_poses):
                pose_4x4 = _pose_rt_to_4x4(R_gt, T_gt)
                poses_4x4.append(pose_4x4.astype(np.float32))
                w.writerow([int(idx), float(dist_gt), float(elev), float(azim)] + [float(x) for x in pose_4x4.reshape(-1)])
        np.save(gt_pose_npy, np.stack(poses_4x4, axis=0).astype(np.float32))
        print(f"\n{'='*80}")
        print(f"[GT] GT位姿生成完成")
        print(f"{'='*80}")
        print(f"  生成的GT位姿数: {len(target_poses)}")
        print(f"  保存的位姿列表: {gt_csv}")
        print(f"  保存的位姿矩阵: {gt_pose_npy}")
        print(f"{'='*80}\n")
    except Exception as _e:
        print(f"[Warning] Failed to write gt_poses.csv/gt_poses_pose4x4.npy: {_e}")

    if int(os.environ.get("VAL_PRECOMPUTE_GT", "1")) != 0:
        print(f"\n{'='*80}")
        print(f"[GT] 预计算GT渲染结果")
        print(f"{'='*80}")
        print(f"  位姿数量: {len(target_poses)}")
        print(f"  缓存路径: {output_root}")
        print(f"{'='*80}")
        for idx, (R_gt, T_gt, dist_gt, elev, azim) in enumerate(target_poses):
            tag = f"gt_{idx:02d}"
            exp_dir = os.path.join(output_root, tag)
            os.makedirs(exp_dir, exist_ok=True)
            # Save GT spherical explicitly
            with open(os.path.join(exp_dir, "gt_camera_spherical.txt"), "w", encoding="utf-8") as f:
                f.write(f"dist={float(dist_gt):.6f}\n")
                f.write(f"elev_deg={float(elev):.6f}\n")
                f.write(f"azim_deg={float(azim):.6f}\n")

            # Target refs (full + search) and cache to disk for later stages
            alpha_ref_full, depth_ref_full, normals_ref_full, ref_points_view_full = render_targets(mesh, R_gt, T_gt, device, render_h, render_w)
            alpha_ref_s, depth_ref_s, normals_ref_s, ref_points_view_s = render_targets(mesh, R_gt, T_gt, device, search_h, search_w)
            _save_npz_tensors(
                os.path.join(exp_dir, "gt_refs_full.npz"),
                alpha=alpha_ref_full,
                depth=depth_ref_full,
                normals=normals_ref_full,
                ref_points_view=ref_points_view_full,
            )
            _save_npz_tensors(
                os.path.join(exp_dir, "gt_refs_search.npz"),
                alpha=alpha_ref_s,
                depth=depth_ref_s,
                normals=normals_ref_s,
                ref_points_view=ref_points_view_s,
            )

            # Save target/gt images (for quick inspection)
            _save_silhouette_pair(exp_dir, "target", alpha_ref_full)
            with torch.no_grad():
                rgba_gt = sil_full(meshes_world=mesh, R=R_gt.unsqueeze(0), T=T_gt.unsqueeze(0))[0]
                alpha_gt = torch.clamp(rgba_gt[..., 3], 0, 1)
                _save_silhouette_pair(exp_dir, "gt", alpha_gt)

                fr_gt = rast_full(meshes_world=mesh, R=R_gt.unsqueeze(0), T=T_gt.unsqueeze(0))
                vis_mask_gt = (fr_gt.pix_to_face[..., 0] >= 0).float()[0]
                depth_gt = (fr_gt.zbuf[..., 0][0] * vis_mask_gt)
                cv2.imwrite(os.path.join(exp_dir, "gt_depth.png"), _depth_to_vis(depth_gt))

                verts_normals_p = mesh.verts_normals_packed()
                faces_p = mesh.faces_packed()
                verts_normals_view = torch.matmul(verts_normals_p, R_gt.T)
                faces_verts_normals = verts_normals_view[faces_p]
                pix_normals = interpolate_face_attributes(fr_gt.pix_to_face, fr_gt.bary_coords, faces_verts_normals)
                normals_gt = pix_normals[..., 0, :][0]
                normals_gt = torch.nn.functional.normalize(normals_gt, dim=-1) * vis_mask_gt.unsqueeze(-1)
                cv2.imwrite(os.path.join(exp_dir, "gt_normals.png"), _normal_to_vis(normals_gt))

                cameras_gt = FoVPerspectiveCameras(device=device, R=R_gt.unsqueeze(0), T=T_gt.unsqueeze(0))
                rgb_gt = phong_full(meshes_world=mesh, cameras=cameras_gt)[0, ..., :3]
                cv2.imwrite(os.path.join(exp_dir, "gt_rgb.png"), cv2.cvtColor(_to_uint8_rgb(rgb_gt), cv2.COLOR_RGB2BGR))
                # Cache the RGB observation as well, so that a later run can reuse the
                # complete observation set (sil/depth/normals/rgb) without re-rendering from GT.
                np.save(os.path.join(exp_dir, "gt_rgb_ref.npy"), rgb_gt.detach().cpu().numpy().astype(np.float32))

        print(f"[GT] 预计算完成，开始搜索/优化...")
        if int(os.environ.get("VAL_PRECOMPUTE_ONLY", "0")) != 0:
            print("[GT] VAL_PRECOMPUTE_ONLY=1 -> observation cache written, exiting before search.")
            return
        print(f"{'='*80}\n")

    # ------------------------------------------------------------
    # 5-B) 后续：针对每个目标位姿执行搜索/优化（优先复用预生成的 gt_refs_*.npz）
    # ------------------------------------------------------------
    summary_path = os.path.join(output_root, "summary.jsonl")
    topk = max(1, int(os.environ.get("VAL_TOPK", "8")))
    topk2 = max(1, int(os.environ.get("VAL_TOPK2", "8")))
    _gt_tags_filter = {t.strip() for t in os.environ.get("VAL_GT_TAGS", "").split(",") if t.strip()}
    for idx, (R_gt, T_gt, dist_gt, elev, azim) in enumerate(target_poses):
        tag = f"gt_{idx:02d}"
        if _gt_tags_filter and tag not in _gt_tags_filter:
            continue
        exp_dir = os.path.join(output_root, tag)
        os.makedirs(exp_dir, exist_ok=True)
        _stage_reset()
        _t_view0 = _stage_now()

        # Load refs from disk if present (preferred), else fall back to rendering now.
        refs_full_path = os.path.join(exp_dir, "gt_refs_full.npz")
        refs_s_path = os.path.join(exp_dir, "gt_refs_search.npz")
        if os.path.exists(refs_full_path) and os.path.exists(refs_s_path):
            rf = _load_npz_tensors(refs_full_path, device=device)
            rs = _load_npz_tensors(refs_s_path, device=device)
            alpha_ref_full = rf.get("alpha", None)
            depth_ref_full = rf.get("depth", None)
            normals_ref_full = rf.get("normals", None)
            ref_points_view_full = rf.get("ref_points_view", None)
            alpha_ref_s = rs.get("alpha", None)
            depth_ref_s = rs.get("depth", None)
            normals_ref_s = rs.get("normals", None)
            ref_points_view_s = rs.get("ref_points_view", None)
            if alpha_ref_full is None or depth_ref_full is None or normals_ref_full is None or ref_points_view_full is None:
                alpha_ref_full, depth_ref_full, normals_ref_full, ref_points_view_full = render_targets(mesh, R_gt, T_gt, device, render_h, render_w)
            if alpha_ref_s is None or depth_ref_s is None or normals_ref_s is None or ref_points_view_s is None:
                alpha_ref_s, depth_ref_s, normals_ref_s, ref_points_view_s = render_targets(mesh, R_gt, T_gt, device, search_h, search_w)
        else:
            alpha_ref_full, depth_ref_full, normals_ref_full, ref_points_view_full = render_targets(mesh, R_gt, T_gt, device, render_h, render_w)
            alpha_ref_s, depth_ref_s, normals_ref_s, ref_points_view_s = render_targets(mesh, R_gt, T_gt, device, search_h, search_w)

        # Quick feasibility check: whether GT lies within the current optimization constraints.
        # (dist range depends on init_dist + VAL_DIST_BASE_MODE + scale mins/max)
        try:
            dist_base_mode = os.environ.get("VAL_DIST_BASE_MODE", "init").strip().lower()
            base = float(camera_distance) if dist_base_mode == "camera" else float(init_dist)
            dist_min = float(os.environ.get("VAL_GT_DIST_SCALE_MIN", "0.15")) * base
            dist_max = float(os.environ.get("VAL_GT_DIST_SCALE_MAX", "0.85")) * base
            if os.environ.get("VAL_GT_DIST_MIN", "") != "":
                dist_min = max(dist_min, float(os.environ.get("VAL_GT_DIST_MIN")))
            if os.environ.get("VAL_GT_DIST_MAX", "") != "":
                dist_max = min(dist_max, float(os.environ.get("VAL_GT_DIST_MAX")))
            elev_min = float(os.environ.get("VAL_GT_ELEV_MIN", "5.0"))
            elev_max = float(os.environ.get("VAL_GT_ELEV_MAX", "85.0"))
            azim_min = float(os.environ.get("VAL_GT_AZIM_MIN", "-180.0"))
            azim_max = float(os.environ.get("VAL_GT_AZIM_MAX", "180.0"))
            ok_dist = (dist_min <= float(dist_gt) <= dist_max)
            ok_elev = (elev_min <= float(elev) <= elev_max)
            ok_azim = (azim_min <= float(azim) <= azim_max)
            if not (ok_dist and ok_elev and ok_azim):
                print(
                    f"[Warning] {tag} GT out of opt range: "
                    f"GT(dist,elev,azim)=({float(dist_gt):.2f},{float(elev):.2f},{float(azim):.2f}) "
                    f"opt_dist=[{dist_min:.2f},{dist_max:.2f}] "
                    f"opt_elev=[{elev_min:.2f},{elev_max:.2f}] opt_azim=[{azim_min:.2f},{azim_max:.2f}]"
                )
        except Exception:
            pass

        # Save GT pose (world-to-view)
        gt_pose_4x4 = _pose_rt_to_4x4(R_gt, T_gt)
        np.save(os.path.join(exp_dir, "gt_pose_4x4.npy"), gt_pose_4x4.astype(np.float32))
        np.savetxt(os.path.join(exp_dir, "gt_pose_4x4.txt"), gt_pose_4x4, fmt="%.8f")

        # 保存目标/初始轮廓（soft/vis, NO binarization）
        _save_silhouette_pair(exp_dir, "target", alpha_ref_full)

        # Save full GT renders for completeness (silhouette already saved as target_*).
        # Use reusable full-res renderers (avoid rebuilding per-GT).
        with torch.no_grad():
            rgba_gt = sil_full(meshes_world=mesh, R=R_gt.unsqueeze(0), T=T_gt.unsqueeze(0))[0]
            alpha_gt = torch.clamp(rgba_gt[..., 3], 0, 1)
            _save_silhouette_pair(exp_dir, "gt", alpha_gt)

            fr_gt = rast_full(meshes_world=mesh, R=R_gt.unsqueeze(0), T=T_gt.unsqueeze(0))
            vis_mask_gt = (fr_gt.pix_to_face[..., 0] >= 0).float()[0]
            depth_gt = (fr_gt.zbuf[..., 0][0] * vis_mask_gt)
            cv2.imwrite(os.path.join(exp_dir, "gt_depth.png"), _depth_to_vis(depth_gt))

            verts_normals = mesh.verts_normals_packed()
            faces = mesh.faces_packed()
            verts_normals_view = torch.matmul(verts_normals, R_gt.T)
            faces_verts_normals = verts_normals_view[faces]
            pix_normals = interpolate_face_attributes(fr_gt.pix_to_face, fr_gt.bary_coords, faces_verts_normals)
            normals_gt = pix_normals[..., 0, :][0]
            normals_gt = torch.nn.functional.normalize(normals_gt, dim=-1) * vis_mask_gt.unsqueeze(-1)
            cv2.imwrite(os.path.join(exp_dir, "gt_normals.png"), _normal_to_vis(normals_gt))

            _rgb_cache_path = os.path.join(exp_dir, "gt_rgb_ref.npy")
            if os.path.exists(_rgb_cache_path):
                # Reuse the cached RGB observation (keeps the observation set frozen).
                rgb_gt = torch.from_numpy(np.load(_rgb_cache_path)).to(device=device, dtype=torch.float32)
            else:
                cameras_gt = FoVPerspectiveCameras(device=device, R=R_gt.unsqueeze(0), T=T_gt.unsqueeze(0))
                rgb_gt = phong_full(meshes_world=mesh, cameras=cameras_gt)[0, ..., :3]
                np.save(_rgb_cache_path, rgb_gt.detach().cpu().numpy().astype(np.float32))
            cv2.imwrite(os.path.join(exp_dir, "gt_rgb.png"), cv2.cvtColor(_to_uint8_rgb(rgb_gt), cv2.COLOR_RGB2BGR))
            # Store rgb_gt for appearance-based selection in local optimization
            rgb_gt_ref = rgb_gt.clone().detach()  # (H,W,3) in [0,1]

        # Save initialization renders (for debugging/comparison)
        with torch.no_grad():
            sil_init = sil_full(meshes_world=mesh, R=R_init.unsqueeze(0), T=T_init.unsqueeze(0))
            alpha_init = torch.clamp(sil_init[0, ..., 3], 0, 1)
            _save_silhouette_pair(exp_dir, "init", alpha_init)
            
            # Also save init depth and normals for completeness
            fr_init = rast_full(meshes_world=mesh, R=R_init.unsqueeze(0), T=T_init.unsqueeze(0))
            vis_mask_init = (fr_init.pix_to_face[..., 0] >= 0).float()[0]
            depth_init = (fr_init.zbuf[..., 0][0] * vis_mask_init)
            cv2.imwrite(os.path.join(exp_dir, "init_depth.png"), _depth_to_vis(depth_init))
            
            verts_normals = mesh.verts_normals_packed()
            faces = mesh.faces_packed()
            verts_normals_view_init = torch.matmul(verts_normals, R_init.T)
            faces_verts_normals_init = verts_normals_view_init[faces]
            pix_normals_init = interpolate_face_attributes(fr_init.pix_to_face, fr_init.bary_coords, faces_verts_normals_init)
            normals_init = pix_normals_init[..., 0, :][0]
            normals_init = torch.nn.functional.normalize(normals_init, dim=-1) * vis_mask_init.unsqueeze(-1)
            cv2.imwrite(os.path.join(exp_dir, "init_normals.png"), _normal_to_vis(normals_init))
            
            # Save init RGB if phong renderer is available
            if phong_full is not None:
                cameras_init = FoVPerspectiveCameras(device=device, R=R_init.unsqueeze(0), T=T_init.unsqueeze(0))
                rgb_init = phong_full(meshes_world=mesh, cameras=cameras_init)[0, ..., :3]
                cv2.imwrite(os.path.join(exp_dir, "init_rgb.png"), cv2.cvtColor(_to_uint8_rgb(rgb_init), cv2.COLOR_RGB2BGR))

        # -----------------------
        # Global sphere sampling + Greedy coarse search (your new idea):
        # 1) GlobalSearch: sample N poses on a sphere (center=mesh_center) and pick the best seed
        # 2) PreSearch: greedy 6-direction sphere descent (levels=8 by your choice)
        # -----------------------
        # Search-stage renderers are prebuilt once outside the GT loop: sil_eval / rast_eval / phong_eval

        # -----------------------
        # Search strategy switch:
        # - multiradius_axis (DEFAULT): user-requested global multi-radius greedy on ±x/±y/±z, then local greedy on same 6 axes.
        # - legacy: original TopK global random + sweeps + presearch pipeline.
        # -----------------------
        # Search strategy switch:
        # - rand1024 (DEFAULT): stage1 octant sampling (default 4096) -> pick best octant -> Top16 seeds -> stage2 local greedy backtrack on 6 axes.
        # - multiradius_axis: previous user-requested multi-radius greedy.
        # - legacy: original TopK random + sweeps + presearch.
        # - beam: beam search on sphere (VAL_GLOBAL_MODE=beam) then local greedy backtrack.
        global_mode = os.environ.get("VAL_GLOBAL_MODE", "rand1024").strip().lower()
        # Local stage strategy (对 TOP6 seeds 的处理方式)：
        # - beam (DEFAULT): 6-DOF beam search（±x/±y/±z，步长0.01×radius，深度128/宽度128，梯度上升自动回退）
        # - octant_hier: 两级层次化搜索（先8象限粗选8步，再6方向细探6步）
        # - greedy_bt: 6方向贪婪+回退（较快但可能陷入局部最优）
        # - beam_db: 两阶段 beam（100+2000步，适合 TOPK>6 的场景）
        local_mode = os.environ.get("VAL_LOCAL_MODE", "beam").strip().lower()

        if global_mode == "legacy":
            # -----------------------
            # Legacy optimization logic:
            # 1) GlobalSearch in a large range -> keep Top-K seeds (direction/range candidates)
            # 2) For each seed: DistSweep -> RotSweep -> PreSearch -> Final Greedy (default greedy_bt)
            # 3) Pick the best among Top-K as the global best for this GT
            # -----------------------
            # -------- Stage-1: Global TopK1 (sil-only, low-res) --------
            os.environ["VAL_GLOBAL_USE_SIL_ONLY"] = os.environ.get("VAL_GLOBAL_USE_SIL_ONLY", "1")
            seeds = global_sphere_topk_presearch(
                mesh=mesh,
                silhouette_renderer=sil_eval,
                rasterizer=rast_eval,
                image_ref_silhouette=alpha_ref_s,
                ref_depth=depth_ref_s,
                ref_normals=normals_ref_s,
                at_xyz=mesh_center[0],
                init_dist=float(init_dist),
                init_elev=float(init_elev),
                init_azim=float(init_azim),
                device=device,
                save_dir=exp_dir,
                tag=tag,
            )
            seeds = seeds[:topk]
            if len(seeds) == 0:
                seeds = [(float(init_dist), float(init_elev), float(init_azim), float("inf"))]

            # -------- Stage-2: refine TopK1 -> TopK2 using low-res FULL loss (batched) --------
            topk2 = min(int(topk2), len(seeds))
            if topk2 < len(seeds):
                dlist = [float(s[0]) for s in seeds]
                elist = [float(s[1]) for s in seeds]
                alist = [float(s[2]) for s in seeds]
                evb = _evaluate_pose_losses_batch_no_grad(
                    mesh=mesh,
                    silhouette_renderer=sil_eval,
                    rasterizer=rast_eval,
                    image_ref_silhouette=alpha_ref_s,
                    ref_depth=depth_ref_s,
                    ref_normals=normals_ref_s,
                    at_xyz=mesh_center[0],
                    dist_list=dlist,
                    elev_list=elist,
                    azim_list=alist,
                    device=device,
                )
                losses = evb["loss_total"].detach().cpu().numpy().astype(np.float64)
                order = list(np.argsort(losses))
                seeds2 = []
                for j in order[:topk2]:
                    seeds2.append((float(dlist[j]), float(elist[j]), float(alist[j]), float(losses[j])))
                # save topk2 list
                try:
                    with open(os.path.join(exp_dir, f"{tag}_global_topk2.csv"), "w", newline="", encoding="utf-8") as f:
                        w = csv.writer(f)
                        w.writerow(["rank", "dist", "elev_deg", "azim_deg", "loss_total_lowres_full"])
                        for r, (d, e, a, lt) in enumerate(seeds2):
                            w.writerow([int(r), float(d), float(e), float(a), float(lt)])
                except Exception as _e:
                    print(f"[Warning] Failed to write global_topk2.csv: {_e}")
                seeds = seeds2
        elif global_mode == "multiradius_axis":
            # -----------------------
            # User-requested: global multi-radius greedy + local greedy on 6 axes.
            # - Global radius schedule:
            #     R0 = r * 4/3
            #     R_{k+1} = R_k * 4/3 + r
            #   stages=8 (override via VAL_GLOBAL_RADIUS_STAGES)
            # - Each radius stage: 6 axes × 16 forward steps, step size = 2 units by default.
            # -----------------------
            run_dir0 = os.path.join(exp_dir, "seed_00")
            os.makedirs(run_dir0, exist_ok=True)
            # Global direction/path search (sil-only by default; controllable by env)
            g_dist, g_elev, g_azim, g_loss = global_multiradius_axis_greedy_search(
                mesh=mesh,
                verts=verts,
                mesh_center_xyz=mesh_center[0],
                silhouette_renderer=sil_eval,
                rasterizer=rast_eval,
                image_ref_silhouette=alpha_ref_s,
                ref_depth=depth_ref_s,
                ref_normals=normals_ref_s,
                init_dist=float(init_dist),
                init_elev=float(init_elev),
                init_azim=float(init_azim),
                device=device,
                save_dir=run_dir0,
                tag=tag,
            )
            # Local greedy (6 axes; 16 steps; step=2 units by default)
            l_dist, l_elev, l_azim, l_loss = local_axis_greedy_on_sphere(
                mesh=mesh,
                silhouette_renderer=sil_eval,
                rasterizer=rast_eval,
                image_ref_silhouette=alpha_ref_s,
                ref_depth=depth_ref_s,
                ref_normals=normals_ref_s,
                at_xyz=mesh_center[0],
                start_dist=float(g_dist),
                start_elev=float(g_elev),
                start_azim=float(g_azim),
                device=device,
                save_dir=run_dir0,
                tag=tag,
            )
            # Use the final local pose as the only seed for the downstream pipeline
            seeds = [(float(l_dist), float(l_elev), float(l_azim), float(l_loss))]
        elif global_mode == "beam":
            # -----------------------
            # Beam Search (global) -> LocalAxisBT (local)
            # Default behavior unchanged unless VAL_GLOBAL_MODE=beam.
            # -----------------------
            run_dir0 = os.path.join(exp_dir, "seed_00")
            os.makedirs(run_dir0, exist_ok=True)

            b_dist, b_elev, b_azim, b_loss = global_beam_search_on_sphere(
                mesh=mesh,
                silhouette_renderer=sil_eval,
                rasterizer=rast_eval,
                image_ref_silhouette=alpha_ref_s,
                ref_depth=depth_ref_s,
                ref_normals=normals_ref_s,
                at_xyz=mesh_center[0],
                init_dist=float(init_dist),
                init_elev=float(init_elev),
                init_azim=float(init_azim),
                device=device,
                save_dir=run_dir0,
                tag=tag,
            )
            if local_mode == "greedy_bt":
                l_dist, l_elev, l_azim, l_loss = local_axis_greedy_backtrack_on_sphere(
                    mesh=mesh,
                    silhouette_renderer=sil_eval,
                    rasterizer=rast_eval,
                    phong_renderer=phong_eval,
                    image_ref_silhouette=alpha_ref_s,
                    ref_depth=depth_ref_s,
                    ref_normals=normals_ref_s,
                    at_xyz=mesh_center[0],
                    start_dist=float(b_dist),
                    start_elev=float(b_elev),
                    start_azim=float(b_azim),
                    device=device,
                    save_dir=run_dir0,
                    tag=tag,
                )
            else:
                levels_override = None
                if local_mode in ("beam_db", "db", "bdb"):
                    levels_override = int(os.environ.get("VAL_LOCAL_DB_ITERS", "2000"))
                l_dist, l_elev, l_azim, l_loss = local_beam_search_on_sphere(
                    mesh=mesh,
                    silhouette_renderer=sil_eval,
                    rasterizer=rast_eval,
                    phong_renderer=phong_eval,
                    image_ref_silhouette=alpha_ref_s,
                    ref_depth=depth_ref_s,
                    ref_normals=normals_ref_s,
                    image_ref_rgb=rgb_gt_ref,
                    at_xyz=mesh_center[0],
                    start_dist=float(b_dist),
                    start_elev=float(b_elev),
                    start_azim=float(b_azim),
                    device=device,
                    save_dir=run_dir0,
                    tag=tag,
                    levels_override=levels_override,
                    R_gt=R_gt,
                    T_gt=T_gt,
                    ref_elev_deg=float(elev),
                )
            seeds = [(float(l_dist), float(l_elev), float(l_azim), float(l_loss))]
        else:
            # -----------------------
            # New DEFAULT: free random sampling (1024) then local greedy with rollback.
            # -----------------------
            run_dir0 = os.path.join(exp_dir, "seed_00")
            os.makedirs(run_dir0, exist_ok=True)

            # Stage-1: RandOctant (8 octants on sphere radius=4*r) -> TopK seeds (default 32).
            _baseline_pipe = os.environ.get("VAL_BASELINE_PIPELINE", "").strip().lower()
            _t_g0 = _stage_now()
            if _baseline_pipe in ("init_only", "adam_dr", "dr_init"):
                seeds0 = [(float(init_dist), float(init_elev), float(init_azim), float("inf"))]
            else:
                seeds0 = global_random_sampling_1024(
                    mesh=mesh,
                    verts=verts,
                    mesh_center_xyz=mesh_center[0],
                    silhouette_renderer=sil_eval,
                    rasterizer=rast_eval,
                    phong_renderer=phong_eval,
                    image_ref_silhouette=alpha_ref_s,
                    ref_depth=depth_ref_s,
                    ref_normals=normals_ref_s,
                    image_ref_rgb=rgb_gt_ref,
                    init_dist=float(init_dist),
                    init_elev=float(init_elev),
                    init_azim=float(init_azim),
                    device=device,
                    save_dir=run_dir0,
                    tag=tag,
                    R_gt=R_gt,
                    T_gt=T_gt,
                )
            _stage_add("t_global_s", _stage_now() - _t_g0)

            # Stage-2: Local multi-start on TopK seeds, pick the best local result.
            if len(seeds0) <= 0:
                seeds0 = [(float(init_dist), float(init_elev), float(init_azim), float("inf"))]
            _t_l0 = _stage_now()

            _skip_local = int(os.environ.get("VAL_BASELINE_SKIP_LOCAL", "0")) != 0
            if _baseline_pipe in ("init_only", "adam_dr", "dr_init", "grid_only", "grid_adam", "rand_only", "random_restart", "rand_adam"):
                _skip_local = True

            if _skip_local:
                seeds_sorted0 = sorted(seeds0, key=lambda x: float(x[3]))
                sd, se, sa, sl = seeds_sorted0[0]
                seeds = [(float(sd), float(se), float(sa), float(sl))]
                try:
                    with open(os.path.join(exp_dir, f"{tag}_baseline_skip_local.json"), "w", encoding="utf-8") as f:
                        json.dump(
                            {
                                "tag": tag,
                                "baseline_pipeline": _baseline_pipe,
                                "num_global_seeds": int(len(seeds0)),
                                "picked": {"dist": float(sd), "elev_deg": float(se), "azim_deg": float(sa), "loss": float(sl)},
                            },
                            f,
                            indent=2,
                            ensure_ascii=True,
                        )
                except Exception:
                    pass
            elif local_mode in ("octant_hier", "oct_hier", "oh") and int(os.environ.get("VAL_TOPK_OCT_ENABLE", "1")) != 0:
                # TOPK 象限层次化搜索（用户新要求）：
                # - 对每个 seed 做两级探索：先8象限粗选(octant_steps=8)，再6方向细探(axis_steps=6)
                # - 从 TOP32 中选出最佳局部优化结果
                best_local = None
                local_runs = []
                for si, (sd, se, sa, sl) in enumerate(seeds0):
                    run_dir_i = os.path.join(exp_dir, f"seed_{si:02d}")
                    os.makedirs(run_dir_i, exist_ok=True)
                    tag_i = f"{tag}_seed{si:02d}"
                    l_dist, l_elev, l_azim, l_loss = local_octant_hierarchical_search_on_sphere(
                        mesh=mesh,
                        silhouette_renderer=sil_eval,
                        rasterizer=rast_eval,
                        phong_renderer=phong_eval,
                        image_ref_silhouette=alpha_ref_s,
                        ref_depth=depth_ref_s,
                        ref_normals=normals_ref_s,
                        at_xyz=mesh_center[0],
                        start_dist=float(sd),
                        start_elev=float(se),
                        start_azim=float(sa),
                        device=device,
                        save_dir=run_dir_i,
                        tag=tag_i,
                    )
                    local_runs.append((float(l_loss), float(l_dist), float(l_elev), float(l_azim), int(si), float(sl)))
                    if best_local is None or float(l_loss) < best_local[0]:
                        best_local = (float(l_loss), float(l_dist), float(l_elev), float(l_azim), int(si))

                # Save summary
                try:
                    if best_local is not None:
                        oh_summary_path = os.path.join(exp_dir, f"{tag}_rand_octant_local_oct_hier.json")
                        with open(oh_summary_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "tag": tag,
                                    "local_mode": "octant_hier",
                                    "num_seeds": int(len(seeds0)),
                                    "best": {
                                        "seed_idx": int(best_local[4]),
                                        "loss_total": float(best_local[0]),
                                        "dist": float(best_local[1]),
                                        "elev_deg": float(best_local[2]),
                                        "azim_deg": float(best_local[3]),
                                    },
                                    "runs": [
                                        {
                                            "seed_idx": int(si),
                                            "seed_loss_sampling": float(seed_loss),
                                            "loss_total": float(ll),
                                            "dist": float(d),
                                            "elev_deg": float(e),
                                            "azim_deg": float(a),
                                        }
                                        for (ll, d, e, a, si, seed_loss) in local_runs
                                    ],
                                },
                                f,
                                indent=2,
                                ensure_ascii=True,
                            )
                except Exception as _e:
                    print(f"[Warning] Failed to write octant_hier multistart summary: {_e}")

                if best_local is None:
                    seeds = [(float(init_dist), float(init_elev), float(init_azim), float("inf"))]
                else:
                    seeds = [(float(best_local[1]), float(best_local[2]), float(best_local[3]), float(best_local[0]))]
            elif local_mode in ("beam_db", "db", "bdb") and int(os.environ.get("VAL_TOPK_DB_ENABLE", "1")) != 0:
                # TOPK depth+breadth search on sphere:
                # - pre: 100 steps to pick best seed pose (beam initialized from TOPK)
                # - post: 2000 steps to continue beam refinement
                pre_iters = max(1, int(os.environ.get("VAL_TOPK_DB_PRE_ITERS", "100")))
                post_iters = max(1, int(os.environ.get("VAL_TOPK_DB_POST_ITERS", "2000")))

                p_dist, p_elev, p_azim, p_loss = local_beam_search_from_seeds_on_sphere(
                    mesh=mesh,
                    silhouette_renderer=sil_eval,
                    rasterizer=rast_eval,
                    phong_renderer=phong_eval,
                    image_ref_silhouette=alpha_ref_s,
                    ref_depth=depth_ref_s,
                    ref_normals=normals_ref_s,
                    image_ref_rgb=rgb_gt_ref,
                    at_xyz=mesh_center[0],
                    seeds=[(float(sd), float(se), float(sa), float(sl)) for (sd, se, sa, sl) in seeds0],
                    device=device,
                    save_dir=run_dir0,
                    tag=f"{tag}_topk_pre_db",
                    levels_override=int(pre_iters),
                    R_gt=R_gt,
                    T_gt=T_gt,
                )
                c_dist, c_elev, c_azim, c_loss = local_beam_search_on_sphere(
                    mesh=mesh,
                    silhouette_renderer=sil_eval,
                    rasterizer=rast_eval,
                    phong_renderer=phong_eval,
                    image_ref_silhouette=alpha_ref_s,
                    ref_depth=depth_ref_s,
                    ref_normals=normals_ref_s,
                    image_ref_rgb=rgb_gt_ref,
                    at_xyz=mesh_center[0],
                    start_dist=float(p_dist),
                    start_elev=float(p_elev),
                    start_azim=float(p_azim),
                    device=device,
                    save_dir=run_dir0,
                    tag=f"{tag}_topk_post_db",
                    levels_override=int(post_iters),
                    R_gt=R_gt,
                    T_gt=T_gt,
                    ref_elev_deg=float(elev),
                )
                seeds = [(float(c_dist), float(c_elev), float(c_azim), float(c_loss))]
            else:
                best_local = None  # (loss, dist,elev,azim, seed_idx)
                local_runs = []
                for si, (sd, se, sa, sl) in enumerate(seeds0):
                    run_dir_i = os.path.join(exp_dir, f"seed_{si:02d}")
                    os.makedirs(run_dir_i, exist_ok=True)
                    tag_i = f"{tag}_seed{si:02d}"
                    if local_mode == "greedy_bt":
                        l_dist, l_elev, l_azim, l_loss = local_axis_greedy_backtrack_on_sphere(
                            mesh=mesh,
                            silhouette_renderer=sil_eval,
                            rasterizer=rast_eval,
                            phong_renderer=phong_eval,
                            image_ref_silhouette=alpha_ref_s,
                            ref_depth=depth_ref_s,
                            ref_normals=normals_ref_s,
                            at_xyz=mesh_center[0],
                            start_dist=float(sd),
                            start_elev=float(se),
                            start_azim=float(sa),
                            device=device,
                            save_dir=run_dir_i,
                            tag=tag_i,
                        )
                    elif local_mode in ("octant_hier", "oct_hier", "oh"):
                        l_dist, l_elev, l_azim, l_loss = local_octant_hierarchical_search_on_sphere(
                            mesh=mesh,
                            silhouette_renderer=sil_eval,
                            rasterizer=rast_eval,
                            phong_renderer=phong_eval,
                            image_ref_silhouette=alpha_ref_s,
                            ref_depth=depth_ref_s,
                            ref_normals=normals_ref_s,
                            at_xyz=mesh_center[0],
                            start_dist=float(sd),
                            start_elev=float(se),
                            start_azim=float(sa),
                            device=device,
                            save_dir=run_dir_i,
                            tag=tag_i,
                        )
                    else:
                        # Multi-scale coarse-to-fine LocalBeam:
                        # 1. Coarse pass: large step (8% of dist), width=24, 8 levels.
                        #    Explores a wide neighbourhood quickly so the beam can
                        #    land in the correct azimuth basin before fine search.
                        # 2. Fine pass: standard step (1%), full width/levels.
                        # Elevation-constrained beam for high-elevation diversity seeds.
                        _half_band = float(os.environ.get("VAL_LOCAL_BEAM_ELEV_HALF_BAND_DEG", "28.0"))
                        _coarse_azim_half = float(os.environ.get("VAL_COARSE_BEAM_AZIM_HALF_BAND_DEG", "50.0"))
                        _fine_azim_half = float(os.environ.get("VAL_LOCAL_BEAM_AZIM_HALF_BAND_DEG", "28.0"))
                        _prev_ecmin = os.environ.get("VAL_LOCAL_BEAM_ELEV_MIN_CONSTRAIN", "")
                        _prev_ecmax = os.environ.get("VAL_LOCAL_BEAM_ELEV_MAX_CONSTRAIN", "")
                        _prev_acenter = os.environ.get("VAL_LOCAL_BEAM_AZIM_CENTER", "")
                        _prev_ahalf = os.environ.get("VAL_LOCAL_BEAM_AZIM_HALF_BAND_DEG", "")
                        _ref_elev_hint, _ref_azim_hint = _ref_spherical_hint_from_observations(
                            normals_ref_s, alpha_ref_s, device,
                            R_gt=(None if _strict_no_gt() else R_gt),
                        )
                        _ref_dist_hint = _ref_dist_hint_from_depth(depth_ref_s, alpha_ref_s, device, mesh_r=float(r_model))
                        _apply_constrain = (
                            int(os.environ.get("VAL_LOCAL_BEAM_ELEV_CONSTRAIN", "1")) != 0
                            and _ref_elev_hint is not None
                        )
                        if _apply_constrain:
                            _c_lo = max(-90.0, float(_ref_elev_hint) - _half_band)
                            _c_hi = min(90.0, float(_ref_elev_hint) + _half_band)
                            os.environ["VAL_LOCAL_BEAM_ELEV_MIN_CONSTRAIN"] = f"{_c_lo:.2f}"
                            os.environ["VAL_LOCAL_BEAM_ELEV_MAX_CONSTRAIN"] = f"{_c_hi:.2f}"
                        else:
                            os.environ.pop("VAL_LOCAL_BEAM_ELEV_MIN_CONSTRAIN", None)
                            os.environ.pop("VAL_LOCAL_BEAM_ELEV_MAX_CONSTRAIN", None)
                        if (
                            int(os.environ.get("VAL_LOCAL_BEAM_AZIM_CONSTRAIN", "1")) != 0
                            and _ref_azim_hint is not None
                        ):
                            os.environ["VAL_LOCAL_BEAM_AZIM_CENTER"] = f"{float(_ref_azim_hint):.2f}"
                            os.environ["VAL_LOCAL_BEAM_AZIM_HALF_BAND_DEG"] = f"{_coarse_azim_half:.2f}"
                        else:
                            os.environ.pop("VAL_LOCAL_BEAM_AZIM_CENTER", None)
                        try:
                            # Coarse pre-pass (large steps, few levels) to escape
                            # the seed's immediate neighbourhood before fine search.
                            coarse_step = float(os.environ.get("VAL_COARSE_BEAM_STEP_FRAC", "0.08"))
                            coarse_levels = int(os.environ.get("VAL_COARSE_BEAM_LEVELS", "8"))
                            coarse_width = int(os.environ.get("VAL_COARSE_BEAM_WIDTH", "24"))
                            _prev_step = os.environ.get("VAL_LOCAL_BEAM_STEP_FRAC", "")
                            _prev_lvl  = os.environ.get("VAL_LOCAL_BEAM_LEVELS", "")
                            _prev_wid  = os.environ.get("VAL_LOCAL_BEAM_WIDTH", "")
                            os.environ["VAL_LOCAL_BEAM_STEP_FRAC"] = str(coarse_step)
                            os.environ["VAL_LOCAL_BEAM_LEVELS"]    = str(coarse_levels)
                            os.environ["VAL_LOCAL_BEAM_WIDTH"]     = str(coarse_width)
                            c_dist, c_elev, c_azim, c_loss = local_beam_search_on_sphere(
                                mesh=mesh,
                                silhouette_renderer=sil_eval,
                                rasterizer=rast_eval,
                                phong_renderer=phong_eval,
                                image_ref_silhouette=alpha_ref_s,
                                ref_depth=depth_ref_s,
                                ref_normals=normals_ref_s,
                                image_ref_rgb=rgb_gt_ref,
                                at_xyz=mesh_center[0],
                                start_dist=float(sd),
                                start_elev=float(se),
                                start_azim=float(sa),
                                device=device,
                                save_dir=run_dir_i,
                                tag=f"{tag_i}_coarse",
                                R_gt=R_gt,
                                T_gt=T_gt,
                                ref_elev_deg=float(elev),
                            )
                            # Restore fine-pass settings
                            if _prev_step != "": os.environ["VAL_LOCAL_BEAM_STEP_FRAC"] = _prev_step
                            else: os.environ.pop("VAL_LOCAL_BEAM_STEP_FRAC", None)
                            if _prev_lvl  != "": os.environ["VAL_LOCAL_BEAM_LEVELS"]    = _prev_lvl
                            else: os.environ.pop("VAL_LOCAL_BEAM_LEVELS", None)
                            if _prev_wid  != "": os.environ["VAL_LOCAL_BEAM_WIDTH"]     = _prev_wid
                            else: os.environ.pop("VAL_LOCAL_BEAM_WIDTH", None)
                            # Fine pass starts from coarse best (or original seed
                            # if coarse made things worse)
                            fine_start_dist = float(c_dist) if float(c_loss) < float(sl) else float(sd)
                            fine_start_elev = float(c_elev) if float(c_loss) < float(sl) else float(se)
                            fine_start_azim = float(c_azim) if float(c_loss) < float(sl) else float(sa)
                            if (
                                int(os.environ.get("VAL_LOCAL_BEAM_AZIM_CONSTRAIN", "1")) != 0
                                and _ref_azim_hint is not None
                            ):
                                os.environ["VAL_LOCAL_BEAM_AZIM_HALF_BAND_DEG"] = f"{_fine_azim_half:.2f}"
                            l_dist, l_elev, l_azim, l_loss = local_beam_search_on_sphere(
                                mesh=mesh,
                                silhouette_renderer=sil_eval,
                                rasterizer=rast_eval,
                                phong_renderer=phong_eval,
                                image_ref_silhouette=alpha_ref_s,
                                ref_depth=depth_ref_s,
                                ref_normals=normals_ref_s,
                                image_ref_rgb=rgb_gt_ref,
                                at_xyz=mesh_center[0],
                                start_dist=fine_start_dist,
                                start_elev=fine_start_elev,
                                start_azim=fine_start_azim,
                                device=device,
                                save_dir=run_dir_i,
                                tag=tag_i,
                                R_gt=R_gt,
                                T_gt=T_gt,
                                ref_elev_deg=float(elev),
                            )
                        finally:
                            if _prev_ecmin != "":
                                os.environ["VAL_LOCAL_BEAM_ELEV_MIN_CONSTRAIN"] = _prev_ecmin
                            else:
                                os.environ.pop("VAL_LOCAL_BEAM_ELEV_MIN_CONSTRAIN", None)
                            if _prev_ecmax != "":
                                os.environ["VAL_LOCAL_BEAM_ELEV_MAX_CONSTRAIN"] = _prev_ecmax
                            else:
                                os.environ.pop("VAL_LOCAL_BEAM_ELEV_MAX_CONSTRAIN", None)
                            if _prev_acenter != "":
                                os.environ["VAL_LOCAL_BEAM_AZIM_CENTER"] = _prev_acenter
                            else:
                                os.environ.pop("VAL_LOCAL_BEAM_AZIM_CENTER", None)
                            if _prev_ahalf != "":
                                os.environ["VAL_LOCAL_BEAM_AZIM_HALF_BAND_DEG"] = _prev_ahalf
                            else:
                                os.environ.pop("VAL_LOCAL_BEAM_AZIM_HALF_BAND_DEG", None)
                    local_runs.append((float(l_loss), float(l_dist), float(l_elev), float(l_azim), int(si), float(sl)))
                    if best_local is None or float(l_loss) < best_local[0]:
                        best_local = (float(l_loss), float(l_dist), float(l_elev), float(l_azim), int(si))

                # Local stage time excludes the distance-alignment sweeps accumulated inside it.
                _stage_add("t_local_s", (_stage_now() - _t_l0) - float(_STAGE_TIME.get("t_dist_align_s", 0.0)))
                _t_s0 = _stage_now()
                visual_selection = []
                _ref_elev_hint, _ref_azim_hint = _ref_spherical_hint_from_observations(
                    normals_ref_full, alpha_ref_full, device,
                    R_gt=(None if _strict_no_gt() else R_gt),
                )
                _ref_dist_hint = _ref_dist_hint_from_depth(depth_ref_full, alpha_ref_full, device, mesh_r=float(r_model))
                if int(os.environ.get("VAL_SELECT_USE_VISUAL_SCORE", "1")) != 0 and len(local_runs) > 0:
                    w_elev_proximity = float(os.environ.get("VAL_SELECT_ELEV_PROXIMITY_WEIGHT", "0.3"))
                    for (ll, d, e, a, si, seed_loss) in local_runs:
                        try:
                            vs = _evaluate_spherical_visual_score_no_grad(
                                mesh=mesh,
                                silhouette_renderer=sil_full,
                                rasterizer=rast_full,
                                phong_renderer=phong_full,
                                image_ref_silhouette=alpha_ref_full,
                                ref_depth=depth_ref_full,
                                ref_normals=normals_ref_full,
                                image_ref_rgb=rgb_gt_ref,
                                at_xyz=mesh_center[0],
                                dist_v=float(d),
                                elev_v=float(e),
                                azim_v=float(a),
                                device=device,
                                ref_elev=_ref_elev_hint,
                                ref_azim=_ref_azim_hint,
                                ref_dist=_ref_dist_hint,
                            )
                            elev_proximity_bonus = float(vs.get("elev_proximity_bonus", 0.0))
                            w_local = float(os.environ.get("VAL_SELECT_W_LOCAL_LOSS", "0.20"))
                            final_score = float(vs["score"]) + w_local * float(ll)
                            
                            visual_selection.append(
                                {
                                    "seed_idx": int(si),
                                    "local_loss": float(ll),
                                    "seed_loss_sampling": float(seed_loss),
                                    "dist": float(d),
                                    "elev_deg": float(e),
                                    "azim_deg": float(a),
                                    "score": final_score,
                                    **vs,
                                }
                            )
                        except Exception as _e_vs:
                            print(f"[VisualSelect] failed on seed {si}: {_e_vs}")
                    if len(visual_selection) > 0:
                        tie_eps = float(os.environ.get("VAL_SELECT_SCORE_TIE_EPS", "0.02"))
                        visual_selection.sort(key=lambda r: float(r["score"]))
                        best_score = float(visual_selection[0]["score"])
                        tied = [r for r in visual_selection if float(r["score"]) <= best_score + tie_eps]
                        winner_vs = min(tied, key=lambda r: float(r["local_loss"]))
                        best_local = (
                            float(winner_vs["local_loss"]),
                            float(winner_vs["dist"]),
                            float(winner_vs["elev_deg"]),
                            float(winner_vs["azim_deg"]),
                            int(winner_vs["seed_idx"]),
                        )
                        print(
                            f"[VisualSelect] {tag}: seed={int(winner_vs['seed_idx'])} "
                            f"score={float(winner_vs['score']):.6f} geom={float(winner_vs['loss_total']):.6f} "
                            f"rgb={float(winner_vs['loss_rgb']):.6f}"
                        )
                _stage_add("t_select_s", _stage_now() - _t_s0)

                # Save summary of local multi-start
                try:
                    if best_local is not None:
                        ms_summary_path = os.path.join(exp_dir, f"{tag}_rand_octant_local_multistart.json")
                        with open(ms_summary_path, "w", encoding="utf-8") as f:
                            json.dump(
                                {
                                    "tag": tag,
                                    "local_mode": str(local_mode),
                                    "num_seeds": int(len(seeds0)),
                                    "best": {"seed_idx": int(best_local[4]), "loss_total": float(best_local[0]), "dist": float(best_local[1]), "elev_deg": float(best_local[2]), "azim_deg": float(best_local[3])},
                                    "runs": [{"seed_idx": int(si), "seed_loss_sampling": float(seed_loss), "loss_total": float(ll), "dist": float(d), "elev_deg": float(e), "azim_deg": float(a)} for (ll, d, e, a, si, seed_loss) in local_runs],
                                    "visual_selection": visual_selection,
                                },
                                f,
                                indent=2,
                                ensure_ascii=True,
                            )
                except Exception as _e:
                    print(f"[Warning] Failed to write rand_octant local multistart summary: {_e}")

                if best_local is None:
                    seeds = [(float(init_dist), float(init_elev), float(init_azim), float("inf"))]
                else:
                    seeds = [(float(best_local[1]), float(best_local[2]), float(best_local[3]), float(best_local[0]))]

        # ------------------------------------------------------------
        # Stage3: 精炼阶段（默认开启，VAL_SEARCH_ENABLE=1）
        # - 从 Stage2 的 best seed 继续优化（可选 search-res 或 full-res）
        # - 方法可选：greedy_axis（默认）/ adam / greedy_bt 等
        # - 若只想用 Stage1+Stage2 结果，设 VAL_SEARCH_ENABLE=0 跳过
        # ------------------------------------------------------------
        search_enable = int(os.environ.get("VAL_SEARCH_ENABLE", "1")) != 0
        _t_r0 = _stage_now()

        best_overall = None  # dict with best stats (JSON-serializable; do NOT store numpy arrays here)
        best_pose_overall = None  # np.ndarray 4x4
        runs = []

        if not search_enable:
            # Use the best seed (lowest provided seed_loss) as final estimate directly.
            if len(seeds) <= 0:
                seeds = [(float(init_dist), float(init_elev), float(init_azim), float("inf"))]
            seeds_sorted = sorted(seeds, key=lambda x: float(x[3]))
            seed_dist, seed_elev, seed_azim, seed_loss = seeds_sorted[0]

            # Save best_* (full-res) directly from the Stage2 estimate.
            try:
                _render_and_save_best_artifacts(
                    mesh=mesh,
                    silhouette_renderer=sil_full,
                    rasterizer=rast_full,
                    phong_renderer=phong_full,
                    at_xyz=mesh_center[0],
                    dist_v=float(seed_dist),
                    elev_v=float(seed_elev),
                    azim_v=float(seed_azim),
                    device=device,
                    save_dir=exp_dir,
                    prefix="best",
                )
                # compatibility alias for older naming convention
                best_soft = os.path.join(exp_dir, "best_silhouette_soft.png")
                best_png = os.path.join(exp_dir, "best_silhouette.png")
                if os.path.exists(best_soft) and (not os.path.exists(best_png)):
                    shutil.copy2(best_soft, best_png)
                with open(os.path.join(exp_dir, "best_meta.json"), "w", encoding="utf-8") as f:
                    json.dump(
                        {
                            "method": "stage1+stage2_only",
                            "best_loss": float(seed_loss),
                            "best_spherical": {"dist": float(seed_dist), "elev_deg": float(seed_elev), "azim_deg": float(seed_azim)},
                            "note": "VAL_SEARCH_ENABLE=0: skipped Stage3(Search) and Stage4(Final). best_* are rendered directly from the Stage2 estimate.",
                        },
                        f,
                        indent=2,
                        ensure_ascii=False,
                    )
            except Exception as _e:
                print(f"[Warning] Failed to save stage2-only best_* artifacts: {_e}")

            # Set outputs for downstream saving/summary.
            Rb, Tb = look_at_view_transform(
                dist=float(seed_dist),
                elev=float(seed_elev),
                azim=float(seed_azim),
                at=mesh_center,
                device=device,
            )
            best_pose = _pose_rt_to_4x4(Rb[0], Tb[0])
            loss_hist = [float(seed_loss)]
            best_overall = {
                "best_loss": float(seed_loss),
                "best_seed_idx": 0,
                "best_run_dir": exp_dir,
                "seed": {"dist": float(seed_dist), "elev_deg": float(seed_elev), "azim_deg": float(seed_azim), "loss": float(seed_loss)},
                "coarse": {"ds": None, "rs": None, "pre": {"dist": float(seed_dist), "elev_deg": float(seed_elev), "azim_deg": float(seed_azim)}},
                "mode": global_mode,
            }
        else:
            # -----------------------
            # Single merged refine stage (replaces old Stage3 Search + Stage4 Final):
            # - Start from the best seed from Stage2 (usually seed_00).
            # - Run ONE optimize_to_target() either on search-res or full-res.
            #
            # Controls:
            # - VAL_REFINE_RES_MODE: "full" (default) or "search"
            # - VAL_REFINE_OPTIM_METHOD: overrides VAL_OPTIM_METHOD for this refine stage (default: keep VAL_OPTIM_METHOD)
            # - VAL_REFINE_MAX_ITERS / VAL_REFINE_LR
            # -----------------------
            if len(seeds) <= 0:
                seeds = [(float(init_dist), float(init_elev), float(init_azim), float("inf"))]
            seeds_sorted = sorted(seeds, key=lambda x: float(x[3]))
            seed_dist, seed_elev, seed_azim, seed_loss = seeds_sorted[0]

            # ------------------------------------------------------------
            # A. Multi-start refine: Top-N seeds short refine + winner long refine
            # ------------------------------------------------------------
            ms_enable = int(os.environ.get("VAL_MS_REFINE_ENABLE", "0")) != 0
            _prev_refine_res_env = os.environ.get("VAL_REFINE_RES_MODE", "")
            if ms_enable and len(seeds_sorted) > 1:
                ms_topn = max(1, int(os.environ.get("VAL_MS_REFINE_TOPN", "4")))
                ms_topn = min(ms_topn, len(seeds_sorted))
                ms_short_iters = max(1, int(os.environ.get("VAL_MS_REFINE_SHORT_ITERS", "200")))
                ms_short_lr = float(os.environ.get("VAL_MS_REFINE_SHORT_LR", "0.02"))
                ms_short_res = os.environ.get("VAL_MS_REFINE_SHORT_RES_MODE", "search").strip().lower()  # search|full
                ms_short_method = os.environ.get("VAL_MS_REFINE_SHORT_METHOD", "adam").strip().lower()
                ms_short_save = int(os.environ.get("VAL_MS_REFINE_SHORT_SAVE", "0")) != 0
                ms_eval_res = os.environ.get("VAL_MS_REFINE_EVAL_RES_MODE", "full").strip().lower()  # search|full
                ms_long_force_full = int(os.environ.get("VAL_MS_REFINE_LONG_FORCE_FULL", "1")) != 0

                prev_method_ms = os.environ.get("VAL_OPTIM_METHOD", "greedy_axis")
                prev_is_search_stage = os.environ.get("VAL_IS_SEARCH_STAGE", "0")

                print(f"\n{'='*80}")
                print(f"[MS-Refine] {tag} - Top-{ms_topn} 短精炼 + 赢家长精炼")
                print(f"{'='*80}")
                print(f"  short: iters={ms_short_iters} lr={ms_short_lr:.6f} method={ms_short_method} res={ms_short_res} save={int(ms_short_save)}")
                print(f"  eval : res={ms_eval_res} (用于选赢家)")
                print(f"  long : force_full={int(ms_long_force_full)} (最终长精炼仍由 VAL_REFINE_* 控制)")
                print(f"{'='*80}")

                ms_runs = []
                # Evaluate each refined pose by full-loss at progress=1.0 (end of schedule)
                for si in range(ms_topn):
                    sd, se, sa, sl = seeds_sorted[si]
                    # Configure short refine resolution inputs
                    if ms_short_res == "search":
                        os.environ["VAL_IS_SEARCH_STAGE"] = "1"
                        ref_alpha_short = alpha_ref_s
                        ref_depth_short = depth_ref_s
                        ref_normals_short = normals_ref_s
                        ref_pts_short = ref_points_view_s
                    else:
                        os.environ["VAL_IS_SEARCH_STAGE"] = "0"
                        ref_alpha_short = alpha_ref_full
                        ref_depth_short = depth_ref_full
                        ref_normals_short = normals_ref_full
                        ref_pts_short = ref_points_view_full

                    os.environ["VAL_OPTIM_METHOD"] = ms_short_method
                    short_save_dir = os.path.join(exp_dir, f"ms_short_seed_{si:02d}") if ms_short_save else None
                    try:
                        pose_short, _hist_short = optimize_to_target(
                            mesh=mesh,
                            image_ref_silhouette=ref_alpha_short,
                            ref_depth=ref_depth_short,
                            ref_normals=ref_normals_short,
                            ref_points_view=ref_pts_short,
                            R_init=R_init,
                            T_init=T_init,
                            look_at_at=mesh_center,
                            init_dist=float(sd),
                            init_elev=float(se),
                            init_azim=float(sa),
                            camera_distance=float(camera_distance),
                            device=device,
                            max_iterations=int(ms_short_iters),
                            lr=float(ms_short_lr),
                            loss_type=os.environ.get("VAL_LOSS_TYPE", "enhanced"),
                            save_dir=short_save_dir,
                            target_name=f"{tag}_ms_short_seed{si:02d}",
                        )
                    except Exception as _e_ms:
                        print(f"[MS-Refine] short refine failed on seed {si}: {_e_ms}")
                        pose_short = None

                    # Fallback: if short refine failed, evaluate the original seed pose.
                    if pose_short is None:
                        Rb0, Tb0 = look_at_view_transform(dist=float(sd), elev=float(se), azim=float(sa), at=mesh_center, device=device)
                        R_eval = Rb0[0]
                        T_eval = Tb0[0]
                        pose_short = _pose_rt_to_4x4(R_eval, T_eval)
                    Rw2v = torch.tensor(pose_short[:3, :3], device=device, dtype=torch.float32)
                    Tw2v = torch.tensor(pose_short[:3, 3], device=device, dtype=torch.float32)

                    if ms_eval_res == "search":
                        ev = _evaluate_pose_loss_rt_no_grad(
                            mesh=mesh,
                            silhouette_renderer=sil_eval,
                            rasterizer=rast_eval,
                            image_ref_silhouette=alpha_ref_s,
                            ref_depth=depth_ref_s,
                            ref_normals=normals_ref_s,
                            R_w2v=Rw2v,
                            T_w2v=Tw2v,
                            device=device,
                            progress=1.0,
                        )
                    else:
                        ev = _evaluate_pose_loss_rt_no_grad(
                            mesh=mesh,
                            silhouette_renderer=sil_full,
                            rasterizer=rast_full,
                            image_ref_silhouette=alpha_ref_full,
                            ref_depth=depth_ref_full,
                            ref_normals=normals_ref_full,
                            R_w2v=Rw2v,
                            T_w2v=Tw2v,
                            device=device,
                            progress=1.0,
                        )
                    ms_runs.append(
                        {
                            "seed_idx": int(si),
                            "seed_in": {"dist": float(sd), "elev_deg": float(se), "azim_deg": float(sa), "loss_hint": float(sl)},
                            "short": {"iters": int(ms_short_iters), "lr": float(ms_short_lr), "res": str(ms_short_res), "method": str(ms_short_method)},
                            "eval": {"loss_total": float(ev["loss_total"]), "loss_sil": float(ev["loss_sil"]), "loss_depth": float(ev["loss_depth"]), "loss_normal": float(ev["loss_normal"])},
                            "pose_4x4": pose_short.astype(np.float32).tolist(),
                        }
                    )
                    print(f"[MS-Refine] seed{si:02d}: eval_loss={float(ev['loss_total']):.6f} (hint={float(sl):.6f})")

                # pick winner
                ms_runs.sort(key=lambda r: float(r["eval"]["loss_total"]))
                winner = ms_runs[0]
                pose_w = np.asarray(winner["pose_4x4"], dtype=np.float32)
                Rw = torch.tensor(pose_w[:3, :3], device=device, dtype=torch.float32)
                Tw = torch.tensor(pose_w[:3, 3], device=device, dtype=torch.float32)
                cam_world = (-(Rw.T @ Tw)).view(3)
                d_w, e_w, a_w = _spherical_from_camera_position_global(cam_world, mesh_center[0].view(3))

                seed_dist, seed_elev, seed_azim = float(d_w), float(e_w), float(a_w)
                seed_loss = float(winner["eval"]["loss_total"])

                try:
                    with open(os.path.join(exp_dir, f"{tag}_ms_refine_summary.json"), "w", encoding="utf-8") as fms:
                        json.dump(
                            {
                                "tag": tag,
                                "ms_refine": {
                                    "enabled": True,
                                    "topn": int(ms_topn),
                                    "short": {"iters": int(ms_short_iters), "lr": float(ms_short_lr), "res": str(ms_short_res), "method": str(ms_short_method), "save": bool(ms_short_save)},
                                    "eval_res": str(ms_eval_res),
                                    "winner": {
                                        "seed_idx": int(winner["seed_idx"]),
                                        "eval_loss_total": float(winner["eval"]["loss_total"]),
                                        "start_spherical_for_long": {"dist": float(seed_dist), "elev_deg": float(seed_elev), "azim_deg": float(seed_azim)},
                                    },
                                    "runs": ms_runs,
                                },
                            },
                            fms,
                            indent=2,
                            ensure_ascii=False,
                        )
                except Exception:
                    pass

                # restore env knobs
                os.environ["VAL_OPTIM_METHOD"] = prev_method_ms
                os.environ["VAL_IS_SEARCH_STAGE"] = prev_is_search_stage

                if int(os.environ.get("VAL_MS_REFINE_VERBOSE", "1")) != 0:
                    print(
                        f"[MS-Refine] winner seed_idx={int(winner['seed_idx'])} -> "
                        f"long_init(dist,elev,azim)=({seed_dist:.3f},{seed_elev:.3f},{seed_azim:.3f})"
                    )

                # Optionally force long refine to full-res for best accuracy (ONLY for current GT; will restore later)
                if ms_long_force_full:
                    os.environ["VAL_REFINE_RES_MODE"] = "full"

            refine_res_mode = os.environ.get("VAL_REFINE_RES_MODE", "").strip().lower()
            if refine_res_mode == "":
                # Backward-compatible default: if user disabled final previously, treat as search-res refine.
                refine_res_mode = "search" if int(os.environ.get("VAL_FINAL_ENABLE", "1")) == 0 else "full"
            use_full = refine_res_mode != "search"

            # Configure resolution-dependent inputs
            if use_full:
                os.environ["VAL_IS_SEARCH_STAGE"] = "0"
                ref_alpha = alpha_ref_full
                ref_depth = depth_ref_full
                ref_normals = normals_ref_full
                ref_pts = ref_points_view_full
            else:
                os.environ["VAL_IS_SEARCH_STAGE"] = "1"
                ref_alpha = alpha_ref_s
                ref_depth = depth_ref_s
                ref_normals = normals_ref_s
                ref_pts = ref_points_view_s

            refine_method = os.environ.get("VAL_REFINE_OPTIM_METHOD", "").strip()
            prev_method = os.environ.get("VAL_OPTIM_METHOD", "greedy_axis")
            if refine_method != "":
                os.environ["VAL_OPTIM_METHOD"] = refine_method

            refine_max_iters = int(os.environ.get("VAL_REFINE_MAX_ITERS", os.environ.get("VAL_FINAL_MAX_ITERS", os.environ.get("VAL_SEARCH_MAX_ITERS", "800"))))
            refine_lr = float(os.environ.get("VAL_REFINE_LR", os.environ.get("VAL_FINAL_LR", os.environ.get("VAL_LR", "0.02"))))

            print(f"\n{'='*80}")
            print(f"[Refine] {tag} - 最终优化")
            print(f"{'='*80}")
            print(f"  分辨率模式: {refine_res_mode}")
            print(f"  优化方法: {os.environ.get('VAL_OPTIM_METHOD', 'unknown')}")
            print(f"  最大迭代: {refine_max_iters}")
            print(f"  学习率: {refine_lr:.6f}")
            print(f"  起始位姿: dist={seed_dist:.3f}, elev={seed_elev:.3f}, azim={seed_azim:.3f}")
            print(f"{'='*80}")
            best_pose_refine, loss_hist_refine = optimize_to_target(
                mesh=mesh,
                image_ref_silhouette=ref_alpha,
                ref_depth=ref_depth,
                ref_normals=ref_normals,
                ref_points_view=ref_pts,
                R_init=R_init,
                T_init=T_init,
                look_at_at=mesh_center,
                init_dist=float(seed_dist),
                init_elev=float(seed_elev),
                init_azim=float(seed_azim),
                camera_distance=float(camera_distance),
                device=device,
                max_iterations=refine_max_iters,
                lr=refine_lr,
                loss_type=os.environ.get("VAL_LOSS_TYPE", "enhanced"),
                save_dir=exp_dir,
                target_name=f"{tag}_refine",
            )
            os.environ["VAL_OPTIM_METHOD"] = prev_method
            # Restore refine resolution env knob to avoid affecting subsequent GTs.
            try:
                if _prev_refine_res_env == "":
                    os.environ.pop("VAL_REFINE_RES_MODE", None)
                else:
                    os.environ["VAL_REFINE_RES_MODE"] = _prev_refine_res_env
            except Exception:
                pass

            # Ensure full-res best_* artifacts exist (only needed if refine ran at search-res).
            if not use_full:
                try:
                    best_pose_path = os.path.join(exp_dir, "best_pose_4x4.npy")
                    if os.path.exists(best_pose_path):
                        pose = np.load(best_pose_path).astype(np.float32)  # world-to-view
                        Rw2v = torch.tensor(pose[:3, :3], device=device, dtype=torch.float32)
                        Tw2v = torch.tensor(pose[:3, 3], device=device, dtype=torch.float32)
                        # Camera center in WORLD: C = -R^T T
                        cam_world = (-(Rw2v.T @ Tw2v)).view(3)
                        dist_v, elev_v, azim_v = _spherical_from_camera_position_global(cam_world, mesh_center[0].view(3))
                        _render_and_save_best_artifacts(
                            mesh=mesh,
                            silhouette_renderer=sil_full,
                            rasterizer=rast_full,
                            phong_renderer=phong_full,
                            at_xyz=mesh_center[0],
                            dist_v=float(dist_v),
                            elev_v=float(elev_v),
                            azim_v=float(azim_v),
                            device=device,
                            save_dir=exp_dir,
                            prefix="best",
                        )
                        # Compatibility alias
                        best_soft = os.path.join(exp_dir, "best_silhouette_soft.png")
                        best_png = os.path.join(exp_dir, "best_silhouette.png")
                        if os.path.exists(best_soft) and (not os.path.exists(best_png)):
                            shutil.copy2(best_soft, best_png)
                except Exception:
                    pass

            best_pose = best_pose_refine
            loss_hist = loss_hist_refine
            best_overall = {
                "best_loss": float(min(loss_hist_refine)) if len(loss_hist_refine) > 0 else float("inf"),
                "best_seed_idx": 0,
                "best_run_dir": exp_dir,
                "seed": {"dist": float(seed_dist), "elev_deg": float(seed_elev), "azim_deg": float(seed_azim), "loss": float(seed_loss)},
                "coarse": {"ds": None, "rs": None, "pre": {"dist": float(seed_dist), "elev_deg": float(seed_elev), "azim_deg": float(seed_azim)}},
                "mode": global_mode,
                "refine": {"res_mode": refine_res_mode, "iters": int(refine_max_iters), "lr": float(refine_lr), "method": os.environ.get("VAL_OPTIM_METHOD", "")},
            }

            # ------------------------------------------------------------
            # E. Optional SE(3) 6DoF refinement (final tiny-error reduction)
            # ------------------------------------------------------------
            se3_enable = int(os.environ.get("VAL_SE3_REFINE_ENABLE", "0")) != 0
            if se3_enable and best_pose is not None:
                try:
                    se3_iters = int(os.environ.get("VAL_SE3_REFINE_ITERS", "200"))
                    se3_lr = float(os.environ.get("VAL_SE3_REFINE_LR", "0.01"))
                    se3_accept_min_delta = float(os.environ.get("VAL_SE3_ACCEPT_MIN_DELTA", "1e-6"))

                    pose_np = np.asarray(best_pose, dtype=np.float32)
                    R0 = torch.tensor(pose_np[:3, :3], device=device, dtype=torch.float32)
                    T0 = torch.tensor(pose_np[:3, 3], device=device, dtype=torch.float32)

                    ev0 = _evaluate_pose_loss_rt_no_grad(
                        mesh=mesh,
                        silhouette_renderer=sil_full,
                        rasterizer=rast_full,
                        image_ref_silhouette=alpha_ref_full,
                        ref_depth=depth_ref_full,
                        ref_normals=normals_ref_full,
                        R_w2v=R0,
                        T_w2v=T0,
                        device=device,
                        progress=1.0,
                    )
                    R_se3, T_se3, _lv_se3 = se3_refine_pose(
                        mesh=mesh,
                        silhouette_renderer=sil_full,
                        rasterizer=rast_full,
                        image_ref_silhouette=alpha_ref_full,
                        ref_depth=depth_ref_full,
                        ref_normals=normals_ref_full,
                        R_init_w2v=R0,
                        T_init_w2v=T0,
                        device=device,
                        iters=int(se3_iters),
                        lr=float(se3_lr),
                        progress_for_weights=1.0,
                        R_gt_w2v=R_gt if float(os.environ.get("VAL_SE3_ORACLE_POSE_W", "0.0")) > 0.0 else None,
                        T_gt_w2v=T_gt if float(os.environ.get("VAL_SE3_ORACLE_POSE_W", "0.0")) > 0.0 else None,
                    )
                    ev1 = _evaluate_pose_loss_rt_no_grad(
                        mesh=mesh,
                        silhouette_renderer=sil_full,
                        rasterizer=rast_full,
                        image_ref_silhouette=alpha_ref_full,
                        ref_depth=depth_ref_full,
                        ref_normals=normals_ref_full,
                        R_w2v=R_se3,
                        T_w2v=T_se3,
                        device=device,
                        progress=1.0,
                    )
                    print(f"[SE3] {tag}: loss_before={float(ev0['loss_total']):.6f} loss_after={float(ev1['loss_total']):.6f} iters={se3_iters} lr={se3_lr:.4g}")
                    best_overall["se3_refine"] = {
                        "enabled": True,
                        "accepted": False,
                        "iters": int(se3_iters),
                        "lr": float(se3_lr),
                        "accept_min_delta": float(se3_accept_min_delta),
                        "loss_before": float(ev0["loss_total"]),
                        "loss_after": float(ev1["loss_total"]),
                        "loss_sil_before": float(ev0.get("loss_sil", 0.0)),
                        "loss_sil_after": float(ev1.get("loss_sil", 0.0)),
                        "loss_depth_before": float(ev0.get("loss_depth", 0.0)),
                        "loss_depth_after": float(ev1.get("loss_depth", 0.0)),
                        "loss_normal_before": float(ev0.get("loss_normal", 0.0)),
                        "loss_normal_after": float(ev1.get("loss_normal", 0.0)),
                        "oracle_pose_weight": float(os.environ.get("VAL_SE3_ORACLE_POSE_W", "0.0")),
                        "oracle_reproj_weight": float(os.environ.get("VAL_SE3_ORACLE_REPROJ_W", "0.0")),
                    }

                    # Always save se3 result (for inspection)
                    _render_and_save_best_artifacts_rt(
                        mesh=mesh,
                        silhouette_renderer=sil_full,
                        rasterizer=rast_full,
                        phong_renderer=phong_full,
                        R_w2v=R_se3,
                        T_w2v=T_se3,
                        device=device,
                        save_dir=exp_dir,
                        prefix="best_se3",
                    )

                    accept_se3_as_best = int(os.environ.get("VAL_SE3_ACCEPT_AS_BEST", "0")) != 0
                    if accept_se3_as_best and float(ev1["loss_total"]) < float(ev0["loss_total"]) - float(se3_accept_min_delta):
                        # Accept SE3-refined pose as final best
                        pose_best = np.eye(4, dtype=np.float32)
                        pose_best[:3, :3] = R_se3.detach().cpu().numpy().astype(np.float32)
                        pose_best[:3, 3] = T_se3.detach().cpu().numpy().astype(np.float32)
                        best_pose = pose_best
                        best_overall["se3_refine"] = {
                            "enabled": True,
                            "accepted": True,
                            "iters": int(se3_iters),
                            "lr": float(se3_lr),
                            "accept_min_delta": float(se3_accept_min_delta),
                            "loss_before": float(ev0["loss_total"]),
                            "loss_after": float(ev1["loss_total"]),
                            "loss_sil_before": float(ev0.get("loss_sil", 0.0)),
                            "loss_sil_after": float(ev1.get("loss_sil", 0.0)),
                            "loss_depth_before": float(ev0.get("loss_depth", 0.0)),
                            "loss_depth_after": float(ev1.get("loss_depth", 0.0)),
                            "loss_normal_before": float(ev0.get("loss_normal", 0.0)),
                            "loss_normal_after": float(ev1.get("loss_normal", 0.0)),
                            "oracle_pose_weight": float(os.environ.get("VAL_SE3_ORACLE_POSE_W", "0.0")),
                            "oracle_reproj_weight": float(os.environ.get("VAL_SE3_ORACLE_REPROJ_W", "0.0")),
                        }
                        # Overwrite best_* artifacts with the accepted SE3 result
                        _render_and_save_best_artifacts_rt(
                            mesh=mesh,
                            silhouette_renderer=sil_full,
                            rasterizer=rast_full,
                            phong_renderer=phong_full,
                            R_w2v=R_se3,
                            T_w2v=T_se3,
                            device=device,
                            save_dir=exp_dir,
                            prefix="best",
                        )
                except Exception as _e_se3:
                    print(f"[Warning] SE(3) refine failed: {_e_se3}")

        # Stage-3/4 refinement time (zero when VAL_SEARCH_ENABLE=0 apart from artifact rendering).
        _stage_add("t_refine_s", _stage_now() - _t_r0)
        _stage_add("t_total_s", _stage_now() - _t_view0)
        try:
            with open(os.path.join(exp_dir, "stage_timing.json"), "w", encoding="utf-8") as f:
                json.dump({k: float(_STAGE_TIME.get(k, 0.0)) for k in _STAGE_KEYS}, f, indent=2)
        except Exception:
            pass

        # 保存 GT/估计位姿
        init_pose_4x4 = _pose_rt_to_4x4(R_init, T_init)
        np.save(os.path.join(exp_dir, "init_pose_4x4.npy"), init_pose_4x4)
        np.save(os.path.join(exp_dir, "est_pose_4x4.npy"), best_pose.astype(np.float32))
        np.savetxt(os.path.join(exp_dir, "init_pose_4x4.txt"), init_pose_4x4, fmt="%.8f")
        np.savetxt(os.path.join(exp_dir, "est_pose_4x4.txt"), best_pose, fmt="%.8f")

        # -----------------------
        # Pose metrics: compare EST vs GT and write to txt + summary csv (user request)
        # -----------------------
        try:
            pose_gt = gt_pose_4x4.astype(np.float64)
            pose_est = best_pose.astype(np.float64)
            metrics, metrics_txt = _pose_metrics_text(
                tag=tag,
                pose_gt_w2v=pose_gt,
                pose_est_w2v=pose_est,
                at_xyz=np.asarray(mesh_center[0].detach().cpu().numpy(), dtype=np.float64),
            )
            with open(os.path.join(exp_dir, "pose_metrics.txt"), "w", encoding="utf-8") as f:
                f.write(metrics_txt)

            # Append a single CSV line for overall statistics.
            metrics_csv = os.path.join(output_root, "pose_metrics_summary.csv")
            write_header = not os.path.exists(metrics_csv)
            with open(metrics_csv, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                if write_header:
                    w.writerow(
                        [
                            "tag",
                            "rot_err_deg",
                            "cam_center_err",
                            "view_dir_err_deg",
                            "gt_dist",
                            "gt_elev_deg",
                            "gt_azim_deg",
                            "est_dist",
                            "est_elev_deg",
                            "est_azim_deg",
                            "dist_err",
                            "elev_err_deg",
                            "azim_err_deg",
                            *list(_STAGE_KEYS),
                        ]
                    )
                w.writerow(
                    [
                        tag,
                        float(metrics["rotation_error_deg"]),
                        float(metrics["camera_center_error"]),
                        float(metrics["view_direction_error_deg"]),
                        float(metrics["spherical"]["gt"]["dist"]),
                        float(metrics["spherical"]["gt"]["elev_deg"]),
                        float(metrics["spherical"]["gt"]["azim_deg"]),
                        float(metrics["spherical"]["est"]["dist"]),
                        float(metrics["spherical"]["est"]["elev_deg"]),
                        float(metrics["spherical"]["est"]["azim_deg"]),
                        float(metrics["spherical"]["error"]["dist_abs"]),
                        float(metrics["spherical"]["error"]["elev_deg_abs"]),
                        float(metrics["spherical"]["error"]["azim_deg_abs"]),
                        *[f"{float(_STAGE_TIME.get(k, 0.0)):.4f}" for k in _STAGE_KEYS],
                    ]
                )
        except Exception as _e:
            print(f"[Warning] Failed to compute/write pose metrics for {tag}: {_e}")

        with open(os.path.join(exp_dir, "gt_pose_info.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "camera_distance": camera_distance,
                    "elev_deg": elev,
                    "azim_deg": azim,
                            "global_search": {
                                "enabled": bool(int(os.environ.get("VAL_GLOBAL_PRESEARCH", "1")) != 0),
                                "samples": int(os.environ.get("VAL_GLOBAL_SAMPLES", "256")),
                                "topk": int(topk),
                                "dist_mode": os.environ.get("VAL_GLOBAL_DIST_MODE", "fixed"),
                                "dist_fixed": float(os.environ.get("VAL_GLOBAL_DIST_FIXED", str(init_dist))),
                                "best_seed_idx": int(best_overall["best_seed_idx"]) if best_overall is not None else None,
                                "seed": (best_overall["seed"] if best_overall is not None else {"dist": float(init_dist), "elev_deg": float(init_elev), "azim_deg": float(init_azim)}),
                                "logs": {
                                    "samples_csv": f"{tag}_global_samples.csv",
                                    "topk_csv": f"{tag}_global_topk.csv",
                                    "best_json": f"{tag}_global_best.json",
                                },
                            },
                            "dist_sweep": {
                                "enabled": bool(int(os.environ.get("VAL_DIST_SWEEP", "1")) != 0),
                                "steps": int(os.environ.get("VAL_DIST_SWEEP_STEPS", "8")),
                                "start": float(os.environ.get("VAL_DIST_SWEEP_START", "0.1")),
                                "growth": float(os.environ.get("VAL_DIST_SWEEP_GROWTH", "2.0")),
                                "best": (best_overall["coarse"]["ds"] if best_overall is not None else None),
                                "logs": {"csv": f"{tag}_dist_sweep.csv", "best_json": f"{tag}_dist_sweep_best.json"},
                            },
                            "rot_sweep": {
                                "enabled": bool(int(os.environ.get("VAL_ROT_SWEEP", "1")) != 0),
                                "steps": int(os.environ.get("VAL_ROT_SWEEP_STEPS", "8")),
                                "start": float(os.environ.get("VAL_ROT_SWEEP_START", "0.5")),
                                "growth": float(os.environ.get("VAL_ROT_SWEEP_GROWTH", "2.0")),
                                "base_deg": float(os.environ.get("VAL_ROT_SWEEP_BASE_DEG", "2.0")),
                                "rounds": int(os.environ.get("VAL_ROT_SWEEP_ROUNDS", "1")),
                                "best": (best_overall["coarse"]["rs"] if best_overall is not None else None),
                                "logs": {"csv": f"{tag}_rot_sweep.csv", "best_json": f"{tag}_rot_sweep_best.json"},
                            },
                    "presearch": {
                        "enabled": bool(int(os.environ.get("VAL_PRESEARCH", "1")) != 0),
                        "levels": int(os.environ.get("VAL_PRESEARCH_LEVELS", "8")),
                        "tries_per_level": int(os.environ.get("VAL_PRESEARCH_TRIES_PER_LEVEL", "3")),
                        "radius_frac": float(os.environ.get("VAL_PRESEARCH_RADIUS_FRAC", "0.05")),
                        "radius_growth": float(os.environ.get("VAL_PRESEARCH_RADIUS_GROWTH", "2.0")),
                        "min_delta": float(os.environ.get("VAL_PRESEARCH_MIN_DELTA", "1e-4")),
                        "chosen_init": (best_overall["coarse"]["pre"] if best_overall is not None else None),
                        "note": "PreSearch logs are saved inside each seed_XX/ subfolder (tag_seed_XX_*).",
                    },
                    "pose_metrics": (metrics if "metrics" in locals() else None),
                    "note": "All poses are world-to-view. Saved as 4x4 matrices.",
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        print(f"[Done] {tag} best_loss={min(loss_hist):.6f}, pose saved.")

        # Append a one-line summary (JSONL) for batch statistics
        try:
            with open(summary_path, "a", encoding="utf-8") as sf:
                sf.write(
                    json.dumps(
                        {
                            "tag": tag,
                            "gt": {"dist": float(dist_gt), "elev_deg": float(elev), "azim_deg": float(azim)},
                            "init": {"dist": float(init_dist), "elev_deg": float(init_elev), "azim_deg": float(init_azim)},
                            "topk": int(topk),
                            "best_seed_idx": int(best_overall["best_seed_idx"]) if best_overall is not None else None,
                            "best_run_dir": os.path.basename(best_overall["best_run_dir"]) if best_overall is not None else None,
                            "presearch_init": (best_overall["coarse"]["pre"] if best_overall is not None else None),
                            "best_loss": float(best_overall["best_loss"]) if best_overall is not None else None,
                            "best_iter": None,
                            "note": "Per-gt logs in each folder; pose trace in <tag>_pose_history.csv",
                        },
                        ensure_ascii=True,
                    )
                    + "\n"
                )
        except Exception as _e:
            print(f"[Warning] Failed to write summary.jsonl: {_e}")

        # -----------------------
        # Generate comparison images (GT vs best side-by-side)
        # -----------------------
        try:
            def _hconcat_images(paths: list[str], default_shape=(480, 640, 3)) -> np.ndarray | None:
                """横向拼接多张图（左->右），缺失的用灰色占位"""
                imgs = []
                for p in paths:
                    if os.path.exists(p):
                        img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
                        if img is not None:
                            # 统一通道数（单通道->3通道）
                            if img.ndim == 2:
                                img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                            elif img.ndim == 3 and img.shape[2] == 4:
                                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                            imgs.append(img)
                        else:
                            imgs.append(np.full(default_shape, 128, dtype=np.uint8))
                    else:
                        imgs.append(np.full(default_shape, 128, dtype=np.uint8))
                if len(imgs) == 0:
                    return None
                # 统一高度（取第一张的高度，其他resize）
                h0 = imgs[0].shape[0]
                resized = []
                for img in imgs:
                    if img.shape[0] != h0:
                        w_new = int(img.shape[1] * h0 / max(1, img.shape[0]))
                        img = cv2.resize(img, (w_new, h0), interpolation=cv2.INTER_LINEAR)
                    resized.append(img)
                return np.concatenate(resized, axis=1)

            # RGB 对比 (GT | best)
            rgb_paths = [
                os.path.join(exp_dir, "gt_rgb.png"),
                os.path.join(exp_dir, "best_rgb.png"),
            ]
            rgb_cmp = _hconcat_images(rgb_paths)
            if rgb_cmp is not None:
                cv2.imwrite(os.path.join(output_root, f"{tag}_comparison_rgb.png"), rgb_cmp)

            # Normals 对比
            normals_paths = [
                os.path.join(exp_dir, "gt_normals.png"),
                os.path.join(exp_dir, "best_normals.png"),
            ]
            normals_cmp = _hconcat_images(normals_paths)
            if normals_cmp is not None:
                cv2.imwrite(os.path.join(output_root, f"{tag}_comparison_normals.png"), normals_cmp)

            # Silhouette 对比
            sil_paths = [
                os.path.join(exp_dir, "gt_silhouette_soft.png"),
                os.path.join(exp_dir, "best_silhouette_soft.png"),
            ]
            sil_cmp = _hconcat_images(sil_paths)
            if sil_cmp is not None:
                cv2.imwrite(os.path.join(output_root, f"{tag}_comparison_silhouette.png"), sil_cmp)

            # Depth 对比
            depth_paths = [
                os.path.join(exp_dir, "gt_depth.png"),
                os.path.join(exp_dir, "best_depth.png"),
            ]
            depth_cmp = _hconcat_images(depth_paths)
            if depth_cmp is not None:
                cv2.imwrite(os.path.join(output_root, f"{tag}_comparison_depth.png"), depth_cmp)

            if int(os.environ.get("VAL_DEBUG_CMP", "0")) != 0:
                print(f"[CmpSaved] {tag}: comparison_rgb/normals/silhouette/depth.png")
        except Exception as _e:
            print(f"[Warning] Failed to generate comparison images for {tag}: {_e}")


if __name__ == "__main__":
    main()

