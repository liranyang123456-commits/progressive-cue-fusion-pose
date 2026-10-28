# Benchmark 综合报告

生成时间 (UTC): 2026-06-13T06:01:19.171901+00:00

主日志: `F:\Diff_Pose_Estimate_VRHI\benchmark_results\master.jsonl`（含 6 项 pose 指标 / 视角）

指标: $e_R$ (rot), $e_t$ (cam_center), $e_v$ (view_dir), $|\Delta d|$, $|\Delta$elev, $|\Delta$azim; SR@5/10/15° = $e_R$ 成功率。

## bunny_100gt

### 主表：平均 $e_R$ (°)

| 方法 | mean $e_R$ | std | SR@5° | SR@10° | SR@15° | family |
|------|------------|-----|-------|--------|--------|--------|
| p3d_softsil_gt_init | 0.000 | 0.000 | 100% | 100% | 100% | dr_oracle |
| ours_v10 | 1.033 | 1.066 | 99% | 100% | 100% |  |
| ours_v9_geom | 7.148 | 22.857 | 75% | 90% | 96% | ours |
| ours_v12c | 9.363 | 29.396 | 67% | 90% | 96% | ours |
| ours_v9 | 9.387 | 28.482 | 66% | 89% | 96% | ours |
| ours_v9_se3 | 9.387 | 28.482 | 66% | 89% | 96% | ours |
| ours_v12f | 9.532 | 29.520 | 66% | 89% | 96% | ours |
| greedy_bt_local | 12.927 | 25.891 | 27% | 67% | 86% | ablation |
| rand_topk_only | 36.672 | 38.722 | 13% | 29% | 35% | ablation |
| grid_adam_top16 | 37.486 | 39.906 | 7% | 20% | 28% | dr_search |
| grid_adam | 40.509 | 41.457 | 7% | 19% | 29% | dr_search |
| grid_adam_gauss_soft | 41.730 | 41.495 | 3% | 18% | 27% | dr_soft |
| rand_restart_adam | 47.664 | 48.204 | 7% | 18% | 29% | dr_search |
| grid_hybrid | 49.248 | 44.761 | 2% | 10% | 22% | dr_search |
| adam_dr_sil_only | 82.583 | 62.519 | 20% | 22% | 23% | dr |
| p3d_softsil_reg | 102.032 | 48.378 | 0% | 0% | 2% | dr_p3d |
| p3d_softsil_spherical | 102.036 | 48.364 | 0% | 0% | 2% | dr_p3d |
| p3d_softsil_xyz | 106.342 | 48.220 | 0% | 0% | 2% | dr_p3d |
| adam_dr_hard | 106.640 | 48.179 | 0% | 2% | 2% | dr |
| adam_dr_gauss_soft | 107.099 | 49.023 | 0% | 2% | 2% | dr_soft |
| adam_dr | 107.340 | 50.087 | 1% | 4% | 4% | dr |
| hybrid_dr | 109.125 | 53.772 | 1% | 3% | 5% | dr |
| hybrid_dr_gauss_soft | 110.316 | 53.407 | 0% | 3% | 5% | dr_soft |
| p3d_softsil_grid | 115.678 | 45.126 | 0% | 0% | 0% | dr_p3d |

### 全指标均值（完整 $N$ 的方法）

| 方法 | e_R° | e_t | e_v° | |Δd| | |Δelev|° | |Δazim|° |
|------|------|------|------|------|------|------|
| p3d_softsil_gt_init | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ours_v10 | 1.033 | 0.005 | 0.951 | 0.003 | 0.791 | 0.520 |
| ours_v9_geom | 7.148 | 0.019 | 4.070 | 0.004 | 3.262 | 5.387 |
| ours_v12c | 9.363 | 0.018 | 4.139 | 0.005 | 3.058 | 7.599 |
| ours_v9 | 9.387 | 0.019 | 4.186 | 0.005 | 3.096 | 7.573 |
| ours_v9_se3 | 9.387 | 0.019 | 4.186 | 0.005 | 3.096 | 7.573 |
| ours_v12f | 9.532 | 0.019 | 4.235 | 0.004 | 3.154 | 7.720 |
| greedy_bt_local | 12.927 | 0.036 | 7.178 | 0.011 | 5.332 | 10.003 |
| rand_topk_only | 36.672 | 0.115 | 22.684 | 0.023 | 10.674 | 33.057 |
| grid_adam_top16 | 37.486 | 0.118 | 26.250 | 0.042 | 15.153 | 31.409 |
| grid_adam | 40.509 | 0.126 | 28.946 | 0.040 | 15.535 | 34.480 |
| grid_adam_gauss_soft | 41.730 | 0.130 | 29.779 | 0.042 | 15.979 | 35.760 |
| rand_restart_adam | 47.664 | 0.142 | 32.308 | 0.038 | 18.619 | 41.033 |
| grid_hybrid | 49.248 | 0.150 | 33.936 | 0.067 | 17.387 | 43.386 |
| adam_dr_sil_only | 82.583 | 0.266 | 57.125 | 0.054 | 20.129 | 77.765 |
| p3d_softsil_reg | 102.032 | 0.625 | 68.544 | 0.397 | 23.709 | 97.461 |
| p3d_softsil_spherical | 102.036 | 0.620 | 68.528 | 0.392 | 23.691 | 97.466 |
| p3d_softsil_xyz | 106.342 | 0.801 | 74.405 | 0.555 | 46.927 | 90.559 |
| adam_dr_hard | 106.640 | 0.403 | 75.769 | 0.144 | 32.770 | 97.266 |
| adam_dr_gauss_soft | 107.099 | 0.395 | 74.751 | 0.141 | 32.136 | 98.140 |
| adam_dr | 107.340 | 0.389 | 74.224 | 0.134 | 31.606 | 98.686 |
| hybrid_dr | 109.125 | 0.340 | 73.490 | 0.084 | 29.637 | 102.194 |
| hybrid_dr_gauss_soft | 110.316 | 0.346 | 74.936 | 0.091 | 29.092 | 103.269 |
| p3d_softsil_grid | 115.678 | 0.619 | 79.387 | 0.358 | 25.182 | 111.845 |

### SOTA / 外部 DR 子集（论文 Table SOTA）

| 方法 | 文献 | $e_R$↓ | $e_t$↓ | SR@10° |
|------|------|--------|--------|--------|
| adam_dr | PyTorch3D/nvdiffrast pose Adam (2020) | 107.34 (+98.0) | 0.389 | 4% |
| adam_dr_gauss_soft | Gaussian-soft sil (SoftRas-like) (2019) | 107.10 (+97.7) | 0.395 | 2% |
| grid_adam | Grid search + Adam DR (2020) | 40.51 (+31.1) | 0.126 | 19% |
| ours_v9 | VRHI v9 RandMultiR+Beam (2025) | 9.39 | 0.019 | 89% |
| p3d_softsil_grid | PyTorch3D SoftSil + grid init (2020) | 115.68 (+106.3) | 0.619 | 0% |
| p3d_softsil_spherical | PyTorch3D tutorial SoftSil spherical (2020) | 102.04 (+92.6) | 0.620 | 0% |
| p3d_softsil_xyz | PyTorch3D tutorial camera_position xyz (2020) | 106.34 (+97.0) | 0.801 | 0% |
| rand_restart_adam | Random-restart Adam DR (2020) | 47.66 (+38.3) | 0.142 | 18% |

## bunny_10gt

### 主表：平均 $e_R$ (°)

| 方法 | mean $e_R$ | std | SR@5° | SR@10° | SR@15° | family |
|------|------------|-----|-------|--------|--------|--------|
| p3d_softsil_gt_init | 0.000 | 0.000 | 100% | 100% | 100% | dr_oracle |
| ours_v10 | 0.882 | 1.235 | 100% | 100% | 100% |  |
| ours_v12c | 0.924 | 0.000 | 100% | 100% | 100% | ours |
| ours_v12f | 0.924 | 0.000 | 100% | 100% | 100% | ours |
| ours_v9_geom | 0.924 | 0.000 | 100% | 100% | 100% | ours |
| ours_v9_production | 3.828 | 3.712 | 80% | 90% | 100% |  |
| ours_v9 | 4.088 | 3.645 | 80% | 90% | 100% | ours |
| ours_v9_se3 | 4.088 | 3.645 | 80% | 90% | 100% | ours |
| greedy_bt_local | 10.027 | 0.000 | 0% | 0% | 100% | ablation |
| rand_topk_only | 39.160 | 0.000 | 0% | 0% | 0% | ablation |
| grid_adam_top16 | 50.373 | 0.000 | 0% | 0% | 0% | dr_search |
| grid_adam | 52.646 | 51.573 | 10% | 20% | 20% | dr_search |
| grid_adam_gauss_soft | 53.112 | 48.848 | 0% | 20% | 20% | dr_soft |
| grid_hybrid | 66.450 | 61.616 | 0% | 10% | 10% | dr_search |
| rand_restart_adam | 84.242 | 65.904 | 10% | 20% | 20% | dr_search |
| p3d_softsil_xyz | 89.617 | 55.686 | 0% | 0% | 0% | dr_p3d |
| adam_dr_sil_only | 95.114 | 70.071 | 20% | 20% | 20% | dr |
| adam_dr_gauss_soft | 97.198 | 52.265 | 0% | 0% | 0% | dr_soft |
| adam_dr_hard | 97.466 | 52.659 | 0% | 0% | 0% | dr |
| p3d_softsil_grid | 97.753 | 53.922 | 0% | 0% | 0% | dr_p3d |
| adam_dr | 98.947 | 53.900 | 0% | 0% | 0% | dr |
| p3d_softsil_spherical | 104.330 | 51.681 | 0% | 0% | 0% | dr_p3d |
| p3d_softsil_reg | 104.394 | 51.623 | 0% | 0% | 0% | dr_p3d |
| hybrid_dr | 106.985 | 50.661 | 0% | 0% | 0% | dr |
| hybrid_dr_gauss_soft | 108.123 | 51.805 | 0% | 0% | 0% | dr_soft |

### 全指标均值（完整 $N$ 的方法）

| 方法 | e_R° | e_t | e_v° | |Δd| | |Δelev|° | |Δazim|° |
|------|------|------|------|------|------|------|
| p3d_softsil_gt_init | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |
| ours_v10 | 0.882 | 0.004 | 0.765 | 0.002 | 0.524 | 0.677 |
| ours_v9_production | 3.828 | 0.017 | 2.946 | 0.003 | 2.539 | 1.967 |
| ours_v9 | 4.088 | 0.018 | 3.177 | 0.003 | 2.765 | 2.107 |
| ours_v9_se3 | 4.088 | 0.018 | 3.177 | 0.003 | 2.765 | 2.107 |
| grid_adam | 52.646 | 0.140 | 36.071 | 0.044 | 16.526 | 47.577 |
| grid_adam_gauss_soft | 53.112 | 0.146 | 36.487 | 0.049 | 17.237 | 47.707 |
| grid_hybrid | 66.450 | 0.152 | 40.202 | 0.070 | 18.840 | 62.227 |
| rand_restart_adam | 84.242 | 0.201 | 49.233 | 0.038 | 25.044 | 79.205 |
| p3d_softsil_xyz | 89.617 | 0.843 | 78.137 | 0.565 | 58.274 | 76.876 |
| adam_dr_sil_only | 95.114 | 0.217 | 49.979 | 0.056 | 15.374 | 94.419 |
| adam_dr_gauss_soft | 97.198 | 0.351 | 66.476 | 0.136 | 43.313 | 81.282 |
| adam_dr_hard | 97.466 | 0.357 | 67.062 | 0.141 | 43.959 | 80.800 |
| p3d_softsil_grid | 97.753 | 0.546 | 59.411 | 0.387 | 31.822 | 90.309 |
| adam_dr | 98.947 | 0.337 | 64.774 | 0.126 | 41.121 | 83.577 |
| p3d_softsil_spherical | 104.330 | 0.585 | 63.870 | 0.408 | 31.107 | 97.009 |
| p3d_softsil_reg | 104.394 | 0.554 | 63.970 | 0.374 | 31.171 | 97.035 |
| hybrid_dr | 106.985 | 0.303 | 68.259 | 0.081 | 36.763 | 90.559 |
| hybrid_dr_gauss_soft | 108.123 | 0.317 | 71.007 | 0.088 | 39.079 | 91.816 |

### SOTA / 外部 DR 子集（论文 Table SOTA）

| 方法 | 文献 | $e_R$↓ | $e_t$↓ | SR@10° |
|------|------|--------|--------|--------|
| adam_dr | PyTorch3D/nvdiffrast pose Adam (2020) | 98.95 (+95.1) | 0.337 | 0% |
| adam_dr_gauss_soft | Gaussian-soft sil (SoftRas-like) (2019) | 97.20 (+93.4) | 0.351 | 0% |
| grid_adam | Grid search + Adam DR (2020) | 52.65 (+48.8) | 0.140 | 20% |
| ours_v9 | VRHI v9 RandMultiR+Beam (2025) | 4.09 (+0.3) | 0.018 | 90% |
| ours_v9_production | ours_v9_production () | 3.83 | 0.017 | 90% |
| p3d_softsil_grid | PyTorch3D SoftSil + grid init (2020) | 97.75 (+93.9) | 0.546 | 0% |
| p3d_softsil_spherical | PyTorch3D tutorial SoftSil spherical (2020) | 104.33 (+100.5) | 0.585 | 0% |
| p3d_softsil_xyz | PyTorch3D tutorial camera_position xyz (2020) | 89.62 (+85.8) | 0.843 | 0% |
| rand_restart_adam | Random-restart Adam DR (2020) | 84.24 (+80.4) | 0.201 | 20% |

## bunny_3gt

### 主表：平均 $e_R$ (°)

| 方法 | mean $e_R$ | std | SR@5° | SR@10° | SR@15° | family |
|------|------------|-----|-------|--------|--------|--------|
| ours_v9 | 3.172 | 1.709 | 67% | 100% | 100% | ours |
| grid_adam | 101.044 | 65.708 | 0% | 0% | 0% | dr_search |
| hybrid_dr | 103.063 | 40.795 | 0% | 0% | 0% | dr |
| adam_dr | 105.177 | 54.180 | 0% | 0% | 0% | dr |

### 全指标均值（完整 $N$ 的方法）

| 方法 | e_R° | e_t | e_v° | |Δd| | |Δelev|° | |Δazim|° |
|------|------|------|------|------|------|------|

### SOTA / 外部 DR 子集（论文 Table SOTA）

| 方法 | 文献 | $e_R$↓ | $e_t$↓ | SR@10° |
|------|------|--------|--------|--------|
| adam_dr | PyTorch3D/nvdiffrast pose Adam (2020) | 105.18 (+102.0) | 0.392 | 0% |
| grid_adam | Grid search + Adam DR (2020) | 101.04 (+97.9) | 0.197 | 0% |
| ours_v9 | VRHI v9 RandMultiR+Beam (2025) | 3.17 | 0.008 | 100% |

## dragon_100gt

### 主表：平均 $e_R$ (°)

| 方法 | mean $e_R$ | std | SR@5° | SR@10° | SR@15° | family |
|------|------------|-----|-------|--------|--------|--------|
| ours_v10 | 0.742 | 0.737 | 100% | 100% | 100% |  |
| p3d_softsil_gt_init | 3.700 | 2.598 | 64% | 100% | 100% | dr_oracle |
| ours_v9_geom | 11.037 | 15.356 | 49% | 67% | 81% | ours |
| greedy_bt_local | 20.257 | 14.081 | 3% | 19% | 42% | ablation |
| rand_topk_only | 25.525 | 32.997 | 20% | 45% | 63% | ablation |
| grid_adam_top16 | 31.297 | 33.748 | 15% | 32% | 50% | dr_search |
| ours_v12c | 35.055 | 21.399 | 3% | 8% | 20% | ours |
| rand_restart_adam | 36.686 | 41.653 | 19% | 36% | 52% | dr_search |
| ours_v12f | 37.638 | 22.318 | 1% | 6% | 17% | ours |
| ours_v9 | 39.110 | 23.449 | 2% | 8% | 19% | ours |
| ours_v9_se3 | 39.110 | 23.449 | 2% | 8% | 19% | ours |
| p3d_softsil_xyz | 53.087 | 62.643 | 51% | 51% | 51% | dr_p3d |
| grid_adam_gauss_soft | 63.925 | 39.498 | 0% | 2% | 2% | dr_soft |
| grid_adam | 64.788 | 42.773 | 2% | 4% | 6% | dr_search |
| grid_hybrid | 70.760 | 45.611 | 0% | 5% | 7% | dr_search |
| p3d_softsil_grid | 89.505 | 51.971 | 9% | 9% | 11% | dr_p3d |
| adam_dr_sil_only | 95.638 | 65.988 | 14% | 17% | 20% | dr |
| hybrid_dr_gauss_soft | 97.029 | 43.871 | 0% | 0% | 0% | dr_soft |
| p3d_softsil_reg | 97.202 | 50.485 | 7% | 8% | 8% | dr_p3d |
| adam_dr_hard | 97.417 | 43.059 | 0% | 2% | 2% | dr |
| hybrid_dr | 98.930 | 46.544 | 0% | 1% | 2% | dr |
| adam_dr_gauss_soft | 99.185 | 42.489 | 0% | 0% | 1% | dr_soft |
| p3d_softsil_spherical | 99.191 | 51.371 | 8% | 8% | 8% | dr_p3d |
| adam_dr | 99.517 | 42.756 | 0% | 0% | 1% | dr |

### 全指标均值（完整 $N$ 的方法）

| 方法 | e_R° | e_t | e_v° | |Δd| | |Δelev|° | |Δazim|° |
|------|------|------|------|------|------|------|
| ours_v10 | 0.742 | 5.688 | 0.695 | 4.253 | 0.570 | 0.371 |
| p3d_softsil_gt_init | 3.700 | nan | nan | nan | nan | nan |
| ours_v9_geom | 11.037 | 58.274 | 9.855 | 27.581 | 6.404 | 7.078 |
| greedy_bt_local | 20.257 | 222.348 | 14.921 | 218.700 | 9.810 | 15.359 |
| rand_topk_only | 25.525 | 99.188 | 20.856 | 43.647 | 8.915 | 21.078 |
| grid_adam_top16 | 31.297 | 125.878 | 22.433 | 70.996 | 13.941 | 26.305 |
| ours_v12c | 35.055 | 252.987 | 24.968 | 248.250 | 11.781 | 30.780 |
| rand_restart_adam | 36.686 | 133.799 | 26.529 | 74.370 | 14.188 | 32.022 |
| ours_v12f | 37.638 | 252.206 | 25.867 | 247.151 | 12.265 | 33.324 |
| ours_v9 | 39.110 | 252.307 | 25.906 | 247.214 | 12.150 | 34.921 |
| ours_v9_se3 | 39.110 | 252.307 | 25.906 | 247.214 | 12.150 | 34.921 |
| p3d_softsil_xyz | 53.087 | nan | nan | nan | nan | nan |
| grid_adam_gauss_soft | 63.925 | 242.714 | 40.687 | 225.540 | 24.291 | 54.761 |
| grid_adam | 64.788 | 243.267 | 41.333 | 225.179 | 24.273 | 55.600 |
| grid_hybrid | 70.760 | 248.894 | 42.427 | 233.749 | 25.768 | 61.397 |
| p3d_softsil_grid | 89.505 | nan | nan | nan | nan | nan |
| adam_dr_sil_only | 95.638 | 321.606 | 62.967 | 68.233 | 22.132 | 91.991 |
| hybrid_dr_gauss_soft | 97.029 | 354.862 | 67.971 | 92.526 | 24.742 | 90.501 |
| p3d_softsil_reg | 97.202 | nan | nan | nan | nan | nan |
| adam_dr_hard | 97.417 | 423.630 | 69.123 | 151.334 | 27.040 | 91.089 |
| hybrid_dr | 98.930 | 364.994 | 70.049 | 91.113 | 26.082 | 91.683 |
| adam_dr_gauss_soft | 99.185 | 429.346 | 70.426 | 152.595 | 26.075 | 92.619 |
| p3d_softsil_spherical | 99.191 | nan | nan | nan | nan | nan |
| adam_dr | 99.517 | 430.038 | 70.661 | 153.224 | 26.298 | 92.729 |

### SOTA / 外部 DR 子集（论文 Table SOTA）

| 方法 | 文献 | $e_R$↓ | $e_t$↓ | SR@10° |
|------|------|--------|--------|--------|
| adam_dr | PyTorch3D/nvdiffrast pose Adam (2020) | 99.52 (+60.4) | 430.038 | 0% |
| adam_dr_gauss_soft | Gaussian-soft sil (SoftRas-like) (2019) | 99.19 (+60.1) | 429.346 | 0% |
| grid_adam | Grid search + Adam DR (2020) | 64.79 (+25.7) | 243.267 | 4% |
| ours_v9 | VRHI v9 RandMultiR+Beam (2025) | 39.11 | 252.307 | 8% |
| p3d_softsil_grid | PyTorch3D SoftSil + grid init (2020) | 89.50 (+50.4) | nan | 9% |
| p3d_softsil_spherical | PyTorch3D tutorial SoftSil spherical (2020) | 99.19 (+60.1) | nan | 8% |
| p3d_softsil_xyz | PyTorch3D tutorial camera_position xyz (2020) | 53.09 (+14.0) | nan | 51% |
| rand_restart_adam | Random-restart Adam DR (2020) | 36.69 (-2.4) | 133.799 | 36% |

## dragon_10gt

### 主表：平均 $e_R$ (°)

| 方法 | mean $e_R$ | std | SR@5° | SR@10° | SR@15° | family |
|------|------------|-----|-------|--------|--------|--------|
| ours_v10 | 0.526 | 0.406 | 100% | 100% | 100% |  |
| rand_restart_adam | 20.068 | 26.453 | 20% | 50% | 70% | dr_search |
| ours_v9_se3 | 31.982 | 28.604 | 20% | 30% | 50% | ours |
| ours_v9 | 41.547 | 29.038 | 0% | 20% | 40% | ours |
| grid_hybrid | 64.388 | 39.962 | 10% | 20% | 20% | dr_search |
| grid_adam | 74.135 | 42.958 | 0% | 0% | 0% | dr_search |
| grid_adam_gauss_soft | 76.603 | 37.635 | 0% | 0% | 0% | dr_soft |
| p3d_softsil_spherical | 83.372 | 61.711 | 20% | 20% | 20% | dr_p3d |
| p3d_softsil_grid | 85.293 | 60.060 | 20% | 20% | 20% | dr_p3d |
| hybrid_dr_gauss_soft | 90.924 | 45.983 | 0% | 0% | 0% | dr_soft |
| hybrid_dr | 93.067 | 43.681 | 0% | 0% | 0% | dr |
| adam_dr_sil_only | 96.099 | 65.294 | 20% | 20% | 20% | dr |
| adam_dr_hard | 106.335 | 45.892 | 0% | 0% | 0% | dr |
| adam_dr_gauss_soft | 107.071 | 42.709 | 0% | 0% | 0% | dr_soft |
| adam_dr | 107.260 | 44.312 | 0% | 0% | 0% | dr |

### 全指标均值（完整 $N$ 的方法）

| 方法 | e_R° | e_t | e_v° | |Δd| | |Δelev|° | |Δazim|° |
|------|------|------|------|------|------|------|
| ours_v10 | 0.526 | 3.704 | 0.505 | 2.871 | 0.478 | 0.197 |
| rand_restart_adam | 20.068 | 113.177 | 17.707 | 52.449 | 9.570 | 15.810 |
| ours_v9_se3 | 31.982 | 253.308 | 22.922 | 249.310 | 9.162 | 28.701 |
| ours_v9 | 41.547 | 220.155 | 33.236 | 207.634 | 19.256 | 34.236 |
| grid_hybrid | 64.388 | 246.722 | 42.421 | 229.806 | 25.936 | 50.668 |
| grid_adam | 74.135 | 243.784 | 47.077 | 221.427 | 29.215 | 62.280 |
| grid_adam_gauss_soft | 76.603 | 245.435 | 48.847 | 222.684 | 31.146 | 63.590 |
| p3d_softsil_spherical | 83.372 | nan | nan | nan | nan | nan |
| p3d_softsil_grid | 85.293 | nan | nan | nan | nan | nan |
| hybrid_dr_gauss_soft | 90.924 | 348.964 | 62.525 | 108.542 | 25.328 | 79.826 |
| hybrid_dr | 93.067 | 365.725 | 66.888 | 111.182 | 27.609 | 80.596 |
| adam_dr_sil_only | 96.099 | 288.372 | 58.745 | 77.535 | 27.609 | 90.006 |
| adam_dr_hard | 106.335 | 411.296 | 66.696 | 154.463 | 30.045 | 97.883 |
| adam_dr_gauss_soft | 107.071 | 427.003 | 69.578 | 154.209 | 33.093 | 97.616 |
| adam_dr | 107.260 | 415.951 | 68.997 | 154.038 | 31.226 | 98.141 |

### SOTA / 外部 DR 子集（论文 Table SOTA）

| 方法 | 文献 | $e_R$↓ | $e_t$↓ | SR@10° |
|------|------|--------|--------|--------|
| adam_dr | PyTorch3D/nvdiffrast pose Adam (2020) | 107.26 (+65.7) | 415.951 | 0% |
| adam_dr_gauss_soft | Gaussian-soft sil (SoftRas-like) (2019) | 107.07 (+65.5) | 427.003 | 0% |
| grid_adam | Grid search + Adam DR (2020) | 74.14 (+32.6) | 243.784 | 0% |
| ours_v9 | VRHI v9 RandMultiR+Beam (2025) | 41.55 | 220.155 | 20% |
| p3d_softsil_grid | PyTorch3D SoftSil + grid init (2020) | 85.29 (+43.7) | nan | 20% |
| p3d_softsil_spherical | PyTorch3D tutorial SoftSil spherical (2020) | 83.37 (+41.8) | nan | 20% |
| rand_restart_adam | Random-restart Adam DR (2020) | 20.07 (-21.5) | 113.177 | 50% |

## dragon_3gt

### 主表：平均 $e_R$ (°)

| 方法 | mean $e_R$ | std | SR@5° | SR@10° | SR@15° | family |
|------|------------|-----|-------|--------|--------|--------|
| ours_v10 | 0.407 | 0.540 | 100% | 100% | 100% |  |

### 全指标均值（完整 $N$ 的方法）

| 方法 | e_R° | e_t | e_v° | |Δd| | |Δelev|° | |Δazim|° |
|------|------|------|------|------|------|------|

### SOTA / 外部 DR 子集（论文 Table SOTA）

| 方法 | 文献 | $e_R$↓ | $e_t$↓ | SR@10° |
|------|------|--------|--------|--------|

## 文献索引

详见 `DR_METHODS_LITERATURE.md`、`SOTA_COMPARISON_MATRIX.md`。
