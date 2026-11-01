@echo off
setlocal
set LOG=%USERPROFILE%\scflow2_setup2.log
echo === STEP2 start %DATE% %TIME% > %LOG%
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
echo vcvars done >> %LOG%
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7
set CUDA_PATH=%CUDA_HOME%
set PATH=%CUDA_HOME%\bin;%PATH%
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=7.5
set FORCE_CUDA=1
set MAX_JOBS=8
set MSSdk=1
set PY=%USERPROFILE%\venvs\scflow2\Scripts\python.exe
set SRC=%USERPROFILE%\scflow2_src
if not exist %SRC% mkdir %SRC%
where git >> %LOG% 2>&1
echo --- light deps >> %LOG%
%PY% -m pip install "numpy<2" opencv-python scipy scikit-image kornia==0.6.1 trimesh einops pyyaml terminaltables yapf matplotlib transforms3d imageio pypng pytz tqdm >> %LOG% 2>&1
%PY% -m pip install "git+https://github.com/thodan/bop_toolkit" >> %LOG% 2>&1
%PY% -c "import numpy, torch, kornia, trimesh; from bop_toolkit_lib.inout import load_depth; print('deps ok numpy', numpy.__version__)" >> %LOG% 2>&1
echo --- pointnet2_ops >> %LOG%
if not exist %SRC%\Pointnet2_PyTorch git clone --depth 1 https://github.com/erikwijmans/Pointnet2_PyTorch %SRC%\Pointnet2_PyTorch >> %LOG% 2>&1
%PY% -m pip install --no-build-isolation %SRC%\Pointnet2_PyTorch\pointnet2_ops_lib >> %LOG% 2>&1
%PY% -c "from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation, QueryAndGroup; print('pointnet2_ops ok')" >> %LOG% 2>&1
echo --- lietorch >> %LOG%
if not exist %SRC%\lietorch git clone --recursive https://github.com/princeton-vl/lietorch %SRC%\lietorch >> %LOG% 2>&1
%PY% -m pip install --no-build-isolation %SRC%\lietorch >> %LOG% 2>&1
%PY% -c "from lietorch import SE3; import lietorch_extras; print('lietorch ok')" >> %LOG% 2>&1
echo --- pytorch3d >> %LOG%
if not exist %SRC%\pytorch3d git clone --depth 1 -b v0.7.4 https://github.com/facebookresearch/pytorch3d %SRC%\pytorch3d >> %LOG% 2>&1
%PY% -m pip install --no-build-isolation %SRC%\pytorch3d >> %LOG% 2>&1
%PY% -c "import pytorch3d; from pytorch3d.renderer import MeshRasterizer; from pytorch3d.ops import knn_points; print('pytorch3d ok', pytorch3d.__version__)" >> %LOG% 2>&1
echo === STEP2 done %TIME% >> %LOG%
