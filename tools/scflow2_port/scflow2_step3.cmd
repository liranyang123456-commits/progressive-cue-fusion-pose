@echo off
setlocal
set LOG=%USERPROFILE%\scflow2_setup3.log
echo === STEP3 start %DATE% %TIME% > %LOG%
set PY=%USERPROFILE%\venvs\scflow2\Scripts\python.exe
set SRC=%USERPROFILE%\scflow2_src
set PATH=C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Tools\MSVC\14.29.30133\bin\Hostx64\x64;C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin;%PATH%
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7
set CUDA_PATH=%CUDA_HOME%
set DISTUTILS_USE_SDK=1
set MSSdk=1
set TORCH_CUDA_ARCH_LIST=7.5
set FORCE_CUDA=1
set MAX_JOBS=8
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Professional\VC\Auxiliary\Build\vcvars64.bat"
echo vcvars done >> %LOG%
if not exist %SRC% mkdir %SRC%
powershell -NoProfile -Command "Expand-Archive -Path %USERPROFILE%\scflow2_in\scflow2_src.zip -DestinationPath %USERPROFILE%\scflow2_src -Force" >> %LOG% 2>&1
dir %SRC% >> %LOG%
echo --- bop_toolkit >> %LOG%
%PY% -m pip install --no-deps %SRC%\bop_toolkit >> %LOG% 2>&1
%PY% -c "from bop_toolkit_lib.inout import load_depth; print('bop_toolkit ok')" >> %LOG% 2>&1
echo --- pointnet2_ops >> %LOG%
%PY% -m pip install --no-build-isolation -v %SRC%\Pointnet2_PyTorch\pointnet2_ops_lib > %USERPROFILE%\build_pointnet2.log 2>&1
echo pointnet2 pip exit %ERRORLEVEL% >> %LOG%
%PY% -c "from pointnet2_ops.pointnet2_utils import furthest_point_sample, gather_operation, QueryAndGroup; print('pointnet2_ops ok')" >> %LOG% 2>&1
echo --- lietorch >> %LOG%
%PY% -m pip install --no-build-isolation -v %SRC%\lietorch > %USERPROFILE%\build_lietorch.log 2>&1
echo lietorch pip exit %ERRORLEVEL% >> %LOG%
%PY% -c "from lietorch import SE3; import lietorch_extras; print('lietorch ok')" >> %LOG% 2>&1
echo --- pytorch3d >> %LOG%
%PY% -m pip install --no-build-isolation -v %SRC%\pytorch3d > %USERPROFILE%\build_pytorch3d.log 2>&1
echo pytorch3d pip exit %ERRORLEVEL% >> %LOG%
%PY% -c "import pytorch3d; from pytorch3d.renderer import MeshRasterizer; from pytorch3d.ops import knn_points; print('pytorch3d ok', pytorch3d.__version__)" >> %LOG% 2>&1
echo === STEP3 done %TIME% >> %LOG%
