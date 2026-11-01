@echo off
setlocal
set LOG=%USERPROFILE%\scflow2_setup.log
echo === STEP1 start %DATE% %TIME% >> %LOG%
if not exist C:\Python310\python.exe (
  echo downloading python 3.10 >> %LOG%
  powershell -NoProfile -Command "[Net.ServicePointManager]::SecurityProtocol='Tls12'; Invoke-WebRequest -Uri https://www.python.org/ftp/python/3.10.11/python-3.10.11-amd64.exe -OutFile %USERPROFILE%\python-3.10.11-amd64.exe" >> %LOG% 2>&1
  %USERPROFILE%\python-3.10.11-amd64.exe /quiet InstallAllUsers=1 PrependPath=0 Include_test=0 TargetDir=C:\Python310 >> %LOG% 2>&1
)
C:\Python310\python.exe --version >> %LOG% 2>&1
if not exist %USERPROFILE%\venvs\scflow2\Scripts\python.exe (
  C:\Python310\python.exe -m venv %USERPROFILE%\venvs\scflow2 >> %LOG% 2>&1
)
set PY=%USERPROFILE%\venvs\scflow2\Scripts\python.exe
%PY% -m pip install --upgrade pip==24.0 "setuptools<70" wheel ninja >> %LOG% 2>&1
%PY% -m pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117 >> %LOG% 2>&1
%PY% -c "import torch; print('torch', torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_name(0))" >> %LOG% 2>&1
%PY% -m pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu117/torch2.0.0/index.html >> %LOG% 2>&1
%PY% -c "import mmcv, mmcv.ops; print('mmcv', mmcv.__version__); from mmcv.ops import Correlation, furthest_point_sample, roi_align; print('mmcv ops ok')" >> %LOG% 2>&1
echo === STEP1 done %TIME% >> %LOG%
