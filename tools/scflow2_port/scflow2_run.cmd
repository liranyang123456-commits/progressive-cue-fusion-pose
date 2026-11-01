@echo off
setlocal enabledelayedexpansion
set LOG=%USERPROFILE%\scflow2_run.log
echo === RUN start %DATE% %TIME% > %LOG%
set PY=%USERPROFILE%\venvs\scflow2\Scripts\python.exe
set REPO=%USERPROFILE%\SCFlow2
set PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7\bin;%PATH%
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.7
set TORCH_HOME=%USERPROFILE%\torch_home
set PYTHONUNBUFFERED=1
set PYTHONIOENCODING=utf-8
set KMP_DUPLICATE_LIB_OK=TRUE
if not exist %REPO% (
  powershell -NoProfile -Command "Expand-Archive -Path %USERPROFILE%\scflow2_in\SCFlow2_repo.zip -DestinationPath %USERPROFILE%\SCFlow2 -Force" >> %LOG% 2>&1
)
if not exist %REPO%\data\lmo\image_lists\ours.txt powershell -NoProfile -Command "Expand-Archive -Path %USERPROFILE%\scflow2_in\scflow2_data.zip -DestinationPath %USERPROFILE%\SCFlow2 -Force" >> %LOG% 2>&1
if not exist %TORCH_HOME%\hub (
  powershell -NoProfile -Command "Expand-Archive -Path %USERPROFILE%\scflow2_in\torch_home.zip -DestinationPath %USERPROFILE%\torch_home -Force" >> %LOG% 2>&1
)
if not exist %REPO%\checkpoints mkdir %REPO%\checkpoints
if not exist %REPO%\checkpoints\scflow2_pretrained.pth copy /y %USERPROFILE%\scflow2_in\scflow2_pretrained.pth %REPO%\checkpoints\scflow2_pretrained.pth >> %LOG% 2>&1
cd /d %REPO%
dir data >> %LOG%
dir configs\ours >> %LOG%
for %%D in (lmo lm tless) do (
  for %%B in (15to30 30to60 60to90 90to180) do (
    echo --- %%D bin%%B %TIME% >> %LOG%
    %PY% test.py --config configs/ours/%%D_bin%%B.py --checkpoint checkpoints/scflow2_pretrained.pth --save-dir results/ours/%%D_bin%%B > %USERPROFILE%\scflow2_test_%%D_%%B.log 2>&1
    echo exit !ERRORLEVEL! >> %LOG%
  )
)
echo === RUN done %TIME% >> %LOG%
