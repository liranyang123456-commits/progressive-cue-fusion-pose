@echo off
setlocal
set LOG=%USERPROFILE%\scflow2_setup5.log
echo === STEP5 start %DATE% %TIME% > %LOG%
set PY=%USERPROFILE%\venvs\scflow2\Scripts\python.exe
%PY% -m pip install "numpy==1.26.4" "opencv-python==4.9.0.80" "opencv-python-headless==4.9.0.80" >> %LOG% 2>&1
%PY% -c "import numpy; print('numpy', numpy.__version__)" >> %LOG% 2>&1
%PY% -c "import albumentations, imgaug, iopath, pycocotools, cv2; print('extra deps ok', cv2.__version__)" >> %LOG% 2>&1
cd /d %USERPROFILE%\SCFlow2
set TORCH_HOME=%USERPROFILE%\torch_home
set KMP_DUPLICATE_LIB_OK=TRUE
%PY% -c "import sys; sys.path.insert(0,'.'); from datasets import build_dataset; from models import build_refiner; print('scflow2 imports ok')" >> %LOG% 2>&1
echo === STEP5 done %TIME% >> %LOG%
