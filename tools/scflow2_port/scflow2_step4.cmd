@echo off
setlocal
set LOG=%USERPROFILE%\scflow2_setup4.log
echo === STEP4 start %DATE% %TIME% > %LOG%
set PY=%USERPROFILE%\venvs\scflow2\Scripts\python.exe
%PY% -m pip install albumentations==1.4.2 imgaug==0.4.0 iopath pycocotools >> %LOG% 2>&1
%PY% -c "import albumentations, imgaug, iopath, pycocotools; print('extra deps ok')" >> %LOG% 2>&1
rem scikit-sparse (CHOLMOD) has no Windows build; it is only used by an alternative sparse solver that the refiner test path does not call
%PY% -c "import re,io;p=r'%USERPROFILE%\SCFlow2\models\utils\raft_3d_basic_blocks.py';s=open(p,encoding='utf-8').read();s=s.replace('from sksparse import cholmod','try:\n    from sksparse import cholmod\nexcept ImportError:  # Windows: scikit-sparse not available; only needed by the sparse-solver variant\n    cholmod = None',1);open(p,'w',encoding='utf-8').write(s);print('patched raft_3d_basic_blocks.py')" >> %LOG% 2>&1
cd /d %USERPROFILE%\SCFlow2
set TORCH_HOME=%USERPROFILE%\torch_home
set KMP_DUPLICATE_LIB_OK=TRUE
%PY% -c "import sys; sys.path.insert(0,'.'); from datasets import build_dataset; from models import build_refiner; print('scflow2 imports ok')" >> %LOG% 2>&1
echo === STEP4 done %TIME% >> %LOG%
