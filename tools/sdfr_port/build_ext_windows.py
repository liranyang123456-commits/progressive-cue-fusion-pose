import os, runpy, sys
import torch.utils.cpp_extension as ce
ce.SUBPROCESS_DECODE_ARGS = ("utf-8", "ignore")  # no console code page in a headless build on a Chinese-locale system
os.chdir(sys.argv[1])
sys.argv = ["setup.py", "build_ext", "--inplace"]
runpy.run_path("setup.py", run_name="__main__")
