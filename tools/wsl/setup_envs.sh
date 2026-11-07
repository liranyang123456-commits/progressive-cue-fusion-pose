#!/bin/bash
# Environments for the two Linux-only baselines (GoTrack, MegaPose/HappyPose) in WSL 2 Ubuntu or Linux, with PyTorch
# cu128 so that they run on Blackwell GPUs (the released environments pin CUDA 11). Requires conda.
#   GOTRACK_DIR (default ~/gotrack), HAPPYPOSE_DIR (default ~/happypose), HAPPYPOSE_DATA_DIR (default ~/happypose_data)
set -e
export PIP_DEFAULT_TIMEOUT=120
source "$(conda info --base)/etc/profile.d/conda.sh"
GOTRACK_DIR=${GOTRACK_DIR:-$HOME/gotrack}
HAPPYPOSE_DIR=${HAPPYPOSE_DIR:-$HOME/happypose}
HAPPYPOSE_DATA_DIR=${HAPPYPOSE_DATA_DIR:-$HOME/happypose_data}

# ---------------- GoTrack (refiner) ----------------
[ -d "$GOTRACK_DIR" ] || git clone --recurse-submodules https://github.com/facebookresearch/gotrack "$GOTRACK_DIR"
if [ "$(stat -c %s "$GOTRACK_DIR/gotrack_checkpoint.pt")" -lt 1000000 ]; then  # git-lfs pointer: fetch the 1.6 GB checkpoint
  curl -L --retry 8 -o "$GOTRACK_DIR/gotrack_checkpoint.pt" \
    https://media.githubusercontent.com/media/facebookresearch/gotrack/main/gotrack_checkpoint.pt
fi
conda env list | grep -q '^gotrack ' || conda create -y -n gotrack python=3.10
conda activate gotrack
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install "numpy<2" omegaconf hydra-core==1.3.2 hydra-colorlog==1.2.0 trimesh pyrender==0.1.45 kornia==0.7.2 \
  opencv-python imageio scipy einops pypng scikit-image scikit-learn matplotlib distinctipy huggingface-hub tqdm pillow \
  fvcore iopath torchinfo pytorch-lightning webdataset pycocotools faiss-cpu
(cd "$GOTRACK_DIR" && pip install -e external/bop_toolkit && pip install --no-deps -e external/dinov2)
conda deactivate

# ---------------- MegaPose through HappyPose ----------------
[ -d "$HAPPYPOSE_DIR" ] || git clone --recurse-submodules https://github.com/agimus-project/happypose.git "$HAPPYPOSE_DIR"
conda env list | grep -q '^happypose ' || conda create -y -n happypose python=3.10
conda activate happypose
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install git+https://github.com/agimus-project/bop_toolkit
pip install "setuptools<70" wheel && pip install --no-build-isolation visdom   # visdom does not build with setuptools>=70
(cd "$HAPPYPOSE_DIR" && pip install -e happypose/pose_estimators/cosypose && pip install -e ".[pypi]")
# released networks (the INRIA host may be unreachable; the LAAS mirror serves the same files)
for m in coarse-rgb-906902141 refiner-rgb-653307694 refiner-rgbd-288182519; do
  mkdir -p "$HAPPYPOSE_DATA_DIR/megapose-models/$m"
  for f in config.yaml checkpoint.pth.tar; do
    [ -s "$HAPPYPOSE_DATA_DIR/megapose-models/$m/$f" ] || curl -L --retry 8 -o "$HAPPYPOSE_DATA_DIR/megapose-models/$m/$f" \
      "https://gepettoweb.laas.fr/data/happypose/megapose/megapose-models/$m/$f"
  done
done
python -c "import torch, happypose; print('happypose', torch.__version__, torch.cuda.is_available())"
