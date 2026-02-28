# Jetson Mamba Install (ARM64 / JetPack)

A **tested, reproducible guide** for installing:

-   PyTorch (NVIDIA Jetson wheel)
-   torchvision (ABI-matched)
-   causal-conv1d
-   Mamba (mamba-ssm)
-   Triton + Transformers stack

on **NVIDIA Jetson (JetPack / ARM64)**.

This setup was validated on a **Jetson Orin Nano** and focuses on
building from source reliably without breaking CUDA or PyTorch.

------------------------------------------------------------------------

## Overview

Installing Mamba on Jetson is difficult because:

-   No official ARM wheels exist for many dependencies
-   CUDA extensions must compile locally
-   Dependency managers frequently overwrite Jetson PyTorch builds
-   Builds fail due to memory limits

This guide documents a working configuration and rules that prevent
common failures.

------------------------------------------------------------------------

## Key Tips (Read First)

### Use a micromamba environment

This setup was done inside a micromamba environment which allows:

-   clean dependency isolation
-   reproducible builds
-   environment backups using `conda-pack`

------------------------------------------------------------------------

### Increase swap size (VERY IMPORTANT)

Building CUDA extensions on Jetson often fails due to RAM limits.

``` bash
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

swapon --show
free -h
```

------------------------------------------------------------------------

### Package Management Rules (IMPORTANT)

Jetson environments break easily if dependency managers modify PyTorch.

#### Torch stack → pip ONLY

Install these using NVIDIA wheels only:

-   torch
-   torchvision
-   torchaudio

Never install them through conda.

#### PyTorch-adjacent libraries → pip

Examples:

-   timm
-   transformers
-   huggingface_hub
-   triton
-   einops

Install safely:

``` bash
pip install --no-deps <package>
```

This prevents pip from replacing torch.

#### Numeric / system libraries → micromamba

Examples:

-   numpy\<2
-   scipy
-   matplotlib
-   libsndfile

Always pin NumPy:

``` bash
numpy<2
```

------------------------------------------------------------------------

## 1. Create Environment + Install PyTorch

``` bash
micromamba create -n causal python=3.10 -y
micromamba activate causal

export PYTHONNOUSERSITE=1

python -m pip install -U pip setuptools wheel "numpy<2"

python -m pip install --no-cache-dir \
"https://developer.download.nvidia.com/compute/redist/jp/v61/pytorch/torch-2.5.0a0+872d972e41.nv24.08.17622132-cp310-cp310-linux_aarch64.whl"
```

### Verify CUDA

``` bash
python - <<'EOF'
import torch
print("torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
EOF
```

------------------------------------------------------------------------

## 2. Install torchvision (ABI must match torch)

``` bash
micromamba activate causal
export PYTHONNOUSERSITE=1

python -m pip install -U "setuptools<70" "wheel<0.44"
```

### Build torchvision

``` bash
git clone https://github.com/pytorch/vision.git
cd vision
git checkout v0.20.0
python setup.py install
```

### Smoke Test

``` bash
python - <<'EOF'
import torch, torchvision
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("CUDA:", torch.cuda.is_available())
EOF
```

If issues occur, reference:

https://github.com/azimjaan21/jetpack-6.1-pytorch-torchvision-

------------------------------------------------------------------------

## 3. Build causal-conv1d (v1.2.2)

``` bash
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
git checkout v1.2.2
```

Clean previous builds:

``` bash
rm -rf build dist *.egg-info
```

### CUDA Environment

``` bash
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### Build Settings

``` bash
export FORCE_BUILD=1
export CAUSAL_CONV1D_FORCE_BUILD=1
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="8.7"
export PIP_NO_CACHE_DIR=1
```

### Install

``` bash
FORCE_BUILD=1 \
CAUSAL_CONV1D_FORCE_BUILD=1 \
PIP_NO_CACHE_DIR=1 \
MAX_JOBS=1 \
TORCH_CUDA_ARCH_LIST="8.7" \
python -m pip install -v . --no-build-isolation --no-deps
```

### causal-conv1d Smoke Test

``` bash
python - <<'PY'
import torch
from causal_conv1d import causal_conv1d_fn

B, L, D, K = 2, 64, 32, 3
w = torch.randn(D, K, device="cuda", dtype=torch.float16, requires_grad=True)
x = torch.randn(B, D, L, device="cuda", dtype=torch.float16, requires_grad=True)

y = causal_conv1d_fn(x, w)
y.sum().backward()

print("Layout B works:", y.shape)
PY
```

Expected:

    Layout B works: torch.Size([2, 32, 64])

------------------------------------------------------------------------

## 4. Install Mamba (v2.2.4)

``` bash
git clone https://github.com/state-spaces/mamba.git
cd mamba
git checkout v2.2.4
```

### Build Environment

``` bash
export CUDA_HOME=/usr/local/cuda
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

export MAMBA_FORCE_BUILD=TRUE
export MAX_JOBS=1
export TORCH_CUDA_ARCH_LIST="8.7"
export PIP_NO_CACHE_DIR=1
```

### Install

``` bash
MAMBA_FORCE_BUILD=TRUE \
PIP_NO_CACHE_DIR=1 \
python -m pip install -v . \
  --no-build-isolation \
  --no-deps \
  --force-reinstall
```

------------------------------------------------------------------------

## 5. Required Python Dependencies (pip only)

``` bash
pip install --no-deps \
einops \
packaging \
transformers \
triton \
timm
```

------------------------------------------------------------------------

## 6. Transformers Compatibility Fix

Add this shim near the top of your model file:

``` python
try:
    from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput
except ImportError:
    class _DummyGenOutput(dict):
        def __init__(self, sequences=None, scores=None, **kwargs):
            super().__init__(sequences=sequences, scores=scores, **kwargs)
            self.sequences = sequences
            self.scores = scores

    class GreedySearchDecoderOnlyOutput(_DummyGenOutput):
        pass

    class SampleDecoderOnlyOutput(_DummyGenOutput):
        pass

    import transformers.generation as _gen_mod
    _gen_mod.GreedySearchDecoderOnlyOutput = GreedySearchDecoderOnlyOutput
    _gen_mod.SampleDecoderOnlyOutput = SampleDecoderOnlyOutput
```

------------------------------------------------------------------------

## 7. Environment Backup (Recommended)

``` bash
conda-pack -n causal -o causal_env.tar.gz
```

Restore:

``` bash
mkdir -p ~/micromamba/envs/causal_restore
tar -xzf causal_env.tar.gz -C ~/micromamba/envs/causal_restore
~/micromamba/envs/causal_restore/bin/conda-unpack
micromamba activate causal_restore
```

------------------------------------------------------------------------

## Final Notes

-   Always prevent dependency managers from modifying torch.
-   Prefer `pip install --no-deps`.
-   Build CUDA extensions with limited parallel jobs.
-   Increase swap before compiling.
-   Backup working environments immediately.

------------------------------------------------------------------------

## Tested Hardware

-   Jetson Orin Nano
-   JetPack 6.x
-   CUDA 12.x
-   Python 3.10

------------------------------------------------------------------------

## License

MIT
