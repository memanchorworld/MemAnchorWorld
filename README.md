# MemAnchorWorld: A Sparse World Model with Horizon-Conditioned Memory Retrieval for Autonomous Driving

## Overview

MemAnchorWorld extends SparseWorld for future semantic occupancy forecasting.
It stores historical scene queries, retrieves memory anchors for each forecast
horizon, and uses point-level residuals to correct semantic scores, occupancy
confidence, and position.

## Getting Started

### Installation

Use Linux x86-64 with an NVIDIA GPU, Python 3.9, a C/C++ toolchain, and the
CUDA 11.8 Toolkit. The Toolkit provides `nvcc` for compiling CUDA operators.
Install the matching PyTorch build before compiling extensions.

1. Check out SparseWorld and copy in the MemAnchorWorld source files. Download
   the MemAnchorWorld archive from this repository and set
   `MEMANCHORWORLD_SRC` to the extracted directory containing
   `configs/`, `mmdet3d/`, and `tools/`.

   ```bash
   git clone https://github.com/MSunDYY/SparseWorld.git
   cd SparseWorld
   git checkout 14f5fe3bdd3428002d913988fbfd9b653a1a207e
   MEMANCHORWORLD_SRC=/path/to/extracted/MemAnchorWorld
   cp -a "$MEMANCHORWORLD_SRC"/{configs,mmdet3d,tools} .
   ```

2. Create the environment and select the CUDA Toolkit. Adjust `CUDA_HOME` to
   your installation path.

   ```bash
   python3.9 -m venv .venv
   source .venv/bin/activate
   export CUDA_HOME=/usr/local/cuda-11.8
   export PATH="$CUDA_HOME/bin:$PATH"
   nvidia-smi
   nvcc --version
   python -m pip install 'setuptools==69.5.1' 'wheel==0.45.1' ninja packaging
   python -m pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 \
     --index-url https://download.pytorch.org/whl/cu118
   ```

3. Install the OpenMMLab stack and runtime dependencies. SparseWorld uses
   MMCV 1.x APIs. The following commands build MMCV 1.7.0 against the installed
   PyTorch and install the CUDA 11.8 wheel for Torch Scatter.

   ```bash
   python -m pip install numpy==1.23.4 yapf==0.40.1 \
     opencv-python==4.11.0.86 opencv-python-headless==4.11.0.86
   MMCV_WITH_OPS=1 FORCE_CUDA=1 python -m pip install -v \
     --no-build-isolation --no-binary=mmcv-full mmcv-full==1.7.0
   python -m pip install numpy==1.23.4 mmdet==2.28.0 mmcls==0.25.0 \
     mmsegmentation==0.30.0 nuscenes-devkit==1.2.0 lyft-dataset-sdk==0.0.8 \
     numba==0.53.0 llvmlite==0.36.0 scipy==1.13.1 \
     scikit-image==0.19.3 scikit-learn==1.6.1 pandas==2.3.3 \
     networkx==3.2.1 trimesh plyfile pyquaternion ipython thop tensorboard \
     efficientnet-pytorch fvcore yacs timm
   python -m pip install torch-scatter==2.1.2 --only-binary=torch-scatter \
     -f https://data.pyg.org/whl/torch-2.0.1+cu118.html
   ```

4. Build the BEV pooling and multi-scale multi-view sampling operators. Run the
   import check from the SparseWorld root directory.

   ```bash
   FORCE_CUDA=1 python -m pip install -v --no-build-isolation --no-deps -e .
   (cd mmdet3d/models/sparsedetectors/csrc && python setup.py build_ext --inplace)
   python -m pip check
   python - <<'PY'
   import mmcv
   import torch
   import torch_scatter
   import mmcv._ext
   from mmdet3d.ops.bev_pool_v2 import bev_pool_v2_ext
   from mmdet3d.models.sparsedetectors.csrc import _msmv_sampling_cuda
   from mmdet3d.models.sparsedetectors.query_memory import FutureMemoryAdapterV2

   assert torch.version.cuda == '11.8' and torch.cuda.is_available()
   assert mmcv.__version__ == '1.7.0'
   print(torch.__version__, torch.version.cuda, mmcv.__version__)
   PY
   ```

See the [SparseWorld installation guide](https://github.com/MSunDYY/SparseWorld/blob/master/docs/install.md)
and the [MMCV build guide](https://mmcv.readthedocs.io/en/1.x/get_started/build.html)
for the corresponding upstream build steps.

### Dataset and Memory Cache

Download nuScenes v1.0 trainval and the matching lidarseg data from
[nuScenes](https://www.nuscenes.org/download). Extract them under
`data/nuscenes/`. Add the nuScenes occupancy labels from
[Occ3D](https://github.com/Tsinghua-MARS-Lab/Occ3D) as `data/nuscenes/gts/`.
Download the two BEVDet metadata files, the `admlp` and `occworld` inputs, and
`epoch_56.pth` from the
[SparseWorld data and weights repository](https://huggingface.co/MSunDYY2001/SparseWorld/tree/main).
Keep the paths below because the dataset loader and the metadata files refer to
these locations.

```text
SparseWorld/
├── ckpts/
│   └── epoch_56.pth
├── data/
│   └── nuscenes/
│       ├── gts/<scene>/<sample>/labels.npz
│       ├── lidarseg/
│       ├── maps/
│       ├── samples/
│       ├── sweeps/
│       ├── v1.0-trainval/
│       ├── bevdetv2-nuscenes_infos_train.pkl
│       └── bevdetv2-nuscenes_infos_val.pkl
├── admlp/
│   ├── fengze_nuscenes_infos_train.pkl
│   ├── fengze_nuscenes_infos_val.pkl
│   └── stp3_val/
│       └── data_nuscene.pkl
└── occworld/
    ├── nuscenes_infos_train_temporal_v3_scene.pkl
    └── nuscenes_infos_val_temporal_v3_scene.pkl
```

Generate the observation-query caches from the SparseWorld checkpoint. Run both
commands for training or only the validation command to run inference on the
validation split.

```bash
CACHE_CONFIG=configs/sparseworld/nuscenes-temporal/memanchorworld-query-memory.py

python tools/query_memory/precompute_query_memory.py \
  --config "$CACHE_CONFIG" --checkpoint ckpts/epoch_56.pth --split train \
  --output-dir data/query_memory/sparseworld_epoch56_schema2_train

python tools/query_memory/precompute_query_memory.py \
  --config "$CACHE_CONFIG" --checkpoint ckpts/epoch_56.pth --split val \
  --output-dir data/query_memory/sparseworld_epoch56_schema2_val
```

The script writes one `<scene>/<sample>.pt` record for each processed sample.
Use `--skip-existing` to continue an interrupted run with the same settings.
If you store the caches elsewhere, set `query_memory_train_cache_root` and
`query_memory_val_cache_root` in
[`memanchorworld-query-memory.py`](configs/sparseworld/nuscenes-temporal/memanchorworld-query-memory.py).

### Training and Inference

From the SparseWorld root directory, run the following commands with local
checkpoints. Training resumes from a compatible checkpoint containing the
optimizer state. For validation, use a trained MemAnchorWorld checkpoint.
The single-GPU validation command reports occupancy metrics and writes
predicted trajectories to `admlp/output_data.pkl`.

```bash
CONFIG=configs/sparseworld/nuscenes-temporal/memanchorworld-r50.py
python tools/train.py "$CONFIG" --resume-from /path/to/resume_checkpoint.pth
python tools/test_temporal.py "$CONFIG" /path/to/model_checkpoint.pth --eval segm
```

## Configuration

| Model | Configuration |
| --- | --- |
| MemAnchorWorld with ResNet-50 | [memanchorworld-r50.py](configs/sparseworld/nuscenes-temporal/memanchorworld-r50.py) |

## Acknowledgments

This implementation builds on [SparseWorld](https://github.com/MSunDYY/SparseWorld)
and the OpenMMLab ecosystem.
