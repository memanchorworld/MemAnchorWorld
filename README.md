# MemAnchorWorld: A Sparse World Model with Horizon-Conditioned Memory Retrieval for Autonomous Driving

## Overview

MemAnchorWorld extends SparseWorld for future semantic occupancy forecasting.
It stores historical scene queries, retrieves memory anchors for each forecast
horizon, and uses point-level residuals to correct semantic scores, occupancy
confidence, and position.

## Getting Started

### Installation

Set up [SparseWorld](https://github.com/MSunDYY/SparseWorld) using its
[installation guide](https://github.com/MSunDYY/SparseWorld/blob/master/docs/install.md).
Copy the files in this repository into the corresponding paths in the SparseWorld
checkout. The code uses SparseWorld's backbone, operators, and dependencies.

### Dataset and Memory Cache

Follow the SparseWorld
[dataset preparation guide](https://github.com/MSunDYY/SparseWorld/blob/master/docs/prepare_datasets.md)
for nuScenes, Occ3D, and the preprocessed inputs. Use
[`tools/query_memory/precompute_query_memory.py`](tools/query_memory/precompute_query_memory.py)
to build the observation-query caches with a SparseWorld checkpoint. Set the
dataset and cache paths in the configuration and its inherited files.

### Training and Inference

Run the commands from the SparseWorld checkout after integrating the MemAnchorWorld
files. Replace the example paths with your local checkpoints. Training resumes
from a compatible checkpoint that includes the optimizer state.

```bash
CONFIG=configs/sparseworld/nuscenes-temporal/memanchorworld-r50.py
python tools/train.py "$CONFIG" --resume-from /path/to/resume_checkpoint.pth
torchrun --nproc_per_node=1 tools/test_temporal.py "$CONFIG" /path/to/model_checkpoint.pth \
  --launcher pytorch --dump_dir predictions --dump-only
```

## Configuration

| Model | Configuration |
| --- | --- |
| MemAnchorWorld with ResNet-50 | [memanchorworld-r50.py](configs/sparseworld/nuscenes-temporal/memanchorworld-r50.py) |

## Acknowledgments

This implementation builds on [SparseWorld](https://github.com/MSunDYY/SparseWorld)
and the OpenMMLab ecosystem.
