# MemAnchorWorld

Core implementation of historical memory-anchor retrieval and task- and
horizon-gated point-level residual adaptation for sparse 4D semantic occupancy
forecasting.

## Core modules

- `mmdet3d/models/sparsedetectors/query_memory.py` implements memory storage,
  alignment, retrieval, gated residual adaptation, and point/voxel supervision.
- `mmdet3d/models/sparsedetectors/sparseworld_4d_traj.py` integrates the modules
  with recurrent occupancy forecasting and trajectory planning.
- `mmdet3d/datasets/pipelines/loading_query_memory.py` loads historical queries.
- `tools/query_memory/precompute_query_memory.py` builds observation-query caches.
- `configs/` contains the selected model configuration and its base configurations.

## Integration

The files retain their original SparseWorld/OpenMMLab package paths. Integrate
them into the corresponding forecasting codebase, retaining its existing
backbone, operators, and dependencies. Dataset locations, memory-cache paths,
and checkpoint paths are configured for the local environment.

The selected configuration is
`configs/sparseworld/nuscenes-temporal/sparseworld-traj-memory-future-adapter-v2-ab-softvoxel-center-t1.py`.
Training and evaluation entry points are `tools/train.py`, `tools/test.py`, and
`tools/test_temporal.py`.
