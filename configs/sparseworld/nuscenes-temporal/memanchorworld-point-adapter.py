"""Point-level memory adapter with geometric and semantic retrieval.
"""

_base_ = ['./memanchorworld-query-adapter.py']

future_memory_adapter_version = 'v2'
future_memory_target_horizons = [1.0, 2.0, 3.0]

future_memory_adapter_v2 = dict(
    enabled=True,
    embed_dims=256,
    num_classes=17,
    num_points=48,
    num_heads=8,
    coarse_topk_geometry=8,
    coarse_topk_semantic=8,
    point_topk=8,
    coarse_radius=6.0,
    point_radius=6.0,
    max_effective_age=8.0,
    semantic_uncertainty_gamma=2.0,
    semantic_weight_floor=0.10,
    semantic_dropout_probability=0.10,
    # Process point attention in chunks to limit activation memory.
    query_chunk_size=40,
    # Recompute V2 during backward rather than retaining three horizons of
    # point-attention activations. Full CPU offload is disabled because the
    # thousands of query chunks can exhaust host RAM before the first step.
    activation_checkpoint=True,
    save_on_cpu=False,
    # The coarse read and half of the point heads retain an explicit
    # geometry-only route, so wrong Baseline semantics cannot suppress all
    # spatially plausible Memory evidence.
    coarse_geometry_context_weight=0.50,
    point_geometry_head_fraction=0.50,
    gate_bias=-1.0,
    dropout=0.0,
    voxel_size=(0.4, 0.4, 0.4),
    loss_occ_weight=0.25,
    # Anchor occupancy corrections to the frozen backbone predictions.
    loss_occ_anchor_weight=0.05,
    loss_sem_weight=0.25,
    loss_threshold_weight=0.10,
    loss_soft_voxel_weight=0.10,
    loss_base_cls_weight=1.0,
    loss_pts_weight=0.5,
    threshold_margin=0.10,
    # None derives half of the 3-D voxel diagonal (0.3464 m for 0.4 m cubic
    # voxels). Exact non-empty voxel membership is always positive regardless
    # of centre distance, matching the final voxel evaluator.
    positive_radius=None,
    # Positive and negative point groups are averaged independently before
    # these weights are applied, preventing the much larger negative group
    # from overwhelming occupancy and threshold supervision.
    occupancy_positive_weight=1.0,
    occupancy_negative_weight=1.0,
    threshold_positive_weight=1.0,
    threshold_negative_weight=1.0,
)

model = dict(
    future_memory_adapter_version='v2',
    query_memory_cfg=dict(
        future_memory_adapter_version='v2',
        future_memory_adapter_enabled=True,
        future_memory_adapter_finetune_mode=True,
        future_memory_target_horizons=future_memory_target_horizons,
        future_memory_adapter_v2=future_memory_adapter_v2,
        # Disable query-memory fusion and the separate refiner.
        enabled=False,
        memory_conditioned_refiner=False,
        memory_finetune_mode=False,
        memory_joint_finetune_mode=False,
        memory_joint_head_finetune_mode=False,
        memory_refiner_finetune_mode=False,
        memory_refiner_future_only=False,
    ),
)

optimizer = dict(
    type='AdamW',
    constructor='TrainableOnlyOptimizerConstructor',
    lr=1e-4,
    weight_decay=1e-2,
    paramwise_cfg=dict(bypass_duplicate=True))
