"""Query-level memory adapter with a frozen backbone.
"""

_base_ = ['./memanchorworld-query-memory.py']

load_from = 'ckpts/epoch_56.pth'
resume_from = None

future_memory_target_horizons = [1.0, 2.0, 3.0]

# Inherit the trajectory dataset and memory-cache configuration.
# Apply memory corrections through the output adapter.
model = dict(
    query_memory_cfg=dict(
        enabled=False,
        source='cache',
        freeze_base_model=True,
        future_memory_adapter_enabled=True,
        future_memory_adapter_finetune_mode=True,
        future_memory_target_horizons=future_memory_target_horizons,
        future_memory_adapter=dict(
            enabled=True,
            embed_dims=256,
            num_classes=17,
            num_points=48,
            num_heads=8,
            horizon_count=3,
            topk=16,
            spatial_radius=6.0,
            max_age=8.0,
            dropout=0.0,
            gate_bias=-1.0,
        ),
        # Disable query-memory fusion and the separate refiner.
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
