"""Task-specific residual gates with horizon-conditioned biases.
"""
_base_ = ['./memanchorworld-point-adapter.py']

work_dir = 'work_dirs/memanchorworld-task-gates'
load_from = work_dir + '/init_weights.pth'
resume_from = None
auto_resume = False
model = dict(query_memory_cfg=dict(future_memory_adapter_v2=dict(
    gate_mode='task_split', horizon_gate_bias=True,
    query_chunk_size=48, activation_checkpoint=True, save_on_cpu=False)))
data = dict(samples_per_gpu=1, workers_per_gpu=1)
optimizer = dict(lr=2e-5)
# Use a fixed learning rate with linear warmup.
lr_config = dict(_delete_=True, policy='fixed', warmup='linear',
                 warmup_iters=100, warmup_ratio=0.5)
runner = dict(type='EpochBasedRunner', max_epochs=3)
fp16 = dict(loss_scale='dynamic')
checkpoint_config = dict(interval=1, max_keep_ckpts=3)
log_config = dict(interval=50)
workflow = [('train', 1)]
