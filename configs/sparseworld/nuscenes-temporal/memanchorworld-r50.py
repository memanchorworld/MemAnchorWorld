"""MemAnchorWorld with a ResNet-50 backbone and class-balanced occupancy loss.
"""
_base_ = ['./memanchorworld-voxel-loss.py']

work_dir = ('work_dirs/'
            'memanchorworld-r50')
load_from = None
resume_from = ('work_dirs/'
               'memanchorworld-voxel-loss/'
               'epoch_2.pth')
auto_resume = False

model = dict(query_memory_cfg=dict(future_memory_adapter_v2=dict(
    # At 3 s, mix the positive point mean with the mean across observed
    # ground-truth classes using weights 0.75 and 0.25.
    occupancy_positive_class_balance_beta={'1s': 0.0, '2s': 0.0, '3s': 0.25},
    occupancy_class_balance_diagnostics=True)))

# Resume after epoch 2; exactly one new epoch, numbered 3. Do not reset LR,
# optimizer moments, FP16 scaler or runner iteration count with load_from.
runner = dict(type='EpochBasedRunner', max_epochs=3)
