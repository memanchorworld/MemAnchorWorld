"""Soft voxel loss with centers aligned to the occupancy grid.
"""
_base_ = ['./memanchorworld-task-gates.py']

work_dir = 'work_dirs/memanchorworld-voxel-loss'
load_from = None
resume_from = ('work_dirs/memanchorworld-task-gates/'
               'epoch_1.pth')
auto_resume = False

model = dict(query_memory_cfg=dict(future_memory_adapter_v2=dict(
    soft_voxel_center_offset=0.5,
    soft_voxel_renormalize_valid_corners=True)))

# The source checkpoint has completed local epoch 1. Runner epoch 2 performs
# precisely one further training epoch, using the resumed optimizer state.
runner = dict(type='EpochBasedRunner', max_epochs=2)
