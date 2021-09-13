_base_ = './atss_r50_fpn_1x_coco.py'

# 2x settings learning policy
lr_config = dict(step=[16, 22])
runner = dict(type='EpochBasedRunner', max_epochs=24)

# work dir
work_dir = './work_dirs/atss_r50_fpn_2x_coco'
