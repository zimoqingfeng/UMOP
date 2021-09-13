_base_ = './umop_r50_fpn_1x_coco.py'

# LR & epoch settings: 1.5x
lr_config = dict(step=[12, 16])
runner = dict(type='EpochBasedRunner', max_epochs=18)
