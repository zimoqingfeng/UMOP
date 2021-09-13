_base_ = './umop_r50_fpn_1.5x_coco.py'
model = dict(
    backbone=dict(
        # ResNet-101
        depth=101,
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet101')))
