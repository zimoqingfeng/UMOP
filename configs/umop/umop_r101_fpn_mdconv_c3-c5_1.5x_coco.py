_base_ = './umop_r50_fpn_1.5x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
        init_cfg=dict(
            type='Pretrained',
            checkpoint='torchvision://resnet101')))
