_base_ = './umop_swin-S_fpn_2x_coco_mstrain.py'

model = dict(
    backbone=dict(
        # SwinTransformer-Large
        type='SwinTransformer',
        embed_dim=192,
        num_heads=[6, 12, 24, 48],
        pretrain_model_path='./pretrained_backbones/swin_large_patch4_window7_224_22k.pth'),
    neck=dict(
        in_channels=[192, 384, 768, 1536]))
