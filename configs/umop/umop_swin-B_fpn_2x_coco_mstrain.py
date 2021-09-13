_base_ = './umop_swin-S_fpn_2x_coco_mstrain.py'

model = dict(
    backbone=dict(
        # SwinTransformer-Base
        type='SwinTransformer',
        embed_dim=128,
        num_heads=[4, 8, 16, 32]),
    neck=dict(
        in_channels=[128, 256, 512, 1024]))
