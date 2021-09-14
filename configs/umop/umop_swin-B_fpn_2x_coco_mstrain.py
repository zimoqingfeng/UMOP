_base_ = './umop_swin-S_fpn_2x_coco_mstrain.py'

model = dict(
    backbone=dict(
        # SwinTransformer-Base
        type='SwinTransformer',
        embed_dim=128,
        num_heads=[4, 8, 16, 32],
        pretrain_model_path='./pretrained_backbones/swin_base_patch4_window7_224_22k.pth'),
    neck=dict(
        in_channels=[128, 256, 512, 1024]))
