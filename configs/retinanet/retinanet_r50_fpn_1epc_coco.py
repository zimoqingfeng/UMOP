_base_ = [
    '../_base_/models/retinanet_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
# runner
runner = dict(type='EpochBasedRunner', max_epochs=1)
# work dir
work_dir = './work_dirs/retinanet_r50_fpn_1epc_coco'