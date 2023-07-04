img_scale = (3000, 2000)     # expected resizing image shape (1280, 720)  width, height

img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

infer_pipeline = [
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='DefaultFormatBundle'),
            dict(type='Collect', keys=['img']),
        ])
]

current_dir = 'extra'
device = 'cuda:0'


show_score_thr = 0.7

evaluation = dict(metric=['bbox', 'segm'])      # choise in ['bbox', 'segm']
