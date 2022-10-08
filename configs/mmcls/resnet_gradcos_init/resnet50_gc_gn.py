_base_ = [
    '../resnet/resnet50_8xb32_in1k.py',
]

initializer = dict(
    samples_per_gpu=64,
    gradinit_min_scale=0.01,
    gradinit_lr=0.003,
    gradinit_grad_clip=1.,
    overlap=0.2,
    gradinit_gamma=10,
    num_iters=100,
    num_batch_splits=4,
    gn_mul=1.,
    resnet=True,
    tasks=('val1', 'train1', 'val2'),
    eval_interval=100,
    opt_target="gc_gn_sqrt",
)
