
# dataset config
data = dict(
    name='cifar10',
    root='',
    train_only=False,
    transforms=None,
)


# model config
model = dict(
    type='simsiam',
    backbone=dict(
        name='resnet18',
        eval_mode=True,),
    neck=dict(),
    head=dict(),
)
