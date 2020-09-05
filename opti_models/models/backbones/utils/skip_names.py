# File with pretrain model layer names and other meta


RESNET_LAYERS = (
    ['conv1', 'bn1', 'relu'],
    ['maxpool', 'layer1'],
    ['layer2'],
    ['layer3'],
    ['layer4']
)

MOBILENETV2_LAYERS = (
    ['init_block', 'stage1'],
    ['stage2'],
    ['stage3'],
    ['stage4'],
    ['stage5', 'final_block']
)

EFFICIENTNET_LAYERS = (
    ['init_block', 'stage1'],
    ['stage2'],
    ['stage3'],
    ['stage4'],
    ['stage5', 'final_block']
)