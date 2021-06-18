# File with pretrain model layer names and other meta


RESNET_LAYERS = (['conv1', 'bn1', 'relu'], ['maxpool', 'layer1'], ['layer2'], ['layer3'], ['layer4'])

MOBILENET_LAYERS = (['init_block', 'stage1'], ['stage2'], ['stage3'], ['stage4'], ['stage5', 'final_block'])

EFFICIENTNET_LAYERS = (['init_block', 'stage1'], ['stage2'], ['stage3'], ['stage4'], ['stage5', 'final_block'])

MIXNET_LAYERS = (['init_block'], ['stage1'], ['stage2'], ['stage3'], ['stage4', 'final_block'])

GENET_LAYERS = (
    ['0', '1', "2"],
    ['3'],
    ['4'],
    ['5'],
    ['6', '7', '8', '9', '10'],
)

RES2NET_LAYERS = (['conv1', 'bn1', 'act1'], ['maxpool', 'layer1'], ['layer2'], ['layer3'], ['layer4'])
