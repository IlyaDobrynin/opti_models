from .skip_names import RESNET_LAYERS, MOBILENETV2_LAYERS, EFFICIENTNET_LAYERS


encoder_dict = {
    # RESNET
    'resnet18': {'skip': RESNET_LAYERS, 'filters': (64, 64, 128, 256, 512), 'features': False},
    'resnet34': {'skip': RESNET_LAYERS, 'filters': (64, 64, 128, 256, 512), 'features': False},
    'resnet50': {'skip': RESNET_LAYERS, 'filters': (64, 256, 512, 1024, 2048), 'features': False},
    'resnet101': {'skip': RESNET_LAYERS, 'filters': (64, 256, 512, 1024, 2048), 'features': False},
    'resnet152': {'skip': RESNET_LAYERS, 'filters': (64, 256, 512, 1024, 2048), 'features': False},

    # MOBILENET
    'mobilenetv2_w1': {'skip': MOBILENETV2_LAYERS, 'filters': (16, 24, 32, 96, 1280), 'features': True},
    'mobilenetv2_wd2': {'skip': MOBILENETV2_LAYERS, 'filters': (8, 12, 16, 48, 1280), 'features': True},
    'mobilenetv2_wd4': {'skip': MOBILENETV2_LAYERS, 'filters': (4, 6, 8, 24, 1280), 'features': True},
    'mobilenetv2_w3d4': {'skip': MOBILENETV2_LAYERS, 'filters': (12, 18, 24, 72, 1280), 'features': True},

    # EFFICIENTNET
    'efficientnet_b0': {'skip': EFFICIENTNET_LAYERS, 'filters': (16, 24, 40, 112, 1280), 'features': True},
    'efficientnet_b1': {'skip': EFFICIENTNET_LAYERS, 'filters': (16, 24, 40, 112, 1280), 'features': True},
    'efficientnet_b0b': {'skip': EFFICIENTNET_LAYERS, 'filters': (16, 24, 40, 112, 1280), 'features': True},
    'efficientnet_b1b': {'skip': EFFICIENTNET_LAYERS, 'filters': (16, 24, 40, 112, 1280), 'features': True},
    'efficientnet_b2b': {'skip': EFFICIENTNET_LAYERS, 'filters': (16, 24, 48, 120, 1408), 'features': True},
    'efficientnet_b3b': {'skip': EFFICIENTNET_LAYERS, 'filters': (24, 32, 48, 136, 1536), 'features': True},
    'efficientnet_b4b': {'skip': EFFICIENTNET_LAYERS, 'filters': (24, 32, 56, 160, 1792), 'features': True},
    'efficientnet_b5b': {'skip': EFFICIENTNET_LAYERS, 'filters': (24, 40, 64, 176, 2048), 'features': True},
    'efficientnet_b6b': {'skip': EFFICIENTNET_LAYERS, 'filters': (32, 40, 72, 200, 2304), 'features': True},
    'efficientnet_b7b': {'skip': EFFICIENTNET_LAYERS, 'filters': (32, 48, 80, 224, 2560), 'features': True},
    'efficientnet_b0с': {'skip': EFFICIENTNET_LAYERS, 'filters': (16, 24, 40, 112, 1280), 'features': True},
    'efficientnet_b1с': {'skip': EFFICIENTNET_LAYERS, 'filters': (16, 24, 40, 112, 1280), 'features': True},
    'efficientnet_b2с': {'skip': EFFICIENTNET_LAYERS, 'filters': (16, 24, 48, 120, 1408), 'features': True},
    'efficientnet_b3c': {'skip': EFFICIENTNET_LAYERS, 'filters': (24, 32, 48, 136, 1536), 'features': True},
    'efficientnet_b4c': {'skip': EFFICIENTNET_LAYERS, 'filters': (24, 32, 56, 160, 1792), 'features': True},
    'efficientnet_b5c': {'skip': EFFICIENTNET_LAYERS, 'filters': (24, 40, 64, 176, 2048), 'features': True},
    'efficientnet_b6c': {'skip': EFFICIENTNET_LAYERS, 'filters': (32, 40, 72, 200, 2304), 'features': True},
    'efficientnet_b7c': {'skip': EFFICIENTNET_LAYERS, 'filters': (32, 48, 80, 224, 2560), 'features': True},
    'efficientnet_b8c': {'skip': EFFICIENTNET_LAYERS, 'filters': (32, 56, 88, 248, 2816), 'features': True},
}