from .skip_names import (
    EFFICIENTNET_LAYERS,
    GENET_LAYERS,
    MIXNET_LAYERS,
    MOBILENET_LAYERS,
    REGNET_LAYERS,
    RES2NET_LAYERS,
    RESNEST_LAYERS,
    RESNET_LAYERS,
)

encoder_dict = {
    # TORCHVISION MODELS
    'resnet18': {'skip': RESNET_LAYERS, 'filters': (64, 64, 128, 256, 512), 'features': False},
    'resnet34': {'skip': RESNET_LAYERS, 'filters': (64, 64, 128, 256, 512), 'features': False},
    'resnet50': {'skip': RESNET_LAYERS, 'filters': (64, 256, 512, 1024, 2048), 'features': False},
    'resnet101': {'skip': RESNET_LAYERS, 'filters': (64, 256, 512, 1024, 2048), 'features': False},
    'resnet152': {'skip': RESNET_LAYERS, 'filters': (64, 256, 512, 1024, 2048), 'features': False},
    'resnext50_32x4d': {'skip': RESNET_LAYERS, 'filters': (64, 256, 512, 1024, 2048), 'features': False},
    'resnext101_32x8d': {'skip': RESNET_LAYERS, 'filters': (64, 256, 512, 1024, 2048), 'features': False},
    'wide_resnet50_2': {'skip': RESNET_LAYERS, 'filters': (64, 256, 512, 1024, 2048), 'features': False},
    'wide_resnet101_2': {'skip': RESNET_LAYERS, 'filters': (64, 256, 512, 1024, 2048), 'features': False},
    # MOBILENET
    'mobilenetv2_w1': {'skip': MOBILENET_LAYERS, 'filters': (16, 24, 32, 96, 1280), 'features': True},
    'mobilenetv2_wd2': {'skip': MOBILENET_LAYERS, 'filters': (8, 12, 16, 48, 1280), 'features': True},
    'mobilenetv2_wd4': {'skip': MOBILENET_LAYERS, 'filters': (4, 6, 8, 24, 1280), 'features': True},
    'mobilenetv2_w3d4': {'skip': MOBILENET_LAYERS, 'filters': (12, 18, 24, 72, 1280), 'features': True},
    'mobilenetv3_w1': {'skip': MOBILENET_LAYERS, 'filters': (16, 24, 40, 112, 960), 'features': True},
    # MIXNET
    'mixnet_s': {'skip': MIXNET_LAYERS, 'filters': (16, 24, 40, 80, 1536), 'features': True},
    'mixnet_m': {'skip': MIXNET_LAYERS, 'filters': (24, 32, 40, 80, 1536), 'features': True},
    'mixnet_l': {'skip': MIXNET_LAYERS, 'filters': (32, 40, 56, 104, 1536), 'features': True},
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
    # GENET
    'genet_small': {'skip': GENET_LAYERS, 'filters': (13, 48, 48, 384, 1920), 'features': True},
    'genet_normal': {'skip': GENET_LAYERS, 'filters': (32, 128, 192, 640, 2560), 'features': True},
    'genet_large': {'skip': GENET_LAYERS, 'filters': (32, 128, 192, 640, 2560), 'features': True},
    # TIMM RES2NETS
    'timm_res2net50_14w_8s': {'skip': RES2NET_LAYERS, 'filters': (64, 256, 512, 1024, 2048), 'features': False},
    'timm_res2net50_48w_2s': {'skip': RES2NET_LAYERS, 'filters': (64, 256, 512, 1024, 2048), 'features': False},
    'timm_res2net50_26w_4s': {'skip': RES2NET_LAYERS, 'filters': (64, 256, 512, 1024, 2048), 'features': False},
    'timm_res2net50_26w_6s': {'skip': RES2NET_LAYERS, 'filters': (64, 256, 512, 1024, 2048), 'features': False},
    'timm_res2net50_26w_8s': {'skip': RES2NET_LAYERS, 'filters': (64, 256, 512, 1024, 2048), 'features': False},
    'timm_res2net101_26w_4s': {'skip': RES2NET_LAYERS, 'filters': (64, 256, 512, 1024, 2048), 'features': False},
    'timm_res2next50': {'skip': RES2NET_LAYERS, 'filters': (64, 256, 512, 1024, 2048), 'features': False},
    # TIMM RESNETS
    'timm_ig_resnext101_32x48d': {'skip': RESNET_LAYERS, 'filters': (64, 256, 512, 1024, 2048), 'features': False},
    'timm_resnetrs420': {'skip': RESNET_LAYERS, 'filters': (64, 256, 512, 1024, 2048), 'features': False},
    # TIMM REGNETS
    'timm_regnetx_002': {'skip': REGNET_LAYERS, 'filters': (32, 24, 56, 152, 368), 'features': False},
    'timm_regnetx_004': {'skip': REGNET_LAYERS, 'filters': (32, 32, 64, 160, 384), 'features': False},
    'timm_regnetx_006': {'skip': REGNET_LAYERS, 'filters': (32, 48, 96, 240, 528), 'features': False},
    'timm_regnetx_008': {'skip': REGNET_LAYERS, 'filters': (32, 64, 128, 288, 672), 'features': False},
    'timm_regnetx_016': {'skip': REGNET_LAYERS, 'filters': (32, 72, 168, 408, 912), 'features': False},
    'timm_regnetx_032': {'skip': REGNET_LAYERS, 'filters': (32, 96, 192, 432, 1008), 'features': False},
    'timm_regnetx_040': {'skip': REGNET_LAYERS, 'filters': (32, 80, 240, 560, 1360), 'features': False},
    'timm_regnetx_064': {'skip': REGNET_LAYERS, 'filters': (32, 168, 392, 784, 1624), 'features': False},
    'timm_regnetx_080': {'skip': REGNET_LAYERS, 'filters': (32, 80, 240, 720, 1920), 'features': False},
    'timm_regnetx_120': {'skip': REGNET_LAYERS, 'filters': (32, 224, 448, 896, 2240), 'features': False},
    'timm_regnetx_160': {'skip': REGNET_LAYERS, 'filters': (32, 256, 512, 896, 2048), 'features': False},
    'timm_regnetx_320': {'skip': REGNET_LAYERS, 'filters': (32, 336, 672, 1344, 2520), 'features': False},
    'timm_regnety_002': {'skip': REGNET_LAYERS, 'filters': (32, 24, 56, 152, 368), 'features': False},
    'timm_regnety_004': {'skip': REGNET_LAYERS, 'filters': (32, 48, 104, 208, 440), 'features': False},
    'timm_regnety_006': {'skip': REGNET_LAYERS, 'filters': (32, 48, 112, 256, 608), 'features': False},
    'timm_regnety_008': {'skip': REGNET_LAYERS, 'filters': (32, 64, 128, 320, 768), 'features': False},
    'timm_regnety_016': {'skip': REGNET_LAYERS, 'filters': (32, 48, 120, 336, 888), 'features': False},
    'timm_regnety_032': {'skip': REGNET_LAYERS, 'filters': (32, 72, 216, 576, 1512), 'features': False},
    'timm_regnety_040': {'skip': REGNET_LAYERS, 'filters': (32, 128, 192, 512, 1088), 'features': False},
    'timm_regnety_064': {'skip': REGNET_LAYERS, 'filters': (32, 144, 288, 576, 1296), 'features': False},
    'timm_regnety_080': {'skip': REGNET_LAYERS, 'filters': (32, 168, 448, 896, 2016), 'features': False},
    'timm_regnety_120': {'skip': REGNET_LAYERS, 'filters': (32, 224, 448, 896, 2240), 'features': False},
    'timm_regnety_160': {'skip': REGNET_LAYERS, 'filters': (32, 224, 448, 1232, 3024), 'features': False},
    'timm_regnety_320': {'skip': REGNET_LAYERS, 'filters': (32, 232, 696, 1392, 3712), 'features': False},
    # TIMM RESNEST
    'timm_resnest14d': {'skip': RESNEST_LAYERS, 'filters': (64, 256, 512, 1024, 2048), 'features': False},
}
