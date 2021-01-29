# Models
To get list of available models via python API, you can do:
```
from opti_models.models.backbones.backbone_factory import show_available_backbones
model_names = show_available_backbones()
```                                     
## Usecases
By default, you can use any name from lists below, to convert this model.

For example, if you want to convert to ONNX `densenet121` with batch_size = 100 and 
number of output classes = 10, you need tu run:
```
python opti_models/convertations/cvt_onnx.py --model-name densenet121 --model-path CKPT-PATH --batch_size 100 --num-classes 10
```
where `CKPT-PATH` - path to the pretrained weights.

Next, if you want to make TensorRT convertation from this ONNX model, run:
```
python opti_models/convertations/cvt_tensorrt.py --onnx-path data/onnx_export/resnet18/resnet18_bs-100_res-3x224x224_simplified.onnx
```

## Full list of backbones:
### Pretrained
```
resnet152
resnet101
resnet50
resnet34
resnet18
vgg11
vgg11_bn
vgg13
vgg13_bn
vgg16
vgg16_bn
vgg19
vgg19_bn
densenet121
densenet169
densenet161
densenet201
inception_v3
resnext50_32x4d
resnext101_32x8d
wide_resnet50
wide_resnet101
mobilenetv2_w1
mobilenetv2_wd2
mobilenetv2_wd4
mobilenetv2_w3d4
mobilenetv3_large_w1
mixnet_s
mixnet_m
mixnet_l
efficientnet_b0
efficientnet_b1
efficientnet_b0b
efficientnet_b1b
efficientnet_b2b
efficientnet_b3b
efficientnet_b4b
efficientnet_b5b
efficientnet_b6b
efficientnet_b7b
genet_small
genet_normal
genet_large
```
### Not pretrained
```
efficientnet_b0c
efficientnet_b1c
efficientnet_b2c
efficientnet_b3c
efficientnet_b4c
efficientnet_b5c
efficientnet_b6c
efficientnet_b7c
efficientnet_b8c
```
## Small Imagenet Validation Protocol (SIVP)
SIVP is the small protocol on the subset of Imagenet Validation Set.

Data are [here](https://drive.google.com/file/d/1Yi_SZ400LKMXeA08BvDip4qBJonaThae/view?usp=sharing)

All fps given for batch_size=1, i7-7800x + 1080ti<br>
For reproduce instructions see [README.md](../../README.md), pt.5 - Simple pipeline example

| Model Name            | TOP@1 Acc / Err| TOP@5 Acc / Err  | TensorRT FPS  | Torch FPS |
|-----------------------|----------------|------------------|---------------|-----------|
| resnet18              | 70.19 / 29.81  | 90.49 / 9.51     | 907.35        |  |
| resnet34              | 73.33 / 26.67  | 92.45 / 7.55     | 590.06        |  |
| resnet50              | 76.28 / 23.72  | 94.04 / 5.96     | 410.80        |  |
| resnet101             | 77.94 / 22.06  | 94.73 / 5.27     | 233.89        |  |
| resnet152             | 79.29 / 20.71  | 95.63 / 4.37     | 160.71        |  |
| vgg11                 | 68.96 / 31.04  | 89.52 / 10.48    | 357.62        |  |
| vgg11_bn              | 71.52 / 28.48  | 91.18 / 8.82     | 350.98        |  |
| vgg13                 | 70.19 / 29.81  | 90.19 / 9.81     | 313.81        |  |
| vgg13_bn              | 72.07 / 27.93  | 91.65 / 8.50     | 317.57        |  |
| vgg16                 | 72.27 / 27.73  | 91.50 / 4.37     | 271.28        |  |
| vgg16_bn              | 74.08 / 25.92  | 92.93 / 7.07     | 271.02        |  |
| vgg19                 | 72.91 / 27.09  | 92.10 / 7.90     | 241.38        |  |
| vgg19_bn              | 75.22 / 24.78  | 93.29 / 6.71     | 240.54        |  |
| densenet121           | 75.43 / 24.57  | 93.47 / 6.53     | 271.10        |  |
| densenet169           | 76.30 / 23.70  | 93.81 / 6.19     | 171.46        |  |
| densenet161           | 78.52 / 21.48  | 94.70 / 5.30     | 129.95        |  |
| densenet201           | 77.44 / 22.56  | 94.66 / 5.34     | 122.38        |  |
| inception_v3          | 71.08 / 28.92  | 89.96 / 10.04    | 336.42        |  |
| resnext50_32x4d       | 75.22 / 24.78  | 93.29 / 6.71     | 215.39        |  |
| resnext101_32x8d      | 79.70 / 20.30  | 95.62 / 4.38     | 63.72         |  |
| wide_resnet50         | 78.28 / 21.72  | 95.01 / 4.99     | 257.51        |  |
| wide_resnet101        | 79.20 / 20.80  | 95.50 / 4.50     | 133.75        |  |
| mobilenetv2_w1        | 72.92 / 27.08  | 92.17 / 7.83     | 875.57        | 139.35 |
| mobilenetv2_wd2       | 63.14 / 36.86  | 85.45 / 14.55    | 1162.90       |  |
| mobilenetv2_wd4       | 49.23 / 50.77  | 74.72 / 25.28    | 1462.98       |  |
| mobilenetv2_w3d4      | 69.45 / 30.55  | 89.85 / 10.15    | 990.63        |  |
| mobilenetv3_large_w1  | 74.97 / 25.03  | 93.16 / 6.84     | 701.99        |  |
| mixnet_s              | 76.63 / 23.37  | 93.87 / 6.13     | 473.14        |  |
| mixnet_m              | 77.82 / 22.18  | 94.36 / 5.64     | 380.59        |  |
| mixnet_l              | 78.92 / 21.08  | 94.98 / 5.02     | 334.07        |  |
| efficientnet_b0       | 75.89 / 24.11  | 93.40 / 6.60     | 527.71        |  |
| efficientnet_b1       | 76.46 / 23.54  | 94.18 / 5.82     | 388.76        |  |
| efficientnet_b0b      | 76.65 / 23.35  | 93.91 / 6.09     | 507.28        |  |
| genet_small           | 77.44 / 22.56  | 94.33 / 5.67     | 919.32        |  |
| genet_normal          | 81.64 / 18.36  | 96.10 / 3.90     | 502.96        |  |
| genet_large**         | 82.39 / 17.61  | 96.54 / 3.46     | 381.89        |  |

** genet_large model tested on 256x256 image size