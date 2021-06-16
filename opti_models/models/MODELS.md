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
cvt-onnx --model-name densenet121 --model-path CKPT-PATH --batch_size 100 --num-classes 10
```
where `CKPT-PATH` - path to the pretrained weights.

Next, if you want to make TensorRT convertation from this ONNX model, run:
```
cvt-trt --onnx-path data/onnx_export/resnet18/resnet18_bs-100_res-3x224x224_simplified.onnx
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

Data is [here](https://drive.google.com/file/d/1Yi_SZ400LKMXeA08BvDip4qBJonaThae/view?usp=sharing). <br>
All fps given for batch_size=1, Ryzen 3950x + 2080ti.<br>
For reproduce instructions see [README.md](../../README.md), pt.5 - Simple pipeline example.<br>
Source code is here: [torch](../benchmarks/imagenet_torch_benchmark.py), [tensorrt](../benchmarks/imagenet_tensorrt_benchmark.py).

| Model Name            | TOP@1 Acc / Err| TOP@5 Acc / Err  | TensorRT FP32 FPS | TensorRT FP16 FPS | TensorRT INT8 FPS  | Torch FPS |
|-----------------------|----------------|------------------|-------------------|-------------------|--------------------|-----------|
| resnet18              | 70.19 / 29.81  | 90.49 / 9.51     |   791.76          |   1622.17         |    800.47          | 366.29    |
| resnet34              | 73.33 / 26.67  | 92.45 / 7.55     |   459.72          |   1014.03         |    1225.64         | 262.77    |
| resnet50              | 76.28 / 23.72  | 94.04 / 5.96     |   381.97          |   944.52          |    475.40          | 192.20    |
| resnet101             | 77.94 / 22.06  | 94.73 / 5.27     |   293.69          |   532.80          |    293.87          | 105.61    |
| resnet152             | 79.29 / 20.71  | 95.63 / 4.37     |   208.78          |   373.74          |    208.00          | 73.23     |
| vgg11                 | 68.96 / 31.04  | 89.52 / 10.48    |   416.81          |   893.61          |    417.91          | 403.87    |
| vgg11_bn              | 71.52 / 28.48  | 91.18 / 8.82     |   408.01          |   888.04          |    413.58          | 383.22    |
| vgg13                 | 70.19 / 29.81  | 90.19 / 9.81     |   352.15          |   776.40          |    342.61          | 332.17    |
| vgg13_bn              | 72.07 / 27.93  | 91.65 / 8.35     |   349.14          |   776.64          |    342.86          | 311.18    |
| vgg16                 | 72.27 / 27.73  | 91.50 / 8.50     |   337.35          |   656.71          |              | 271.88    |
| vgg16_bn              | 74.08 / 25.92  | 92.93 / 7.07     |   339.68          |   658.25          |              | 256.03    |
| vgg19                 | 72.91 / 27.09  | 92.10 / 7.90     |   297.02          |   573.72          |              | 227.05    |
| vgg19_bn              | 75.22 / 24.78  | 93.29 / 6.71     |   297.02          |   573.56          |              | 214.85    |
| densenet121           | 75.43 / 24.57  | 93.47 / 6.53     |   290.06          |   337.07          |              | 78.09     |
| densenet169           | 76.30 / 23.70  | 93.81 / 6.19     |   180.21          |   192.54          |              | 56.16     |
| densenet161           | 78.52 / 21.48  | 94.70 / 5.30     |   132.53          |   202.47          |              | 59.52     |
| densenet201           | 77.44 / 22.56  | 94.66 / 5.34     |   129.03          |   130.30          |              | 46.09     |
| inception_v3**        | 71.08 / 28.92  | 89.96 / 10.04    |   235.48          |   523.53          |              | 111.67    |
| resnext50_32x4d       | 75.22 / 24.78  | 93.29 / 6.71     |   348.19          |   913.06          |              | 125.94    |
| resnext101_32x8d      | 79.70 / 20.30  | 95.62 / 4.38     |   100.94          |   309.41          |              | 60.77     |
| wide_resnet50         | 78.28 / 21.72  | 95.01 / 4.99     |   312.80          |   629.27          |              | 198.42    |
| wide_resnet101        | 79.20 / 20.80  | 95.50 / 4.50     |   161.92          |   348.85          |              | 107.00    |
| mobilenetv2_w1        | 72.92 / 27.08  | 92.17 / 7.83     |   939.36          |   1704.42         |              | 205.81    |
| mobilenetv2_wd2       | 63.14 / 36.86  | 85.45 / 14.55    |   1278.88         |   1981.82         |              | 206.88    |
| mobilenetv2_wd4       | 49.23 / 50.77  | 74.72 / 25.28    |   1602.74         |   2126.85         |              | 211.05    |
| mobilenetv2_w3d4      | 69.45 / 30.55  | 89.85 / 10.15    |   1037.34         |   1838.11         |              | 207.38    |
| mobilenetv3_large_w1  | 74.97 / 25.03  | 93.16 / 6.84     |   825.02          |   1213.68         |              | 146.42    |
| mixnet_s              | 76.63 / 23.37  | 93.87 / 6.13     |   506.84          |   619.63          |              | 101.15    |
| mixnet_m              | 77.82 / 22.18  | 94.36 / 5.64     |   401.94          |   501.20          |              | 83.77     |
| mixnet_l              | 78.92 / 21.08  | 94.98 / 5.02     |   362.70          |   463.91          |              | 84.89     |
| efficientnet_b0       | 75.89 / 24.11  | 93.40 / 6.60     |   584.42          |   899.18          |              | 130.61    |
| efficientnet_b1       | 76.46 / 23.54  | 94.18 / 5.82     |   400.64          |   664.44          |              | 92.79     |
| efficientnet_b0b      | 76.65 / 23.35  | 93.91 / 6.09     |   578.89          |   904.07          |              | 130.02    |
| genet_small           | 77.44 / 22.56  | 94.33 / 5.67     |   776.80          |   1583.03         |              | 226.11    |
| genet_normal          | 81.64 / 18.36  | 96.10 / 3.90     |   470.24          |   1197.32         |              | 225.79    |
| genet_large**         | 82.39 / 17.61  | 96.54 / 3.46     |   346.88          |   931.01          |              | 183.94    |

** inception_v3 model tested on 299х299 image size

** genet_large model tested on 256x256 image size
