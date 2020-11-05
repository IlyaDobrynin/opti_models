# Models

## Small Imagenet Validation Protocol (SIVP)
### TensorRT

| Model | TRT TOP@1 Acc / Err | TRT TOP@5 Acc / Err  | FPS |
|---|---|---|---|
| resnet18  | 70.19 / 29.81 |  90.49 / 9.51  | 4809.1  |
| resnet34  | 73.33 / 26.67 |  92.45 / 7.55 | 3957.8  |
| resnet50  | 76.28 / 23.72  | 94.04 / 5.96  | 2753.2 |
| mobilenetv2_w1  | 72.92 / 27.08  | 92.17 / 7.83  | 2771.0  |
| mobilenetv2_wd2  | 63.14 / 36.86  | 85.45 / 14.55  | 3115.1  |
| mobilenetv2_wd4  | 49.23 / 50.77  | 74.72 / 25.28  | 4846.6  |
| mobilenetv2_w3d4  | 69.45 / 30.55  | 89.85 / 10.15  | 2696.3  |
| mobilenetv3_large_w1  | 74.97 / 25.03  | 93.16 / 6.84  | 1848.3  |
| mixnet_s  | 76.63 / 23.37  | 93.87 / 6.13  | 754.1  |
| mixnet_m  | 77.82 / 22.18  | 94.36 / 5.64  | 570.6  |
| mixnet_l  | 78.92 / 21.08  | 94.98 / 5.02  | 592.9  |
| efficientnet_b0  | 75.89 / 24.11  | 93.40 / 6.60  | 1128.9  |
| efficientnet_b1  | 76.46 / 23.54  | 94.18 / 5.82  | 810.5  |
| genet_small  | 77.44 / 22.56  | 94.33 / 5.67  | 2747.9  |
| genet_normal  | 81.64 / 18.36  | 96.10 / 3.90  | 2579.9  |
| genet_large**  | 82.39 / 17.61  | 96.54 / 3.46  | 1734.2 |

*All fps given for batch_size=1, i7-7800x + 1080ti<br>
** genet_large model tested on 256x256 image size