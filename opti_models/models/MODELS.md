# Models

## Small Imagenet Validation Protocol (SIVP)
SIVP is the small protocol on the subset of Imagenet Validation Set.

Data: https://drive.google.com/file/d/1Yi_SZ400LKMXeA08BvDip4qBJonaThae/view?usp=sharing <br>

All fps given for batch_size=1, i7-7800x + 1080ti<br>
For reproduce instructions see [README.md](../../README.md) 

| Model Name | TOP@1 Acc / Err | TOP@5 Acc / Err  | TRT FPS | Torch FPS |
|---|---|---|---|---|
| resnet18  | 70.19 / 29.81 |  90.49 / 9.51  | 4809.1  | 224.9  |
| resnet34  | 73.33 / 26.67 |  92.45 / 7.55 | 3957.8  | 196.3  |
| resnet50  | 76.28 / 23.72  | 94.04 / 5.96  | 2753.2 | 180.8  |
| mobilenetv2_w1  | 72.92 / 27.08  | 92.17 / 7.83  | 2771.0  | 179.7  |
| mobilenetv2_wd2  | 63.14 / 36.86  | 85.45 / 14.55  | 3115.1  | 178.8  |
| mobilenetv2_wd4  | 49.23 / 50.77  | 74.72 / 25.28  | 4846.6  | 179.8  |
| mobilenetv2_w3d4  | 69.45 / 30.55  | 89.85 / 10.15  | 2696.3  | 179.9  |
| mobilenetv3_large_w1  | 74.97 / 25.03  | 93.16 / 6.84  | 1848.3  | 135.5  |
| mixnet_s  | 76.63 / 23.37  | 93.87 / 6.13  | 754.1  | 97.1  |
| mixnet_m  | 77.82 / 22.18  | 94.36 / 5.64  | 570.6  | 81.3  |
| mixnet_l  | 78.92 / 21.08  | 94.98 / 5.02  | 592.9  | 81.3  |
| efficientnet_b0  | 75.89 / 24.11  | 93.40 / 6.60  | 1128.9  | 119.2  |
| efficientnet_b1  | 76.46 / 23.54  | 94.18 / 5.82  | 810.5  | 91.5  |
| genet_small  | 77.44 / 22.56  | 94.33 / 5.67  | 2747.9  | 186.8  |
| genet_normal  | 81.64 / 18.36  | 96.10 / 3.90  | 2579.9  | 190.2  |
| genet_large**  | 82.39 / 17.61  | 96.54 / 3.46  | 1734.2 | 176.4  |

** genet_large model tested on 256x256 image size