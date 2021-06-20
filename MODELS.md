# Models
To get list of available models via python API, you can do:
```
from opti_models.models.backbones.backbone_factory import show_available_backbones
model_names = show_available_backbones()
```
## Use
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

## Small Imagenet Validation Protocol (SIVP)
SIVP is the small protocol on the subset of Imagenet Validation Set.

Data is [here](https://drive.google.com/file/d/1Yi_SZ400LKMXeA08BvDip4qBJonaThae/view?usp=sharing). <br>

For reproduce instructions see [README.md](README.md), pt.5 - Simple pipeline example.<br>
Source code is here: [torch](opti_models/benchmarks/imagenet_torch_benchmark.py), [tensorrt](opti_models/benchmarks/imagenet_tensorrt_benchmark.py).

All fps given for batch_size=1<br>

### [Ryzen 3950x + 2080ti results](stats/r3050x_2080ti_bs1.csv)
