# opti_models

## Install

0. Clone the repo:
```
git clone git@github.com:IlyaDobrynin/opti_models.git && cd opti_models
git checkout dev
```

1. Create a clean virtual environment 
```
python3 -m venv venv
source venv/bin/activate
```
2. Install dependencies
````
pip install --upgrade pip
pip install .
pip install nvidia-pyindex
pip install --upgrade nvidia-tensorrt
````

## Convertation
**CURRENTLY IN DEV MODE**

### ONNX Convertation

1. Run:
```
    python opti_models/convertations/cvt_onnx.py --model-name BACKBONE_NAME
```

For instance, in order to convert ResNet18 with ImageNet pretraning run:
```
    python opti_models/convertations/cvt_onnx.py --model-name resnet18
```
For instance, in order to convert you own ResNet18 torchvision model with batch size 1, image size 224x224, num classes 1 and custom weights run:
```
    python opti_models/convertations/cvt_onnx.py --model-name resnet18 --model-path CKPT-PATH --batch_size 1 --size 224 224 --num-classes 1
```

Parameters cheatsheet:

- `model-name` (str, required) - Name of the model.
- `model-type` (str, optional) - Type of the model. Default: `backbone`. Options:
  - `backbone` - simply backbone classification model
  - `classifier` - opti-models classification model
  - `custom` - your own torchvision models.    
- `model-path` (str, optional) - Path to the model. Default: `ImageNet` - model with imagenet pretrain.
- `batch-size` (int, optional) - Batch size for converted model. Default: `1`.
- `size` (int int, optional) - Image size. Default: `224 224`.
- `num-classes` (int, optional) - Num classes in the head for backbone model.  Default: `1000`.
- `export-dir` (str, optional) - Directory to export onnx converted files. Default: `data/onnx-export`.

If you're converting your own model with custom `num-classes`, opti_models simply changes the last FC layer of the network,
so that the output dimention is equal to `num-classes`, instead of 1000 in the ImageNet pretraining. 

If you have a custom head (or the entire model) — check [ONNX Convertation with Python — Custom Model](#onnx-convertation-with-python--custom-model)

By default, cvt_onnx.py will generate 2 outputs: regular .onnx file and a simplified version of it, obtained with ONNX simplifier. 


### ONNX Convertation — Custom Model

The script for convertation is `cvt_onxx.py`. In order to convert something entirely custom, you need to use python api and 
provide `model` argument to the `make_onnx_convertation` function.
Example:

````
from opti_models import make_onnx_convertation
model = MyModel(**my_model_parameters)
make_onnx_convertation(model=model, **other_parameters)
````
**Important**: It's not guaranteed for custom models to successfully convert to ONNX or TRT, since some operations 
simply are not supported by either ONNX or TRT.   


### TensorRT Convertation

1. Run:
```
    python opti_models/convertations/cvt_tensorrt.py --onnx-path 
```

For instance, in order to convert the previously converted ResNet18 model to TRT run:

```
    python opti_models/convertations/cvt_tensorrt.py --onnx-path data/onnx_export/resnet18/resnet18_bs-1_res-224x224_simplified.onnx 
```

Parameters cheatsheet:

- `onnx-path` (str, required) - Path to the exported onnx model.
- `batch-size` (int, optional) - Batch size for converted model. Default: `1`.
- `precision` (str, optional) - Precision of the TRT engine, 32 (for FP32) or 16 (for FP16). Default: `32`.
- `export-dir` (str, optional) - Directory to export TRT engine . Default: `data/trt-export`.


## Models
For list of all models see [MODELS.md](/opti_models/models/MODELS.md)

## Benchmarks

### Imagenet
For all imagenet benchmarks you should download and untar: https://drive.google.com/file/d/1Yi_SZ400LKMXeA08BvDip4qBJonaThae/view?usp=sharing

#### Torch Imagenet Benchmark
1. In `scripts/benchmarks/torch_imagenet_benchmark.sh` change:
    - `path_to_images` - path to the `imagenetv2-top-images-format-val` folder
    - `model_name` - name of the model to bench
    - `batch_size` - anount of the images in every batch
    - `workers` - number of workersл 
2. Run: 
```
    bash scripts/benchmarks/torch_imagenet_benchmark.sh
```
#### Tensorrt Imagenet Benchmark
1.  In `scripts/benchmarks/tensorrt_imagenet_benchmark.sh` change:
    - `path_to_images` - path to the `imagenetv2-top-images-format-val` folder
    - `trt_path` - path to the TensorRT .engine model (for convertation see [Convertation](#Convertation))
2. Run: 
```
    bash scripts/benchmarks/tensorrt_imagenet_benchmark.sh
```

