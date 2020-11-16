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



### ONNX Convertation with Python

1. Run:
```
    python opti_models/convertations/cvt_onnx.py --model_name MODEL_NAME --export_dir EXPORT_ONNX_DIR --model_mode 'torchvision' --ckpt_path CKPT_PATH --batch_size BATCH_SIZE --in_size IN_SIZE --num_classes NUM_CLASSES    
```
For instance, in order to convert ResNet18 with ImageNet pretraning run:
```
    python opti_models/convertations/cvt_onnx.py --model_name 'resnet18' --export_dir 'data/onnx_export' --model_mode 'torchvision' True --batch_size 1 --in_size 224 224
```
For instance, in order to convert you own ResNet18 torchvision model with custom weights run:
```
    python opti_models/convertations/cvt_onnx.py --model_name 'resnet18' --export_dir 'data/onnx_export' --model_mode 'torchvision' True --batch_size 1 --in_size 224 224 --ckpt_path CKPT_PATH --num_classes NUM_CLASSES
```

If you're converting your own model with custom `NUM_CLASSES`, opti_models simply changes the last FC layer of the network, so that the output dimention is equal to `NUM_CLASSES`, instead of 1000 in the ImageNet pretraining. If you have a custom head (or the entire model) — check [ONNX Convertation with Python — Custom Model](#onnx-convertation-with-python--custom-model)

By default, `cvt_onnx.py` will generate 2 outputs: regular `.onnx` file and a simplified version of it, obtained with ONNX simplifier. 


### ONNX Convertation with Python — Custom Model

The script for convertation is `cvt_onxx.py`. In order to convert something entirely custom, you need to change just 1 line of script. In particular, 96th line:

````
# Implement your own Model Class
model = None
````
Change this to `model = GetCustomModel()`, and you're good to go. Note that it's not guaranteed for custom models to successfully convert to ONNX or TRT, since some operations simply are not supported by either ONNX or TRT.   


### ONNX Convertation with Bash:

1. In `scripts/convertations/onnx_convertation.sh` change:
    - `model_name` - name of the model to convert
    - `export_dir` - directory to export onnx converted file (default `data/onnx_export`)
    - `model_mode` - whether this model is from torchvision, opti_models, or your custom model (default `torchvision`)
    - `ckpt_path` - path to checkpoint (default `imagenet` for regular ImageNet pretraining)
    - `batch_size` - batch size for converted model (default `1`) 
    - `in_size` - image size (default `224 224`)
    - `num_classes` - when loading torchvision model, specify the num_classes in the head (default `1000`)
2. Run:
```
    bash scripts/convertations/onnx_convertation.sh
```

### TensorRT Convertation with Python

1. Run:
```
    python opti_models/convertations/cvt_tensorrt.py --onnx_path ONNX_MODEL_PATH --export_dir EXPORT_TRT_DIR --batch_size BATCH_SIZE--in_size IN_SIZE -fp_type PRECISION_TYPE    
```

For instance, in order to convert the previously converted ResNet18 model to TRT run:

```
    python opti_models/convertations/cvt_tensorrt.py --onnx_path data/onnx_export/resnet18/resnet18_bs-1_res-224x224_simplified.onnx --export_dir data/trt_export --batch_size 1 --in_size 224 224 --fp_type 32
```
### TensorRT Convertation with Bash

1. In `scripts/convertations/tensorrt_convertation.sh` change:
    - `onnx_path` - path to the ONNX file
    - `export_dir` - directory to export converted file (default `data/trt_export`)
    - `batch_size` - batch size for converted model (default = 1) 
    - `fp_type` - type of float point precision, could be "16" or "32" (default = "32") 
2. Run:
```
    bash scripts/convertations/tensorrt_convertation.sh
```

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

