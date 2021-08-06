![logo](https://github.com/IlyaDobrynin/opti_models/blob/master/logo.jpeg)
# Opti Models
Repo for the easy-way models convertation.

## Content
1. [Install](#Install)
    - [With Docker](#With-Docker)
    - [Without docker](#Without-docker)
2. [Convertation](#Convertation)
    - [ONNX](#ONNX-Convertation)
    - [TensorRT](#Tensorrt-Convertation)
3. [Models](#Models)
4. [Benchmarks](#Benchmarks)
5. [Simple pipeline example](#Simple-pipeline-example)
6. [Citing](#Citing)
7. [License](#License)

## Install
[Back to Content](#Content)


#### **NOTE** You need to have nvidia divers and CUDA installed. [Details](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

0. Clone the repo:
```
git clone https://github.com/IlyaDobrynin/opti_models.git && cd opti_models
```
### With Docker
[Back to Content](#Content)

**We highly advice you to work with this project inside the Docker that we've built for you, since
there are tested dependencies.
Otherwise, we can't guarantee that it will work due to various environmental reasons.**

<details>
<summary>Expand</summary>

Steps for work with docker:
1. Set up Docker:
    - [Install docker](https://docs.docker.com/engine/install/ubuntu/)
    - [Make Docker run without root](https://docs.docker.com/engine/install/linux-postinstall/)
    - [Install nvidia-docker](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#setting-up-nvidia-container-toolkit)
2. Build docker image (from PROJECT_DIR - directory, where opti_models is located):
    ```
    docker build -t opti_models .
    ```
    At first it may take a while.
3. **_OPTIONAL STEP_** If you want to run imagenet benchmarks on the predefined backbones, you need to download data
and place in to the `/usr/local/opti_models` directory. Details [here](#Benchmarks)

4. Run docker container:
    - Without mounting data folder for benchmarks:
        ```
        bash run_docker_no-images.sh <PROJECT_DIR> <CONTAINER_VERSION:-latest>

        or

        docker run \
           --gpus all \
           --ipc=host \
           -v <PROJECT_DIR>:/workspace \
           -it opti_models
        ```
    - With mounting data folder for benchmarks (only if you have step 3 done):
        ```
        bash run_docker.sh <PROJECT_DIR> <CONTAINER_VERSION:-latest> <DATA_STORAGE:-/usr/local/opti_models>

        or

        docker run \
           --gpus all \
           --ipc=host \
           -v <PROJECT_DIR>:/workspace \
           -v <STORAGE>:/usr/local/opti_models/ \
           -it opti_models
        ```
- <PROJECT_DIR> - directory, where opti_models is located
- <CONTAINER_VERSION> - version of the container, default - latest
- <DATA_STORAGE> - directory with the calibration and validation images, default - /usr/local/opti_models

</details>

### Without docker
[Back to Content](#Content)

<details>
<summary>Expand</summary>

1. Create a clean virtual environment
```
python3 -m venv venv
source venv/bin/activate
```
2. Install dependencies
````
pip install --upgrade pip
pip install -r requirements/requirements.txt
pip install nvidia-pyindex
pip install nvidia-tensorrt==7.2.2.3
pip install -e .
````

</details>

## Convertation
[Back to Content](#Content)

### ONNX Convertation
[Back to Content](#Content)

<details>
<summary>Expand</summary>

#### Predefined models
1. Run:
```
cvt-onnx --model-name BACKBONE_NAME
```

Parameters cheatsheet:
- `model-name` (str, required) - Name of the model.
- `model-type` (str, optional) - Type of the model. Default: `classifier`. Options:
  - `classifier` - simply backbone classification model
  - `opti-classifier` - opti-models classification model
  - `custom` - your own torchvision models.
- `model-path` (str, optional) - Path to the model. Default: `ImageNet` - model with imagenet pretrain.
- `batch-size` (int, optional) - Batch size for converted model. Default: `1`.
- `size` (int int int, optional) - Image size `[Ch x H x W]`. Default: `3 224 224`.
- `num-classes` (int, optional) - Num classes in the head for backbone model. Default: `1000`.
- `export-name` (str, optional) - Name of the exported onnx file. Default: `{model-name}_bs-{batch-size}_res-{size}`.

If you're converting your own model with custom `num-classes`, opti_models simply changes the last FC layer of the network,
so that the output dimension is equal to `num-classes`, instead of 1000 in the ImageNet pretraining.

If you have a custom head (or the entire model) — check [ONNX Convertation — Custom Model](#onnx-convertation--custom-model)

By default, cvt_onnx.py will generate 2 outputs: regular .onnx file and a simplified version of it, obtained with ONNX simplifier.

**Examples:**
```
# Convert simple resnet with imagenet pretrain
cvt-onnx --model-name resnet18

# Convert you own ResNet18 torchvision model with batch size 1,
# image size 224x224, num classes 1 and custom weights
cvt-onnx --model-name resnet18 --model-path CKPT-PATH --num-classes 1
```

#### Entirely custom model

The script for convertation is `cvt_onxx.py`. In order to convert something entirely custom, you need to use python api
and provide `model` argument to the `make_onnx_convertation` function.

**Example:**

```
from opti_models import make_onnx_convertation
model = MyModel(**my_model_parameters)
make_onnx_convertation(model=model, **other_parameters)
```
**Important**: It's not guaranteed for custom models to successfully convert to ONNX or TRT, since some operations
simply are not supported by either ONNX or TRT.

</details>

### TensorRT Convertation
[Back to Content](#Content)

<details>
<summary>Expand</summary>

1. Run:
```
cvt-trt --onnx-path
```

Parameters cheatsheet:

- `onnx-path` (str, required) - Path to the exported onnx model.
- `precision` (str, optional) - Precision of the TRT engine, 32 (for FP32), 16 (for FP16) or 8 (for INT8). Default: `32`.
- `export-name` (str, optional) - Name of the exported TRT engine . Default: `{model_name}_prec-{precision}_bs-{bs}_res-{c}x{h}x{w}.engine`.
- `calibration-images-dir` (str, optional) - Path to the directory with images for int8 calibration. If `precision==8`
  this parameter required.

**Example:**
```
# Convert previously converted ResNet18 ONNX model
cvt-trt --onnx-path data/onnx_export/resnet18/resnet18_bs-1_res-3x224x224_simplified.onnx
```
</details>

## Models
[Back to Content](#Content)

For list of all models see [MODELS.md](MODELS.md). <br>
There you can find list of all available pretrained backbones and results of benchmarks for this models.

## Benchmarks
[Back to Content](#Content)



<details>
<summary>Imagenet</summary>

#### 1. Prepare data
For all imagenet benchmarks you need to prepare data:
1. Download data from [GoogleDrive](https://drive.google.com/file/d/1Yi_SZ400LKMXeA08BvDip4qBJonaThae/view?usp=sharing)
2. Untar it to the `usr/local/opti_models`:
```
sudo mkdir -p /usr/local/opti_models
sudo mv imagenetv2-topimages.tar /usr/local/opti_models/
sudo tar -xvf imagenetv2-topimages.tar
```

#### 2. Run imagenet Benchmark with PyTorch models
```
bench-torch --model-name MODEL-NAME
```
Parameters cheatsheet:
- `model-name` (str, required) - name of the model to test.
- `export-name` (str, optional) - file where to store bench statistics. If None, no statistics will be saved. Default: None
- `path-to-images` (str, optional) - path to the validation set. Default - `/usr/local/opti_models/imagenetv2-top-images-format-val`.
- `size` (int int, optional) - Image size. Default: `224 224`.
- `batch-size` (int, optional) - Batch size for converted model. Default: `1`.
- `workers` (int, optional) - Number of the workers. Default: `1`

#### 3. Run imagenet Benchmark with TensorRT models
```
bench-trt --trt-path TRT-PATH
```
Parameters cheatsheet:
- `trt-path` (str, required) - path to the TensorRT model.
- `export-name` - file where to store bench statistics. If None, no statistics will be saved. Default: None
- `path-to-images` (str, required) - path to the validation set. Default - `/usr/local/opti_models/imagenetv2-top-images-format-val`.

</details>

## Simple pipeline example
[Back to Content](#Content)

<details>
<summary>Expand</summary>

#### Let's sum up simple end2end pipeline for convertations and benchmarking:
1. First let's run simple pytorch speed benchmark with resnet18:
    ```
    bench-torch --model-name resnet18
    Output:
    INFO:root:      TORCH BENCHMARK FOR resnet18: START
    100%|█████████████| 10000/10000 [01:28<00:00, 112.92it/s]
    INFO:root:      Average fps: 213.2855109396426
    INFO:root:      TOP 1 ACCURACY: 70.19   TOP 1 ERROR: 29.81
    INFO:root:      TOP 5 ACCURACY: 90.49   TOP 5 ERROR: 9.51
    INFO:root:      BENCHMARK FOR resnet18: SUCCESS
    ```
2. Then, convert this model to ONNX:
    ```
    cvt-onnx --model-name resnet18

    Output:
    INFO:root:      Convert to ONNX: START
    INFO:root:      Convert to ONNX: SUCCESS
    INFO:root:      ONNX check: SUCCESS
    INFO:root:      Convert to ONNX Simplified: START
    INFO:root:      Convert to ONNX Simplified: SUCCESS
    INFO:root:      Result validation: START
    INFO:root:      Result validation: SUCCESS
    INFO:root:      >>>>> Result dim = (1, 1000)
    ```
3. After that, try to convert ONNX model to TensorRT:
    ```
    cvt-trt --onnx-path data/onnx-export/resnet18/resnet18_bs-1_res-3x224x224_simplified.onnx

    Output:
    INFO:root:      Convert to TensorRT: START
    INFO:root:      >>>>> TensorRT inference engine settings:
    INFO:root:      >>>>>   * Inference precision - DataType.FLOAT
    INFO:root:      >>>>>   * Max batch size - 1
    INFO:root:      >>>>> ONNX file parsing: START
    INFO:root:      >>>>> Num of network layers: 51
    INFO:root:      >>>>> Building TensorRT engine. This may take a while...
    INFO:root:      >>>>> TensorRT engine build: SUCCESS
    INFO:root:      >>>>> Saving TensorRT engine
    INFO:root:      >>>>> TensorRT engine save: SUCCESS
    INFO:root:      Convert to TensorRT: SUCCESS
    ```
4. Last step - let's see the TRT model performance on the same data as in step 1:
    ```
    bench-trt --trt-path data/trt-export/resnet18/resnet18_prec-32_bs-1_res-3x224x224.engine

    Output:
    INFO:root:      TENSORRT BENCHMARK FOR resnet18: START
    100%|█████████████| 10000/10000 [01:17<00:00, 129.49it/s]
    INFO:root:      Average fps: 1005.4750549267463
    INFO:root:      TOP 1 ACCURACY: 70.19   TOP 1 ERROR: 29.81
    INFO:root:      TOP 5 ACCURACY: 90.49   TOP 5 ERROR: 9.51
    INFO:root:      BENCHMARK FOR resnet18: SUCCESS
    ```

</details>

## Citing
[Back to Content](#Content)

```
@misc{Dobrynin:2021,
  Author = {Dobrynin, Ilya and Panshin, Ivan},
  Title = {Opti Models: Easy-Way Models Convertations},
  Year = {2021},
  Publisher = {GitHub},
  Journal = {GitHub repository},
  Howpublished = {\url{https://github.com/IlyaDobrynin/opti_models}}
}
```

## License
[Back to Content](#Content)

Project is distributed under [MIT License](LICENSE)
