# opti_models
## Description

## Install
```
git clone git@github.com:IlyaDobrynin/opti_models.git && cd opti_models
git checkout dev
pip install .
```

## Benchmarks
### Default Imagenet Benchmark
1. Download and untar: https://drive.google.com/file/d/1Yi_SZ400LKMXeA08BvDip4qBJonaThae/view?usp=sharing
2. In `scripts/default_imagenet_benchmark.sh` change:
    - `path_to_images` - path to the `imagenetv2-top-images-format-val` folder
    - `model_name` - name of the model to bench
    - `in_size` - height and width of the input image, default imagenet value = 224
    - `batch_size` - anount of the images in every batch
    - `workers` - number of workers
3. Run: `bash scripts/default_imagenet_benchmark.sh`
