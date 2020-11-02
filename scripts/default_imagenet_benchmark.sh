path_to_images="/mnt/Disk_G/DL_Data/source/imagenet/imagenetv2-topimages/imagenetv2-top-images-format-val"
model_name="genet_small"
in_size=256
batch_size=32
workers=11

python opti_models/benchmarks/imagenet_torch_benchmark.py \
  --path_to_images ${path_to_images}\
  --model_name ${model_name}\
  --in_size ${in_size}\
  --batch_size ${batch_size}\
  --workers ${workers}

