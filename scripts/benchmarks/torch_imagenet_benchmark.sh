path_to_images="/mnt/Disk_G/DL_Data/source/imagenet/imagenetv2-topimages/imagenetv2-top-images-format-val"
model_name="genet_small"
batch_size=1
workers=1

python opti_models/benchmarks/imagenet_torch_benchmark.py \
  --path_to_images ${path_to_images}\
  --model_name ${model_name}\
  --in_size 224 224\
  --batch_size ${batch_size}\
  --workers ${workers}

