path_to_images="/mnt/Disk_G/DL_Data/source/imagenet/imagenetv2-topimages/imagenetv2-top-images-format-val"
trt_path="/mnt/Disk_F/Programming/pet_projects/opti_models/data/trt_export/genet_small_bs-1_res-(224,224).engine"

python opti_models/benchmarks/imagenet_tensorrt_benchmark.py \
  --path_to_images ${path_to_images}\
  --trt_path ${trt_path}\
  --in_size 224 224
