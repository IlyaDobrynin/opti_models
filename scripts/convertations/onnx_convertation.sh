model_name="resnet18"
export_dir="data/onnx_export"
batch_size=1
num_classes=5

python opti_models/convertations/cvt_onnx.py \
  --model_name ${model_name}\
  --export_dir ${export_dir}\
  --batch_size ${batch_size}\
  --num_classes ${num_classes}\
  --in_size 224 224
