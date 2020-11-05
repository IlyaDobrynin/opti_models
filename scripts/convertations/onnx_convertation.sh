model_name="genet_small"
export_dir="data/onnx_export"
batch_size=1

python opti_models/convertations/cvt_onnx.py \
  --model_name ${model_name}\
  --export_dir ${export_dir}\
  --batch_size ${batch_size}\
  --in_size 224 224
