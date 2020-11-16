onnx_path="data/onnx_export/resnet18/resnet18_bs-1_res-224x224_simplified.onnx"
export_dir="data/trt_export"
batch_size=1
fp_type="32"

python opti_models/convertations/cvt_tensorrt.py \
  --onnx_path ${onnx_path}\
  --export_dir ${export_dir}\
  --batch_size ${batch_size}\
  --in_size 224 224\
  --fp_type ${fp_type}

