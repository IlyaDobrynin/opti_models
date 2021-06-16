import logging
import os
import typing as t

import numpy as np
import onnx

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
except Exception:
    print("Can't import pycuda")
import tensorrt as trt

from ..utils.image_utils import imagenet_preprocess

logging.basicConfig(level=logging.INFO)


def get_input_shape(model_path: str) -> t.Tuple:
    model = onnx.load(model_path)
    layer = model.graph.input[0]
    tensor_type = layer.type.tensor_type
    size = []
    if tensor_type.HasField("shape"):
        for d in tensor_type.shape.dim:
            if d.HasField("dim_value"):
                size.append(d.dim_value)
    return tuple(size)


def load_data(data_dir: str, onnx_model_path: str, preprocess_method: callable = None):
    bs, c, h, w = get_input_shape(model_path=onnx_model_path)

    image_paths = []
    for root_dir, folders, files in os.walk(data_dir):
        if len(files) > 0:
            image_paths += [os.path.join(root_dir, file) for file in files]

    # image_paths = [os.path.join(data_dir, x) for x in os.listdir(data_dir)]
    logging.info(f"\tNumber of calibration images: {len(image_paths)}")
    batches = np.zeros(shape=(len(image_paths), c, h, w), dtype=np.float32)
    if preprocess_method is None:
        logging.info(f"\tUsing default preprocess_method: imagenet_preprocess")
        preprocess_method = imagenet_preprocess
    for i, image_path in enumerate(image_paths):
        batches[i] = preprocess_method(image_path, (w, h))
    return batches


class Int8EntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self,
        cache_file: str,
        calibration_images_dir: str,
        batch_size: int,
        onnx_model_path: str,
        preprocess_method: callable = None,
    ):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.data = load_data(
            data_dir=calibration_images_dir, onnx_model_path=onnx_model_path, preprocess_method=preprocess_method
        )
        self.batch_size = batch_size
        self.current_index = 0
        self.device_input = cuda.mem_alloc(self.data[0].nbytes * self.batch_size)

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names, p_str=None):
        if self.current_index + self.batch_size > self.data.shape[0]:
            return None
        batch = self.data[self.current_index : self.current_index + self.batch_size].ravel()
        cuda.memcpy_htod(self.device_input, batch)
        self.current_index += self.batch_size
        return [self.device_input]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)
