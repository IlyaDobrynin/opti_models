import json
import logging
import os
import typing as t

import numpy as np
import pandas as pd

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
except Exception:
    logging.warning(f"\tCan't import pycuda")
try:
    import tensorrt as trt
except Exception:
    logging.warning(f"\tCan't import tensorrt")


class HostDeviceMem:
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def load_engine(trt_runtime, engine_path):
    with open(engine_path, 'rb') as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


def allocate_buffers(engine):
    """Allocates host and device buffer for TRT engine inference.
    This function is similair to the one in ../../common.py, but
    converts network outputs (which are np.float32) appropriately
    before writing them to Python buffer. This is needed, since
    TensorRT plugins doesn't support output type description, and
    in our particular case, we use NMS plugin as network output.
    Args:
        engine (trt.ICudaEngine): TensorRT engine
    Returns:
        inputs [HostDeviceMem]: engine input memory
        outputs [HostDeviceMem]: engine output memory
        bindings [int]: buffer to device bindings
        stream (cuda.Stream): cuda stream for engine inference synchronization
    """
    inputs = []
    outputs = []
    bindings = []
    stream = cuda.Stream()

    # Current NMS implementation in TRT only supports DataType.FLOAT but
    # it may change in the future, which could brake this sample here
    # when using lower precision [e.g. NMS output would not be np.float32
    # anymore, even though this is assumed in binding_to_type]
    binding_to_type = {"input": np.float32, "output": np.float32}
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding))  # * engine.max_batch_size
        dtype = binding_to_type[str(binding)]
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem))
        else:
            outputs.append(HostDeviceMem(host_mem, device_mem))
    return inputs, outputs, bindings, stream


def get_shapes(engine):
    sizes = []
    for binding in engine:
        sizes.append(trt.volume(engine.get_binding_shape(binding)))
    return sizes


def prepare_data(path_to_images: str):
    img_classes = [int(s) for s in os.listdir(path_to_images)]
    out_dict = {}
    for img_cls in img_classes:
        images_folder_path = os.path.join(path_to_images, str(img_cls))
        for img_name in os.listdir(images_folder_path):
            path_to_image = os.path.join(images_folder_path, img_name)
            out_dict[path_to_image] = img_cls

    out_df = pd.DataFrame()
    out_df['names'] = list(out_dict.keys())
    out_df['labels'] = list(out_dict.values())

    return out_df


def top_n_accuracy(preds: t.List, truths: t.List, n: int):
    best_n = np.argsort(-np.asarray(preds), axis=1)[:, :n]
    successes = 0
    for i, truth in enumerate(truths):
        if truth in best_n[i, :]:
            successes += 1
    return float(successes) / len(truths)


def compute_metrics(trues_df: pd.DataFrame, preds: t.Dict, top_n_ranks: t.Tuple = (1, 5)):
    true_labels = []
    pred_labels = []
    for name in preds.keys():
        true_labels.append(trues_df[trues_df['names'] == name]['labels'].values.tolist()[0])
        pred_labels.append(preds[name])

    return (top_n_accuracy(preds=pred_labels, truths=true_labels, n=rank) for rank in top_n_ranks)


def combine_statistics(
    trt_models_path: str, excluded_stats: t.List = ('top_1_err', 'top_5_err'), sort_by: str = '16_images_per_second'
) -> pd.DataFrame:
    stats_dict = {}
    for path, folders, files in os.walk(trt_models_path):
        if (len(files) > 0) and (all([file.endswith('.json') for file in files])):
            model_name = path.split(os.sep)[-2]
            stats_dict[model_name] = {}
            for file in files:
                precision = file.split("_")[0]
                with open(os.path.join(path, file)) as f:
                    stats = json.load(f)
                stats_dict[model_name][precision] = {k: v for k, v in stats.items()}
    out_df = pd.DataFrame()
    out_df['model_name'] = [model_name for model_name in stats_dict.keys()]

    df_dict = {'32': {}, '16': {}, '8': {}}
    for model_name in stats_dict.keys():
        for precision in sorted(stats_dict[model_name].keys()):
            for key, value in stats_dict[model_name][precision].items():
                if key not in excluded_stats:
                    if key not in df_dict[precision]:
                        df_dict[precision][key] = [value]
                    else:
                        df_dict[precision][key].append(value)

    for precision, values_dict in df_dict.items():
        for k, v in values_dict.items():
            out_df[f"{precision}_{k}"] = v
    out_df.sort_values(by=[sort_by], inplace=True, ascending=False, ignore_index=True)
    return out_df
