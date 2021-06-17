import argparse
import json
import logging
import os

import pandas as pd

from opti_models.benchmarks.imagenet_tensorrt_benchmark import main

logging.basicConfig(level=logging.INFO)


def parse_args():
    # Default args
    path = "/usr/local/opti_models/imagenetv2-top-images-format-val"
    parser = argparse.ArgumentParser(description='Simple speed benchmark, based on TRT models')
    parser.add_argument('--trt-path', type=str, help="Path to TRT model")
    parser.add_argument(
        '--path-to-images', default=path, type=str, help=f"Path to the validation images, default: {path}"
    )
    return parser.parse_args()


def get_precision(trt_model_name: str):
    name_list = trt_model_name.split("_")
    precision = None
    for n in name_list:
        if n.startswith("prec"):
            precision = n.split("-")[1]
    if precision is None:
        raise ValueError(f"Can't find precision in trt_model_name: {trt_model_name}")
    return precision


def combine_statistics(trt_models_path: str):

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
    out_df['model_name'] = [model_name for model_name in stats_dict.keys() for _ in stats_dict[model_name].keys()]
    out_df['precision'] = [
        precision for model_name in stats_dict.keys() for precision in sorted(stats_dict[model_name].keys())
    ]

    df_dict = {}
    for model_name in stats_dict.keys():
        for precision in sorted(stats_dict[model_name].keys()):
            for key, value in stats_dict[model_name][precision].items():
                if key not in df_dict:
                    df_dict[key] = [value]
                else:
                    df_dict[key].append(value)
    for key, values in df_dict.items():
        out_df[key] = values
    return out_df


def bench_all():
    from opti_models.models.backbones.backbone_factory import show_available_backbones

    excluded_names = [
        'efficientnet_b1b',
        'efficientnet_b2b',
        'efficientnet_b3b',
        'efficientnet_b4b',
        'efficientnet_b5b',
        'efficientnet_b6b',
        'efficientnet_b7b',
        'efficientnet_b0c',
        'efficientnet_b1c',
        'efficientnet_b2c',
        'efficientnet_b3c',
        'efficientnet_b4c',
        'efficientnet_b5c',
        'efficientnet_b6c',
        'efficientnet_b7c',
        'efficientnet_b8c',
    ]
    trt_models_path = "data/trt-export"
    model_names = [
        name
        for name in show_available_backbones()
        if (name not in excluded_names) and (name in os.listdir(trt_models_path))
    ]
    for i, model_name in enumerate(model_names):
        logging.info(f"\t{i + 1}/{len(model_names)}: {model_name.upper()}")
        trt_model_path = os.path.join(trt_models_path, model_name)
        trt_model_names = [f for f in os.listdir(trt_model_path) if f.endswith(".engine")]
        for trt_model_name in trt_model_names:

            precision = get_precision(trt_model_name=trt_model_name)
            logging.info(f"\tPRECISION: {precision} -------------")

            trt_path = os.path.join(trt_model_path, trt_model_name)
            export_name = os.path.join(trt_model_path, f"statistics/{precision}_{trt_model_name.split('.')[0]}.json")
            args = parse_args()
            args.trt_path = trt_path
            args.export_name = export_name
            try:
                main(args=args)
            except Exception as e:
                logging.info(f"\tCan't bench {model_name} {precision}\n{repr(e)}")

        logging.info(f"-" * 100)

    statistics_df = combine_statistics(trt_models_path=trt_models_path)
    statistics_df.to_csv(os.path.join(trt_models_path, "statistics.csv"), index=False)


if __name__ == '__main__':
    bench_all()
