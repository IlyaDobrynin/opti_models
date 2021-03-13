"""
Script for dumping references for the models
"""
import argparse
import logging
import os
from time import time

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from opti_models import show_available_backbones
from opti_models.utils.common_utils import seed_everything
from opti_models.utils.model_utils import get_model

logging.basicConfig(level=logging.INFO)


def initialize_decoder(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.normal_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight)
            nn.init.normal_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.normal_(m.bias)
        else:
            if hasattr(m, 'weight'):
                nn.init.xavier_uniform_(m.weight)
            if hasattr(m, 'bias'):
                nn.init.normal_(m.bias)


def make_reference(model_name: str, device: str, image_path: str) -> torch.Tensor:
    seed_everything()
    assert device in ['cpu', 'cuda'], f'Wrong device: {device}. Shold be "cpu" or "cuda"'
    model = get_model(model_type='classifier', model_name=model_name, model_path=None, show=False)
    model.apply(initialize_decoder)
    model.to(device).eval()

    image = cv2.cvtColor(cv2.imread(image_path, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    image = torch.unsqueeze(torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32)), 0)
    image = image.to(device)
    with torch.no_grad():
        model_out = F.softmax(model(image), dim=-1)
    return model_out


def save_reference(tensor: torch.Tensor, model_name: str, device: str, out_path: str):
    out_ref_path = os.path.join(out_path, model_name)
    if not os.path.exists(out_ref_path):
        os.makedirs(out_ref_path, exist_ok=True)
    torch.save(tensor, os.path.join(out_ref_path, f'{model_name}-lena-{device}.ref'))


def dump_referenes(out_path: str):
    if not os.path.exists(out_path):
        os.makedirs(out_path, exist_ok=True)
    model_names = [name for name in show_available_backbones() if name not in os.listdir(out_path)]
    path_to_image = 'tests/res/lenna.png'
    for i, model_name in enumerate(model_names):
        logging.info(f"\t{i + 1}/{len(model_names)} - {model_name.upper()} DUMP")
        for device in ['cpu', 'cuda']:
            try:
                args = parse_args()
                args.model_name = model_name
                if model_name == "genet_large":
                    args.size = (3, 256, 256)
                elif model_name == 'inception_v3':
                    args.size = (3, 299, 299)
                t = time()
                reference_tensor = make_reference(model_name=model_name, device=device, image_path=path_to_image)
                save_reference(tensor=reference_tensor, model_name=model_name, device=device, out_path=out_path)
                logging.info(f"\t{device.upper()} done in {time() - t} sec.")
            except Exception as e:
                logging.error(f"\t Can't export model: {model_name}:\n\t{repr(e)}")
                # raise e
        logging.info(f"-" * 100)


def parse_args():
    parser = argparse.ArgumentParser(description='Dump references')
    parser.add_argument('--export-dir', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    dump_referenes(out_path=args.export_dir)
