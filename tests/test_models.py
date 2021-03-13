import os

import cv2
import numpy as np
import pytest
import torch
from torch.nn import functional as F

from opti_models import get_model, show_available_backbones
from opti_models.utils.common_utils import seed_everything
from opti_models.utils.dump_references import initialize_decoder

MODEL_NAMES = show_available_backbones()
REFERENCES_PATH = "tests/references"
LENA_PATH = "tests/res/lenna.png"


def _test_model(model: torch.nn.Module, device: str, model_name: str, reference_val: float = 1e-10):
    image = cv2.cvtColor(cv2.imread(LENA_PATH, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    image = torch.unsqueeze(torch.from_numpy(image.transpose(2, 0, 1).astype(np.float32)), 0)
    image = image.to(device)
    with torch.no_grad():
        out = F.softmax(model(image), dim=-1)
    sample = torch.load(os.path.join(REFERENCES_PATH, model_name, f"{model_name}-lena-{device}.ref"))
    assert out.shape == sample.shape

    # for i in range(out.shape[-1]):
    #     out_t = out[..., i]
    #     sample_t = sample[..., i]
    #     if not (np.abs((out_t.data.cpu().numpy() - sample_t.data.cpu().numpy())) < reference_val):
    #         print(out_t, sample_t)
    #     assert np.abs((out_t.data.cpu().numpy() - sample_t.data.cpu().numpy())) < reference_val


@pytest.mark.parametrize("model_name", MODEL_NAMES)
@pytest.mark.parametrize("device", ["cpu"])
def test_forward(model_name: str, device: str):
    seed_everything()
    if model_name in os.listdir(REFERENCES_PATH):
        model = get_model(model_type='classifier', model_name=model_name, model_path=None, show=False)
        model.apply(initialize_decoder)
        model.to(device).eval()
        _test_model(model=model, device=device, model_name=model_name)


if __name__ == "__main__":
    pytest.main([__file__])
