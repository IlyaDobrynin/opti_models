import typing as t

import cv2
import numpy as np
from albumentations import Compose, Normalize, Resize
from albumentations.pytorch.transforms import ToTensorV2


def imagenet_preprocess(image_path: str, size: t.Tuple) -> np.ndarray:
    transforms = Compose(
        [
            Resize(height=size[0], width=size[1], interpolation=cv2.INTER_AREA),
            Normalize(),
            ToTensorV2(),
        ],
        p=1,
    )

    image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    input_data = transforms(image=image)["image"]

    return input_data
