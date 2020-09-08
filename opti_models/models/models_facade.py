# -*- coding: utf-8 -*-
"""
    Facade for all models classes
"""
from typing import Union
from torch.nn import Module
from . import ClassificationFactory
from .backbones.backbone_factory import BACKBONES

# import segmentation_models_pytorch as smp


class ModelFacade:
    """
        Class realize facade pattern for all models
        Arguments:
            task:           Task for the model:
                                - classification
                                - segmentation
                                - detection
            model_name:     Name of the architecture for the given task. See in documentation.

    """

    _models_dict = {
        "backbones": BACKBONES,
        "classification": {
            "basic_classifier": ClassificationFactory,
        },
        "segmentation": {
        },
        "detection": {
        },
        "ocr": {
        }
    }

    def __init__(self, task: str):
        tasks = self._models_dict.keys()
        if task not in tasks:
            raise ValueError(
                f"Wrong task parameter: {task}. "
                f"Should be: {[t for t in tasks]}"
            )
        self.task = task

    def get_model_class(
            self,
            model_definition: Union[str, Module]
    ):
        """ Metod returns model class

        :return:
        """

        if isinstance(model_definition, str) and \
                model_definition in self._models_dict[self.task].keys():
            model_class = self._models_dict[self.task][model_definition]
        elif isinstance(model_definition, Module):
            model_class = model_definition
        else:
            raise ValueError(
                f"Wrong model_definition parameter: {model_definition}. "
                f"Should be in {self._models_dict[self.task].keys()} "
                f"or an instance of torch.nn.Module."
            )

        return model_class
