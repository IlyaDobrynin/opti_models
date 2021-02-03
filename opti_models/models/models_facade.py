# -*- coding: utf-8 -*-
"""
    Facade for all models classes
"""
from typing import Union
from torch.nn import Module
from . import ClassificationFactory
from .backbones.backbone_factory import BACKBONES
import logging
logging.basicConfig(level=logging.INFO)


class ModelFacade:
    """
        Class realize facade pattern for all models
        Arguments:
            task:           Task for the model:
                                - classification
                                - opti-classification
                                - detection
                                - ocr
            model_name:     Name of the architecture for the given task. See in documentation.

    """

    _models_dict = {
        "classification": BACKBONES,
        "opti-classification": {
            "basic_classifier": ClassificationFactory,
        },
        # "segmentation": {
        # },
        # "detection": {
        # },
        # "ocr": {
        # }
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

    def show_available_models(self):
        for m_type, m_dict in self._models_dict.items():
            logging.info(f"\tModels, available for {m_type}:")
            for m_name in m_dict.keys():
                logging.info(f"\t\t{m_name}")

