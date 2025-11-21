from typing import Dict, Union

import torch
from mmdet.models.detectors import (CascadeRCNN, DETR, DINO,
                                    FasterRCNN, FCOS, RetinaNet, TOOD)
from mmengine.optim import OptimWrapper
from mmengine.registry import MODELS
from mmyolo.models.detectors import YOLODetector


def train_step_with_fs_module(self, data: Union[dict, tuple, list],
                              optim_wrapper: OptimWrapper) -> Dict[str, torch.Tensor]:
    """With the UnitModule loss"""
    with optim_wrapper.optim_context(self):
        data, fs_losses = self.data_preprocessor(data, True)
        losses = self._run_forward(data, mode='loss')
    losses.update(unit_losses)
    parsed_losses, log_vars = self.parse_losses(losses)
    optim_wrapper.update_params(parsed_losses)
    return log_vars


def with_fs_module(cls):
    cls.train_step = train_step_with_fs_module
    return cls


@MODELS.register_module()
@with_fs_module
class FSYOLODetector(YOLODetector):
    """YOLODetector with FSModule"""


@MODELS.register_module()
@with_fs_module
class FSFasterRCNN(FasterRCNN):
    """FasterRCNN with FSModule"""

