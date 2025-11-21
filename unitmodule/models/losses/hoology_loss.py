import torch.nn as nn
from mmengine.registry import MODELS
from torch import Tensor


@MODELS.register_module()
class HomologyLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        # from color channel (0, 1, 2) corresponding to (1, 2, 0)
        loss = self.loss_fn(a, b)
        return self.loss_weight * loss

