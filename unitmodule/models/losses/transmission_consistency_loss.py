import torch.nn as nn
from mmengine.registry import MODELS
import torch
from torch import Tensor


def transmission_consistency_loss(transmission_map):
    """
    使用 PyTorch 实现透射率一致性损失 (Transmission Consistency Loss)

    参数:
    - transmission_map (torch.Tensor): 形状为 [B, C, H, W] 的张量，包含 B 个样本，每个样本有 C=3 个通道 (RGB)，H 和 W 为图像的高度和宽度。

    返回:
    - total_loss (float): 总透射率一致性损失值。
    """

    # 确保输入是一个张量
    if not isinstance(transmission_map, torch.Tensor):
        raise TypeError("transmission_map 应该是一个 PyTorch 张量")

    # 确保传入的张量是四维 [B, C, H, W]，其中 B 是批次大小，C=3 表示 RGB 通道
    if transmission_map.shape[1] != 3:
        raise ValueError("transmission_map 应该有三个通道 (C=3)")

    # 提取每个批次中的三个通道
    T_R = transmission_map[:, 0, :, :]
    T_G = transmission_map[:, 1, :, :]
    T_B = transmission_map[:, 2, :, :]

    # 对数变换
    log_T_R = torch.log(T_R)
    log_T_G = torch.log(T_G)
    log_T_B = torch.log(T_B)

    # 颜色通道组合 Ω (R-G, R-B, G-B)
    channel_pairs = [(log_T_R, log_T_G), (log_T_R, log_T_B), (log_T_G, log_T_B)]

    # 初始化损失值
    total_loss = 0

    # 遍历每对颜色通道
    for (log_T_c1, log_T_c2) in channel_pairs:
        # 计算对数透射率比值
        ratio = log_T_c1 / log_T_c2

        # 计算比值的均值
        mean_ratio = torch.mean(ratio, dim=[1, 2], keepdim=True)  # 对每个样本计算均值

        # 计算 L2 范数的平方
        loss_term = torch.sum((ratio - mean_ratio) ** 2, dim=[1, 2])  # 对每个样本计算 L2 损失

        # 累加所有样本的损失
        total_loss += torch.sum(loss_term)

    return total_loss


@MODELS.register_module()
class TransmissionConsistencyLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, x: Tensor) -> Tensor:
        loss = transmission_consistency_loss(x)
        return self.loss_weight * loss
