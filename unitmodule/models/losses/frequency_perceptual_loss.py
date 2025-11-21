import torch
import torch.nn as nn
from mmengine.registry import MODELS
from torch import Tensor


def to_frequency_domain(img):
    fft = torch.fft.fft2(img, norm='ortho')  # 频域表示
    amplitude = torch.abs(fft).real  # 振幅谱，确保为实数
    phase = torch.angle(fft).real    # 相位谱，确保为实数
    return amplitude, phase


def hybrid_loss(gen_img, ref_img, alpha=0.5, beta=0.5, gamma=1.0):
    gen_amp, gen_phase = to_frequency_domain(gen_img)
    ref_amp, ref_phase = to_frequency_domain(ref_img)

    # 振幅损失
    amp_loss = torch.mean((gen_amp - ref_amp) ** 2)

    # 优化后的相位损失（使用余弦相似性）
    phase_loss = torch.mean(1 - torch.cos(gen_phase - ref_phase))

    # 时域重建损失
    recon_loss = torch.nn.functional.l1_loss(gen_img, ref_img)

    # 综合损失
    total_loss = alpha * amp_loss + beta * phase_loss + gamma * recon_loss
    return total_loss


@MODELS.register_module()
class FrequencyPerceptualLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0, alpha=0.5, beta=0.5, gamma=1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        loss = hybrid_loss(a, b, self.alpha, self.beta, self.gamma)
        return self.loss_weight * loss


