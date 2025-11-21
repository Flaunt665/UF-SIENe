from typing import Optional, Tuple, Union

import mmcv.cnn as cnn
import torch
import torch.nn as nn
from einops import rearrange
from mmcv.cnn import build_activation_layer, build_norm_layer
from mmengine.model import BaseModule
from mmengine.registry import MODELS
from torch import Tensor

from .gb import *
from .layers import *


class MultiScaleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(MultiScaleConv, self).__init__()
        # 定义多尺度卷积层
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 5, padding=2)
        self.conv3 = nn.Conv2d(in_ch, out_ch, 7, padding=3)

    def forward(self, x):
        # 对每种卷积核大小的卷积进行并行计算
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        # 将多尺度的输出进行融合
        return x1 + x2 + x3


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False, padding_mode='reflect'),
            nn.GroupNorm(num_channels=out_ch, num_groups=8, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False, padding_mode='reflect'),
            nn.GroupNorm(num_channels=out_ch, num_groups=8, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class InDoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(InDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 9, stride=4, padding=4, bias=False, padding_mode='reflect'),
            nn.GroupNorm(num_channels=out_ch, num_groups=8, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False, padding_mode='reflect'),
            nn.GroupNorm(num_channels=out_ch, num_groups=8, affine=True),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class LKA(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        u = x.clone()
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class LKABlock(nn.Module):
    def __init__(self, d_model, norm_cfg: Optional[dict] = None):
        super().__init__()
        if norm_cfg is None:
            norm_cfg = dict(type='GN', num_groups=8)
        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
        self.norm = build_norm_layer(norm_cfg, d_model)[1]

    def forward(self, x):
        shorcut = x.clone()
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        x = self.norm(x)
        return x



class FELKBlock(BaseModule):
    def __init__(self,
                 d_model,
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None,
                 is_filter=False):

        super().__init__(init_cfg)
        if norm_cfg is None:
            norm_cfg = dict(type='GN', num_groups=8)
        if act_cfg is None:
            act_cfg = dict(type='ReLU')

        self.proj_1 = nn.Conv2d(d_model, d_model, 1)
        self.spatial_conv = LKA(d_model)
        self.proj_2 = nn.Conv2d(d_model, d_model, 1)
        self.norm = build_norm_layer(norm_cfg, d_model)[1]

        self.fft_conv = ResBlock(d_model*2, d_model*2, is_filter=is_filter)
        # self.fft_conv2 = cnn.ConvModule(d_model, d_model, 3, padding=1,
        #                                  padding_mode='reflect', norm_cfg=norm_cfg, act_cfg=act_cfg)
        # self.pw = cnn.ConvModule(d_model, d_model, 1, 1,
        #                          padding_mode='reflect', norm_cfg=norm_cfg, act_cfg=None)

    def forward(self, x) -> Tensor:
        batch, c, h, w = x.size()
        spatial_feat = self.proj_1(x)
        spatial_feat = self.spatial_conv(spatial_feat)



        # 频域卷积
        # 1. 先转换到频域
        fft_feat = torch.fft.rfft2(x, norm='ortho')
        x_fft_real = torch.unsqueeze(torch.real(fft_feat), dim=-1)
        x_fft_imag = torch.unsqueeze(torch.imag(fft_feat), dim=-1)
        fft_feat = torch.cat((x_fft_real, x_fft_imag), dim=-1)
        fft_feat = rearrange(fft_feat, 'b c h w d -> b (c d) h w').contiguous()

        # 2. 频域卷积处理
        fft_feat = self.fft_conv(fft_feat)


        # 3. 还原回时域
        fft_feat = rearrange(fft_feat, 'b (c d) h w -> b c h w d', d=2).contiguous()
        fft_feat = torch.view_as_complex(fft_feat)
        fft_feat = torch.fft.irfft2(fft_feat, s=(h, w), norm='ortho')

        out = spatial_feat + fft_feat
        out = self.proj_2(out)
        out = self.norm(out)

        # fft_feat = self.fft_conv2(fft_feat)
        #
        # out = self.pw(spatial_feat + fft_feat)

        return x+out


class ChannelAttention(nn.Module):
    """Channel-attention module https://github.com/open-mmlab/mmdetection/tree/v3.0.0rc1/configs/rtmdet."""

    def __init__(self, channels: int) -> None:
        """Initializes the class and sets the basic configurations and instance variables required."""
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Conv2d(channels, channels, 1, 1, 0, bias=True)
        self.act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Applies forward pass using activation on convolutions of the input, optionally using batch normalization."""
        return x * self.act(self.fc(self.pool(x)))


class SpatialAttention(nn.Module):
    """Spatial-attention module."""

    def __init__(self, kernel_size=7):
        """Initialize Spatial-attention module with kernel size argument."""
        super().__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.cv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.act = nn.Sigmoid()

    def forward(self, x):
        """Apply channel and spatial attention on input for feature recalibration."""
        return x * self.act(self.cv1(torch.cat([torch.mean(x, 1, keepdim=True), torch.max(x, 1, keepdim=True)[0]], 1)))


class CBAM(nn.Module):
    """Convolutional Block Attention Module."""

    def __init__(self, c1, kernel_size=7):
        """Initialize CBAM with given input channel (c1) and kernel size."""
        super().__init__()
        self.channel_attention = ChannelAttention(c1)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        """Applies the forward pass through C1 module."""
        return self.spatial_attention(self.channel_attention(x))


@MODELS.register_module()
class FSBackbone(BaseModule):
    def __init__(self,
                 stem_channels: Tuple[int],
                 embed_dims: Tuple[int],
                 in_channels: int = 3,
                 padding_mode: str = 'reflect',
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)
        if norm_cfg is None:
            norm_cfg = dict(type='GN', num_groups=8)
        if act_cfg is None:
            act_cfg = dict(type='ReLU')
        inc = in_channels

        stem_layers = []
        for outc in stem_channels:
            stem_layers.append(
                cnn.ConvModule(inc, outc, 3, 2,
                               padding=1, padding_mode=padding_mode,
                               norm_cfg=norm_cfg, act_cfg=act_cfg))
            inc = outc
        self.stem = nn.Sequential(*stem_layers)

        layers = [FELKBlock(d_model=embed_dims[i], is_filter=True) for i in
                  range(len(embed_dims) - 1)]
        layers.append(FELKBlock(d_model=embed_dims[-1], is_filter=False))
        self.layers = nn.Sequential(*layers)

    def forward(self, x) -> Tensor:
        x = self.stem(x)
        x_1 = self.layers[0](x)
        out = self.layers[1](x_1) + x
        return out


@MODELS.register_module()
class FSAHead(BaseModule):
    def __init__(self,
                 init_cfg=None):
        super().__init__(init_cfg)
        self.conv1 = MultiScaleConv(3, 64)
        self.conv2 = DoubleConv(64, 64)
        self.dense = nn.Linear(64, 3, bias=False)
        # self.threshold = 0.1
        self.maxpool = nn.MaxPool2d(15, 7)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.cbam_1 = CBAM(64)
        self.cbam_2 = CBAM(64)

    def forward(self, x) -> Tensor:
        a = self.conv1(x)
        a = self.cbam_1(a)
        a = self.maxpool(a)
        a = self.conv2(a)
        a = self.cbam_2(a)
        a = self.pool(a)
        a = a.view(-1, 64)
        a = torch.sigmoid(self.dense(a))
        a = a.view(-1, 3, 1, 1)
        return a

@MODELS.register_module()
class GBAHead(BaseModule):
    def __init__(self, init_cfg=None):
        super().__init__(init_cfg)

    def forward(self, x) -> Tensor:
        a = get_A(x).cuda()
        a = torch.mean(a, dim=[2, 3], keepdim=True)
        a = a.view(-1, 3, 1, 1)
	
        return a


@MODELS.register_module()
class FSTHead(BaseModule):
    def __init__(self,
                 in_channels: int,
                 hid_channels: int,
                 out_channels: int = 3,
                 padding_mode: str = 'reflect',
                 norm_cfg: Optional[dict] = None,
                 act_cfg: Optional[dict] = None,
                 init_cfg: Optional[dict] = None):
        super().__init__(init_cfg)

        # 将 upscale_factor 固定为 2
        self.upscale_factor = 2  # 内部固定的上采样因子

        if norm_cfg is None:
            norm_cfg = dict(type='GN', num_groups=8)
        if act_cfg is None:
            act_cfg = dict(type='ReLU')

        # 第一个卷积层输出的通道数调整为 hid_channels * (self.upscale_factor ** 2)
        self.conv1 = cnn.ConvModule(
            in_channels,
            hid_channels * (self.upscale_factor ** 2),
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg
        )

        # 使用 PixelShuffle 进行上采样
        self.pixel_shuffle1 = nn.PixelShuffle(self.upscale_factor)

        # 第二个卷积层
        self.conv2 = cnn.ConvModule(
            hid_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            padding_mode=padding_mode,
            norm_cfg=None,
            act_cfg=None
        )
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x) -> Tensor:
        # 第一次卷积并通过 PixelShuffle 上采样
        x = self.conv1(x)
        x = self.pixel_shuffle1(x)

        # 第二次卷积
        x = self.conv2(x)
        x = self.up2(x)

        # 应用 Sigmoid 激活函数
        x = torch.sigmoid(x)

        return x

class TransmissionLoss(nn.Module):
    def __init__(self, loss_weight: float = 1.0):
        super().__init__()
        self.loss_weight = loss_weight
        self.loss_fn = nn.MSELoss(reduction='mean')

    def forward(self, a: Tensor, b: Tensor) -> Tensor:
        loss = self.loss_fn(a, b)
        return self.loss_weight * loss


@MODELS.register_module()
class FSModule(BaseModule):
    def __init__(self,
                 unit_backbone: dict,
                 t_head: dict,
                 a_head: dict,
                 loss_t,
                 loss_acc: Optional[dict] = None,
                 loss_cc: Optional[dict] = None,
                 loss_sp: Optional[dict] = None,
                 loss_tv: Optional[dict] = None,
                 loss_fp: Optional[dict] = None,
                 loss_tc: Optional[dict] = None,
                 alpha: float = 0.9,
                 t_min: float = 0.001,
                 init_cfg=None):
        super().__init__(init_cfg)
        assert 0 < alpha < 1
        assert 0 <= t_min < 0.1

        self.alpha = alpha
        self.t_min = t_min

        self.unit_backbone = MODELS.build(unit_backbone)
        self.t_head = MODELS.build(t_head)
        self.a_head = MODELS.build(a_head)

        self.loss_t = MODELS.build(loss_t)
        self.loss_acc = MODELS.build(loss_acc) if loss_acc else None
        self.loss_cc = MODELS.build(loss_cc) if loss_cc else None
        self.loss_sp = MODELS.build(loss_sp) if loss_sp else None
        self.loss_tv = MODELS.build(loss_tv) if loss_tv else None
        self.loss_fp = MODELS.build(loss_fp) if loss_fp else None
        self.loss_tc = MODELS.build(loss_tc) if loss_tc else None

    def forward(self, x, training: bool = False) -> Union[Tensor, Tuple[Tensor, dict]]:
        if training:
            return self.loss(x)
        else:  # training == False
            return self.predict(x)

    def _forward(self, x) -> Tuple[Tensor, Tensor]:
        feature = self.unit_backbone(x)
        t = self.t_head(feature)
        a = self.a_head(x)
        # print("t:", t.shape)
        # print("a:", a.shape)
        return t, a

    def predict(self, x, show: bool = False) -> Union[Tensor, tuple]:
        t, a = self._forward(x)
        t = torch.clamp(t, min=self.t_min)

        # 保存最终的 x 图像
        x = self.denoise(x, t, a)
        x = torch.clamp(x, 0, 1)
        

        return (x, t, a) if show else x




    def loss(self, x) -> Tuple[Tensor, dict]:
        feature = self.unit_backbone(x)
        t = self.t_head(feature)
        a = self.a_head(x)

        t = torch.clamp(t, min=self.t_min)

        # get x of denoise
        x_denoise = self.denoise(x, t, a)

        # create fake x with noise and predict its t and A
        x_fake = self.noise(x, self.alpha, a)
        t_fake, a_fake = self._forward(x_fake)
        x_fake_denoise = self.denoise(x_fake, t_fake, a_fake)

        loss_t = self.loss_t(self.alpha * t, t_fake)
        losses = dict(loss_t=loss_t)
        if self.loss_acc:
            loss_acc = self.loss_acc(feature, a)
            losses.update(loss_acc=loss_acc)

        if self.loss_cc:
            loss_cc = self.loss_cc(x_denoise)
            losses.update(loss_cc=loss_cc)

        if self.loss_sp:
            loss_sp = self.loss_sp(x_denoise, x_fake_denoise)
            losses.update(loss_sp=loss_sp)

        if self.loss_tv:
            loss_tv = self.loss_tv(x_denoise)
            losses.update(loss_tv=loss_tv)

        if self.loss_fp:
            loss_fp = self.loss_fp(x, x_denoise)
            losses.update(loss_fp=loss_fp)

        if self.loss_tc:
            loss_tc = self.loss_tc(self.alpha * t)
            losses.update(loss_tc=loss_tc)

        x_denoise = torch.clamp(x_denoise, 0, 1)
        return x_denoise, losses

    @staticmethod
    def noise(x, t, a) -> Tensor:
        """Noise image"""
        return x * t + (1 - t) * a

    @staticmethod
    def denoise(x, t, a) -> Tensor:
        """Denoise image"""
        return (x - (1 - t) * a) / t


