import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

train_size = (1, 3, 256, 256)


class AvgPool2d(nn.Module):
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad

        # only used for fast implementation
        self.fast_imp = fast_imp
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        if self.kernel_size is None and self.base_size:
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // train_size[-1]

            # only used for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // train_size[-1])

        if self.fast_imp:  # Non-equivalent implementation but faster
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                r1 = [r for r in self.rs if h % r == 0][0]
                r2 = [r for r in self.rs if w % r == 0][0]
                # reduction_constraint
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h, w = s.shape
                k1, k2 = min(h - 1, self.kernel_size[0] // r1), min(w - 1, self.kernel_size[1] // r2)
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else:
            n, c, h, w = x.shape
            s = x.cumsum(dim=-1).cumsum(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # pad 0 for convenience
            k1, k2 = min(h, self.kernel_size[0]), min(w, self.kernel_size[1])

            s1, s2, s3, s4 = s[:, :, :-k1, :-k2], s[:, :, :-k1, k2:], s[:, :, k1:, :-k2], s[:, :, k1:, k2:]
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out


class BasicConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, bias=True, norm=False, relu=True, transpose=False):
        super(BasicConv, self).__init__()
        if bias and norm:
            bias = False

        padding = kernel_size // 2
        layers = list()
        if transpose:
            padding = kernel_size // 2 - 1
            layers.append(
                nn.ConvTranspose2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        else:
            layers.append(
                nn.Conv2d(in_channel, out_channel, kernel_size, padding=padding, stride=stride, bias=bias))
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if relu:
            layers.append(nn.GELU())
        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


class Gap(nn.Module):
    def __init__(self, in_channel) -> None:
        super().__init__()

        self.fscale_d = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        self.fscale_h = nn.Parameter(torch.zeros(in_channel), requires_grad=True)
        # self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.gap = AvgPool2d(base_size=80)

    def forward(self, x):
        x_d = self.gap(x)
        x_h = (x - x_d) * (self.fscale_h[None, :, None, None] + 1.)
        x_d = x_d * self.fscale_d[None, :, None, None]
        return x_d + x_h


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, is_filter=False):
        super(ResBlock, self).__init__()
        self.conv1 = BasicConv(in_channel, out_channel, kernel_size=3, stride=1, norm=True)
        self.conv2 = BasicConv(out_channel, out_channel, kernel_size=3, stride=1, norm=True)
        self.is_filter = is_filter

        self.dyna = dynamic_filter(in_channel // 2) if self.is_filter else nn.Identity()
        self.dyna_2 = dynamic_filter(in_channel // 2, kernel_size=5) if self.is_filter else nn.Identity()

        # self.dyna = FixedFilter(in_channel // 2, sigma=3.0) if self.is_filter else nn.Identity()
        # self.dyna_2 = FixedFilter(in_channel // 2, sigma=5.0) if self.is_filter else nn.Identity()
        # self.dyna = RetinexFilter(in_channel // 2) if self.is_filter else nn.Identity()
        # self.dyna_2 = RetinexFilter(in_channel // 2) if self.is_filter else nn.Identity()
        # self.localap = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.global_ap = Gap(in_channel)

    def forward(self, x):
        out = self.conv1(x)

        if self.is_filter:
            k3, k5 = torch.chunk(out, 2, dim=1)
            out_k3 = self.dyna(k3)
            out_k5 = self.dyna_2(k5)
            out = torch.cat((out_k3, out_k5), dim=1)
        # MCSF
        # non_local, local = torch.chunk(out, 2, dim=1)
        # non_local = self.global_ap(non_local)
        # print("nonlocal:", non_local.shape)
        # local = self.localap(local)
        # out = torch.cat((non_local, local), dim=1)
        out = self.global_ap(out)
        out = self.conv2(out)

        return out + x


class dynamic_filter(nn.Module):
    def __init__(self, inchannels, kernel_size=3, stride=1, group=8):
        super(dynamic_filter, self).__init__()
        self.stride = stride
        self.kernel_size = kernel_size
        self.group = group

        # 卷积层
        self.conv = nn.Conv2d(inchannels, group * kernel_size ** 2, kernel_size=1, stride=1, bias=False)

        # 替换为 GroupNorm
        self.norm = nn.GroupNorm(num_groups=8, num_channels=group * kernel_size ** 2, affine=True)

        # Softmax 激活
        self.act = nn.Softmax(dim=-2)

        # Padding
        self.pad = nn.ReflectionPad2d(kernel_size // 2)

        # 自适应平均池化
        self.ap = nn.AdaptiveAvgPool2d((1, 1))

        # 模块化操作
        self.modulate = SFconv(inchannels)

    def forward(self, x):
        identity_input = x  # 输入的原始值
        low_filter = self.ap(x)

        # 卷积操作
        low_filter = self.conv(low_filter)
        low_filter = self.norm(low_filter)  # 使用 GroupNorm 替代 BatchNorm

        # 输入形状和 unfold 操作
        n, c, h, w = x.shape
        x = F.unfold(self.pad(x), kernel_size=self.kernel_size).reshape(
            n, self.group, c // self.group, self.kernel_size ** 2, h * w
        )

        # 低频滤波器处理
        n, c1, p, q = low_filter.shape
        low_filter = low_filter.reshape(
            n, c1 // self.kernel_size ** 2, self.kernel_size ** 2, p * q
        ).unsqueeze(2)

        # Softmax 激活
        low_filter = self.act(low_filter)

        # 低频部分计算
        low_part = torch.sum(x * low_filter, dim=3).reshape(n, c, h, w)

        # 高频部分计算
        out_high = identity_input - low_part

        # 模块化融合
        out = self.modulate(low_part, out_high)
        return out



class SFconv(nn.Module):
    def __init__(self, features, M=2, r=2, L=32) -> None:
        super().__init__()

        d = max(int(features / r), L)
        self.features = features

        self.fc = BasicConv(features, d, kernel_size=1, stride=1, norm=True)
        self.fcs = nn.ModuleList([])
        for i in range(M):
            self.fcs.append(
                BasicConv(features, d, kernel_size=1, stride=1, norm=True)
            )
        self.softmax = nn.Softmax(dim=1)
        # self.gap = nn.AdaptiveAvgPool2d(1)
        self.gap = AvgPool2d(base_size=80)

        self.out = BasicConv(features, d, kernel_size=1, stride=1, norm=True)

    def forward(self, low, high):
        emerge = low + high
        emerge = self.gap(emerge)

        fea_z = self.fc(emerge)

        high_att = self.fcs[0](fea_z)
        low_att = self.fcs[1](fea_z)

        attention_vectors = torch.cat([high_att, low_att], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        high_att, low_att = torch.chunk(attention_vectors, 2, dim=1)

        fea_high = high * high_att
        fea_low = low * low_att

        out = self.out(fea_high + fea_low)
        return out


# 输入图像分成多个固定大小的块（patches），然后对每个块进行处理。提取每个块的低频信息，并结合可学习的参数进行高频和低频的分离处理。
class Patch_ap(nn.Module):
    def __init__(self, inchannel, patch_size):
        super(Patch_ap, self).__init__()

        # self.ap = nn.AdaptiveAvgPool2d((1,1))
        self.ap = AvgPool2d(base_size=80)

        self.patch_size = patch_size
        self.channel = inchannel * patch_size ** 2
        self.h = nn.Parameter(torch.zeros(self.channel))
        self.l = nn.Parameter(torch.zeros(self.channel))

    def forward(self, x):
        # 将输入张量 x 中的高（h）和宽（w）维度重排成多个小块（patches）
        patch_x = rearrange(x, 'b c (p1 w1) (p2 w2) -> b c p1 w1 p2 w2', p1=self.patch_size, p2=self.patch_size)
        # 进一步重排 patch_x 的维度，将通道和 patch 维度合并在一起
        patch_x = rearrange(patch_x, ' b c p1 w1 p2 w2 -> b (c p1 p2) w1 w2', p1=self.patch_size, p2=self.patch_size)

        low = self.ap(patch_x)
        high = (patch_x - low) * self.h[None, :, None, None]
        out = high + low * self.l[None, :, None, None]
        out = rearrange(out, 'b (c p1 p2) w1 w2 -> b c (p1 w1) (p2 w2)', p1=self.patch_size, p2=self.patch_size)

        return out

class RetinexDecomposer(nn.Module):
    def __init__(self, in_channels, sigma_list=(15, 80), eps=1e-6,
                 clamp_log=5.0):
        super().__init__()
        self.in_channels = in_channels
        self.sigma_list = sigma_list
        self.eps = eps
        self.clamp_log = clamp_log
        self.filters = nn.ModuleList([self._make_gaussian_layer(s) for s in sigma_list])

    def _make_gaussian_layer(self, sigma):
        radius = int(3.0 * sigma)
        k = 2 * radius + 1
        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        g = torch.exp(-0.5 * (x / sigma) ** 2)
        g = (g / g.sum()).view(1, 1, 1, k)
        h = nn.Conv2d(self.in_channels, self.in_channels, (1, k),
                      padding=(0, radius), groups=self.in_channels, bias=False)
        h.weight.data.copy_(g.repeat(self.in_channels, 1, 1, 1))
        v = nn.Conv2d(self.in_channels, self.in_channels, (k, 1),
                      padding=(radius, 0), groups=self.in_channels, bias=False)
        v.weight.data.copy_(g.permute(0,1,3,2).repeat(self.in_channels,1,1,1))
        for p in h.parameters(): p.requires_grad = False
        for p in v.parameters(): p.requires_grad = False
        return nn.Sequential(h, v)

    def forward(self, x):
        # 1) 保证正数域，避免 log(<=0)
        x_pos = F.softplus(x, beta=1.0) + self.eps   # 比 ReLU 更平滑

        # 2) log 域分解
        logx = torch.log(x_pos)
        illum_list = [f(logx) for f in self.filters]
        illum = torch.stack(illum_list, dim=0).mean(0)
        reflect = logx - illum

        # 3) 回到线性域并限幅（防爆）
        illum = torch.clamp(illum, -self.clamp_log, self.clamp_log)
        low_lin = torch.exp(illum)                   # 低频 ~ 照明（正）
        # 反射：把比例映射到 [-1,1]，保持类似“高频残差”的尺度
        high_ratio = torch.exp(torch.clamp(reflect, -self.clamp_log, self.clamp_log))
        high_lin = torch.tanh(high_ratio - 1.0)      # 有界，避免后续 GN/Conv 爆

        # 4) 与输入对齐个大致尺度（可选）
        # 把 low 的均值拉回接近 x 的均值，减少分布漂移
        s = (x_pos.mean(dim=(2,3), keepdim=True) + self.eps) / (low_lin.mean(dim=(2,3), keepdim=True) + self.eps)
        low_lin = low_lin * s

        return low_lin, high_lin

class RetinexFilter(nn.Module):
    """用 RetinexDecomposer 替代 dynamic_filter，再复用原 SFconv 做跨分量能量重分配"""
    def __init__(self, in_channels):
        super().__init__()
        self.retinex = RetinexDecomposer(in_channels)
        self.modulate = SFconv(in_channels)  # 直接复用你现有的 SFconv

    def forward(self, x):
        low, high = self.retinex(x)
        # 与 dynamic_filter 输出语义保持一致：传低频+高频到 SFconv
        out = self.modulate(low, high)
        return out

# ====== 添加到 layers.py ======


class FixedGaussianLPF(nn.Module):
    """固定高斯低通（深度可分离 1D×1D），不学习参数，通道独立"""
    def __init__(self, in_channels, sigma=3.0):
        super().__init__()
        self.in_channels = in_channels
        self.sigma = float(sigma)
        radius = int(3.0 * self.sigma)
        k = 2 * radius + 1

        x = torch.arange(-radius, radius + 1, dtype=torch.float32)
        g = torch.exp(-0.5 * (x / self.sigma) ** 2)
        g = (g / g.sum()).view(1, 1, 1, k)  # 水平核

        self.h = nn.Conv2d(in_channels, in_channels, kernel_size=(1, k),
                           padding=(0, radius), groups=in_channels, bias=False)
        self.v = nn.Conv2d(in_channels, in_channels, kernel_size=(k, 1),
                           padding=(radius, 0), groups=in_channels, bias=False)

        with torch.no_grad():
            self.h.weight.copy_(g.repeat(in_channels, 1, 1, 1))
            self.v.weight.copy_(g.permute(0,1,3,2).repeat(in_channels, 1, 1, 1))
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x):
        return self.v(self.h(x))


class FixedFilter(nn.Module):
    """用固定低通+高频残差替换 DFSM 的分解；再复用原 SFconv 做融合"""
    def __init__(self, in_channels, sigma=3.0):
        super().__init__()
        self.lpf = FixedGaussianLPF(in_channels, sigma=sigma)
        # 复用你现有的跨分量融合模块（名字一般叫 SFconv 或类似）
        self.modulate = SFconv(in_channels)

    def forward(self, x):
        low  = self.lpf(x)
        high = x - low
        out  = self.modulate(low, high)
        return out
