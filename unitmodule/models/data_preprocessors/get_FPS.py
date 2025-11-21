import torch
import time
from fsie_moule import FSModule



c_s1, c_s2 = 32, 32
norm_cfg = dict(type='GN', num_groups=8)
act_cfg = dict(type='ReLU')

# 创建 FsBackbone, FSThead 和 FSAhead 实例
unit_backbone = dict(type='FSBackbone',
        stem_channels=(c_s1, c_s2),
        embed_dims=(32, 32),
        norm_cfg=norm_cfg,
        act_cfg=act_cfg)  # 在此处传入 FsBackbone 所需的实际参数
t_head = dict( type='FSTHead',
        in_channels=32,
        hid_channels=32,
        out_channels=3,
        norm_cfg=norm_cfg,
        act_cfg=act_cfg)  # 在此处传入 FSThead 所需的实际参数
a_head = dict(type='FSAHead')  # 在此处传入 FSAhead 所需的实际参数


# 初始化 FSModule
model = FSModule(unit_backbone=unit_backbone, t_head=t_head, a_head=a_head)

# 创建一个输入张量，大小为 (batch_size, channels, height, width)
input_tensor = torch.randn(1, 3, 640, 640)  # 假设输入为3通道，224x224图像，批量大小为1

# 计算 FLOPs 和参数数量
num_iterations = 100
start_time = time.time()

for _ in range(num_iterations):
    with torch.no_grad():  # Don't compute gradients for FPS measurement
        output = model(input_tensor)

end_time = time.time()

# Calculate FPS
elapsed_time = end_time - start_time
print(elapsed_time)