import os
import torch
import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import ToTensor


def my_save_image(name, image_np, output_path=""):
    if not os.path.exists(output_path):
        os.mkdir(output_path)
        
    p = np_to_pil(image_np)
    p.save(output_path + "{}".format(name))


def pil_to_np(img_PIL, with_transpose=True):
    """
    Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    """
    ar = np.array(img_PIL)
    if len(ar.shape) == 3 and ar.shape[-1] == 4:
        ar = ar[:, :, :3]
        # this is alpha channel
    if with_transpose:
        if len(ar.shape) == 3:
            ar = ar.transpose(2, 0, 1)
        else:
            ar = ar[None, ...]

    return ar.astype(np.float32) / 255.


def np_to_pil(img_np):
    """
    Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    :param img_np:
    :return:
    """
    ar = np.clip(img_np * 255, 0, 255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        assert img_np.shape[0] == 3, img_np.shape
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)


def np_to_torch(img_np):
    """
    Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]

    :param img_np:
    :return:
    """
    return torch.from_numpy(img_np)[None, :]


def torch_to_np(img_var):
    """
    Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    :param img_var:
    :return:
    """
    return img_var.detach().cpu().numpy()[0]


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


def get_A(x):
    # 先保证输入 Tensor 的第一个维度是批次维度
    batch_size = x.shape[0]
    
    # 创建一个空的列表来保存处理后的结果
    result = []
    
    for i in range(batch_size):
        # 取出当前批次的图像
        img = x[i].unsqueeze(0)  # 形状为 [3, 640, 640]
        
        # 将 Tensor 转换为 NumPy 数组，确保值在[0, 1]之间
        img_np = np.clip(torch_to_np(img), 0, 1)
      
        # 转换为 PIL 图像
        img_pil = np_to_pil(img_np)
        
        # 获取图像的大小
        h, w = img_pil.size
        windows = (h + w) / 2
        
        # 应用高斯模糊
        img_pil = img_pil.filter(ImageFilter.GaussianBlur(windows))
        
        # 转换回 Tensor
        img_tensor = ToTensor()(img_pil)  # 形状 [3, 640, 640]
        
        # 将处理后的图像加入结果列表
        result.append(img_tensor)
    
    # 将所有处理后的图像拼接成一个 Tensor
    A = torch.stack(result)  # 形状 [4, 3, 640, 640]
    
    # 最后我们可以选择加一个 batch 维度 (如果需要)
    return A
