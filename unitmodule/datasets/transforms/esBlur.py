import numpy as np
import cv2
# import matplotlib.pyplot as plt
# import time


def relu_map(arr, a):
    out = np.maximum(0, arr-a)
    out[out > 0] = 1
    return out


def sigmoid(x):

    # 避免溢出问题
    array = np.clip(x, -128, 127)
    return 1 / (1 + np.exp(-x))


def normalize_to_range(data, new_min=-5, new_max=5):
    """
    将一个二维数组标准化到指定的区间 [new_min, new_max]。

    参数:
    - data: 输入的二维数组，待标准化的数据。
    - new_min: 目标区间的最小值，默认为 -10。
    - new_max: 目标区间的最大值，默认为 10。

    返回:
    - 标准化后的二维数组，范围在 [new_min, new_max] 之间。
    """
    # 计算原始数据的最小值和最大值
    data_min = np.min(data)
    data_max = np.max(data)

    # 将数据线性标准化到 [new_min, new_max] 范围
    normalized_data = (data - data_min) / (data_max - data_min) * (new_max - new_min) + new_min

    return normalized_data


def boxfilter(image, radius):
    ksize = (2 * radius + 1, 2 * radius + 1)
    filtered_image = cv2.boxFilter(image, -1, ksize,
                                   borderType=cv2.BORDER_REPLICATE)
    return filtered_image


def fast_guidedfilter_color(I, P, radius, eps, step):
    """
    Fast guide filter for colorful guidance image

    Parameters
    ----------
    I: colorful guidance image (3 channels)
    P: input image, may be gray-scale or colorful
    radius: radius for box-filter
    step: step for down sample
    eps: regularization factor
    """
    # check parameters
    I = np.squeeze(I)
    P = np.squeeze(P)
    if I.ndim < 3 or I.shape[2] != 3:
        raise ValueError("guidance image must have 3 channels.")

    # cache original data type
    original_data_type = P.dtype

    # change data type to float32
    I = np.float32(I)
    P = np.float32(P)

    # initialize result
    result = P.copy()
    if result.ndim == 2:
        result = np.expand_dims(result, axis=2)

    # down sample
    height, width = I.shape[:2]
    down_size = (width // step, height // step)
    I_down = cv2.resize(I, dsize=down_size, fx=None, fy=None,
                        interpolation=cv2.INTER_NEAREST)
    P_down = cv2.resize(P, dsize=down_size, fx=None, fy=None,
                        interpolation=cv2.INTER_NEAREST)
    radius_down = radius // step

    # guide filter - processing guidance image I
    mean_I = boxfilter(I_down, radius_down)

    var_I_00 = boxfilter(I_down[..., 0] * I_down[..., 0], radius_down) - \
               mean_I[..., 0] * mean_I[..., 0] + eps
    var_I_11 = boxfilter(I_down[..., 1] * I_down[..., 1], radius_down) - \
               mean_I[..., 1] * mean_I[..., 1] + eps
    var_I_22 = boxfilter(I_down[..., 2] * I_down[..., 2], radius_down) - \
               mean_I[..., 2] * mean_I[..., 2] + eps
    var_I_01 = boxfilter(I_down[..., 0] * I_down[..., 1], radius_down) - \
               mean_I[..., 0] * mean_I[..., 1]
    var_I_02 = boxfilter(I_down[..., 0] * I_down[..., 2], radius_down) - \
               mean_I[..., 0] * mean_I[..., 2]
    var_I_12 = boxfilter(I_down[..., 1] * I_down[..., 2], radius_down) - \
               mean_I[..., 1] * mean_I[..., 2]

    inv_00 = var_I_11 * var_I_22 - var_I_12 * var_I_12
    inv_11 = var_I_00 * var_I_22 - var_I_02 * var_I_02
    inv_22 = var_I_00 * var_I_11 - var_I_01 * var_I_01
    inv_01 = var_I_02 * var_I_12 - var_I_01 * var_I_22
    inv_02 = var_I_01 * var_I_12 - var_I_02 * var_I_11
    inv_12 = var_I_02 * var_I_01 - var_I_00 * var_I_12

    det = var_I_00 * inv_00 + var_I_01 * inv_01 + var_I_02 * inv_02

    inv_00 = inv_00 / det
    inv_11 = inv_11 / det
    inv_22 = inv_22 / det
    inv_01 = inv_01 / det
    inv_02 = inv_02 / det
    inv_12 = inv_12 / det

    # guide filter - filter input image P for every single channel
    mean_P = boxfilter(P_down, radius_down)
    if mean_P.ndim == 2:
        mean_P = np.expand_dims(mean_P, axis=[2])
        P_down = np.expand_dims(P_down, axis=[2])

    channels = np.min([3, mean_P.shape[2]])
    for ch in range(channels):
        mean_P_channel = mean_P[..., ch:ch + 1]
        P_channel = P_down[..., ch:ch + 1]
        mean_Ip = boxfilter(I_down * P_channel, radius_down)
        cov_Ip = mean_Ip - mean_I * mean_P_channel

        a0 = inv_00 * cov_Ip[..., 0] + inv_01 * cov_Ip[..., 1] + \
             inv_02 * cov_Ip[..., 2]
        a1 = inv_01 * cov_Ip[..., 0] + inv_11 * cov_Ip[..., 1] + \
             inv_12 * cov_Ip[..., 2]
        a2 = inv_02 * cov_Ip[..., 0] + inv_12 * cov_Ip[..., 1] + \
             inv_22 * cov_Ip[..., 2]
        b = mean_P[..., ch] - a0 * mean_I[..., 0] - a1 * mean_I[..., 1] - \
            a2 * mean_I[..., 2]
        a = np.concatenate((a0[..., np.newaxis], a1[..., np.newaxis],
                            a2[..., np.newaxis]), axis=2)

        mean_a = boxfilter(a, radius_down)
        mean_b = boxfilter(b, radius_down)

        mean_a_up = cv2.resize(mean_a, dsize=(width, height), fx=None,
                               fy=None, interpolation=cv2.INTER_LINEAR)
        mean_b_up = cv2.resize(mean_b, dsize=(width, height), fx=None,
                               fy=None, interpolation=cv2.INTER_LINEAR)
        gf_one_channel = np.sum(mean_a_up * I, axis=2) + mean_b_up
        result[..., ch] = gf_one_channel

    # post process data type
    result = np.squeeze(result)
    if original_data_type == np.uint8:
        result = np.clip(np.round(result), 0, 255).astype(np.uint8)
    return result


def estBlur(I, win):
    r = 36
    eps = 1e-8
    s = int(r/4)

    # 确保I是一个 numpy 数组，如果I是一个 PIL 图像对象，可以转换为 numpy 数组
    if isinstance(I, np.ndarray) is False:
        I = np.array(I)  # 转换为 numpy 数组
    I = I.astype(np.float32) / 255.0  # 将图像转换为浮动类型并归一化
    height, width, _ = I.shape  # 获取图像的高度、宽度和通道数
    # 将RGB图像转换为YCbCr色彩空间
    imYUV = cv2.cvtColor(I, cv2.COLOR_RGB2YCrCb)

    # 提取亮度分量（Y通道）
    imY = imYUV[:, :, 0]

    # 设置半径数组
    radius = [9, 17, 33]

    # 初始化一个全零的3D数组
    DiffImage = np.zeros((height, width, 4))
    for idx, r in enumerate(radius):
        # 确定高斯核的大小（确保为奇数）
        kernel_size = (r * 2 + 1, r * 2 + 1)  # 窗口大小必须是奇数
        # 使用 OpenCV 的高斯滤波器平滑亮度图像
        GFImage = cv2.GaussianBlur(imY, kernel_size, sigmaX=r)
        # 计算平滑后图像与原图像的绝对差
        DiffImage[:, :, idx] = np.abs(imY - GFImage)

    # 计算 DiffImage 的均值，生成初步模糊图
    roughBlurMap = np.mean(DiffImage, axis=2)
    kernel = np.ones((win, win), np.uint8)  # 使用方形结构元素
    DBlur = cv2.dilate(roughBlurMap, kernel)

    # 第二次膨胀操作，用于标记
    # marker = cv2.dilate(DBlur, kernel)

    # 孔洞填充：使用图像重建
    # 1. 计算反转图像
    # complement_marker = cv2.bitwise_not(marker)
    complement_DBlur = cv2.bitwise_not(DBlur)

    # 2. 使用自定义的图像重建过程
    # FDBlur = reconstruction(complement_DBlur, complement_marker)
    FDBlur = cv2.morphologyEx(complement_DBlur, cv2.MORPH_CLOSE, kernel)
    FDBlur = cv2.bitwise_not(FDBlur)  # 反转填充后的图像
    # Refine blurriness map using guided filter
    Blur = fast_guidedfilter_color(I, FDBlur, r, eps, s)
    # r_blur = blur_relu(Blur, 0.15)
    # print("sum", np.sum(r_blur>0))
    # plt.figure(figsize=(6, 6))
    # plt.imshow(r_blur, cmap='gray')
    # plt.title('r_lur (After Refining and ReLU)')
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()
    # plt.figure(figsize=(6, 6))
    # plt.imshow(Blur, cmap='gray')
    # plt.title('Blur (After Refining)')
    # plt.axis('off')  # 关闭坐标轴
    # plt.show()
    return Blur


# def gamma_correction(image, gamma):
#     # 将图像像素值归一化到[0, 1]
#     image = image / 255.0
#
#     # 应用gamma矫正
#     corrected_image = np.power(image, gamma)
#
#     # 将图像值重新缩放回[0, 255]
#     corrected_image = np.uint8(corrected_image * 255)
#
#     return corrected_image
#
# def main_2():
#     # Load a sample image
#     I = cv2.imread('test4.jpg')  # Ensure the image is in the working directory
#     win = 9  # Window size for dilation
#
#     # Estimate the blurriness map
#     blur_map = estBlur(I, win)
#
#
#     # Print the estimated blur map
#     print("Estimated blur map:")
#     print(blur_map)
#
#     # # Save the estimated blur map as a jpg image
#     # save_path = "blur4.jpg"
#     # cv2.imwrite(save_path, (blur_map * 255).astype(np.uint8))  # Convert to uint8 and save
#     # # 显示 Blur 图像
#     plt.figure(figsize=(6, 6))
#     plt.imshow(blur_map, cmap='gray')
#     plt.title('Blur (After Refining)')
#     plt.axis('off')  # 关闭坐标轴
#     plt.show()
#     # print(f"Estimated blur map saved to {save_path}")
#
#
# if __name__ == "__main__":
#     # I = cv2.imread('blur4.jpg')
#     # # I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
#     # print(I.dtype)
#     # print("i")
#     # result = gamma_correction(I, 0.5)
#     # cv2.imwrite('gammma4.jpg', I)
#
#     main_2()

# def main():
#     # Create a sample image (e.g., 10x10 matrix)
#     test_image = np.random.rand(10, 10)
#
#     # Define the radius for the box filter
#     radius = 2
#
#     # Apply the box filter
#     filtered_image = boxfilter(test_image, radius)
#
#     # Print the results
#     print("Original Image:")
#     print(test_image)
#     print("\nFiltered Image:")
#     print(filtered_image)
#
#
# # 测试快速引导滤波函数
# def main_1():
#     # 读取彩色图像（引导图像）
#     I = cv2.imread('test4.jpg')
#     I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)  # 转换为 RGB 格式
#
#     # 读取灰度图像（需要滤波的输入图像）
#     p = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
#
#     # 设置引导滤波参数
#     r = 8 # 半径
#     eps = 0.01  # 正则化参数
#     s = 2  # 子采样比例（例如：r/4）
#
#     # 执行快速引导滤波
#     q = fast_guidedfilter_color(I, p, r, eps, s)
#     q_bgr = cv2.cvtColor(q.astype(np.uint8), cv2.COLOR_RGB2BGR)
#
#     # 保存滤波后的图像
#     cv2.imwrite('filtered_image.jpg', q_bgr)
#

