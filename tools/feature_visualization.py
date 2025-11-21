import cv2
import mmcv
import numpy as np
import os
import torch
import matplotlib.pyplot as plt


def featuremap_2_heatmap(feature_map, save_intermediate=False, save_dir="intermediate_features", index=0):
    """将特征图转换为单通道热力图，并保存中间结果。"""
    assert isinstance(feature_map, torch.Tensor)
    feature_map = feature_map.detach()
    heatmap = feature_map[:, 0, :, :] * 0
    heatmaps = []

    # 按通道累加
    for c in range(feature_map.shape[1]):
        heatmap += feature_map[:, c, :, :]

    # 转换为 NumPy 格式并归一化
    heatmap = heatmap.cpu().numpy()
    heatmap = np.mean(heatmap, axis=0)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmaps.append(heatmap)

    # 保存中间归一化的特征图
    if save_intermediate:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, f"feature_map_{index}.png")
        cv2.imwrite(save_path, np.uint8(255 * heatmap))
        #print(f"Saved intermediate feature map: {save_path}")

    return heatmaps


def draw_feature_map(features, save_dir="feature_map/540our", img_path="/data/DN_7/UOD_project/UnitModule-FSModule/data/DUO/images/test/540.jpg", i=0, save_intermediate=False, save_padded_img=True):
    """绘制热力图并叠加到输入图像上，保持与预处理一致性。"""
    # 加载输入图像
    img = mmcv.imread(img_path)
    ori_h, ori_w = img.shape[:2]

    # 按预处理调整输入图像尺寸
    img_scale = (640, 640)
    pad_val = (114, 114, 114)  # 预处理填充值
    scale_factor = min(img_scale[0] / ori_h, img_scale[1] / ori_w)
    resized_h, resized_w = int(ori_h * scale_factor), int(ori_w * scale_factor)
    img_resized = mmcv.imresize(img, (resized_w, resized_h), return_scale=False)

    # 创建填充后的图像
    img_padded = np.full((*img_scale, 3), pad_val, dtype=np.uint8)
    img_padded[:resized_h, :resized_w, :] = img_resized

    # 保存填充后的输入图像
    if save_padded_img:
        padded_img_save_dir = os.path.join(save_dir, "padded_inputs")
        if not os.path.exists(padded_img_save_dir):
            os.makedirs(padded_img_save_dir)
        padded_img_path = os.path.join(padded_img_save_dir, "padded_input.png")
        cv2.imwrite(padded_img_path, img_padded)
        #print(f"Saved padded input image: {padded_img_path}")

    # 处理特征图并绘制热力图
    if isinstance(features, torch.Tensor):
        for idx, heat_maps in enumerate(features):
            heat_maps = heat_maps.unsqueeze(0)
            heatmaps = featuremap_2_heatmap(heat_maps, save_intermediate=save_intermediate, save_dir="intermediate_features", index=i + idx)
            for heatmap in heatmaps:
                heatmap = cv2.resize(heatmap, img_scale)  # 调整到预处理后的大小
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap * 0.5 + img_padded * 0.3  # 热力图叠加到填充后的图像

                # 去除填充并调整为输入图像的原始大小
                cropped_superimposed = superimposed_img[:resized_h, :resized_w]
                resized_superimposed = cv2.resize(cropped_superimposed, (ori_w, ori_h))

                # 显示最终图像
                plt.imshow(resized_superimposed[:, :, ::-1])
                plt.show()

                # 保存最终结果
                final_save_path = os.path.join(save_dir, f"final_yolo_{i + idx}.png")
                cv2.imwrite(final_save_path, resized_superimposed)
                #print(f"Saved final resized superimposed image: {final_save_path}")
    else:
        for idx, featuremap in enumerate(features):
            heatmaps = featuremap_2_heatmap(featuremap, save_intermediate=save_intermediate, save_dir="intermediate_features", index=i + idx)
            for heatmap in heatmaps:
                heatmap = cv2.resize(heatmap, img_scale)  # 调整到预处理后的大小
                heatmap = np.uint8(255 * heatmap)
                heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
                superimposed_img = heatmap * 0.5 + img_padded * 0.3  # 热力图叠加到填充后的图像

                # 去除填充并调整为输入图像的原始大小
                cropped_superimposed = superimposed_img[:resized_h, :resized_w]
                resized_superimposed = cv2.resize(cropped_superimposed, (ori_w, ori_h))

                # 显示最终图像
                #plt.imshow(resized_superimposed[:, :, ::-1])
                #plt.show()

                # 保存最终结果
                final_save_path = os.path.join(save_dir, f"final_yolo_{i + idx}.png")
                cv2.imwrite(final_save_path, resized_superimposed)
                #print(f"Saved final resized superimposed image: {final_save_path}")
