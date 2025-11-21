# import numpy as np
# import matplotlib.pyplot as plt
from mmcv.transforms import BaseTransform
from mmcv.transforms.utils import cache_randomness
from mmengine.registry import TRANSFORMS
from .esBlur import *


@TRANSFORMS.register_module()
class BlurGuidedColorRandomTransfer(BaseTransform):
    """Transfer underwater image color by converting HSV color space.

    HSV is (Hue, Saturation, Value).
    The uint8 image(255)(h, w, c) convert to HSV that
    H in [0, 180),
    S in [0, 255],
    V in [0, 255].

    Required Keys:

    - img

    Modified Keys:

    - img

    Args:
        hue_prob (float): The probability for hue in range [0, 1]. Defaults to 0.5.
        saturation_prob (float): The probability for saturation in range [0, 1]. Defaults to 0.5.
        value_prob (float): The probability for value in range [0, 1]. Defaults to 0.5.
        hue_delta (int): delta of hue. Defaults to 5.
        saturation_delta (int): delta of saturation. Defaults to 30.
        value_delta (int): delta of value. Defaults to 30.

    Notes:
        The underwater_hue_interval got from the hue mean in underwater dataset,
        which get the hue mean by convert color from BGR to HSV.
        dataset     |    hue min     |     hue max
        ------------|----------------|-------------
        DUO         |    18.7551     |     95.4836
        URPC2020    |    17.9668     |     99.6359
        URPC2021    |    17.9668     |     103.2373
        UIEB        |    25.5417     |     116.3379
        TrashCan |           0        |    179
         ------------|----------------|-------------
        hue interval       18                116
    """
    underwater_hue_interval = (0, 179)

    def __init__(self,
                 hue_prob: float = 0.5,
                 saturation_prob: float = 0.5,
                 value_prob: float = 0.5,
                 hue_delta: int = 5,
                 saturation_delta: int = 30,
                 value_delta: int = 30,
                 ) -> None:
        assert 0 <= hue_prob <= 1.0
        assert 0 <= saturation_prob <= 1.0
        assert 0 <= value_prob <= 1.0

        self.hue_prob = hue_prob
        self.saturation_prob = saturation_prob
        self.value_prob = value_prob
        self.hue_delta = hue_delta
        self.saturation_delta = saturation_delta
        self.value_delta = value_delta

        self._hue_min, self._hue_max = self.underwater_hue_interval
        self._hue_middle = (self._hue_min + self._hue_max) / 2

    @cache_randomness
    def _random_hue(self):
        return np.random.rand() < self.hue_prob

    @cache_randomness
    def _random_saturation(self):
        return np.random.rand() < self.saturation_prob

    @cache_randomness
    def _random_value(self):
        return np.random.rand() < self.value_prob

    @staticmethod
    def _random_mult():
        return np.random.uniform(-1, 1)

    @cache_randomness
    def _get_hue_gain(self, img):
        """Get hue gain value and keep it in underwater hue interval."""
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hue_mean = np.mean(img_hsv[..., 0])
        hue_gain = self._random_mult() * self.hue_delta

        # img_hue is in the underwater hue interval
        if self._hue_min < hue_mean < self._hue_max:
            hue = np.clip(hue_mean + hue_gain, self._hue_min, self._hue_max)
            hue_gain = hue - hue_mean

        # img_hue is out of the underwater hue interval
        else:
            hue_gain = np.abs(hue_gain)
            if hue_mean >= self._hue_max:
                hue_gain = -hue_gain

        return np.array(hue_gain, dtype=np.int16)

    @cache_randomness
    def _get_saturation_gain(self):
        gain = self._random_mult() * self.saturation_delta
        return np.array(gain, dtype=np.int16)

    @cache_randomness
    def _get_value_gain(self):
        gain = self._random_mult() * self.value_delta
        return np.array(gain, dtype=np.int16)

    def transform(self, results: dict) -> dict:
        """Gaussian Blur Ablation for BGDA.

        在消融实验中，我们保留 BGDA 的类名和注册方式，
        但仅使用简单高斯模糊作为数据增强，不再依赖模糊图和 HSV 颜色扰动。
        """
        img = results['img']  # uint8, BGR

        # 以一定概率做模糊（例如 0.5，避免所有图片都被弄糊）
        if np.random.rand() < 0.5:
            # 随机选择一个小核大小，保证不会太毁图
            k = np.random.choice([3, 5, 7])
            # kernel 必须是奇数
            img = cv2.GaussianBlur(img, (k, k), 0)

        results['img'] = img
        return results
    # def transform(self, results: dict) -> dict:
    #     win = 9
    #     glb = 0
    #     lc = 1.0
    #     save_path = ""
    #     # gamma = 1/2
    #     hue_able = self._random_hue()
    #     saturation_able = self._random_saturation()
    #     value_able = self._random_value()
    # 
    #     if not any((hue_able, saturation_able, value_able)):
    #         return results
    # 
    #     img = results['img']
    #     blur_map = estBlur(img, win)
    #     alter_map = relu_map(blur_map, 0.19)
    # 
    #     # blur_map = blur_relu(blur_map, 0.1)
    #     # blur_map = normalize_to_range(blur_map)
    #     # blur_map = sigmoid(blur_map)
    #     # alter_mask = np.random.rand(*blur_map.shape) < blur_map
    #     # alter_map = alter_mask.astype(int)
    #     # print("normalized:", np.count_nonzero(alter_map))
    #     # blur_map = (blur_map*255).astype(np.uint8)
    #     # blur_map = gamma_correction(blur_map, gamma)
    #     img_dtype = img.dtype
    # 
    #     assert img_dtype == np.uint8
    #     # convert color uint8 from BGR to HSV
    #     img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
    # 
    #     if hue_able:
    #         hue_gain = self._get_hue_gain(img)
    #         hsv_map = int(hue_gain*lc) * alter_map + int(hue_gain * glb)
    #         img_hsv[..., 0] = (img_hsv[..., 0] + hsv_map) % 180
    #         # print("hue_gain", hue_gain)
    #         # print("alter_num", np.sum(alter_map))
    #         # print("changes", np.sum(hue_gain * alter_map))
    # 
    #     if saturation_able:
    #         saturation_gain = self._get_saturation_gain()
    #         saturation_map = int(saturation_gain*lc) * alter_map + int(saturation_gain * glb)
    #         img_hsv[..., 1] = np.clip(img_hsv[..., 1] + saturation_map, 0, 255)
    #         # print("hue_gain", saturation_gain)
    #         # print("alter_num", np.sum(alter_map))
    #         # print("changes", np.sum(saturation_gain * alter_map))
    # 
    #     if value_able:
    #         value_gain = self._get_value_gain()
    #         value_map = int(value_gain*lc) * alter_map + int(value_gain * glb)
    #         img_hsv[..., 2] = np.clip(img_hsv[..., 2] + value_map, 0, 255)
    # 
    #     # convert color from HSV to BGR
    #     img = cv2.cvtColor(img_hsv.astype(img_dtype), cv2.COLOR_HSV2BGR)
    #     # print("Is diffrent?", np.array_equal(test_img, img))
    # 
    # 
    # 
    #     results['img'] = img
    #     return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(underwater_hue_interval={self.underwater_hue_interval}, '
        repr_str += f'hue_prob={self.hue_prob}, '
        repr_str += f'saturation_prob={self.saturation_prob}, '
        repr_str += f'value_prob={self.value_prob}, '
        repr_str += f'hue_delta={self.hue_delta}, '
        repr_str += f'saturation_delta={self.saturation_delta}, '
        repr_str += f'value_delta={self.value_delta})'
        return repr_str


# def main():
#     # 读取测试图像
#
#     img = cv2.imread('test4.jpg')  # 请替换成您的图像路径
#     img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#     # start_time = time.time()
#     # 实例化数据增强类
#     transform = BlurGuidedColorRandomTransfer(
#         hue_prob=0.7,
#         saturation_prob=0.7,
#         value_prob=0.7,
#         hue_delta=10,
#         saturation_delta=20,
#         value_delta=20
#     )
#
#     # 模拟 results 字典
#     results = {'img': img}
#
#     # 应用数据增强
#     results = transform.transform(results)
#     # end_time = time.time()
#     # # 计算并打印程序运行的时间
#     # elapsed_time = end_time - start_time
#     # print(f"程序运行时间: {elapsed_time:.6f} 秒")
#     # 获取增强后的图像
#     img_transformed = results['img']
#     img_transformed_rgb = cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB)
#
#     # 显示原图与增强后的图像
#     plt.figure(figsize=(10, 5))
#
#     plt.subplot(1, 2, 1)
#     plt.title("Original Image")
#     plt.imshow(img_rgb)
#
#     plt.subplot(1, 2, 2)
#     plt.title("Transformed Image")
#     plt.imshow(img_transformed_rgb)
#
#     plt.show()
#
#
# if __name__ == "__main__":
#     main()