# UF-SIENe

# <center> UnitModule

###  Underwater object detection (UOD) plays a vital role in marine ecological monitoring, facility
 inspection, and resource exploration. However, underwater images often suffer from blurriness,
 noise, and color distortion, severely degrading detection performance. Traditional enhancement
 methodsprioritize visual aesthetics but neglect the needs of detection models. To bridge this gap,
 we propose UF-SIENet, a dedicated enhancement network for underwater detection. Central to
 our method is the Frequency Enhanced Low-level Knowledge Aggregation (FELKA) module,
 which estimates the transmission map by decomposing features into high- and low-frequency
 components using learnable low-pass filters. It then adaptively fuses these components to
 enhance structural and semantic consistency, improving detail preservation under challenging
 underwater conditions. To estimate background light, we integrate Underwater Background
 Attention Module (UBAM), which applies both channel and spatial attention, allowing the
 network to concentrate on informative regions while suppressing background interference. This
 attention-guided mechanism improves estimation robustness in scenes with uneven illumination.
 We further propose Blur-Guided Data Augmentation (BGDA), which utilizes blurred-region
 priors to guide the detection model’s attention toward ambiguous areas, thereby increasing
 robustness to various forms of visual degradation. Extensive experiments on the DUO and
 TrashCan datasets demonstrate that UF-SIENet consistently improves detection accuracy across
 various models, with up to 3.1% AP gain on YOLOV10-S



### Installation

This project is based on [MMDetection](https://github.com/open-mmlab/mmdetection/tree/main).

- Python 3.8
- Pytorch 1.11.0+cu113

**Step 1.** Create a conda virtual environment and activate it.

```bash
conda create -n unitmodule python=3.8 -y
conda activate unitmodule
```

**Step 2.** Install PyTorch following [official instructions](https://pytorch.org/get-started/locally/).

Linux and Windows

```bash
# Wheel CUDA 11.3
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113
```

```bash
# Conda CUDA 11.3
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
```

**Step 3.** Install MMDetection and dependent packages.

```bash
pip install -U openmim
mim install mmengine==0.7.4
mim install mmcv==2.0.0
mim install mmdet==3.0.0
mim install mmyolo==0.5.0
pip install -r requirements.txt
```



### Training

```bash
bash tools/dist_train.sh configs/yolov10/DA_fsmodule_yolov10_s_100e_duo.py 2
```

### Test

```bash
bash tools/dist_test.sh configs/yolox/yolox_s_100e_duo.py yolox_s_100e_duo.pth 2
```
