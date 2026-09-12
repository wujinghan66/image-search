"""特征提取模块。

加载在 ImageNet 上预训练好的 ResNet50，去掉最后的 1000 类分类头，
把任意一张图片编码成 2048 维、并做 L2 归一化的特征向量（图像的“语义指纹”）。
归一化之后，两个向量做点积即为余弦相似度。
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms

# ImageNet 数据集的通道均值与标准差，预训练模型预处理必须保持一致
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD = [0.229, 0.224, 0.225]


class FeatureExtractor:
    """基于预训练 ResNet50 的图像特征提取器。"""

    def __init__(self) -> None:
        # 有显卡用显卡，没有就用 CPU，本项目数据量小，CPU 也足够
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 加载预训练 ResNet50；把最后的全连接分类层替换成恒等层，
        # 这样模型直接输出倒数第二层的 2048 维特征
        self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        self.model.fc = nn.Identity()
        self.model.eval().to(self.device)

        # 与训练时一致的预处理：缩放、中心裁剪、张量化、标准化
        self.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
            ]
        )

    @torch.no_grad()
    def extract(self, image) -> np.ndarray:
        """把一张图片编码为归一化的特征向量。

        参数 image 可以是图片路径(str)、PIL.Image 或 numpy 数组。
        返回：shape=(2048,) 的 numpy 向量，L2 范数为 1。
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        feat = self.model(tensor).squeeze(0).cpu()
        feat = feat / feat.norm()  # L2 归一化：之后向量点积即余弦相似度
        return feat.numpy()
