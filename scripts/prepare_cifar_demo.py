"""可选：快速生成一份示例图库，用于在没有自己图片时跑通整个流程。

从 torchvision 自带的 CIFAR-10 小数据集中，每个类别取若干张图片，
放大后保存到 images/cifar10/ 目录（共 10 类，每类 10 张，合计 100 张）。

用法：python scripts/prepare_cifar_demo.py
"""
from __future__ import annotations

import os

from torchvision import datasets
from PIL import Image

CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
PER_CLASS = 10  # 每个类别保存的图片数量
OUT_DIR = os.path.join("images", "cifar10")
UPSCALE = 128   # CIFAR 原图仅 32x32，放大后更便于查看


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    dataset = datasets.CIFAR10(root="./data", train=True, download=True)

    counters = [0] * len(CLASSES)
    for image, label in dataset:
        if all(c >= PER_CLASS for c in counters):
            break
        if counters[label] >= PER_CLASS:
            continue
        class_name = CLASSES[label]
        class_dir = os.path.join(OUT_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        image = image.resize((UPSCALE, UPSCALE), Image.BICUBIC)
        image.save(os.path.join(class_dir, f"{class_name}_{counters[label]:02d}.png"))
        counters[label] += 1

    print(f"[完成] 已在 {OUT_DIR} 生成示例图片，每类 {PER_CLASS} 张。")
    print("接下来运行：python build_index.py")


if __name__ == "__main__":
    main()
