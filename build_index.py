"""离线构建图像特征库。

扫描 images/ 目录下的所有图片，逐张提取特征向量，
连同图片路径一起序列化保存到 features.pkl，供在线检索时直接加载。

用法：
    python build_index.py
    python build_index.py --images images --output features.pkl
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle

import numpy as np

from feature_extractor import FeatureExtractor

# 支持的图片后缀
_IMAGE_PATTERNS = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")


def gather_images(folder: str) -> list[str]:
    """递归收集目录下所有受支持的图片路径，并去重排序。"""
    paths: list[str] = []
    for pattern in _IMAGE_PATTERNS:
        paths.extend(glob.glob(os.path.join(folder, pattern)))
        paths.extend(glob.glob(os.path.join(folder, "**", pattern), recursive=True))
    return sorted(set(paths))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建图像特征库")
    parser.add_argument("--images", default="images", help="图片库目录，默认 images")
    parser.add_argument("--output", default="features.pkl", help="输出的特征库文件")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = gather_images(args.images)

    if not paths:
        print(f"[提示] 在 {args.images}/ 目录下没有找到图片。")
        print("       请先把图片放入该目录（可分子文件夹），或运行 "
              "python scripts/prepare_cifar_demo.py 快速生成示例图。")
        return

    print(f"共发现 {len(paths)} 张图片，开始提取特征（使用预训练 ResNet50）...")
    extractor = FeatureExtractor()

    features = []
    for index, path in enumerate(paths, start=1):
        features.append(extractor.extract(path))
        if index % 10 == 0 or index == len(paths):
            print(f"  已处理 {index}/{len(paths)}")

    database = {
        "paths": paths,
        "features": np.stack(features),  # shape=(图片数量, 2048)
    }
    with open(args.output, "wb") as f:
        pickle.dump(database, f)

    print(f"[完成] 特征库已保存到 {args.output}，共 {len(paths)} 张图片。")


if __name__ == "__main__":
    main()
