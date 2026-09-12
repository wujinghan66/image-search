"""在线以图搜图演示（Gradio 网页界面）。

启动前请先运行 build_index.py 生成 features.pkl。
运行：python app.py，然后浏览器打开终端里提示的本地地址。
"""
from __future__ import annotations

import os
import pickle

import gradio as gr
import numpy as np

from feature_extractor import FeatureExtractor

DB_PATH = "features.pkl"

# 全局只加载一次模型与特征库，避免每次检索重复加载
extractor = FeatureExtractor()


def load_database(path: str):
    """读取离线构建好的特征库。"""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        database = pickle.load(f)
    database["features"] = np.asarray(database["features"])
    return database


database = load_database(DB_PATH)


def search(query_image, top_k: int):
    """根据上传图片检索最相似的 Top-K 张图片。"""
    if query_image is None:
        return []
    if database is None:
        raise gr.Error("尚未找到特征库 features.pkl，请先运行：python build_index.py")

    # 1) 查询图提特征
    query_vec = extractor.extract(query_image)

    # 2) 与图库所有特征算余弦相似度（向量已归一化，点积即可）
    scores = database["features"] @ query_vec

    # 3) 按相似度从高到低排序，取前 K 个
    top_k = max(1, min(int(top_k), len(database["paths"])))
    top_index = np.argsort(scores)[::-1][:top_k]

    return [
        (database["paths"][i], f"相似度 {scores[i]:.3f}")
        for i in top_index
    ]


with gr.Blocks(title="AI 以图搜图系统") as demo:
    gr.Markdown(
        "# 基于深度学习的以图搜图系统\n"
        "上传一张图片，系统会从图库中检索出视觉上最相似的 Top-K 图片。"
    )
    with gr.Row():
        with gr.Column(scale=1):
            input_image = gr.Image(type="pil", label="查询图片")
            top_k_slider = gr.Slider(1, 20, value=5, step=1, label="返回数量 Top-K")
            search_button = gr.Button("开始检索", variant="primary")
        with gr.Column(scale=2):
            result_gallery = gr.Gallery(label="检索结果", columns=5, height="auto")

    search_button.click(
        search,
        inputs=[input_image, top_k_slider],
        outputs=result_gallery,
    )


if __name__ == "__main__":
    if database is None:
        print("[警告] 未找到 features.pkl，网页可打开但无法检索，请先运行 build_index.py")
    demo.launch()
