# 基于深度学习的以图搜图系统（Content-Based Image Retrieval）

> 上传一张图片，即可从本地图库中快速检索出视觉内容最相似的若干图片。
> 系统采用预训练卷积神经网络提取图像语义特征，结合余弦相似度完成 Top-K 检索，并提供可视化网页界面。

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![Gradio](https://img.shields.io/badge/Gradio-Demo-orange)

---

## 一、项目简介

传统的图片检索依赖人工标注的文件名或标签，既费时又难以表达"视觉相似"。
本项目实现了一个**以图搜图（CBIR）**原型系统：无需人工标注、无需自行训练模型，
借助在 ImageNet 上预训练好的 ResNet50 提取图像的高维语义特征，
通过**余弦相似度**对图库进行匹配排序，并使用 Gradio 搭建可交互的网页演示。

## 二、实现原理

整体分为 **离线建库** 与 **在线检索** 两个阶段：

```
离线阶段：图库图片 ──ResNet50 特征提取──> 2048维特征向量 ──L2归一化──> 特征库 features.pkl
                                                                          │
在线阶段：查询图片 ──ResNet50 特征提取──> 查询向量 ──余弦相似度比对──> 相似度排序 ──> Top-K 结果
```

1. **特征提取（迁移学习）**：去掉 ResNet50 最后的 1000 类分类层，取倒数第二层的 2048 维输出作为图像的"语义指纹"。浅层网络捕捉颜色、纹理，深层网络捕捉物体与语义，因此该向量具备良好的泛化表达能力。
2. **向量归一化**：对特征做 L2 归一化后，两向量的点积结果即等价于**余弦相似度**（取值 -1~1，越接近 1 越相似）。
3. **相似度检索**：查询向量与图库全部向量做点积，按分数降序排列，取前 K 张返回。

## 三、目录结构

```
image-search/
├── feature_extractor.py        # 特征提取器：加载预训练 ResNet50 并编码图像
├── build_index.py              # 离线建库：批量提取特征并保存为 features.pkl
├── app.py                      # 在线检索：Gradio 网页演示
├── requirements.txt            # 依赖清单
├── images/                     # 图库目录（可按类别建子文件夹）
│   └── .gitkeep
├── scripts/
│   └── prepare_cifar_demo.py   # 可选：下载 CIFAR-10 生成 100 张示例图
└── demo/                       # 演示截图 / 录屏
```

## 四、环境安装

建议使用 Python 3.9 及以上版本。

```bash
# 1. 创建并激活虚拟环境（可选）
python -m venv venv
# Windows:
venv\Scripts\activate
# macOS / Linux:
source venv/bin/activate

# 2. 安装依赖
pip install -r requirements.txt

# 无 NVIDIA 显卡的 Windows 用户，可安装 CPU 版 PyTorch：
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 五、快速开始

```bash
# 步骤 1：准备图库
#   方式 A：把自己的图片放入 images/ 目录（支持子文件夹分类）
#   方式 B：没有图片时，用下面命令自动生成 100 张示例图（首次会下载 CIFAR-10）
python scripts/prepare_cifar_demo.py

# 步骤 2：离线构建特征库（生成 features.pkl）
python build_index.py

# 步骤 3：启动网页演示
python app.py
```

启动后浏览器打开终端提示的本地地址（默认 http://127.0.0.1:7860 ），
上传一张图片、选择返回数量，点击"开始检索"即可看到最相似的 Top-K 结果及其相似度分数。

## 六、技术要点

- **迁移学习 / 预训练模型**：直接复用 ResNet50 在大规模数据集上学到的通用视觉特征，免去从零训练，小成本获得强表达能力。
- **特征向量化与归一化**：将非结构化图像转化为可计算的定长向量，L2 归一化后以点积高效计算余弦相似度。
- **两阶段检索架构**：离线批量建库 + 在线毫秒级查询，避免每次请求重复计算。
- **模块化设计**：特征提取、建库、界面三层解耦，便于替换骨干网络或扩展检索后端。
- **可视化交互**：基于 Gradio 快速搭建网页 Demo，支持图片上传、Top-K 调节与结果画廊展示。

## 七、可扩展方向

- **更大规模检索**：引入 FAISS 向量索引 / 近似最近邻（ANN），将暴力遍历优化为索引检索，支撑百万级图库。
- **跨模态搜索**：将骨干网络替换为 CLIP，实现"文字搜图 + 以图搜图"的多模态检索。
- **特征降维与可视化**：使用 PCA / t-SNE 对特征降维并绘制聚类散点图，直观展示特征分布。
- **检索精度评估**：使用 mAP、Recall@K 等指标在公开数据集上量化评测。

## 八、技术栈

Python · PyTorch · torchvision · NumPy · Pillow · Gradio
