# Face Inference

基于 ResNet50 的多任务人脸属性预测模型，可同时预测性别、年龄段和人种。

## 模型能力

| 任务 | 类别数 | 标签 |
|------|--------|------|
| **性别** | 2 | Male, Female |
| **年龄** | 9 | 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+ |
| **人种** | 7 | White, Black, Latino, Asian, Southeast Asian, Indian, Middle Eastern |

## 快速开始

### 1. 安装依赖

```bash
# 使用 uv (推荐)
uv sync

# 或使用 pip
pip install torch torchvision pandas pillow tqdm insightface
```

> **重要**: 如果使用 GPU 进行人脸检测，必须安装 `onnxruntime-gpu` 而不是 `onnxruntime`，否则性能会严重下降！

```bash
# 正确安装方式 (GPU)
pip uninstall onnxruntime onnxruntime-gpu  # 先卸载已有版本
pip install onnxruntime-gpu

# 验证 CUDA 是否生效
python -c "import onnxruntime; print(onnxruntime.get_available_providers())"
# 应该输出: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
```

### 2. 运行推理

```bash
# 将测试图片放入 example/ 目录
python inference.py
```

推理结果将保存到 `results/` 目录，包含标注后的图片。

### 示例输出

```
人脸 1: 女性, 30-39岁, 白人 (置信度: G=0.98, A=0.63, R=0.99)
人脸 1: 男性, 40-49岁, 黑人 (置信度: G=1.00, A=0.54, R=0.95)
人脸 1: 女性, 0-2岁, 白人 (置信度: G=0.99, A=0.99, R=0.95)
```

## 训练数据

使用 87,648 张头像图片训练，标签分布如下：

<details>
<summary>点击展开数据分布详情</summary>

**性别分布**
| 类别 | 数量 | 占比 |
|------|------|------|
| Male | 53,402 | 79.0% |
| Female | 14,172 | 21.0% |

**人种分布**
| 类别 | 数量 | 占比 |
|------|------|------|
| White | 39,152 | 58.0% |
| Latino | 7,525 | 11.1% |
| Middle Eastern | 6,865 | 10.2% |
| Black | 5,513 | 8.2% |
| Asian | 4,011 | 5.9% |
| Indian | 2,531 | 3.7% |
| Southeast Asian | 1,977 | 2.9% |

**年龄分布**
| 类别 | 数量 | 占比 |
|------|------|------|
| 30-39 | 17,603 | 26.1% |
| 20-29 | 16,718 | 24.8% |
| 40-49 | 12,999 | 19.3% |
| 50-59 | 10,733 | 15.9% |
| 60-69 | 5,217 | 7.7% |
| 10-19 | 1,886 | 2.8% |
| 3-9 | 1,322 | 2.0% |
| 70+ | 623 | 0.9% |
| 0-2 | 473 | 0.7% |

</details>

## 数据处理

数据处理代码位于: [huihuaAI/face-inference](https://github.com/huihuaAI/face-inference)

### 处理流程

```
原始图片下载 → 人脸检测 → 质量验证 → 裁剪对齐 → 标注 → 训练/验证集划分
```

### 处理脚本说明

| 脚本 | 功能 |
|------|------|
| `downloaded_avatar.py` | 下载头像图片到 `avatars/` |
| `downloaded_csvs.py` | 下载 CSV 元数据到 `csvs/` |
| `face_detection.py` | 基于 InsightFace 的人脸检测 |
| `face_liveness.py` | 活体检测，过滤 AI 生成的人脸 |
| `face_processor.py` | 人脸裁剪和归一化处理 |
| `extract_train_data.py` | 从 CSV 中提取有效标注 |
| `prepare_face_dataset.py` | 按 80/20 比例划分训练/验证集 |
| `generate_annotations.py` | 自动属性标注 |
| `main.py` | FastAPI 手动标注服务 |

## 训练

```bash
python train.py
```

### 训练配置

| 参数 | 值 |
|------|------|
| Batch Size | 256 |
| Epochs | 20 |
| Optimizer | AdamW (lr=0.0004, weight_decay=0.01) |
| Scheduler | OneCycleLR |
| 混合精度 | AMP (FP16) |

### 训练优化

- **任务损失权重平衡**: Gender 权重 0.5，Race/Age 权重 1.0
- **类别权重**: 使用平滑逆频率处理数据不平衡
- **torch.compile**: PyTorch 2.0+ 图优化加速
- **TF32 + cuDNN benchmark**: CUDA 加速

## 推理 API

```python
from inference import FaceInference

# 初始化
infer = FaceInference('checkpoints/resnet50_face_multitask.pth')

# 预测单张图片
infer.predict('example/test.jpg', output_dir='results')
```

### 输出说明

- 检测到的人脸会用红框标注
- 每个人脸上方显示：`性别, 年龄段, 人种`
- 置信度：G=性别, A=年龄, R=人种

## 项目结构

```
face-inference/
├── train.py              # 训练脚本
├── inference.py          # 推理脚本
├── face_processor.py     # 人脸检测和裁剪工具
├── example/              # 示例图片目录
├── results/              # 推理结果输出目录
├── checkpoints/          # 模型权重目录
└── face_data/            # 训练数据目录
    ├── train/
    │   ├── images/
    │   └── annotations.csv
    └── val/
        ├── images/
        └── annotations.csv
```

## 训练环境

| 配置 | 规格 |
|------|------|
| OS | Ubuntu 24.04.2 LTS |
| CPU | Intel Core i9-13900K (32 threads) |
| GPU | NVIDIA GeForce RTX 4090 |
| Driver | 580.95.05 |
| CUDA | 13.0 |

## License

See [LICENSE](LICENSE) file for details.
