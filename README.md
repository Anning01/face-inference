# Face Inference

基于 ResNet50 的多任务人脸属性预测模型，可同时预测性别、年龄段和人种。

## 模型能力

| 任务 | 类别数 | 标签 |
|------|--------|------|
| **性别** | 2 | Male, Female |
| **年龄** | 9 | 0-2, 3-9, 10-19, 20-29, 30-39, 40-49, 50-59, 60-69, 70+ |
| **人种** | 7 | White, Black, Latino, Asian, Southeast Asian, Indian, Middle Eastern |

## 训练数据

使用 87,648 张头像图片训练，标签分布如下：

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

## 安装

```bash
# 使用 uv (推荐)
uv sync

# 或使用 pip
pip install torch torchvision pandas pillow tqdm
```

## 训练

```bash
python train.py
```

### 训练配置

- **Batch Size**: 256
- **Epochs**: 20
- **Optimizer**: AdamW (lr=0.0004, weight_decay=0.01)
- **Scheduler**: OneCycleLR
- **混合精度**: 启用 (AMP)

### 训练优化

- 任务损失权重平衡 (Gender 权重降低)
- 类别权重处理数据不平衡
- `torch.compile` 加速
- TF32 和 cuDNN benchmark

## 推理

```python
import torch
from torchvision import transforms
from PIL import Image

# 加载模型
model = MultiTaskResNet50()
checkpoint = torch.load('checkpoints/resnet50_face_multitask.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# 预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# 推理
image = Image.open('face.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)

with torch.no_grad():
    outputs = model(input_tensor)
    race_pred = RACE_LIST[outputs['race'].argmax()]
    gender_pred = GENDER_LIST[outputs['gender'].argmax()]
    age_pred = AGE_LIST[outputs['age'].argmax()]

print(f"Race: {race_pred}, Gender: {gender_pred}, Age: {age_pred}")
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

click view [LICENSE](LICENSE) file

