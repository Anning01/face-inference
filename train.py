# multitask 版本 - RTX 4090 优化版 v2

import os
# CPU 核心优化：防止 OpenMP 线程冲突 (针对 i9-13900K P/E 核心架构)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.amp.grad_scaler import GradScaler
from torch.amp.autocast_mode import autocast
from torchvision import models, transforms
import pandas as pd
from PIL import Image
from tqdm import tqdm

# --- 1. 配置与常量 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# RTX 4090 优化配置
BATCH_SIZE = 256
LEARNING_RATE = 0.0001 * (256 / 64)  # 线性缩放学习率
EPOCHS = 20
NUM_WORKERS = 16
SAVE_PATH = 'checkpoints/resnet50_face_multitask.pth'

# CUDA 优化
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# 标签映射
RACE_LIST = ['White', 'Black', 'Latino', 'Asian', 'Southeast Asian', 'Indian', 'Middle Eastern']
GENDER_LIST = ['Male', 'Female']
AGE_LIST = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

# 任务损失权重 (Task Loss Weighting)
# Gender 任务简单(2类)且数据不平衡，降低权重防止过快收敛
# Race 和 Age 任务更复杂，保持较高权重
TASK_WEIGHTS = {
    'race': 1.0,
    'gender': 0.5,  # 降低权重，因为二分类太容易收敛
    'age': 1.0
}

# 类别权重 (Class Weighting) - 基于数据分布的逆频率
# 数据分布:
# Race: White(39152), Black(5513), Latino(7525), Asian(4011), SE Asian(1977), Indian(2531), ME(6865)
# Gender: Male(53402), Female(14172)
# Age: 0-2(473), 3-9(1322), 10-19(1886), 20-29(16718), 30-39(17603), 40-49(12999), 50-59(10733), 60-69(5217), 70+(623)

def compute_class_weights(counts, smoothing=0.1):
    """计算类别权重，使用平滑的逆频率"""
    total = sum(counts)
    weights = []
    for c in counts:
        # 使用 sqrt 平滑，避免极端权重
        w = (total / (len(counts) * c)) ** smoothing
        weights.append(w)
    # 归一化使平均权重为 1
    avg = sum(weights) / len(weights)
    return [w / avg for w in weights]

# 根据实际数据分布计算
RACE_COUNTS = [39152, 5513, 7525, 4011, 1977, 2531, 6865]  # 对应 RACE_LIST 顺序
GENDER_COUNTS = [53402, 14172]  # Male, Female
AGE_COUNTS = [473, 1322, 1886, 16718, 17603, 12999, 10733, 5217, 623]  # 对应 AGE_LIST 顺序

# --- 2. 自定义 Dataset ---
class FaceMultiTaskDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

        self.race_to_idx = {name: i for i, name in enumerate(RACE_LIST)}
        self.gender_to_idx = {name: i for i, name in enumerate(GENDER_LIST)}
        self.age_to_idx = {name: i for i, name in enumerate(AGE_LIST)}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        img_path = os.path.join(self.img_dir, os.path.basename(row['cropped_image_path']))

        image = Image.open(img_path).convert('RGB')

        labels = {
            'race': torch.tensor(self.race_to_idx[row['race']], dtype=torch.long),
            'gender': torch.tensor(self.gender_to_idx[row['gender']], dtype=torch.long),
            'age': torch.tensor(self.age_to_idx[row['age']], dtype=torch.long)
        }

        if self.transform:
            image = self.transform(image)

        return image, labels

# --- 3. 多任务 ResNet50 模型 ---
class MultiTaskResNet50(nn.Module):
    def __init__(self):
        super(MultiTaskResNet50, self).__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.race_head = nn.Linear(num_ftrs, len(RACE_LIST))
        self.gender_head = nn.Linear(num_ftrs, len(GENDER_LIST))
        self.age_head = nn.Linear(num_ftrs, len(AGE_LIST))

    def forward(self, x):
        features = self.backbone(x)
        return {
            'race': self.race_head(features),
            'gender': self.gender_head(features),
            'age': self.age_head(features)
        }

# --- 4. 训练逻辑 ---
def train_model():
    # 增强的数据增强策略
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  # 轻微旋转，人脸通常是正脸所以不要太大
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1)),  # 随机遮挡，提升鲁棒性
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载数据集
    train_dataset = FaceMultiTaskDataset('face_data/train/annotations.csv', 'cropped_faces', train_transform)
    val_dataset = FaceMultiTaskDataset('face_data/val/annotations.csv', 'cropped_faces', val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE * 2,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )

    # 初始化模型
    model = MultiTaskResNet50().to(DEVICE)

    # 编译模型 (PyTorch 2.0+)
    use_compile = hasattr(torch, 'compile') and torch.cuda.is_available()
    if use_compile:
        model = torch.compile(model, mode='reduce-overhead')
        print("✓ Model compiled with torch.compile()")

    # 计算类别权重并创建带权重的损失函数
    race_weights = torch.tensor(compute_class_weights(RACE_COUNTS, smoothing=0.3), dtype=torch.float32).to(DEVICE)
    gender_weights = torch.tensor(compute_class_weights(GENDER_COUNTS, smoothing=0.5), dtype=torch.float32).to(DEVICE)
    age_weights = torch.tensor(compute_class_weights(AGE_COUNTS, smoothing=0.3), dtype=torch.float32).to(DEVICE)

    criterion_race = nn.CrossEntropyLoss(weight=race_weights)
    criterion_gender = nn.CrossEntropyLoss(weight=gender_weights)
    criterion_age = nn.CrossEntropyLoss(weight=age_weights)

    print(f"Class weights:")
    print(f"  Race: {[f'{w:.2f}' for w in race_weights.tolist()]}")
    print(f"  Gender: {[f'{w:.2f}' for w in gender_weights.tolist()]}")
    print(f"  Age: {[f'{w:.2f}' for w in age_weights.tolist()]}")

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        epochs=EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy='cos'
    )

    scaler = GradScaler()
    best_val_loss = float('inf')

    print(f"\nTraining on {DEVICE}")
    print(f"Batch size: {BATCH_SIZE}, Num workers: {NUM_WORKERS}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    print(f"Task weights: {TASK_WEIGHTS}")
    print("-" * 60)

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        running_losses = {'race': 0.0, 'gender': 0.0, 'age': 0.0}

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for images, labels in pbar:
            images = images.to(DEVICE, non_blocking=True)
            labels = {k: v.to(DEVICE, non_blocking=True) for k, v in labels.items()}

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type='cuda'):
                outputs = model(images)

                # 带类别权重的损失
                loss_race = criterion_race(outputs['race'], labels['race'])
                loss_gender = criterion_gender(outputs['gender'], labels['gender'])
                loss_age = criterion_age(outputs['age'], labels['age'])

                # 带任务权重的总损失
                total_loss = (
                    TASK_WEIGHTS['race'] * loss_race +
                    TASK_WEIGHTS['gender'] * loss_gender +
                    TASK_WEIGHTS['age'] * loss_age
                )

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running_loss += total_loss.item()
            running_losses['race'] += loss_race.item()
            running_losses['gender'] += loss_gender.item()
            running_losses['age'] += loss_age.item()

            pbar.set_postfix({
                'loss': f"{total_loss.item():.3f}",
                'R': f"{loss_race.item():.2f}",
                'G': f"{loss_gender.item():.2f}",
                'A': f"{loss_age.item():.2f}",
            })

        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_losses = {'race': 0.0, 'gender': 0.0, 'age': 0.0}
        corrects = {'race': 0, 'gender': 0, 'age': 0}
        total = 0

        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="Validating", leave=False):
                images = images.to(DEVICE, non_blocking=True)
                labels = {k: v.to(DEVICE, non_blocking=True) for k, v in labels.items()}

                with autocast(device_type='cuda'):
                    outputs = model(images)

                    loss_race = criterion_race(outputs['race'], labels['race'])
                    loss_gender = criterion_gender(outputs['gender'], labels['gender'])
                    loss_age = criterion_age(outputs['age'], labels['age'])

                    batch_loss = (
                        TASK_WEIGHTS['race'] * loss_race +
                        TASK_WEIGHTS['gender'] * loss_gender +
                        TASK_WEIGHTS['age'] * loss_age
                    )
                    val_loss += batch_loss.item()
                    val_losses['race'] += loss_race.item()
                    val_losses['gender'] += loss_gender.item()
                    val_losses['age'] += loss_age.item()

                for task in ['race', 'gender', 'age']:
                    _, preds = torch.max(outputs[task], 1)
                    corrects[task] += torch.sum(preds == labels[task].data)
                total += images.size(0)

        # 计算平均值
        n_train = len(train_loader)
        n_val = len(val_loader)
        avg_train_loss = running_loss / n_train
        avg_val_loss = val_loss / n_val

        print(f"\nEpoch {epoch+1}/{EPOCHS} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  Train (R/G/A): {running_losses['race']/n_train:.3f} / {running_losses['gender']/n_train:.3f} / {running_losses['age']/n_train:.3f}")
        print(f"  Val   (R/G/A): {val_losses['race']/n_val:.3f} / {val_losses['gender']/n_val:.3f} / {val_losses['age']/n_val:.3f}")

        for task in ['race', 'gender', 'age']:
            acc = corrects[task].double() / total
            print(f"  {task.capitalize()} Acc: {acc:.4f}")

        # 保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs('checkpoints', exist_ok=True)

            # 提取原始模型（去掉 compile 包装），确保兼容性
            raw_model = model._orig_mod if hasattr(model, '_orig_mod') else model

            torch.save({
                'epoch': epoch,
                'model_state_dict': raw_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'class_weights': {
                    'race': race_weights.cpu(),
                    'gender': gender_weights.cpu(),
                    'age': age_weights.cpu(),
                },
                'task_weights': TASK_WEIGHTS,
            }, SAVE_PATH)
            print(f"  ⭐ Best model saved to {SAVE_PATH}")
        print("-" * 60)

if __name__ == '__main__':
    train_model()
