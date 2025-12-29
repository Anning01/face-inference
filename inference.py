import os

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import models, transforms

from face_processor import FaceProcessor  # 复用你的检测工具

# --- 1. 配置与常量 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = 'checkpoints/resnet50_face_multitask.pth' # 建议使用第7轮保存的最佳模型

RACE_LIST = ['白人', '黑人', '拉丁裔', '亚裔', '东南亚裔', '印度裔', '中东裔']
GENDER_LIST = ['男性', '女性']
AGE_LIST = ['0-2岁', '3-9岁', '10-19岁', '20-29岁', '30-39岁', '40-49岁', '50-59岁', '60-69岁', '70岁以上']

# --- 2. 定义模型结构 (必须与训练时一致) ---
class MultiTaskResNet(nn.Module):
    def __init__(self):
        super(MultiTaskResNet, self).__init__()
        self.backbone = models.resnet50(weights=None)
        num_ftrs = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.fc_race = nn.Linear(num_ftrs, len(RACE_LIST))
        self.fc_gender = nn.Linear(num_ftrs, len(GENDER_LIST))
        self.fc_age = nn.Linear(num_ftrs, len(AGE_LIST))

    def forward(self, x):
        features = self.backbone(x)
        return {
            'race': self.fc_race(features),
            'gender': self.fc_gender(features),
            'age': self.fc_age(features)
        }

# --- 3. 推理类 ---
class FaceInference:
    def __init__(self, model_path):
        self.processor = FaceProcessor(det_size=(640, 640)) # 提高检测分辨率
        self.model = MultiTaskResNet().to(DEVICE)
        
        # 加载权重
        print(f"正在加载模型: {model_path}...")
        checkpoint = torch.load(model_path, map_location=DEVICE)
        # 兼容处理：如果保存时带了 _orig_mod (torch.compile)
        state_dict = checkpoint['model_state_dict']
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def predict(self, image_path, output_dir="results"):
        img = Image.open(image_path).convert('RGB')
        
        # 1. 人脸检测与裁剪
        # 这里直接调用 FaceProcessor 的逻辑获取裁剪后的面部
        # 为了演示，我们假设返回检测到的所有脸
        faces = self.processor.app.get(self.processor.img_to_np(img))
        
        if not faces:
            print(f"未在 {image_path} 中检测到人脸")
            return

        draw = ImageDraw.Draw(img)
        # 尝试加载中文字体（Ubuntu通常在以下路径，如果没有请更换）
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 20)
        except:
            font = ImageFont.load_default()

        for i, face in enumerate(faces):
            # 获取人脸框并裁剪
            bbox = face.bbox.astype(int)
            # 使用你的标准化裁剪逻辑
            face_img = self.processor.crop_and_standardize(img, bbox, target_size=(224, 224))
            
            # 2. 推理
            input_tensor = self.transform(face_img).unsqueeze(0).to(DEVICE)
            with torch.amp.autocast(device_type='cuda'): # 开启 4090 的混合精度推理
                outputs = self.model(input_tensor)
            
            # 解析结果
            res = {}
            for task in ['race', 'gender', 'age']:
                probs = torch.softmax(outputs[task], dim=1)
                conf, pred = torch.max(probs, 1)
                res[task] = (pred.item(), conf.item())

            # 3. 打印与绘制
            race_text = RACE_LIST[res['race'][0]]
            gender_text = GENDER_LIST[res['gender'][0]]
            age_text = AGE_LIST[res['age'][0]]
            
            label = f"{gender_text}, {age_text}, {race_text}"
            print(f"人脸 {i+1}: {label} (置信度: {res['gender'][1]:.2f})")

            # 在原图画框
            draw.rectangle([bbox[0], bbox[1], bbox[2], bbox[3]], outline="red", width=3)
            draw.text((bbox[0], bbox[1] - 30), label, fill="red", font=font)

        # 保存结果
        os.makedirs(output_dir, exist_ok=True)
        save_path = os.path.join(output_dir, os.path.basename(image_path))
        img.save(save_path)
        print(f"结果已保存至: {save_path}")

    def img_to_np(self, pil_img):
        return nn.array(pil_img)[:, :, ::-1] # RGB to BGR for InsightFace

if __name__ == "__main__":
    infer = FaceInference(MODEL_PATH)
    
    # 示例：处理一张图
    # test_image = "test.jpg" 
    # infer.predict(test_image)
    
    # 示例：处理一个文件夹
    test_folder = "test_images"
    if os.path.exists(test_folder):
        for f in os.listdir(test_folder):
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                infer.predict(os.path.join(test_folder, f))