import torch

# 加载原始 283MB 的文件
checkpoint = torch.load('checkpoints/resnet50_face_multitask.pth')

# 只提取模型权重
model_weights = checkpoint['model_state_dict']

# 保存为纯权重文件
torch.save(model_weights, 'checkpoints/resnet50_weights_only.pth')

print("模型权重已成功提取并保存为 'checkpoints/resnet50_weights_only.pth'")