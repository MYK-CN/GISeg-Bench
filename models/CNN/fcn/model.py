import torch
from torchvision import models

# 1. 下载并加载 FCN 模型 (使用 ResNet50 作为主干，最经典且显存占用适中)
# weights='DEFAULT' 会自动下载在 COCO 数据集上训练好的权重
fcn_model = models.segmentation.fcn_resnet50(weights='DEFAULT')

# 2. 【关键修改】医学图像通常只有 2 类 (背景 + 病灶)
# 原模型是针对 COCO 数据集的 (21类)，必须修改最后输出层
fcn_model.classifier[4] = torch.nn.Conv2d(512, 2, kernel_size=(1, 1), stride=(1, 1))

# 3. 如果你的显存很小 (比如 < 4GB)，可以在辅助分类头也做修改或直接禁用
fcn_model.aux_classifier = None  # 简单粗暴，省显存

print("FCN 模型已加载并修改完毕，准备训练！")
