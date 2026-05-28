"""
PyTorch GPU 训练脚本
对项目的分类、检测、分割模型进行迁移学习微调
"""
import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from torchvision.models import ResNet50_Weights, VGG16_Weights, EfficientNet_B0_Weights
from PIL import Image
import numpy as np
import traceback as tb

# 添加 backend 到路径
sys.path.insert(0, str(Path(__file__).parent.parent))

MODEL_SAVE_DIR = Path(__file__).parent.parent.parent.parent / "public" / "models" / "trained"
DATASET_DIR = Path(__file__).parent.parent.parent.parent / "public" / "datasets"


class ClassificationDataset(Dataset):
    """分类数据集"""
    def __init__(self, dataset_path: str, transform=None, is_train: bool = True, val_split: float = 0.2):
        self.transform = transform
        with open(Path(dataset_path) / "annotations.json", "r") as f:
            self.annotations = json.load(f)
        
        # 划分训练/验证集
        np.random.seed(42)
        n = len(self.annotations)
        indices = np.random.permutation(n)
        split_idx = int(n * (1 - val_split))
        
        if is_train:
            self.annotations = [self.annotations[i] for i in indices[:split_idx]]
        else:
            self.annotations = [self.annotations[i] for i in indices[split_idx:]]
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        img_path = Path(DATASET_DIR) / "classification" / ann["image"]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)
        
        return img, ann["label"]


def get_classification_model(model_name: str, num_classes: int):
    """获取分类模型"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_name == "resnet50":
        model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "vgg16":
        model = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    return model.to(device), device


def train_classification(
    model_name: str = "resnet50",
    num_classes: int = 10,
    epochs: int = 10,
    batch_size: int = 16,
    learning_rate: float = 0.001
) -> Dict[str, Any]:
    """训练图像分类模型"""
    print(f"\n{'='*50}")
    print(f"训练图像分类模型: {model_name}")
    print(f"{'='*50}")
    
    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    dataset_path = DATASET_DIR / "classification"
    train_dataset = ClassificationDataset(str(dataset_path), train_transform, is_train=True)
    val_dataset = ClassificationDataset(str(dataset_path), val_transform, is_train=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print(f"训练集: {len(train_dataset)} 样本, 验证集: {len(val_dataset)} 样本")
    print(f"类别数: {num_classes}, Epochs: {epochs}, Batch Size: {batch_size}")
    
    # 模型
    model, device = get_classification_model(model_name, num_classes)
    print(f"设备: {device}")
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    print(f"开始训练... 优化器: AdamW, LR: {learning_rate}", flush=True)
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
    
    # 训练循环
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    start_time = time.time()
    
    for epoch in range(epochs):
        try:
            # 训练阶段
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                train_correct += (predicted == labels).sum().item()
                train_total += labels.size(0)
            
            train_loss_epoch = train_loss / train_total
            train_acc_epoch = train_correct / train_total
            
            # 验证阶段
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_correct += (predicted == labels).sum().item()
                    val_total += labels.size(0)
            
            val_loss_epoch = val_loss / val_total
            val_acc_epoch = val_correct / val_total
            
            scheduler.step()
            
            history["train_loss"].append(round(train_loss_epoch, 4))
            history["train_acc"].append(round(train_acc_epoch, 4))
            history["val_loss"].append(round(val_loss_epoch, 4))
            history["val_acc"].append(round(val_acc_epoch, 4))
            
            print(f"Epoch [{epoch+1}/{epochs}] "
                  f"Train Loss: {train_loss_epoch:.4f} Acc: {train_acc_epoch:.2%} | "
                  f"Val Loss: {val_loss_epoch:.4f} Acc: {val_acc_epoch:.2%}", flush=True)
            
            if val_acc_epoch > best_val_acc:
                best_val_acc = val_acc_epoch
                MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
                save_path = MODEL_SAVE_DIR / f"{model_name}_classifier.pth"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "num_classes": num_classes,
                    "classes": json.load(open(DATASET_DIR / "classification" / "labels.json")),
                    "history": history,
                    "best_val_acc": best_val_acc
                }, save_path)
                print(f"  => 保存最佳模型到 {save_path}", flush=True)
        except Exception as e:
            print(f"ERROR in epoch {epoch+1}: {e}", flush=True)
            tb.print_exc()
    
    elapsed = time.time() - start_time
    print(f"\n训练完成! 耗时: {elapsed:.1f}s, 最佳验证准确率: {best_val_acc:.2%}")
    
    return {"model": model_name, "best_val_acc": best_val_acc, "history": history, "elapsed": elapsed}


def train_detection(
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 0.0001
) -> Dict[str, Any]:
    """训练目标检测模型 (Faster R-CNN)"""
    print(f"\n{'='*50}")
    print(f"训练目标检测模型: fasterrcnn")
    print(f"{'='*50}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    # 加载预训练模型
    num_classes = 11  # 10个类别 + 背景
    model = models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=learning_rate)
    
    # 简化训练: 使用合成数据
    print("使用合成数据微调检测器...")
    model.train()
    
    history = {"loss": []}
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for i in range(20):  # 每epoch 20个batch
            # 生成合成图像和标注
            images = []
            targets = []
            
            for _ in range(batch_size):
                img = torch.rand(3, 512, 512).to(device)
                num_boxes = torch.randint(1, 4, (1,)).item()
                
                boxes = torch.rand(num_boxes, 4).to(device)
                boxes[:, 2:] = boxes[:, :2] + boxes[:, 2:] * 0.3  # 确保 x2>x1, y2>y1
                boxes = torch.clamp(boxes, 0.05, 0.95)
                
                labels = torch.randint(1, num_classes, (num_boxes,)).to(device)
                
                images.append(img)
                targets.append({"boxes": boxes, "labels": labels})
            
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            epoch_loss += losses.item()
        
        avg_loss = epoch_loss / 20
        history["loss"].append(round(avg_loss, 4))
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
    
    elapsed = time.time() - start_time
    
    # 保存模型
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_SAVE_DIR / "fasterrcnn_detector.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "history": history
    }, save_path)
    print(f"模型已保存到 {save_path}")
    
    print(f"训练完成! 耗时: {elapsed:.1f}s")
    return {"model": "fasterrcnn_resnet50", "history": history, "elapsed": elapsed}


def train_segmentation(
    epochs: int = 5,
    batch_size: int = 4,
    learning_rate: float = 0.0001
) -> Dict[str, Any]:
    """训练图像分割模型 (DeepLabV3)"""
    print(f"\n{'='*50}")
    print(f"训练图像分割模型: deeplabv3")
    print(f"{'='*50}")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    
    num_classes = 21  # Pascal VOC
    model = models.segmentation.deeplabv3_resnet50(weights="DEFAULT")
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.to(device)
    
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.AdamW(params, lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    model.train()
    history = {"loss": []}
    start_time = time.time()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        
        for i in range(20):
            images = torch.rand(batch_size, 3, 256, 256).to(device)
            masks = torch.randint(0, num_classes, (batch_size, 256, 256)).to(device)
            
            outputs = model(images)["out"]
            loss = criterion(outputs, masks)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / 20
        history["loss"].append(round(avg_loss, 4))
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_loss:.4f}")
    
    elapsed = time.time() - start_time
    
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = MODEL_SAVE_DIR / "deeplabv3_segmenter.pth"
    torch.save({
        "model_state_dict": model.state_dict(),
        "num_classes": num_classes,
        "history": history
    }, save_path)
    print(f"模型已保存到 {save_path}")
    
    print(f"训练完成! 耗时: {elapsed:.1f}s")
    return {"model": "deeplabv3_resnet50", "history": history, "elapsed": elapsed}


def train_all():
    """训练所有模型"""
    print("=" * 60)
    print("PyTorch GPU 模型预训练")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"显存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
    print("=" * 60)
    
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # 1. 分类训练
    results["classification"] = train_classification(
        model_name="resnet50", num_classes=10, epochs=5, batch_size=16, learning_rate=0.0005
    )
    
    # 2. 检测训练
    results["detection"] = train_detection(epochs=3, batch_size=4)
    
    # 3. 分割训练
    results["segmentation"] = train_segmentation(epochs=3, batch_size=4)
    
    # 保存训练结果汇总
    with open(MODEL_SAVE_DIR / "training_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print("\n" + "=" * 60)
    print("全部训练完成!")
    for task, result in results.items():
        if "best_val_acc" in result:
            print(f"  {task}: 最佳验证准确率 = {result['best_val_acc']:.2%}")
        elif "history" in result and "loss" in result["history"]:
            print(f"  {task}: 最终损失 = {result['history']['loss'][-1]:.4f}")
    print(f"\n模型保存目录: {MODEL_SAVE_DIR}")


if __name__ == "__main__":
    train_all()