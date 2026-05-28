"""
真实模型推理服务 - 使用 GPU 训练后模型进行实际预测
替换各路由中的 mock 预测
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from torchvision.models import (
    ResNet50_Weights, 
    detection as detection_models,
    segmentation as segmentation_models
)

BASE_DIR = Path(__file__).parent.parent.parent / "public"
MODEL_DIR = BASE_DIR / "models" / "trained"
DATASET_DIR = BASE_DIR / "datasets"
ROUTES_DIR = Path(__file__).parent.parent / "routes"

IMAGENET_CLASSES_PATH = ROUTES_DIR / "imagenet_classes.json"
IMAGENET_CLASSES = json.load(open(IMAGENET_CLASSES_PATH, "r", encoding="utf-8")) if IMAGENET_CLASSES_PATH.exists() else []

_COCO_PATH = ROUTES_DIR / "coco_classes.json"
if _COCO_PATH.exists():
    _coco_dict = json.load(open(_COCO_PATH, "r", encoding="utf-8"))
    COCO_CLASSES_91 = [_coco_dict.get(str(i), f"class_{i}") for i in range(91)]
else:
    COCO_CLASSES_91 = [f"class_{i}" for i in range(91)]

_VOC_PATH = ROUTES_DIR / "voc_classes.json"
if _VOC_PATH.exists():
    _voc_dict = json.load(open(_VOC_PATH, "r", encoding="utf-8"))
    VOC_CLASSES_21 = [_voc_dict.get(str(i), f"class_{i}") for i in range(21)]
else:
    VOC_CLASSES_21 = [f"class_{i}" for i in range(21)]

CUSTOM_CLASSES_CN = {
    "airplane": "飞机 (airplane)",
    "bird": "鸟 (bird)",
    "car": "汽车 (car)",
    "cat": "猫 (cat)",
    "dog": "狗 (dog)",
    "elephant": "大象 (elephant)",
    "flower": "花 (flower)",
    "horse": "马 (horse)",
    "ship": "船 (ship)",
    "tree": "树 (tree)",
}


class InferenceEngine:
    """统一推理引擎"""
    
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models: Dict[str, Any] = {}
        self.class_names: Dict[str, List[str]] = {}
        self._load_models()
    
    def _load_models(self):
        """加载所有训练好的模型"""
        print(f"[InferenceEngine] 初始化, 设备: {self.device}")
        
        # 加载分类模型
        self._load_classification_model()
        
        # 加载检测模型
        self._load_detection_model()
        
        # 加载分割模型
        self._load_segmentation_model()
        
        print(f"[InferenceEngine] 模型加载完成: {list(self.models.keys())}")
    
    def _load_classification_model(self):
        """加载分类模型"""
        model_path = MODEL_DIR / "resnet50_classifier.pth"
        
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            num_classes = checkpoint.get("num_classes", 10)
            
            model = models.resnet50()
            # Detect fc architecture from saved state_dict
            sd = checkpoint["model_state_dict"]
            has_seq = any("fc.1." in k for k in sd.keys())
            if has_seq:
                model.fc = nn.Sequential(
                    nn.Dropout(0.3),
                    nn.Linear(model.fc.in_features, num_classes)
                )
            else:
                model.fc = nn.Linear(model.fc.in_features, num_classes)
            model.load_state_dict(sd)
            model.to(self.device)
            model.eval()
            
            self.models["resnet50"] = model
            classes = checkpoint.get("classes", [])
            if isinstance(classes, list):
                self.class_names["resnet50"] = classes
            else:
                self.class_names["resnet50"] = list(classes.keys())
            print(f"  [OK] 加载 resnet50 分类器 ({num_classes}类)")
        else:
            # 使用预训练模型作为后备
            print("  [Fallback] 使用 ImageNet 预训练 ResNet50")
            model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
            model.to(self.device)
            model.eval()
            self.models["resnet50"] = model
    
    def _load_detection_model(self):
        """加载检测模型"""
        model_path = MODEL_DIR / "fasterrcnn_detector.pth"
        
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            num_classes = checkpoint.get("num_classes", 11)
            
            model = detection_models.fasterrcnn_resnet50_fpn(weights=None)
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = detection_models.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.to(self.device)
            model.eval()
            
            self.models["fasterrcnn"] = model
            print(f"  [OK] 加载 fasterrcnn 检测器 ({num_classes}类)")
        else:
            print("  [Fallback] 使用 COCO 预训练 Faster R-CNN")
            model = detection_models.fasterrcnn_resnet50_fpn(weights="DEFAULT")
            model.to(self.device)
            model.eval()
            self.models["fasterrcnn"] = model
    
    def _load_segmentation_model(self):
        """加载分割模型"""
        model_path = MODEL_DIR / "deeplabv3_segmenter.pth"
        
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            num_classes = checkpoint.get("num_classes", 21)
            
            model = segmentation_models.deeplabv3_resnet50(weights=None, weights_backbone=None)
            model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
            model.load_state_dict(checkpoint["model_state_dict"], strict=False)
            model.to(self.device)
            model.eval()
            
            self.models["deeplabv3"] = model
            print(f"  [OK] 加载 deeplabv3 分割器 ({num_classes}类)")
        else:
            print("  [Fallback] 使用 Pascal VOC 预训练 DeepLabV3")
            model = segmentation_models.deeplabv3_resnet50(weights="DEFAULT")
            model.to(self.device)
            model.eval()
            self.models["deeplabv3"] = model
    
    def classify(self, image: Image.Image, model_name: str = "resnet50", top_k: int = 5) -> List[Dict]:
        """图像分类推理"""
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        
        model = self.models.get(model_name)
        if model is None:
            model = self.models.get("resnet50")
        
        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = F.softmax(outputs, dim=1)
            top_probs, top_indices = torch.topk(probabilities, top_k)
        
        predictions = []
        class_names = self.class_names.get(model_name, [])
        
        for i in range(top_k):
            idx = top_indices[0][i].item()
            prob = top_probs[0][i].item()
            
            if class_names and idx < len(class_names):
                name = class_names[idx]
            elif idx < len(IMAGENET_CLASSES):
                name = IMAGENET_CLASSES[idx]
            else:
                name = f"class_{idx}"
            
            name = CUSTOM_CLASSES_CN.get(name, name)
            
            predictions.append({
                "class_id": idx,
                "class_name": name,
                "confidence": round(prob, 4)
            })
        
        return predictions
    
    def detect(self, image: Image.Image, model_name: str = "fasterrcnn", 
               confidence_threshold: float = 0.5) -> List[Dict]:
        """目标检测推理"""
        transform = transforms.Compose([transforms.ToTensor()])
        img_tensor = transform(image).to(self.device)
        
        model = self.models.get(model_name)
        if model is None:
            model = self.models.get("fasterrcnn")
        
        with torch.no_grad():
            predictions = model([img_tensor])[0]
        
        detections = []
        for box, label, score in zip(
            predictions["boxes"].cpu().numpy(),
            predictions["labels"].cpu().numpy(),
            predictions["scores"].cpu().numpy()
        ):
            if score >= confidence_threshold:
                class_name = COCO_CLASSES_91[label] if label < len(COCO_CLASSES_91) else f"class_{label}"
                detections.append({
                    "class_id": int(label),
                    "class_name": class_name,
                    "bbox": {
                        "x1": float(box[0]), "y1": float(box[1]),
                        "x2": float(box[2]), "y2": float(box[3])
                    },
                    "confidence": round(float(score), 4)
                })
        
        return detections
    
    def segment(self, image: Image.Image, model_name: str = "deeplabv3") -> Dict[str, Any]:
        """图像分割推理"""
        transform = transforms.Compose([
            transforms.Resize(520),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        
        model = self.models.get(model_name)
        if model is None:
            model = self.models.get("deeplabv3")
        
        with torch.no_grad():
            output = model(img_tensor)["out"][0]
            output = F.interpolate(output.unsqueeze(0), size=image.size[::-1], 
                                   mode="bilinear", align_corners=False)[0]
            segmentation = output.argmax(0).cpu().numpy()
        
        segments = []
        unique_classes = np.unique(segmentation)
        for cls_id in unique_classes:
            if cls_id == 0:
                continue
            mask = segmentation == cls_id
            area = int(mask.sum())
            if area < 100:
                continue
            
            ys, xs = np.where(mask)
            name = VOC_CLASSES_21[cls_id] if cls_id < len(VOC_CLASSES_21) else f"class_{cls_id}"
            segments.append({
                "class_id": int(cls_id),
                "class_name": name,
                "mask_bbox": {
                    "x1": int(xs.min()), "y1": int(ys.min()),
                    "x2": int(xs.max()), "y2": int(ys.max())
                },
                "area_pixels": area,
                "confidence": 1.0
            })
        
        return {"segments": segments, "num_segments": len(segments)}
    
    def is_ready(self) -> bool:
        """检查推理引擎是否就绪"""
        return len(self.models) > 0

# 全局推理引擎实例
inference_engine = InferenceEngine()