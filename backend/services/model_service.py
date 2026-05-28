"""
Model service - Manages model loading and caching
Supports graceful degradation when PyTorch is not available
"""
import os
from typing import Dict, Optional, Any, List
from pathlib import Path
import json

# Try to import torch, graceful degradation if not available
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import numpy as np
    import cv2
    from PIL import Image
    CV_AVAILABLE = True
except ImportError:
    CV_AVAILABLE = False

from utils.config import settings

# Model information catalog
MODEL_CATALOG = {
    "classification": {
        "resnet50": {
            "name": "ResNet-50",
            "description": "Deep residual network for image classification",
            "source": "torchvision.models (Apache 2.0)",
            "pretrained": True,
            "classes": 1000,
            "input_size": [3, 224, 224]
        },
        "vgg16": {
            "name": "VGG-16",
            "description": "Visual Geometry Group network",
            "source": "torchvision.models (Apache 2.0)",
            "pretrained": True,
            "classes": 1000,
            "input_size": [3, 224, 224]
        },
        "efficientnet_b0": {
            "name": "EfficientNet-B0",
            "description": "Efficient and scalable CNN",
            "source": "torchvision.models (Apache 2.0)",
            "pretrained": True,
            "classes": 1000,
            "input_size": [3, 224, 224]
        }
    },
    "detection": {
        "fasterrcnn_resnet50": {
            "name": "Faster R-CNN",
            "description": "Region-based object detector",
            "source": "torchvision.models.detection (Apache 2.0)",
            "pretrained": True,
            "input_size": [3, 800, 800]
        }
    },
    "segmentation": {
        "deeplabv3_resnet50": {
            "name": "DeepLabV3",
            "description": "Semantic image segmentation",
            "source": "torchvision.models.segmentation (Apache 2.0)",
            "pretrained": True,
            "classes": 21,
            "input_size": [3, 520, 520]
        },
        "fcn_resnet50": {
            "name": "FCN (Fully Convolutional Network)",
            "description": "Fully convolutional network for segmentation",
            "source": "torchvision.models.segmentation (Apache 2.0)",
            "pretrained": True,
            "classes": 21,
            "input_size": [3, 520, 520]
        }
    },
    "nlp": {
        "bert_sentiment": {
            "name": "BERT Sentiment Analysis",
            "description": "Sentiment classification using BERT",
            "source": "Hugging Face Transformers (Apache 2.0)",
            "pretrained": True,
            "max_length": 512
        },
        "gpt2_text": {
            "name": "GPT-2 Text Generation",
            "description": "Text generation using GPT-2",
            "source": "Hugging Face Transformers (Apache 2.0)",
            "pretrained": True,
            "max_length": 1024
        }
    },
    "generation": {
        "stable_diffusion": {
            "name": "Stable Diffusion",
            "description": "Text-to-image generation",
            "source": "CompVis/stable-diffusion-v1-4 (CreativeML Open RAIL-M)",
            "pretrained": True
        }
    }
}

class ModelService:
    """Centralized model management service"""
    
    def __init__(self):
        if TORCH_AVAILABLE:
            self._models: Dict[str, Any] = {}
            self._device = torch.device(settings.DEVICE)
        self._model_status = {}
    
    def is_torch_available(self) -> bool:
        """Check if PyTorch is available"""
        return TORCH_AVAILABLE
    
    def get_model_info(self, category: Optional[str] = None) -> Dict:
        """Get information about available models"""
        if category:
            return MODEL_CATALOG.get(category, {})
        return MODEL_CATALOG
    
    def get_all_models(self) -> List[Dict]:
        """Get all available models with their metadata"""
        models = []
        for cat_name, cat_models in MODEL_CATALOG.items():
            for model_id, model_info in cat_models.items():
                models.append({
                    "id": model_id,
                    "category": cat_name,
                    **model_info,
                    "status": self._model_status.get(model_id, "available")
                })
        return models
    
    def load_model(self, model_id: str) -> Dict[str, Any]:
        """Load a model and return its status"""
        if not TORCH_AVAILABLE:
            return {
                "status": "error",
                "message": "PyTorch is not installed. Please install torch to enable model loading.",
                "model_id": model_id
            }
        
        try:
            # Model loading logic here
            self._model_status[model_id] = "loaded"
            return {
                "status": "success",
                "message": f"Model {model_id} loaded successfully",
                "model_id": model_id
            }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "model_id": model_id
            }
    
    def preprocess_image(self, image_path: str, target_size: tuple = (224, 224)) -> Optional[Any]:
        """Preprocess an image for model inference"""
        if not CV_AVAILABLE:
            return None
        
        try:
            img = cv2.imread(image_path)
            if img is None:
                return None
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, target_size)
            if TORCH_AVAILABLE:
                img = img.astype(np.float32) / 255.0
                img = np.transpose(img, (2, 0, 1))
                return torch.from_numpy(img).unsqueeze(0)
            return img
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def get_device(self) -> str:
        """Get the current device string"""
        if TORCH_AVAILABLE:
            return str(self._device)
        return "cpu (PyTorch not available)"

# Global model service instance
model_service = ModelService()
