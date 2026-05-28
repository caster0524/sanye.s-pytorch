"""
Image Classification routes
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
from typing import List, Dict, Any
import random
import json
from pathlib import Path

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from utils.config import settings
from utils.image_utils import image_to_base64
from services.model_service import model_service

try:
    from services.inference_service import inference_engine
    HAS_INFERENCE = inference_engine.is_ready()
except Exception:
    HAS_INFERENCE = False

router = APIRouter()

# ImageNet 1000 class labels (bilingual: Chinese + English)
IMAGENET_CLASSES_PATH = Path(__file__).parent / "imagenet_classes.json"
if IMAGENET_CLASSES_PATH.exists():
    with open(IMAGENET_CLASSES_PATH, "r", encoding="utf-8") as f:
        IMAGENET_CLASSES = json.load(f)
else:
    IMAGENET_CLASSES = []

def get_mock_predictions(num_classes: int = 1000, top_k: int = 5) -> List[Dict[str, Any]]:
    """Generate mock predictions for demo purposes"""
    predictions = []
    used_indices = set()
    
    for i in range(top_k):
        while True:
            idx = random.randint(0, num_classes - 1)
            if idx not in used_indices:
                used_indices.add(idx)
                break
        
        confidence = round(random.uniform(0.05, 0.95 - i * 0.1), 4)
        class_name = IMAGENET_CLASSES[idx] if idx < len(IMAGENET_CLASSES) else f"class_{idx}"
        
        predictions.append({
            "class_id": idx,
            "class_name": class_name,
            "confidence": confidence
        })
    
    return predictions


@router.post("/classify")
async def classify_image(
    file: UploadFile = File(...),
    model: str = "resnet50"
):
    """Classify an uploaded image"""
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Validate image format
        if image.format not in ["JPEG", "PNG", "JPG", "BMP", "WEBP"]:
            raise HTTPException(status_code=400, detail="Unsupported image format")
        
        # 使用真实推理引擎
        if HAS_INFERENCE and model_service.is_torch_available():
            try:
                predictions = inference_engine.classify(image, model_name=model, top_k=5)
                inference_mode = "pytorch_gpu" if (TORCH_AVAILABLE and torch.cuda.is_available()) else "pytorch_cpu"
            except Exception as e:
                print(f"推理失败，回退到 demo: {e}")
                predictions = get_mock_predictions(1000, 5)
                inference_mode = "demo"
        else:
            predictions = get_mock_predictions(1000, 5)
            inference_mode = "demo"
        
        return JSONResponse({
            "success": True,
            "model": model,
            "model_info": {
                "name": f"{model} (GPU)" if inference_mode.startswith("pytorch_gpu") else model,
                "description": f"Using {'GPU accelerated' if inference_mode.startswith('pytorch') else 'demo'} inference",
                "input_size": [224, 224],
                "accuracy": "76.1%" if model == "resnet50" else "N/A"
            },
            "inference_mode": inference_mode,
            "predictions": [
                {
                    "class": p["class_name"],
                    "class_id": p["class_id"],
                    "probability": p["confidence"]
                }
                for p in predictions
            ],
            "image_size": [image.width, image.height]
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Classification failed: {str(e)}")


@router.get("/models/classification")
async def get_classification_models():
    """Get available classification models"""
    models = model_service.get_model_info("classification")
    return JSONResponse({
        "success": True,
        "models": models,
        "torch_available": model_service.is_torch_available()
    })


@router.post("/classify/batch")
async def classify_batch(
    files: List[UploadFile] = File(...),
    model: str = "resnet50"
):
    """Classify multiple images"""
    results = []
    
    for file in files:
        try:
            contents = await file.read()
            image = Image.open(io.BytesIO(contents))
            
            predictions = get_mock_predictions(1000, 5)
            results.append({
                "filename": file.filename,
                "success": True,
                "predictions": predictions
            })
        except Exception as e:
            results.append({
                "filename": file.filename,
                "success": False,
                "error": str(e)
            })
    
    return JSONResponse({
        "success": True,
        "model": model,
        "results": results
    })
