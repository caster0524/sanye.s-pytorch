"""
Object Detection routes
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
import io
from typing import List, Dict, Any
import random
import json

from pathlib import Path

from utils.config import settings
from services.model_service import model_service

try:
    from services.inference_service import inference_engine
    HAS_INFERENCE = inference_engine.is_ready()
except Exception:
    HAS_INFERENCE = False

router = APIRouter()

_COCO_PATH = Path(__file__).parent / "coco_classes.json"
with open(_COCO_PATH, "r", encoding="utf-8") as f:
    _COCO_DICT = json.load(f)
COCO_CLASSES = [_COCO_DICT.get(str(i), f"class_{i}") for i in range(91)]
COCO_VALID_COUNT = 80


def get_mock_detections(num_detections: int = 5) -> List[Dict[str, Any]]:
    """Generate mock detections for demo purposes"""
    detections = []
    used_classes = set()
    
    for i in range(num_detections):
        while True:
            cls_idx = random.randint(1, COCO_VALID_COUNT)
            if cls_idx not in used_classes:
                used_classes.add(cls_idx)
                break
        
        x1 = random.randint(50, 200)
        y1 = random.randint(50, 200)
        width = random.randint(50, 300)
        height = random.randint(50, 300)
        
        detections.append({
            "class_id": cls_idx,
            "class_name": COCO_CLASSES[cls_idx],
            "bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x1 + width,
                "y2": y1 + height
            },
            "confidence": round(random.uniform(0.3, 0.95), 4)
        })
    
    return detections


@router.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    model: str = "fasterrcnn_resnet50"
):
    """Detect objects in an uploaded image"""
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))
        
        # Validate image format
        if image.format not in ["JPEG", "PNG", "JPG", "BMP", "WEBP"]:
            raise HTTPException(status_code=400, detail="Unsupported image format")
        
        # 使用真实推理引擎
        if HAS_INFERENCE:
            try:
                detections = inference_engine.detect(image, model_name=model, confidence_threshold=0.3)
                inference_mode = "pytorch_gpu"
            except Exception as e:
                print(f"检测推理失败: {e}")
                detections = get_mock_detections(random.randint(3, 8))
                inference_mode = "demo"
        else:
            detections = get_mock_detections(random.randint(3, 8))
            inference_mode = "demo"
        
        return JSONResponse({
            "success": True,
            "model": model,
            "inference_mode": inference_mode,
            "detections": detections,
            "count": len(detections),
            "image_info": {
                "width": image.width,
                "height": image.height,
                "format": image.format
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@router.get("/models/detection")
async def get_detection_models():
    """Get available detection models"""
    models = model_service.get_model_info("detection")
    return JSONResponse({
        "success": True,
        "models": models,
        "torch_available": model_service.is_torch_available()
    })
