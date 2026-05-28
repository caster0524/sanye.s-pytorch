"""
Image Segmentation routes
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

_VOC_PATH = Path(__file__).parent / "voc_classes.json"
with open(_VOC_PATH, "r", encoding="utf-8") as f:
    _VOC_DICT = json.load(f)
SEGMENT_CLASSES = [_VOC_DICT.get(str(i), f"class_{i}") for i in range(21)]


def get_mock_segments(num_segments: int = 5) -> List[Dict[str, Any]]:
    """Generate mock segmentation results for demo purposes"""
    segments = []
    used_classes = set()
    
    for i in range(num_segments):
        while True:
            cls_idx = random.randint(1, len(SEGMENT_CLASSES) - 1)
            if cls_idx not in used_classes:
                used_classes.add(cls_idx)
                break
        
        x1 = random.randint(50, 150)
        y1 = random.randint(50, 150)
        width = random.randint(100, 400)
        height = random.randint(100, 400)
        
        segments.append({
            "class_id": cls_idx,
            "class_name": SEGMENT_CLASSES[cls_idx],
            "mask_bbox": {
                "x1": x1,
                "y1": y1,
                "x2": x1 + width,
                "y2": y1 + height
            },
            "confidence": round(random.uniform(0.4, 0.95), 4),
            "area_pixels": random.randint(10000, 100000)
        })
    
    return segments


@router.post("/segment")
async def segment_image(
    file: UploadFile = File(...),
    model: str = "deeplabv3_resnet50"
):
    """Segment an uploaded image"""
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
                result = inference_engine.segment(image, model_name=model)
                segments = result["segments"]
                inference_mode = "pytorch_gpu"
            except Exception as e:
                print(f"分割推理失败: {e}")
                segments = get_mock_segments(random.randint(3, 7))
                inference_mode = "demo"
        else:
            segments = get_mock_segments(random.randint(3, 7))
            inference_mode = "demo"
        
        return JSONResponse({
            "success": True,
            "model": model,
            "inference_mode": inference_mode,
            "segments": segments,
            "original_size": [image.width, image.height],
            "mask_size": [image.width, image.height],
            "num_classes": len(set(s.get("class_name", "") for s in segments)),
            "class_distribution": {
                s["class_name"]: s.get("area_pixels", 1)
                for s in segments
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Segmentation failed: {str(e)}")


@router.get("/models/segmentation")
async def get_segmentation_models():
    """Get available segmentation models"""
    models = model_service.get_model_info("segmentation")
    return JSONResponse({
        "success": True,
        "models": models,
        "torch_available": model_service.is_torch_available()
    })
