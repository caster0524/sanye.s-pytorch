"""
Model management routes
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse
from typing import List, Optional
from pathlib import Path
import os
import aiofiles

from utils.config import settings
from services.model_service import model_service, MODEL_CATALOG

router = APIRouter()

# Supported model formats
SUPPORTED_FORMATS = [".pt", ".pth", ".onnx"]


@router.get("/models")
async def list_models(category: Optional[str] = None):
    """List all available models"""
    try:
        if category:
            models = []
            if category in MODEL_CATALOG:
                for model_id, model_info in MODEL_CATALOG[category].items():
                    models.append({
                        "id": model_id,
                        "category": category,
                        **model_info,
                        "status": "available"
                    })
            return JSONResponse({
                "success": True,
                "category": category,
                "models": models
            })
        else:
            all_models = model_service.get_all_models()
            return JSONResponse({
                "success": True,
                "models": all_models,
                "torch_available": model_service.is_torch_available()
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/catalog")
async def get_model_catalog():
    """Get the complete model catalog"""
    return JSONResponse({
        "success": True,
        "catalog": MODEL_CATALOG,
        "torch_available": model_service.is_torch_available()
    })


@router.post("/models/upload")
async def upload_model(file: UploadFile = File(...)):
    """Upload a custom model file"""
    try:
        # Validate file format
        filename = file.filename or "model.onnx"
        ext = Path(filename).suffix.lower()
        
        if ext not in SUPPORTED_FORMATS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported format. Supported: {', '.join(SUPPORTED_FORMATS)}"
            )
        
        # Create upload directory if needed
        upload_dir = Path(settings.UPLOAD_DIR) / "models"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Save file
        file_path = upload_dir / filename
        
        async with aiofiles.open(file_path, 'wb') as f:
            content = await file.read()
            await f.write(content)
        
        return JSONResponse({
            "success": True,
            "message": f"Model {filename} uploaded successfully",
            "file_path": str(file_path),
            "file_size": len(content),
            "format": ext[1:]  # Remove the dot
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.get("/models/{model_id}")
async def get_model_details(model_id: str):
    """Get details of a specific model"""
    for cat_name, cat_models in MODEL_CATALOG.items():
        if model_id in cat_models:
            return JSONResponse({
                "success": True,
                "model": {
                    "id": model_id,
                    "category": cat_name,
                    **cat_models[model_id]
                },
                "torch_available": model_service.is_torch_available()
            })
    
    raise HTTPException(status_code=404, detail=f"Model {model_id} not found")


@router.post("/models/{model_id}/load")
async def load_model(model_id: str):
    """Load a specific model into memory"""
    try:
        result = model_service.load_model(model_id)
        return JSONResponse(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/models/uploaded/{filename}")
async def delete_uploaded_model(filename: str):
    """Delete an uploaded model"""
    try:
        file_path = Path(settings.UPLOAD_DIR) / "models" / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        
        file_path.unlink()
        
        return JSONResponse({
            "success": True,
            "message": f"Model {filename} deleted"
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/models/uploaded")
async def list_uploaded_models():
    """List all user-uploaded models"""
    try:
        upload_dir = Path(settings.UPLOAD_DIR) / "models"
        
        if not upload_dir.exists():
            return JSONResponse({
                "success": True,
                "models": []
            })
        
        models = []
        for f in upload_dir.iterdir():
            if f.suffix.lower() in SUPPORTED_FORMATS:
                stat = f.stat()
                models.append({
                    "filename": f.name,
                    "size": stat.st_size,
                    "format": f.suffix[1:],
                    "uploaded_at": stat.st_mtime
                })
        
        return JSONResponse({
            "success": True,
            "models": models
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
