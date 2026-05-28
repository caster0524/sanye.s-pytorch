"""
File Upload routes
"""
from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from PIL import Image
import io
import os
import aiofiles
import hashlib
from pathlib import Path
from typing import List, Optional
import uuid

from utils.config import settings
from utils.image_utils import image_to_base64, get_image_info

router = APIRouter()

# Supported file types
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".flac", ".aac"}


@router.post("/upload/image")
async def upload_image(file: UploadFile = File(...)):
    """Upload a single image file"""
    try:
        # Validate file type
        filename = file.filename or "image.jpg"
        ext = Path(filename).suffix.lower()
        
        if ext not in IMAGE_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image format. Supported: {', '.join(IMAGE_EXTENSIONS)}"
            )
        
        # Read and validate image
        contents = await file.read()
        
        try:
            image = Image.open(io.BytesIO(contents))
            image.verify()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Generate unique filename
        file_id = uuid.uuid4().hex[:8]
        new_filename = f"{file_id}_{filename}"
        
        # Save file
        upload_dir = Path(settings.UPLOAD_DIR) / "images"
        upload_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = upload_dir / new_filename
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(contents)
        
        # Get image info
        image = Image.open(io.BytesIO(contents))
        info = get_image_info(image)
        
        return JSONResponse({
            "success": True,
            "file_id": file_id,
            "filename": new_filename,
            "original_filename": filename,
            "path": f"/uploads/images/{new_filename}",
            "size": len(contents),
            "image_info": info
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@router.post("/upload/images")
async def upload_images(files: List[UploadFile] = File(...)):
    """Upload multiple image files"""
    results = []
    errors = []
    
    for file in files:
        try:
            filename = file.filename or "image.jpg"
            ext = Path(filename).suffix.lower()
            
            if ext not in IMAGE_EXTENSIONS:
                errors.append({
                    "filename": filename,
                    "error": f"Unsupported format: {ext}"
                })
                continue
            
            contents = await file.read()
            
            try:
                image = Image.open(io.BytesIO(contents))
                image.verify()
            except Exception:
                errors.append({
                    "filename": filename,
                    "error": "Invalid image file"
                })
                continue
            
            file_id = uuid.uuid4().hex[:8]
            new_filename = f"{file_id}_{filename}"
            
            upload_dir = Path(settings.UPLOAD_DIR) / "images"
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = upload_dir / new_filename
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(contents)
            
            image = Image.open(io.BytesIO(contents))
            info = get_image_info(image)
            
            results.append({
                "file_id": file_id,
                "filename": new_filename,
                "original_filename": filename,
                "path": f"/uploads/images/{new_filename}",
                "size": len(contents),
                "image_info": info
            })
            
        except Exception as e:
            errors.append({
                "filename": file.filename,
                "error": str(e)
            })
    
    return JSONResponse({
        "success": True,
        "uploaded": results,
        "errors": errors,
        "total_uploaded": len(results),
        "total_errors": len(errors)
    })


@router.post("/upload/dataset")
async def upload_dataset(files: List[UploadFile] = File(...), dataset_name: str = "custom_dataset"):
    """Upload a dataset (folder of images organized by class)"""
    try:
        dataset_dir = Path(settings.UPLOAD_DIR) / "datasets" / dataset_name
        dataset_dir.mkdir(parents=True, exist_ok=True)
        
        uploaded_files = 0
        class_folders = set()
        
        for file in files:
            filename = file.filename or "unknown"
            parts = Path(filename).parts
            
            # Assume structure: class_name/image.jpg or just image.jpg
            if len(parts) > 1:
                class_name = parts[0]
                class_folders.add(class_name)
            else:
                class_name = "unclassified"
            
            # Save file
            file_dir = dataset_dir / class_name
            file_dir.mkdir(parents=True, exist_ok=True)
            
            contents = await file.read()
            file_path = file_dir / Path(filename).name
            
            async with aiofiles.open(file_path, 'wb') as f:
                await f.write(contents)
            
            uploaded_files += 1
        
        return JSONResponse({
            "success": True,
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_dir),
            "files_uploaded": uploaded_files,
            "classes_found": list(class_folders),
            "total_classes": len(class_folders)
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/uploads/{path:path}")
async def get_uploaded_file(path: str):
    """Serve uploaded files"""
    file_path = Path(settings.UPLOAD_DIR) / path
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(str(file_path))


@router.delete("/uploads/{path:path}")
async def delete_uploaded_file(path: str):
    """Delete an uploaded file"""
    file_path = Path(settings.UPLOAD_DIR) / path
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        file_path.unlink()
        return JSONResponse({
            "success": True,
            "message": f"File {path} deleted"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/uploads/list")
async def list_uploads(category: Optional[str] = None):
    """List uploaded files"""
    base_dir = Path(settings.UPLOAD_DIR)
    
    if category:
        base_dir = base_dir / category
    
    if not base_dir.exists():
        return JSONResponse({
            "success": True,
            "files": [],
            "total": 0
        })
    
    files = []
    for f in base_dir.rglob("*"):
        if f.is_file():
            rel_path = f.relative_to(Path(settings.UPLOAD_DIR))
            stat = f.stat()
            files.append({
                "path": str(rel_path),
                "url": f"/uploads/{rel_path}",
                "size": stat.st_size,
                "modified": stat.st_mtime
            })
    
    return JSONResponse({
        "success": True,
        "files": files,
        "total": len(files)
    })
