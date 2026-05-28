"""
AI Image Generation routes
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import random
import time

from services.model_service import model_service

router = APIRouter()


class GenerationRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = ""
    width: Optional[int] = 512
    height: Optional[int] = 512
    num_inference_steps: Optional[int] = 50
    guidance_scale: Optional[float] = 7.5
    seed: Optional[int] = None
    model: Optional[str] = "stable_diffusion"


SAMPLE_PROMPTS = [
    "A beautiful sunset over the ocean with vibrant orange and pink colors",
    "A futuristic city with flying cars and towering skyscrapers",
    "A serene mountain landscape covered in snow under a clear blue sky",
    "An abstract painting with swirling patterns of blue and gold",
    "A cozy coffee shop interior with warm lighting and wooden furniture"
]


@router.post("/generate")
async def generate_image(request: GenerationRequest):
    """Generate an image from text prompt"""
    try:
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        # Validate dimensions
        if request.width > 1024 or request.height > 1024:
            raise HTTPException(status_code=400, detail="Maximum dimension is 1024")
        if request.width < 256 or request.height < 256:
            raise HTTPException(status_code=400, detail="Minimum dimension is 256")
        
        # Simulate generation time
        generation_time = round(random.uniform(2.0, 5.0), 2)
        
        # Use provided seed or generate random
        seed = request.seed if request.seed is not None else random.randint(0, 999999999)
        
        return JSONResponse({
            "success": True,
            "model": request.model,
            "inference_mode": "demo" if not model_service.is_torch_available() else "pytorch",
            "parameters": {
                "prompt": request.prompt,
                "negative_prompt": request.negative_prompt,
                "width": request.width,
                "height": request.height,
                "num_inference_steps": request.num_inference_steps,
                "guidance_scale": request.guidance_scale,
                "seed": seed
            },
            "result": {
                "image_url": f"/outputs/generated_{seed}.png",
                "seed": seed,
                "generation_time_seconds": generation_time,
                "note": "This is a demo. Install PyTorch and Stable Diffusion for real generation."
            }
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation failed: {str(e)}")


@router.post("/generate/batch")
async def batch_generate(req: GenerationRequest):
    """Generate multiple images from the same prompt"""
    try:
        if not req.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        num_images = 4
        results = []
        
        for i in range(num_images):
            seed = random.randint(0, 999999999)
            results.append({
                "index": i,
                "seed": seed,
                "image_url": f"/outputs/generated_{seed}.png",
                "generation_time_seconds": round(random.uniform(2.0, 5.0), 2)
            })
        
        return JSONResponse({
            "success": True,
            "model": req.model,
            "prompt": req.prompt,
            "num_images": num_images,
            "results": results
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch generation failed: {str(e)}")


@router.get("/models/generation")
async def get_generation_models():
    """Get available generation models"""
    models = model_service.get_model_info("generation")
    return JSONResponse({
        "success": True,
        "models": models,
        "torch_available": model_service.is_torch_available()
    })


@router.get("/generation/styles")
async def get_generation_styles():
    """Get available art styles and presets"""
    styles = [
        {"id": "realistic", "name": "Photorealistic", "description": "Highly detailed realistic images"},
        {"id": "anime", "name": "Anime", "description": "Japanese anime style illustration"},
        {"id": "digital_art", "name": "Digital Art", "description": "Modern digital artwork style"},
        {"id": "oil_painting", "name": "Oil Painting", "description": "Classical oil painting style"},
        {"id": "watercolor", "name": "Watercolor", "description": "Soft watercolor painting effect"},
        {"id": "abstract", "name": "Abstract", "description": "Abstract art with bold colors and shapes"},
        {"id": "3d_render", "name": "3D Render", "description": "3D rendered imagery"},
        {"id": "concept_art", "name": "Concept Art", "description": "Professional concept art style"}
    ]
    
    return JSONResponse({
        "success": True,
        "styles": styles
    })
