"""
PyTorch Deep Learning Web Application
Main FastAPI Application Entry Point
"""

import os
import sys
from pathlib import Path

# Add backend directory to path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from routes import upload, classify, detect, segment, nlp, generate, models, train
from utils.config import settings, get_upload_dir, get_output_dir, get_model_dir

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    print("Starting PyTorch Deep Learning Server...")
    print(f"Device: {settings.DEVICE}")
    print(f"Upload directory: {get_upload_dir()}")
    
    yield
    
    # Shutdown
    print("Shutting down server...")

# Create FastAPI app
app = FastAPI(
    title="PyTorch Deep Learning Studio",
    description="A comprehensive web platform for deep learning with PyTorch",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/uploads", StaticFiles(directory=str(get_upload_dir())), name="uploads")
app.mount("/outputs", StaticFiles(directory=str(get_output_dir())), name="outputs")

# Include routers
app.include_router(upload.router, prefix="/api", tags=["Upload"])
app.include_router(classify.router, prefix="/api", tags=["Classification"])
app.include_router(detect.router, prefix="/api", tags=["Object Detection"])
app.include_router(segment.router, prefix="/api", tags=["Image Segmentation"])
app.include_router(nlp.router, prefix="/api", tags=["NLP"])
app.include_router(generate.router, prefix="/api", tags=["AIGC Generation"])
app.include_router(models.router, prefix="/api", tags=["Model Management"])
app.include_router(train.router, prefix="/api", tags=["Training"])

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "PyTorch Deep Learning Studio",
        "version": "1.0.0",
        "status": "running",
        "device": settings.DEVICE,
        "torch_available": False
    }

@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "pytorch-deep-learning-studio",
        "torch_available": False
    }
