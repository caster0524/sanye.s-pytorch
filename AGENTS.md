# AGENTS.md - PyTorch Deep Learning Web Application

## Project Overview

**Project Name**: PyTorch Deep Learning Studio
**Type**: Full-stack Web Application (Frontend: Next.js + Backend: FastAPI)
**Core Functionality**: A comprehensive web platform for deep learning with PyTorch, supporting image classification, object detection, image segmentation, NLP processing, and AI image generation.
**Target Users**: ML engineers, data scientists, students, AI researchers

## Architecture

```
/pytorch-web-app
├── backend/                 # Python FastAPI backend
│   ├── main.py             # FastAPI app entry point
│   ├── routes/             # API route handlers
│   │   ├── upload.py       # File upload endpoints
│   │   ├── classify.py     # Image classification
│   │   ├── detect.py       # Object detection
│   │   ├── segment.py      # Image segmentation
│   │   ├── nlp.py          # NLP processing
│   │   ├── generate.py     # AIGC generation
│   │   ├── models.py       # Model management
│   │   └── train.py        # Training interface
│   ├── services/           # Business logic
│   │   └── model_service.py # Model loading/caching
│   └── utils/              # Utilities
│       ├── config.py       # Configuration
│       └── image_utils.py  # Image processing
├── frontend/               # Next.js frontend
│   ├── src/
│   │   ├── app/            # Next.js App Router pages
│   │   │   ├── page.tsx    # Dashboard
│   │   │   ├── classification/
│   │   │   ├── detection/
│   │   │   ├── segmentation/
│   │   │   ├── nlp/
│   │   │   ├── generation/
│   │   │   ├── training/
│   │   │   └── models/
│   │   ├── components/     # Reusable components
│   │   ├── lib/           # API client
│   │   └── types/         # TypeScript types
│   └── public/
├── public/                 # Static assets
│   ├── uploads/           # User uploads
│   ├── models/            # Custom models
│   └── outputs/           # Generated outputs
├── DESIGN.md              # Design specification
└── README.md              # Project documentation
```

## Development Commands

### Backend
```bash
cd backend
pip install -r requirements.txt
python main.py  # Runs on port 8000
```

### Frontend
```bash
cd frontend
pnpm install
pnpm dev  # Runs on port 5000
```

## Key Technical Decisions

### Backend (Python FastAPI)
- **Framework**: FastAPI for async API handling
- **ML Runtime**: PyTorch with CUDA support (when available)
- **File Handling**: aiofiles for async uploads
- **Models**: torchvision for pretrained models, transformers for NLP

### Frontend (Next.js)
- **Framework**: Next.js 14+ with App Router
- **Styling**: Tailwind CSS with custom theme
- **Charts**: Recharts for data visualization
- **Icons**: Lucide React

## Code Style Guidelines

### Python
- Follow PEP 8
- Use type hints for function parameters
- Async/await for I/O operations
- Centralized model service for caching

### TypeScript/JavaScript
- Use TypeScript for type safety
- Follow Next.js conventions
- Client components marked with 'use client'
- API calls in lib/api.ts

## Testing Approach

1. **Backend API Testing**: Test each endpoint with curl
2. **Frontend Testing**: Browser-based manual testing
3. **Integration Testing**: Full workflow testing

## Common Issues & Solutions

### CORS Issues
- Backend configured with CORS middleware for localhost development
- Update allowed origins in production

### Model Loading
- Models are cached after first load
- Clear cache by restarting the backend server

### GPU Memory
- Models use float32 by default
- Use float16 on CUDA for memory efficiency

## Maintenance Notes

- Check backend logs in console for errors
- Monitor GPU memory when running inference
- Clear uploaded files periodically
- Update dependencies regularly
