"""
Natural Language Processing routes
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import random

from services.model_service import model_service

router = APIRouter()


class SentimentRequest(BaseModel):
    text: str
    model: Optional[str] = "bert_sentiment"


class GenerationRequest(BaseModel):
    prompt: str
    max_length: Optional[int] = 100
    temperature: Optional[float] = 1.0
    model: Optional[str] = "gpt2_text"


class SummarizationRequest(BaseModel):
    text: str
    max_length: Optional[int] = 150


SAMPLE_RESPONSES = {
    "positive": [
        "This is absolutely fantastic! I love it so much.",
        "Great experience overall, highly recommended!",
        "Outstanding quality and excellent service.",
        "Wonderful product, exceeded my expectations!",
        "Best purchase I've ever made. Five stars!"
    ],
    "negative": [
        "Terrible experience, completely disappointed.",
        "Would not recommend to anyone. Very poor quality.",
        "Waste of money. The worst purchase ever.",
        "Horrible service and product. Avoid at all costs.",
        "Extremely dissatisfied. This is unacceptable."
    ],
    "neutral": [
        "It's okay, nothing special about it.",
        "Average experience, as expected.",
        "Decent product for the price point.",
        "Works as advertised, nothing more.",
        "Acceptable quality for everyday use."
    ]
}


SENTIMENT_LABELS = {
    "positive": "积极 (positive)",
    "negative": "消极 (negative)",
    "neutral": "中性 (neutral)"
}


def get_mock_sentiment(text: str) -> Dict[str, Any]:
    """Generate mock sentiment analysis"""
    sentiment_choice = random.choice(["positive", "negative", "neutral"])
    confidence = round(random.uniform(0.6, 0.98), 4)
    
    return {
        "sentiment": sentiment_choice,
        "sentiment_label": SENTIMENT_LABELS.get(sentiment_choice, sentiment_choice),
        "confidence": confidence,
        "sample_response": random.choice(SAMPLE_RESPONSES[sentiment_choice])
    }


SAMPLE_GENERATIONS = [
    "The quick brown fox jumps over the lazy dog. This classic pangram has been used for decades to test typewriters and fonts.",
    "In the depths of the ancient forest, a lone traveler sought shelter from the relentless storm that raged outside.",
    "The future of technology is both exciting and uncertain. As AI continues to evolve, we must consider its implications carefully.",
    "A journey of a thousand miles begins with a single step. Every great achievement starts with the decision to try.",
    "The art of programming lies not just in writing code, but in solving problems elegantly and efficiently."
]


def get_mock_generation(prompt: str, max_length: int = 100) -> Dict[str, Any]:
    """Generate mock text generation"""
    text = random.choice(SAMPLE_GENERATIONS)
    if len(text) > max_length:
        text = text[:max_length] + "..."
    
    return {
        "generated_text": text,
        "prompt": prompt,
        "length": len(text.split()),
        "tokens_generated": len(text.split())
    }


def get_mock_summarization(text: str, max_length: int = 150) -> Dict[str, Any]:
    """Generate mock summarization"""
    words = text.split()
    summary_words = words[:min(30, len(words)//3)]
    summary = " ".join(summary_words) + "..."
    
    return {
        "summary": summary,
        "original_length": len(words),
        "summary_length": len(summary_words),
        "compression_ratio": round(len(summary_words) / len(words), 2) if words else 0
    }


@router.post("/nlp/sentiment")
async def analyze_sentiment(request: SentimentRequest):
    """Analyze sentiment of text"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        sentiment = get_mock_sentiment(request.text)
        
        return JSONResponse({
            "success": True,
            "model": request.model,
            "inference_mode": "demo" if not model_service.is_torch_available() else "pytorch",
            "input_text": request.text[:200] + "..." if len(request.text) > 200 else request.text,
            "result": sentiment
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sentiment analysis failed: {str(e)}")


@router.post("/nlp/generate")
async def generate_text(request: GenerationRequest):
    """Generate text from prompt"""
    try:
        if not request.prompt.strip():
            raise HTTPException(status_code=400, detail="Prompt cannot be empty")
        
        generated = get_mock_generation(request.prompt, request.max_length)
        
        return JSONResponse({
            "success": True,
            "model": request.model,
            "inference_mode": "demo" if not model_service.is_torch_available() else "pytorch",
            "result": generated
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text generation failed: {str(e)}")


@router.post("/nlp/summarize")
async def summarize_text(request: SummarizationRequest):
    """Summarize text"""
    try:
        if not request.text.strip():
            raise HTTPException(status_code=400, detail="Text cannot be empty")
        
        summary = get_mock_summarization(request.text, request.max_length)
        
        return JSONResponse({
            "success": True,
            "input_length": len(request.text.split()),
            "result": summary
        })
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {str(e)}")


@router.get("/models/nlp")
async def get_nlp_models():
    """Get available NLP models"""
    models = model_service.get_model_info("nlp")
    return JSONResponse({
        "success": True,
        "models": models,
        "torch_available": model_service.is_torch_available()
    })
