"""
MFA Service - FastAPI endpoint for Montreal Forced Aligner

Получает аудио + транскрипт, возвращает точные word timestamps.
Точность: ±10-20ms (с RMS refinement ±5-10ms)
"""

import os
import tempfile
import base64
import time
from typing import Optional, List
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from aligner import MFAAligner
from rms_refiner import refine_word_endpoints


# === Models ===

class AlignRequest(BaseModel):
    """Запрос на alignment"""
    audio_base64: str = Field(..., description="WAV audio encoded as base64")
    transcript: str = Field(..., description="Text transcript to align")
    language: str = Field(default="en", description="Language code: en, ru, es, de, pt")
    refine_endpoints: bool = Field(default=True, description="Apply RMS refinement to endTime")
    sample_rate: int = Field(default=24000, description="Input audio sample rate (will be resampled to 16kHz)")


class WordTimestamp(BaseModel):
    """Timestamp для одного слова"""
    word: str
    start: float = Field(..., description="Start time in seconds")
    end: float = Field(..., description="End time in seconds")


class AlignResponse(BaseModel):
    """Ответ с timestamps"""
    words: List[WordTimestamp]
    total_duration: float
    processing_time_ms: int
    model_used: str
    refined: bool


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: List[str]
    version: str


# === App lifecycle ===

aligner: Optional[MFAAligner] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Загрузка моделей при старте"""
    global aligner
    print("[MFA] Initializing aligner...")
    aligner = MFAAligner()
    print(f"[MFA] Loaded models: {aligner.available_languages}")
    yield
    print("[MFA] Shutting down...")


app = FastAPI(
    title="MFA Alignment Service",
    description="Montreal Forced Aligner for precise word-level timestamps",
    version="1.0.0",
    lifespan=lifespan
)


# === Endpoints ===

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded=aligner.available_languages if aligner else [],
        version="1.0.0"
    )


@app.post("/align", response_model=AlignResponse)
async def align_audio(request: AlignRequest):
    """
    Выполняет forced alignment аудио с транскриптом.
    
    Возвращает точные timestamps для каждого слова.
    """
    if not aligner:
        raise HTTPException(status_code=503, detail="Aligner not initialized")
    
    if request.language not in aligner.available_languages:
        raise HTTPException(
            status_code=400, 
            detail=f"Language '{request.language}' not supported. Available: {aligner.available_languages}"
        )
    
    start_time = time.time()
    
    try:
        # Decode audio
        audio_bytes = base64.b64decode(request.audio_base64)
        
        # Create temp files
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, "input.wav")
            
            # Save audio
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
            
            # Run alignment
            words = aligner.align(
                audio_path=audio_path,
                transcript=request.transcript,
                language=request.language,
                input_sample_rate=request.sample_rate
            )
            
            # RMS refinement
            if request.refine_endpoints and words:
                words = refine_word_endpoints(
                    audio_path=audio_path,
                    words=words,
                    sample_rate=request.sample_rate
                )
            
            # Calculate total duration
            total_duration = words[-1]["end"] if words else 0.0
            
            processing_time_ms = int((time.time() - start_time) * 1000)
            
            return AlignResponse(
                words=[WordTimestamp(**w) for w in words],
                total_duration=total_duration,
                processing_time_ms=processing_time_ms,
                model_used=aligner.get_model_name(request.language),
                refined=request.refine_endpoints
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Alignment failed: {str(e)}")


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "MFA Alignment Service",
        "version": "1.0.0",
        "endpoints": ["/health", "/align"]
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
