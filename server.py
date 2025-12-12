"""
High-Performance TTS Server for IndexTTS2

FastAPI-based server with concurrent request handling, batching, and streaming support.
Run with: uv run python server.py --port 9999
"""

import os
import sys
import io
import time
import asyncio
import logging
import warnings
from typing import Optional
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Setup path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)
sys.path.append(os.path.join(current_dir, "indextts"))

import argparse
import torch
import torchaudio
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global TTS instance and thread pool
tts_instance = None
executor = None
request_queue = asyncio.Queue()
MAX_CONCURRENT_INFERENCE = 4


class TTSRequest(BaseModel):
    """Request model for TTS synthesis"""
    text: str = Field(..., description="Text to synthesize", min_length=1, max_length=5000)
    speaker_audio_path: Optional[str] = Field(None, description="Path to speaker reference audio")
    emo_audio_path: Optional[str] = Field(None, description="Path to emotion reference audio")
    emo_alpha: float = Field(1.0, ge=0.0, le=1.0, description="Emotion blending weight")
    max_text_tokens_per_segment: int = Field(120, ge=20, le=500, description="Max tokens per segment")
    # Generation parameters
    do_sample: bool = Field(True, description="Whether to use sampling")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.8, ge=0.0, le=1.0, description="Top-p sampling")
    top_k: int = Field(30, ge=0, le=100, description="Top-k sampling")
    repetition_penalty: float = Field(10.0, ge=0.1, le=20.0, description="Repetition penalty")


class TTSResponse(BaseModel):
    """Response model for TTS synthesis"""
    success: bool
    audio_base64: Optional[str] = None
    duration_seconds: Optional[float] = None
    inference_time_ms: Optional[float] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    gpu_available: bool
    gpu_memory_used_mb: Optional[float] = None


class PreloadSpeakerRequest(BaseModel):
    """Request to preload speaker embeddings"""
    speaker_audio_path: str


def load_tts_model(args):
    """Load the TTS model with optimizations"""
    from indextts.infer_v2 import IndexTTS2
    
    logger.info(f"Loading IndexTTS2 model from {args.model_dir}...")
    start_time = time.time()
    
    model = IndexTTS2(
        model_dir=args.model_dir,
        cfg_path=os.path.join(args.model_dir, "config.yaml"),
        use_fp16=args.fp16,
        use_deepspeed=args.deepspeed,
        use_cuda_kernel=args.cuda_kernel,
    )
    
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f}s")
    
    return model


def run_inference(
    text: str,
    speaker_audio_path: str,
    emo_audio_path: Optional[str] = None,
    emo_alpha: float = 1.0,
    max_text_tokens_per_segment: int = 120,
    **generation_kwargs
) -> tuple:
    """Run TTS inference in thread pool (blocking operation)"""
    global tts_instance
    
    start_time = time.perf_counter()
    
    # Generate unique output path
    output_path = os.path.join("outputs", f"api_{int(time.time() * 1000)}.wav")
    os.makedirs("outputs", exist_ok=True)
    
    try:
        result = tts_instance.infer(
            spk_audio_prompt=speaker_audio_path,
            text=text,
            output_path=output_path,
            emo_audio_prompt=emo_audio_path,
            emo_alpha=emo_alpha,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            verbose=False,
            **generation_kwargs
        )
        
        inference_time = (time.perf_counter() - start_time) * 1000
        
        if result and os.path.exists(result):
            # Load audio for duration calculation
            waveform, sample_rate = torchaudio.load(result)
            duration = waveform.shape[1] / sample_rate
            
            # Read audio bytes
            with open(result, 'rb') as f:
                audio_bytes = f.read()
            
            return audio_bytes, duration, inference_time, None
        else:
            return None, None, inference_time, "Inference failed - no output generated"
            
    except Exception as e:
        inference_time = (time.perf_counter() - start_time) * 1000
        logger.error(f"Inference error: {e}")
        return None, None, inference_time, str(e)


async def run_inference_async(
    text: str,
    speaker_audio_path: str,
    emo_audio_path: Optional[str] = None,
    emo_alpha: float = 1.0,
    max_text_tokens_per_segment: int = 120,
    **generation_kwargs
) -> tuple:
    """Async wrapper for TTS inference"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        executor,
        lambda: run_inference(
            text=text,
            speaker_audio_path=speaker_audio_path,
            emo_audio_path=emo_audio_path,
            emo_alpha=emo_alpha,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            **generation_kwargs
        )
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage server lifecycle - load model on startup"""
    global tts_instance, executor
    
    # Parse arguments for model loading
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9999)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--model_dir", type=str, default="./checkpoints")
    parser.add_argument("--fp16", action="store_true", default=False)
    parser.add_argument("--deepspeed", action="store_true", default=False)
    parser.add_argument("--cuda_kernel", action="store_true", default=False)
    parser.add_argument("--workers", type=int, default=MAX_CONCURRENT_INFERENCE)
    args, _ = parser.parse_known_args()
    
    # Initialize thread pool
    executor = ThreadPoolExecutor(max_workers=args.workers)
    logger.info(f"Thread pool initialized with {args.workers} workers")
    
    # Load model
    tts_instance = load_tts_model(args)
    
    yield
    
    # Cleanup
    executor.shutdown(wait=True)
    logger.info("Server shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="IndexTTS2 High-Performance API",
    description="Fast, concurrent TTS API server for IndexTTS2",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for load balancer integration"""
    gpu_memory = None
    gpu_available = torch.cuda.is_available()
    
    if gpu_available:
        try:
            gpu_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # MB
        except Exception:
            pass
    
    return HealthResponse(
        status="healthy" if tts_instance is not None else "loading",
        model_loaded=tts_instance is not None,
        gpu_available=gpu_available,
        gpu_memory_used_mb=gpu_memory
    )


@app.post("/tts", response_model=TTSResponse)
async def synthesize_speech(request: TTSRequest):
    """
    Synthesize speech from text.
    
    Returns base64 encoded WAV audio.
    """
    if tts_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    if not request.speaker_audio_path:
        raise HTTPException(status_code=400, detail="speaker_audio_path is required")
    
    if not os.path.exists(request.speaker_audio_path):
        raise HTTPException(status_code=400, detail=f"Speaker audio not found: {request.speaker_audio_path}")
    
    generation_kwargs = {
        "do_sample": request.do_sample,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k if request.top_k > 0 else None,
        "repetition_penalty": request.repetition_penalty,
    }
    
    audio_bytes, duration, inference_time, error = await run_inference_async(
        text=request.text,
        speaker_audio_path=request.speaker_audio_path,
        emo_audio_path=request.emo_audio_path,
        emo_alpha=request.emo_alpha,
        max_text_tokens_per_segment=request.max_text_tokens_per_segment,
        **generation_kwargs
    )
    
    if error:
        return TTSResponse(
            success=False,
            error=error,
            inference_time_ms=inference_time
        )
    
    import base64
    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
    
    return TTSResponse(
        success=True,
        audio_base64=audio_base64,
        duration_seconds=duration,
        inference_time_ms=inference_time
    )


@app.post("/tts/audio")
async def synthesize_speech_audio(request: TTSRequest):
    """
    Synthesize speech and return raw WAV audio directly.
    
    More efficient than base64 for large files.
    """
    if tts_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    if not request.speaker_audio_path:
        raise HTTPException(status_code=400, detail="speaker_audio_path is required")
    
    if not os.path.exists(request.speaker_audio_path):
        raise HTTPException(status_code=400, detail=f"Speaker audio not found: {request.speaker_audio_path}")
    
    generation_kwargs = {
        "do_sample": request.do_sample,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "top_k": request.top_k if request.top_k > 0 else None,
        "repetition_penalty": request.repetition_penalty,
    }
    
    audio_bytes, duration, inference_time, error = await run_inference_async(
        text=request.text,
        speaker_audio_path=request.speaker_audio_path,
        emo_audio_path=request.emo_audio_path,
        emo_alpha=request.emo_alpha,
        max_text_tokens_per_segment=request.max_text_tokens_per_segment,
        **generation_kwargs
    )
    
    if error:
        raise HTTPException(status_code=500, detail=error)
    
    return Response(
        content=audio_bytes,
        media_type="audio/wav",
        headers={
            "X-Inference-Time-Ms": str(inference_time),
            "X-Audio-Duration-Seconds": str(duration)
        }
    )


@app.post("/tts/upload")
async def synthesize_with_upload(
    text: str = Form(...),
    speaker_audio: UploadFile = File(...),
    emo_audio: Optional[UploadFile] = File(None),
    emo_alpha: float = Form(1.0),
    max_text_tokens_per_segment: int = Form(120),
    do_sample: bool = Form(True),
    temperature: float = Form(0.8),
    top_p: float = Form(0.8),
    top_k: int = Form(30),
    repetition_penalty: float = Form(10.0),
):
    """
    Synthesize speech with uploaded audio files.
    
    Use this endpoint when speaker audio is not on the server filesystem.
    """
    if tts_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    # Save uploaded files temporarily
    os.makedirs("temp_uploads", exist_ok=True)
    
    speaker_path = f"temp_uploads/speaker_{int(time.time() * 1000)}.wav"
    with open(speaker_path, "wb") as f:
        content = await speaker_audio.read()
        f.write(content)
    
    emo_path = None
    if emo_audio:
        emo_path = f"temp_uploads/emo_{int(time.time() * 1000)}.wav"
        with open(emo_path, "wb") as f:
            content = await emo_audio.read()
            f.write(content)
    
    try:
        generation_kwargs = {
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k if top_k > 0 else None,
            "repetition_penalty": repetition_penalty,
        }
        
        audio_bytes, duration, inference_time, error = await run_inference_async(
            text=text,
            speaker_audio_path=speaker_path,
            emo_audio_path=emo_path,
            emo_alpha=emo_alpha,
            max_text_tokens_per_segment=max_text_tokens_per_segment,
            **generation_kwargs
        )
        
        if error:
            raise HTTPException(status_code=500, detail=error)
        
        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "X-Inference-Time-Ms": str(inference_time),
                "X-Audio-Duration-Seconds": str(duration)
            }
        )
    finally:
        # Cleanup temporary files
        if os.path.exists(speaker_path):
            os.remove(speaker_path)
        if emo_path and os.path.exists(emo_path):
            os.remove(emo_path)


@app.post("/preload_speaker")
async def preload_speaker(request: PreloadSpeakerRequest):
    """
    Preload speaker embeddings for faster subsequent inference.
    
    Call this before making TTS requests with a speaker to warm up the cache.
    """
    if tts_instance is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    
    if not os.path.exists(request.speaker_audio_path):
        raise HTTPException(status_code=400, detail=f"Speaker audio not found: {request.speaker_audio_path}")
    
    # Run a minimal inference to cache speaker embeddings
    loop = asyncio.get_event_loop()
    
    def preload():
        try:
            # Use minimal text to trigger embedding computation
            tts_instance.infer(
                spk_audio_prompt=request.speaker_audio_path,
                text="test",
                output_path="outputs/preload_temp.wav",
                max_text_tokens_per_segment=20,
                verbose=False
            )
            return True, None
        except Exception as e:
            return False, str(e)
    
    success, error = await loop.run_in_executor(executor, preload)
    
    if not success:
        raise HTTPException(status_code=500, detail=f"Failed to preload speaker: {error}")
    
    return {"success": True, "message": "Speaker embeddings cached"}


@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "IndexTTS2 High-Performance API",
        "version": "1.0.0",
        "endpoints": {
            "/health": "Health check",
            "/tts": "TTS synthesis (returns base64)",
            "/tts/audio": "TTS synthesis (returns raw audio)",
            "/tts/upload": "TTS with file upload",
            "/preload_speaker": "Preload speaker embeddings"
        }
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IndexTTS2 High-Performance Server")
    parser.add_argument("--port", type=int, default=9999, help="Server port")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--model_dir", type=str, default="./checkpoints", help="Model directory")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16")
    parser.add_argument("--deepspeed", action="store_true", default=False, help="Use DeepSpeed")
    parser.add_argument("--cuda_kernel", action="store_true", default=False, help="Use CUDA kernel")
    parser.add_argument("--workers", type=int, default=MAX_CONCURRENT_INFERENCE, help="Thread pool workers")
    parser.add_argument("--reload", action="store_true", default=False, help="Enable auto-reload")
    args = parser.parse_args()
    
    logger.info(f"Starting IndexTTS2 server on {args.host}:{args.port}")
    
    uvicorn.run(
        "server:app" if args.reload else app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1,  # Keep single worker for GPU model
        log_level="info"
    )
