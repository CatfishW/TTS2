"""
High-Performance TTS Client for IndexTTS2

Async client with connection pooling, concurrent requests, and retry logic.
Run with: uv run python client.py --text "Hello" --voice examples/prompt1.wav --output out.wav
"""

import os
import sys
import asyncio
import argparse
import base64
import time
import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path

import httpx

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class TTSResult:
    """Result of a TTS synthesis request"""
    success: bool
    audio_bytes: Optional[bytes] = None
    duration_seconds: Optional[float] = None
    inference_time_ms: Optional[float] = None
    error: Optional[str] = None
    output_path: Optional[str] = None


class TTSClient:
    """
    High-performance async TTS client with connection pooling and retry logic.
    
    Features:
    - Connection pooling for efficient HTTP reuse
    - Concurrent request submission
    - Automatic retry with exponential backoff
    - Support for both base64 and raw audio responses
    - Batch processing capabilities
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:9999",
        timeout: float = 120.0,
        max_retries: int = 3,
        max_connections: int = 10
    ):
        """
        Initialize TTS client.
        
        Args:
            base_url: Server base URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            max_connections: Maximum concurrent connections
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Connection pool limits
        limits = httpx.Limits(
            max_keepalive_connections=max_connections,
            max_connections=max_connections * 2
        )
        
        # Async client with connection pooling
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=limits,
            http2=False  # Disable HTTP/2 to avoid missing dependencies
        )
        
        # Sync client for non-async usage
        self._sync_client = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout, connect=10.0),
            limits=limits
        )
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def close(self):
        """Close the client connections"""
        await self._client.aclose()
        self._sync_client.close()
    
    async def health_check(self) -> dict:
        """Check server health status"""
        response = await self._client.get("/health")
        response.raise_for_status()
        return response.json()
    
    async def _retry_request(self, request_func, *args, **kwargs):
        """Execute request with exponential backoff retry"""
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                return await request_func(*args, **kwargs)
            except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError) as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1, 2, 4 seconds
                    logger.warning(f"Request failed (attempt {attempt + 1}/{self.max_retries}), "
                                   f"retrying in {wait_time}s: {e}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"Request failed after {self.max_retries} attempts: {e}")
        
        raise last_error
    
    async def synthesize(
        self,
        text: str,
        speaker_audio_path: str,
        emo_audio_path: Optional[str] = None,
        emo_alpha: float = 1.0,
        max_text_tokens_per_segment: int = 120,
        do_sample: bool = True,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 30,
        repetition_penalty: float = 10.0,
        return_raw_audio: bool = True
    ) -> TTSResult:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to synthesize
            speaker_audio_path: Path to speaker reference audio (on server)
            emo_audio_path: Optional path to emotion reference audio
            emo_alpha: Emotion blending weight (0.0 to 1.0)
            max_text_tokens_per_segment: Max tokens per segment
            do_sample: Whether to use sampling
            temperature: Sampling temperature
            top_p: Top-p sampling
            top_k: Top-k sampling
            repetition_penalty: Repetition penalty
            return_raw_audio: If True, use raw audio endpoint; else use base64
        
        Returns:
            TTSResult with audio data
        """
        payload = {
            "text": text,
            "speaker_audio_path": speaker_audio_path,
            "emo_audio_path": emo_audio_path,
            "emo_alpha": emo_alpha,
            "max_text_tokens_per_segment": max_text_tokens_per_segment,
            "do_sample": do_sample,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty
        }
        
        endpoint = "/tts/audio" if return_raw_audio else "/tts"
        
        async def make_request():
            response = await self._client.post(endpoint, json=payload)
            response.raise_for_status()
            return response
        
        try:
            response = await self._retry_request(make_request)
            
            if return_raw_audio:
                audio_bytes = response.content
                inference_time = float(response.headers.get("X-Inference-Time-Ms", 0))
                duration = float(response.headers.get("X-Audio-Duration-Seconds", 0))
                
                return TTSResult(
                    success=True,
                    audio_bytes=audio_bytes,
                    duration_seconds=duration,
                    inference_time_ms=inference_time
                )
            else:
                data = response.json()
                if data.get("success"):
                    audio_bytes = base64.b64decode(data["audio_base64"])
                    return TTSResult(
                        success=True,
                        audio_bytes=audio_bytes,
                        duration_seconds=data.get("duration_seconds"),
                        inference_time_ms=data.get("inference_time_ms")
                    )
                else:
                    return TTSResult(
                        success=False,
                        error=data.get("error", "Unknown error")
                    )
                    
        except Exception as e:
            return TTSResult(success=False, error=str(e))
    
    async def synthesize_with_upload(
        self,
        text: str,
        speaker_audio_path: str,
        emo_audio_path: Optional[str] = None,
        emo_alpha: float = 1.0,
        max_text_tokens_per_segment: int = 120,
        **generation_kwargs
    ) -> TTSResult:
        """
        Synthesize speech by uploading audio files.
        
        Use this when speaker audio is on the client machine, not server.
        
        Args:
            text: Text to synthesize
            speaker_audio_path: Local path to speaker audio file
            emo_audio_path: Optional local path to emotion audio file
            emo_alpha: Emotion blending weight
            max_text_tokens_per_segment: Max tokens per segment
            **generation_kwargs: Additional generation parameters
        
        Returns:
            TTSResult with audio data
        """
        files = {
            "speaker_audio": ("speaker.wav", open(speaker_audio_path, "rb"), "audio/wav")
        }
        
        if emo_audio_path and os.path.exists(emo_audio_path):
            files["emo_audio"] = ("emo.wav", open(emo_audio_path, "rb"), "audio/wav")
        
        data = {
            "text": text,
            "emo_alpha": str(emo_alpha),
            "max_text_tokens_per_segment": str(max_text_tokens_per_segment),
            "do_sample": str(generation_kwargs.get("do_sample", True)).lower(),
            "temperature": str(generation_kwargs.get("temperature", 0.8)),
            "top_p": str(generation_kwargs.get("top_p", 0.8)),
            "top_k": str(generation_kwargs.get("top_k", 30)),
            "repetition_penalty": str(generation_kwargs.get("repetition_penalty", 10.0)),
        }
        
        try:
            async def make_request():
                response = await self._client.post("/tts/upload", files=files, data=data)
                response.raise_for_status()
                return response
            
            response = await self._retry_request(make_request)
            
            inference_time = float(response.headers.get("X-Inference-Time-Ms", 0))
            duration = float(response.headers.get("X-Audio-Duration-Seconds", 0))
            
            return TTSResult(
                success=True,
                audio_bytes=response.content,
                duration_seconds=duration,
                inference_time_ms=inference_time
            )
            
        except Exception as e:
            return TTSResult(success=False, error=str(e))
        finally:
            for f in files.values():
                f[1].close()
    
    async def preload_speaker(self, speaker_audio_path: str) -> bool:
        """
        Preload speaker embeddings for faster inference.
        
        Args:
            speaker_audio_path: Path to speaker audio on server
        
        Returns:
            True if successful
        """
        try:
            response = await self._client.post(
                "/preload_speaker",
                json={"speaker_audio_path": speaker_audio_path}
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Failed to preload speaker: {e}")
            return False
    
    async def synthesize_batch(
        self,
        requests: List[dict],
        max_concurrent: int = 5
    ) -> List[TTSResult]:
        """
        Process multiple TTS requests concurrently.
        
        Args:
            requests: List of request dictionaries with synthesize() parameters
            max_concurrent: Maximum concurrent requests
        
        Returns:
            List of TTSResult in same order as requests
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def bounded_synthesize(req):
            async with semaphore:
                return await self.synthesize(**req)
        
        tasks = [bounded_synthesize(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Convert exceptions to TTSResult
        processed = []
        for r in results:
            if isinstance(r, Exception):
                processed.append(TTSResult(success=False, error=str(r)))
            else:
                processed.append(r)
        
        return processed
    
    # Synchronous API wrappers
    def synthesize_sync(self, **kwargs) -> TTSResult:
        """Synchronous version of synthesize()"""
        return asyncio.run(self.synthesize(**kwargs))
    
    def health_check_sync(self) -> dict:
        """Synchronous health check"""
        response = self._sync_client.get("/health")
        response.raise_for_status()
        return response.json()


async def run_demo(args):
    """Run demonstration of the client"""
    if args.server_url:
        base_url = args.server_url
    else:
        base_url = f"http://{args.host}:{args.port}"
        
    async with TTSClient(base_url=base_url) as client:
        # Health check
        logger.info("Checking server health...")
        try:
            health = await client.health_check()
            logger.info(f"Server status: {health}")
        except Exception as e:
            logger.error(f"Server not available: {e}")
            return
        
        # Single synthesis
        if args.text and args.voice:
            logger.info(f"Synthesizing: '{args.text[:50]}...' with voice: {args.voice}")
            
            start_time = time.time()
            
            if args.upload:
                result = await client.synthesize_with_upload(
                    text=args.text,
                    speaker_audio_path=args.voice,
                    emo_alpha=args.emo_alpha
                )
            else:
                result = await client.synthesize(
                    text=args.text,
                    speaker_audio_path=args.voice,
                    emo_alpha=args.emo_alpha
                )
            
            total_time = time.time() - start_time
            
            if result.success:
                # Save audio
                output_path = args.output or f"output_{int(time.time())}.wav"
                with open(output_path, "wb") as f:
                    f.write(result.audio_bytes)
                
                logger.info(f"✓ Audio saved to: {output_path}")
                logger.info(f"  Duration: {result.duration_seconds:.2f}s")
                logger.info(f"  Inference time: {result.inference_time_ms:.0f}ms")
                logger.info(f"  Total time: {total_time:.2f}s")
            else:
                logger.error(f"✗ Synthesis failed: {result.error}")
        
        # Concurrent test
        if args.concurrent and args.concurrent > 1:
            logger.info(f"\nRunning concurrent test with {args.concurrent} requests...")
            
            requests = [
                {
                    "text": f"This is concurrent request number {i+1}.",
                    "speaker_audio_path": args.voice
                }
                for i in range(args.concurrent)
            ]
            
            start_time = time.time()
            results = await client.synthesize_batch(requests, max_concurrent=args.concurrent)
            total_time = time.time() - start_time
            
            success_count = sum(1 for r in results if r.success)
            logger.info(f"Completed {success_count}/{len(results)} requests in {total_time:.2f}s")
            
            avg_inference = sum(r.inference_time_ms or 0 for r in results if r.success) / max(success_count, 1)
            logger.info(f"Average inference time: {avg_inference:.0f}ms")


def main():
    parser = argparse.ArgumentParser(
        description="IndexTTS2 High-Performance Client",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument("--port", type=int, default=9999, help="Server port")
    parser.add_argument("--server_url", type=str, help="Full server URL (overrides host/port)")
    parser.add_argument("--text", type=str, help="Text to synthesize")
    parser.add_argument("--voice", type=str, help="Path to speaker reference audio")
    parser.add_argument("--output", "-o", type=str, help="Output audio file path")
    parser.add_argument("--emo_alpha", type=float, default=1.0, help="Emotion weight")
    parser.add_argument("--upload", action="store_true", help="Upload audio instead of using server path")
    parser.add_argument("--concurrent", type=int, help="Number of concurrent test requests")
    
    args = parser.parse_args()
    
    if not args.text and not args.concurrent:
        parser.print_help()
        print("\nExample usage:")
        print("  python client.py --text 'Hello world' --voice examples/prompt1.wav --output hello.wav")
        print("  python client.py --server_url https://example.com/api --text 'Hello'")
        return
    
    asyncio.run(run_demo(args))


if __name__ == "__main__":
    main()
