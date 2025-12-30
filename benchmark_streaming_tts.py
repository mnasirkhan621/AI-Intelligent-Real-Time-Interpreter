import asyncio
import time
import logging
import os
import numpy as np
from dotenv import load_dotenv
from elevenlabs.client import AsyncElevenLabs

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("StreamingBenchmark")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

async def benchmark_streaming_tts():
    logger.info("--- ⚡ BENCHMARKING STREAMING TTS LATENCY ⚡ ---")
    
    text_input = "This is a test of the ultra low latency streaming capability of the system."
    voice_id = "21m00Tcm4TlvDq8ikWAM"
    model_id = "eleven_turbo_v2_5"
    
    el_client = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)
    
    logger.info(f"Target Text: '{text_input}'")
    logger.info(f"Model: {model_id}")
    
    t_start = time.time()
    
    # Request PCM 16000 Stream (No await)
    audio_stream = el_client.text_to_speech.convert(
        text=text_input,
        voice_id=voice_id,
        model_id=model_id,
        output_format="pcm_16000"
    )
    
    first_chunk = True
    chunk_count = 0
    total_bytes = 0
    ttfb = 0
    
    logger.info("request sent... waiting for first byte...")
    
    async for chunk in audio_stream:
        if first_chunk:
            t_first = time.time()
            ttfb = (t_first - t_start) * 1000
            logger.info(f"✅ TTFB (Time To First Byte): {ttfb:.2f}ms")
            first_chunk = False
            
        chunk_count += 1
        total_bytes += len(chunk)
        
    t_end = time.time()
    total_dur = (t_end - t_start) * 1000
    
    logger.info(f"✅ Total Download Time: {total_dur:.2f}ms")
    logger.info(f"Total Chunks: {chunk_count}")
    logger.info(f"Total Bytes: {total_bytes}")
    
    if ttfb < 600:
        logger.info("\n🏆 RESULT: EXCELLENT LATENCY (< 600ms)")
    elif ttfb < 1000:
        logger.info("\n🥇 RESULT: GOOD LATENCY (< 1000ms)")
    else:
        logger.info("\n⚠️ RESULT: HIGH LATENCY (> 1000ms)")

if __name__ == "__main__":
    asyncio.run(benchmark_streaming_tts())
