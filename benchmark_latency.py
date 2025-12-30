import asyncio
import time
import logging
import os
import json
import numpy as np
import soundfile as sf
from dotenv import load_dotenv
from elevenlabs.client import AsyncElevenLabs
from groq import AsyncGroq

# Setup
load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Benchmark")

ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

async def benchmark_pipeline():
    """Simulates the pipeline flow and measures latency of each component."""
    
    logger.info("--- STARTING LATENCY BENCHMARK ---")

    # 1. Simulating Audio Capture (Pre-recorded dummy)
    # Generate 3 seconds of "silence/noise" just to have a buffer, 
    # but for Scribe to work we actually need speech. 
    # Since we can't speak into CI, we will use a text input to test TTS/LLM parts specifically
    # or rely on a real test file if present. assuming 'test_audio.wav' does not exist,
    # we will focus on LLM + TTS Latency (The output side).
    
    text_input = "Hello, how are you doing today? I am testing the system latency."
    target_lang = "Urdu"
    
    # --- LLM BENCHMARK ---
    logger.info(f"Step 1: LLM Translation ({text_input} -> {target_lang})")
    groq = AsyncGroq(api_key=GROQ_API_KEY)
    
    t0 = time.time()
    chat = await groq.chat.completions.create(
        messages=[
            {"role": "system", "content": "Output JSON: {\"translation\": \"...\"}"}, 
            {"role": "user", "content": f"Translate to {target_lang}: {text_input}"}
        ],
        model="llama-3.1-8b-instant",
        response_format={"type": "json_object"}
    )
    t1 = time.time()
    content = json.loads(chat.choices[0].message.content)
    translated_text = content.get("translation")
    logger.info(f"✅ LLM Latency: {(t1-t0)*1000:.2f}ms")
    logger.info(f"Translation: {translated_text}")
    
    # --- TTS BENCHMARK (Current Buffer Approach) ---
    logger.info(f"Step 2: TTS Generation (Buffered) - ElevenLabs Turbo v2.5")
    el_client = AsyncElevenLabs(api_key=ELEVENLABS_API_KEY)
    
    t2 = time.time()
    audio_stream = await el_client.text_to_speech.convert(
        text=translated_text,
        voice_id="21m00Tcm4TlvDq8ikWAM",
        model_id="eleven_turbo_v2_5",
        output_format="mp3_44100_128"
    )
    # Convert returns bytes directly (non-streaming simulation)
    first_byte_time = time.time()
    total_bytes = len(audio_stream)
        
    t3 = time.time()
    
    ttfb = (first_byte_time - t2) * 1000 if first_byte_time else 0
    total_dur = (t3 - t2) * 1000
    
    logger.info(f"✅ TTS Time-To-First-Byte (TTFB): {ttfb:.2f}ms")
    logger.info(f"✅ TTS Total Download Time: {total_dur:.2f}ms")
    logger.info(f"Total Bytes: {total_bytes}")
    
    logger.info("--- BENCHMARK COMPLETE ---")

if __name__ == "__main__":
    asyncio.run(benchmark_pipeline())
