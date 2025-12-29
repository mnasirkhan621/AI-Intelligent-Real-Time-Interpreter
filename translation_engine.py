import asyncio
import os
import time
import sounddevice as sd
import numpy as np
import logging
from collections import deque
from groq import AsyncGroq
from deepgram import AsyncDeepgramClient

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranslationEngine:
    def __init__(self, api_keys, input_device, output_device, source_lang, target_lang, verbose_callback=None):
        self.groq_client = AsyncGroq(api_key=api_keys.get("GROQ_API_KEY"))
        self.deepgram_client = AsyncDeepgramClient(api_key=api_keys.get("DEEPGRAM_API_KEY"))
        
        self.input_device = input_device
        self.output_device = output_device
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.verbose_callback = verbose_callback

        self.is_running = False
        self.audio_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        
        # Audio Settings
        self.samplerate = 16000
        self.channels = 1
        self.chunk_duration = 3.0 # Seconds
        self.chunk_samples = int(self.samplerate * self.chunk_duration)
        self.silence_threshold = 0.01 # Adjustable threshold

    async def start(self):
        """Starts the translation engine pipeline."""
        self.is_running = True
        self.loop = asyncio.get_running_loop()
        logger.info(f"Starting Engine: Input={self.input_device}, Output={self.output_device}")
        
        # Start Producer, Processor, and Player
        producer_task = asyncio.create_task(self._audio_producer())
        processor_task = asyncio.create_task(self._processing_consumer())
        player_task = asyncio.create_task(self._playback_consumer())
        
        await asyncio.gather(producer_task, processor_task, player_task)

    def stop(self):
        """Stops the engine."""
        self.is_running = False
        logger.info("Stopping Engine...")

    def _log(self, message):
        if self.verbose_callback:
            self.verbose_callback(message)
        logger.info(message)

    async def _audio_producer(self):
        """Continuously captures audio and pushes to queue."""
        def callback(indata, frames, time, status):
            if status:
                logger.warning(status)
            if self.is_running:
                self.loop.call_soon_threadsafe(self.audio_queue.put_nowait, indata.copy())

        stream = sd.InputStream(
            device=self.input_device,
            channels=self.channels,
            samplerate=self.samplerate,
            callback=callback,
            blocksize=self.chunk_samples
        )
        
        with stream:
            self._log("Audio Capture Started")
            while self.is_running:
                await asyncio.sleep(0.1)

    async def _processing_consumer(self):
        """Consumes audio chunks, processes them, and pushes result to output queue."""
        while self.is_running:
            try:
                audio_data = await self.audio_queue.get()
                
                volume_norm = np.linalg.norm(audio_data) * 10
                if volume_norm < self.silence_threshold:
                    continue

                start_time = time.time()
                
                text = await self._transcribe(audio_data)
                
                # Filter out empty or noise (e.g., "...", ".")
                if not text or len(text.strip()) < 2 or text.strip() in [".", "...", "?", "!"]:
                    continue

                t_transcribe = time.time()
                
                translated_text = await self._translate(text)
                if not translated_text:
                    continue
                t_translate = time.time()

                audio_bytes = await self._text_to_speech(translated_text)
                if not audio_bytes:
                    continue
                t_tts = time.time()

                # Push to Output Queue instead of playing directly
                await self.output_queue.put(audio_bytes)
                
                t_end = time.time()
                latency_ms = (t_end - start_time) * 1000
                stats = (f"Pipeline: {latency_ms:.0f}ms | "
                         f"STT: {(t_transcribe - start_time)*1000:.0f}ms | "
                         f"LLM: {(t_translate - t_transcribe)*1000:.0f}ms | "
                         f"TTS: {(t_tts - t_translate)*1000:.0f}ms")
                self._log(f"Original: {text} -> Translated: {translated_text}")
                self._log(stats)
                
            except Exception as e:
                logger.error(f"Error in processing pipeline: {e}")
                self._log(f"Error: {e}")

    async def _playback_consumer(self):
        """Consumes generated audio chunks and plays them sequentially."""
        while self.is_running:
            try:
                audio_bytes = await self.output_queue.get()
                # Run playback in a thread to avoid blocking the event loop
                await asyncio.to_thread(self._play_audio, audio_bytes)
            except Exception as e:
                logger.error(f"Error in playback: {e}")

    async def _transcribe(self, audio_data):
        """Step B: Transcribe audio using Groq Whisper."""
        try:
            import io
            import soundfile as sf
            
            buffer = io.BytesIO()
            sf.write(buffer, audio_data, self.samplerate, format='WAV')
            buffer.name = 'audio.wav' 
            buffer.seek(0)
            
            # Determine model based on language
            # 'distil-whisper-large-v3-en' is decommissioned. Using 'whisper-large-v3' for all.
            model_id = "whisper-large-v3"
            
            transcription = await self.groq_client.audio.transcriptions.create(
                file=(buffer.name, buffer.read()),
                model=model_id,
                prompt=f"The audio is in {self.source_lang}", 
                response_format="json",
                language=None 
            )
            return transcription.text.strip()
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return None

    async def _translate(self, text):
        """Step C: Translate text using Groq Llama 3."""
        try:
            chat_completion = await self.groq_client.chat.completions.create(
                messages=[
                    {"role": "system", "content": "You are a professional translator. Output only JSON: {\"translation\": \"...\"}"}, 
                    {"role": "user", "content": f"Translate to {self.target_lang}: {text}"}
                ],
                model="llama-3.1-8b-instant",
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            import json
            content = chat_completion.choices[0].message.content
            data = json.loads(content)
            return data.get("translation", "")
        except Exception as e:
            logger.error(f"Translation failed: {e}")
            return text 

    async def _text_to_speech(self, text):
        """Step D: Generate Audio using Edge TTS (Multilingual & Fast)."""
        try:
            import edge_tts
            
            # Select Voice based on Target Language
            # Full list: https://gist.github.com/jisuk/7e5f03761F34101F4106
            lang = self.target_lang.lower()
            voice = "en-US-AriaNeural" # Default
            
            if "urdu" in lang:
                voice = "ur-PK-UzmaNeural"
            elif "hindi" in lang:
                voice = "hi-IN-SwaraNeural"
            elif "spanish" in lang:
                voice = "es-ES-ElviraNeural"
            elif "french" in lang:
                voice = "fr-FR-DeniseNeural"
            elif "japanese" in lang:
                voice = "ja-JP-NanamiNeural"
            # Add more mappings as needed
            
            communicate = edge_tts.Communicate(text, voice)
            
            import io
            audio_buffer = io.BytesIO()
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    audio_buffer.write(chunk["data"])
                
            audio_buffer.seek(0)
            return audio_buffer.read()
        except Exception as e:
            logger.error(f"TTS failed: {e}")
            return None

    def _play_audio(self, audio_data):
        """Step E: Stream audio to Virtual Cable."""
        try:
            import soundfile as sf
            import io
            
            with io.BytesIO(audio_data) as f:
                data, fs = sf.read(f)
                data = data.astype(np.float32)
                
                if self.output_device is not None:
                     # blocking=True is fine here because we are in a separate Thread (via to_thread)
                     sd.play(data, samplerate=fs, device=self.output_device, blocking=True)
                else:
                    logger.warning("No output device selected.")
        except Exception as e:
            logger.error(f"Playback failed: {e}")

