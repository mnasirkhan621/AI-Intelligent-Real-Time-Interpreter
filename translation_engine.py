import asyncio
import os
import time
import sounddevice as sd
import numpy as np
import logging
from collections import deque
from groq import AsyncGroq
from deepgram import AsyncDeepgramClient
from elevenlabs.client import AsyncElevenLabs

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TranslationEngine:
    def __init__(self, api_keys, input_device, output_device, source_lang, target_lang, verbose_callback=None, engine_name="Engine"):
        self.groq_client = AsyncGroq(api_key=api_keys.get("GROQ_API_KEY"))
        # self.deepgram_client = AsyncDeepgramClient(api_key=api_keys.get("DEEPGRAM_API_KEY")) # Kept for reference or removal
        self.elevenlabs_client = AsyncElevenLabs(api_key=api_keys.get("ELEVENLABS_API_KEY"))
        
        self.input_device = input_device
        self.output_device = output_device
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.verbose_callback = verbose_callback
        self.engine_name = engine_name

        self.is_running = False
        self.audio_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        self.stop_event = asyncio.Event()
        
        # Audio Settings
        self.samplerate = 16000
        self.channels = 1
        self.chunk_duration = 5.0 # Increased to 5s for better Scribe context
        self.chunk_samples = int(self.samplerate * self.chunk_duration)
        self.silence_threshold = 0.01 # Lowered back to 0.01 (rms) to catch mic input
        
        # ISO-639-1 Mapping for ElevenLabs Scribe
        self.lang_map = {
            "English": "en", "Urdu": "ur", "Hindi": "hi", "Spanish": "es", 
            "Japanese": "ja", "French": "fr", "German": "de", "Chinese": "zh",
            "Arabic": "ar", "Russian": "ru", "Portuguese": "pt", "Italian": "it",
            "Korean": "ko", "Turkish": "tr", "Dutch": "nl"
        }

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
                
                # RMS-based Silence Detection
                rms = np.sqrt(np.mean(audio_data**2))
                if rms < self.silence_threshold:
                    continue

                start_time = time.time()
                
                text = await self._transcribe(audio_data)
                
                # Robust Filtering for Noise/Hallucinations
                ignored_phrases = [
                    ".", "...", "?", "!", "you", "thank you", "subtitles", 
                    "watching", "video", "subscribe", "notification", "copyright"
                ]
                
                clean_text = text.strip().lower() if text else ""
                if (not text 
                    or len(clean_text) < 3 
                    or clean_text in ignored_phrases
                    or clean_text.startswith("(")  # (Music), (Applause)
                ):
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
                
                # Calculate individual component times
                stt_time = (t_transcribe - start_time) * 1000
                llm_time = (t_translate - t_transcribe) * 1000
                tts_time = (t_tts - t_translate) * 1000
                total_time = (t_end - start_time) * 1000

                # Log messages with engine name prefix
                if self.verbose_callback:
                    self.verbose_callback(f"Original: {text} -> Translated: {translated_text}")
                logger.info(f"[{self.engine_name}] Original: {text} -> Translated: {translated_text}")
                logger.info(f"[{self.engine_name}] Pipeline: {int(total_time)}ms | STT: {int(stt_time)}ms | LLM: {int(llm_time)}ms | TTS: {int(tts_time)}ms")
                
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
            
            # Resolve language code
            lang_code = self.lang_map.get(self.source_lang, "en")
            
            # Use ElevenLabs Scribe (Speech to Text)
            # Model: scribe_v1
            # Optimization: Tag Audio Events False to remove (traffic noises)
            transcript = await self.elevenlabs_client.speech_to_text.convert(
                file=buffer,
                model_id="scribe_v1",
                language_code=lang_code,
                tag_audio_events=False
            )
            
            # Ensure we extract text properly (transcript is likely an object)
            return transcript.text.strip()
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
        """Step D: Generate Audio using ElevenLabs (Production Quality)."""
        try:
            # Voice ID: "Rachel" (21m00Tcm4TlvDq8ikWAM) - generic pleasant voice
            # Model: "eleven_turbo_v2_5" - Fastest, Multilingual, Low Latency
            
            # Using streaming to reduce TTFB (Time To First Byte)
            # stream() returns an async generator, so we iterate over it directly. Do not await the call itself.
            audio_stream = self.elevenlabs_client.text_to_speech.stream(
                text=text,
                voice_id="21m00Tcm4TlvDq8ikWAM",
                model_id="eleven_turbo_v2_5",
                output_format="mp3_44100_128"
            )
            
            import io
            audio_buffer = io.BytesIO()
            
            # Use async generator if stream=True (default in async client typically returns generator? check SDK)
            # Recent ElevenLabs SDK: client.convert returns generator (or async generator for async client)
            
            async for chunk in audio_stream:
                if chunk:
                    audio_buffer.write(chunk)
            
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



