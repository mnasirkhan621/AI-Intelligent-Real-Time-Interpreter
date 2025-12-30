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
    def __init__(self, api_keys, input_device, output_device, source_lang, target_lang, verbose_callback=None, volume_callback=None, shared_event=None, engine_name="Engine"):
        self.groq_client = AsyncGroq(api_key=api_keys.get("GROQ_API_KEY"))
        # self.deepgram_client = AsyncDeepgramClient(api_key=api_keys.get("DEEPGRAM_API_KEY")) # Kept for reference or removal
        self.elevenlabs_client = AsyncElevenLabs(api_key=api_keys.get("ELEVENLABS_API_KEY"))
        
        self.input_device = input_device
        self.output_device = output_device
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.verbose_callback = verbose_callback
        self.volume_callback = volume_callback # Visualizer Callback
        self.shared_event = shared_event # GLOBAL INTERLOCK
        self.engine_name = engine_name

        self.is_running = False
        self.audio_queue = asyncio.Queue()
        self.output_queue = asyncio.Queue()
        self.stop_event = asyncio.Event()
        self.is_playing_audio = False  # Flag for Half-Duplex (Self-Deafening)
        
        # Audio Settings
        self.samplerate = 16000
        self.channels = 1
        self.chunk_duration = 5.0 
        self.chunk_samples = int(self.samplerate * self.chunk_duration)
        self.silence_threshold = 0.01 
        
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
        """Step A: Capture Audio with VAD (Voice Activity Detection)."""
        loop = asyncio.get_event_loop()
        import webrtcvad
        
        vad = webrtcvad.Vad(3) # Mode 3: Very Aggressive (Filters background noise)
        frame_duration_ms = 30
        frame_samples = int(self.samplerate * frame_duration_ms / 1000) # 480 samples
        
        # VAD State
        triggered = False
        buffer = []
        silence_counter = 0
        SUCCESSIVE_SILENCE_LIMIT = int(1000 / frame_duration_ms) # 1 sec of silence to stop
        
        def callback(indata, frames, time_info, status):
            nonlocal triggered, silence_counter, buffer
            if status:
                pass
            
            # VISUALIZER UPDATE
            if self.volume_callback:
                rms = np.sqrt(np.mean(indata**2))
                # Normalize reasonably (mic input is usually low)
                level = min(rms * 5, 1.0) 
                loop.call_soon_threadsafe(self.volume_callback, level)

            # GLOBAL INTERLOCK: If ANY engine is speaking (shared_event set), DON'T LISTEN.
            if self.shared_event and self.shared_event.is_set():
                triggered = False
                buffer = []
                silence_counter = 0
                return
            
            # SELF-DEAFENING: (Fallback)
            if self.is_playing_audio:
                triggered = False
                buffer = []
                silence_counter = 0
                return

            if self.is_running:
                # Convert float32 -> int16 bytes for WebRTCVAD
                audio_int16 = (indata * 32767).astype(np.int16)
                audio_bytes = audio_int16.tobytes()
                
                try:
                    is_speech = vad.is_speech(audio_bytes, self.samplerate)
                except:
                    is_speech = False
                
                if is_speech:
                    if not triggered:
                        triggered = True
                        if self.verbose_callback:
                            loop.call_soon_threadsafe(self.verbose_callback, f"[{self.engine_name}] Speech Detected...")
                    silence_counter = 0
                    buffer.append(indata.copy())
                else:
                    if triggered:
                        silence_counter += 1
                        buffer.append(indata.copy()) # Keep padding
                        
                        if silence_counter > SUCCESSIVE_SILENCE_LIMIT:
                            triggered = False
                            # Flush buffer
                            if buffer:
                                full_audio = np.concatenate(buffer)
                                loop.call_soon_threadsafe(self.audio_queue.put_nowait, full_audio)
                            buffer = []
                            silence_counter = 0
                            if self.verbose_callback:
                                loop.call_soon_threadsafe(self.verbose_callback, f"[{self.engine_name}] End of Speech. Processing...")

        try:
            logger.info(f"Audio Capture Started on Device: {self.input_device}")
            stream = sd.InputStream(
                device=self.input_device,
                channels=self.channels,
                samplerate=self.samplerate,
                callback=callback,
                blocksize=frame_samples # 30ms frames
            )
            with stream:
                while self.is_running:
                    await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Audio Capture Failed: {e}")
            self.stop_event.set()

    async def _processing_consumer(self):
        """Step B: Process Audio (STT -> LLM -> TTS)."""
        while self.is_running:
            try:
                # Wait for VAD-segmented audio chunk
                if self.audio_queue.empty():
                    await asyncio.sleep(0.05)
                    continue
                
                audio_data = await self.audio_queue.get()
                
                # --- PROCESS AUDIO ---
                try:
                    start_time = time.time()
                    
                    text = await self._transcribe(audio_data)
                    
                    # Robust Filtering
                    ignored_phrases = [
                        ".", "...", "?", "!", "you", "thank you", "subtitles", 
                        "watching", "video", "subscribe", "notification", "copyright"
                    ]
                    
                    clean_text = text.strip().lower() if text else ""
                    if (not text 
                        or len(clean_text) < 2 
                        or clean_text in ignored_phrases
                        or clean_text.startswith("(") 
                    ):
                        continue
                    
                    t_transcribe = time.time()
                    
                    translated_text = await self._translate(text)
                    if not translated_text:
                        continue
                    t_translate = time.time()
                    
                    # Log messages (Show user it's starting)
                    if self.verbose_callback:
                        self.verbose_callback(f"Original: {text} -> Translated: {translated_text}")
                    logger.info(f"[{self.engine_name}] Original: {text} -> Translated: {translated_text}")
                    
                     # --- STREAMING TTS PIPELINE ---
                    tts_start = time.time()
                    first_chunk = True
                    
                    async for chunk in self._text_to_speech(translated_text):
                        await self.output_queue.put(chunk)
                        if first_chunk:
                             first_chunk = False
                             tts_latency = (time.time() - t_translate) * 1000
                             logger.info(f"[{self.engine_name}] TTS First Byte: {int(tts_latency)}ms")
    
                    total_time = (time.time() - start_time) * 1000
                    logger.info(f"[{self.engine_name}] Pipeline Total: {int(total_time)}ms")
                    
                except Exception as inner_e:
                    logger.error(f"Processing Error (Auto-Recovering): {inner_e}")
                    self._log(f"⚠️ Connection Glitch: {inner_e}. Retrying...")
                    await asyncio.sleep(2) # Backoff
                
            except Exception as e:
                logger.error(f"Critical Pipeline Error: {e}")
                await asyncio.sleep(5)

    async def _playback_consumer(self):
        """Consumes PCM audio chunks and plays them via RawOutputStream."""
        import sounddevice as sd
        
        # Open a Persistent Stream for Low Latency
        try:
            stream = sd.RawOutputStream(
                samplerate=16000, 
                channels=1, 
                dtype='int16', 
                device=self.output_device,
                blocksize=1024
            )
            
            with stream:
                while self.is_running:
                    if self.output_queue.empty():
                        await asyncio.sleep(0.05)
                        continue

                    # Mute Microphone (Half-Duplex)
                    self.is_playing_audio = True
                    if self.shared_event:
                        self.shared_event.set() # SIGNAL GLOBAL MUTE
                    
                    try:
                        while not self.output_queue.empty():
                            chunk_bytes = await self.output_queue.get()
                            if chunk_bytes:
                                # Convert raw bytes to numpy Int16 array for sounddevice
                                chunk_np = np.frombuffer(chunk_bytes, dtype=np.int16)
                                stream.write(chunk_np)
                        
                        # Small buffer drain time
                        await asyncio.sleep(0.1)
                        
                    finally:
                        self.is_playing_audio = False
                        if self.shared_event:
                            self.shared_event.clear() # UNMUTE
                    
        except Exception as e:
            logger.error(f"Playback Stream Error: {e}")
            self.is_playing_audio = False
            if self.shared_event:
                self.shared_event.clear()

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
        """Step D: AES (Audio Stream Generation) - PCM 16kHz."""
        try:
            # Use 'pcm_16000' for raw playback without decoding overhead
            audio_stream = self.elevenlabs_client.text_to_speech.convert(
                text=text,
                voice_id="21m00Tcm4TlvDq8ikWAM",
                model_id="eleven_turbo_v2_5",
                output_format="pcm_16000"
            )
            
            async for chunk in audio_stream:
                if chunk:
                    yield chunk

        except Exception as e:
            logger.error(f"TTS Stream failed: {e}")
            yield None
