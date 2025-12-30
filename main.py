import customtkinter as ctk
import sounddevice as sd
import threading
import asyncio
import os
import queue
from dotenv import load_dotenv
from translation_engine import TranslationEngine

# Load Env
load_dotenv()

# Appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class TranslatorApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Real-Time AI Translator (Zoom/Meet/Bi-directional)")
        self.geometry("700x750")
        
        self.engine = None
        self.engine_thread = None
        self.log_queue = queue.Queue()
        
        # Layout Config
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        # Labels & Vars
        self.api_key_groq = ctk.StringVar(value=os.getenv("GROQ_API_KEY", ""))
        self.api_key_elevenlabs = ctk.StringVar(value=os.getenv("ELEVENLABS_API_KEY", ""))
        
        # --- Section 1: API Keys ---
        self.frame_api = ctk.CTkFrame(self)
        self.frame_api.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        self.frame_api.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(self.frame_api, text="Groq API Key (STT/LLM):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.entry_groq = ctk.CTkEntry(self.frame_api, textvariable=self.api_key_groq, show="*")
        self.entry_groq.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        ctk.CTkLabel(self.frame_api, text="ElevenLabs API Key (TTS):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.entry_el = ctk.CTkEntry(self.frame_api, textvariable=self.api_key_elevenlabs, show="*")
        self.entry_el.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        # --- Section 2: Audio Devices (Sender) ---
        self.frame_devices = ctk.CTkFrame(self)
        self.frame_devices.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.frame_devices.grid_columnconfigure(1, weight=1)
        
        # Get Devices
        self.input_devices, self.output_devices = self._get_audio_devices()
        
        self.input_device_id = ctk.IntVar()
        self.output_device_id = ctk.IntVar()
        self.recv_input_device_id = ctk.IntVar()
        self.recv_output_device_id = ctk.IntVar()
        
        # --- Sender Config (You -> Them) ---
        ctk.CTkLabel(self.frame_devices, text="[SENDER] Your Mic:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.opt_input = ctk.CTkOptionMenu(self.frame_devices, values=list(self.input_devices.keys()), command=self._update_input_id)
        self.opt_input.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        ctk.CTkLabel(self.frame_devices, text="[SENDER] Output (Virt Cable):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.opt_output = ctk.CTkOptionMenu(self.frame_devices, values=list(self.output_devices.keys()), command=self._update_output_id)
        self.opt_output.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        # --- Receiver Config (Them -> You) ---
        ctk.CTkLabel(self.frame_devices, text="[RECEIVER] Their Audio (Stereo Mix):").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.opt_recv_input = ctk.CTkOptionMenu(self.frame_devices, values=list(self.input_devices.keys()), command=self._update_recv_input_id)
        self.opt_recv_input.grid(row=2, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(self.frame_devices, text="[RECEIVER] Play to (Headphones):").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        self.opt_recv_output = ctk.CTkOptionMenu(self.frame_devices, values=list(self.output_devices.keys()), command=self._update_recv_output_id)
        self.opt_recv_output.grid(row=3, column=1, padx=10, pady=5, sticky="ew")
        
        # Set defaults
        if self.input_devices:
            in_key = list(self.input_devices.keys())[0]
            self.opt_input.set(in_key)
            self.opt_recv_input.set(in_key)
            self._update_input_id(in_key)
            self._update_recv_input_id(in_key)
            
        if self.output_devices:
            out_key = list(self.output_devices.keys())[0]
            self.opt_output.set(out_key)
            self.opt_recv_output.set(out_key)
            self._update_output_id(out_key)
            self._update_recv_output_id(out_key)

        # --- Section 3: Languages ---
        self.frame_lang = ctk.CTkFrame(self)
        self.frame_lang.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.frame_lang.grid_columnconfigure((0,1,2,3), weight=1)
        
        self.languages = [
            "English", "Urdu", "Hindi", "Spanish", "Japanese", "French", 
            "German", "Chinese", "Arabic", "Russian", "Portuguese", 
            "Italian", "Korean", "Turkish", "Dutch"
        ]
        
        ctk.CTkLabel(self.frame_lang, text="Your Language:").grid(row=0, column=0, padx=10, pady=5)
        self.opt_src_lang = ctk.CTkOptionMenu(self.frame_lang, values=self.languages)
        self.opt_src_lang.grid(row=0, column=1, padx=10, pady=5)
        self.opt_src_lang.set("English")
        
        ctk.CTkLabel(self.frame_lang, text="Their Language:").grid(row=0, column=2, padx=10, pady=5)
        self.opt_tgt_lang = ctk.CTkOptionMenu(self.frame_lang, values=self.languages)
        self.opt_tgt_lang.grid(row=0, column=3, padx=10, pady=5)
        self.opt_tgt_lang.set("Urdu")
        
        # --- Section 4: Controls ---
        self.btn_start = ctk.CTkButton(self, text="START DUAL TRANSLATION", fg_color="green", height=50, command=self.start_translation)
        self.btn_start.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        self.btn_stop = ctk.CTkButton(self, text="STOP", fg_color="red", command=self.stop_translation, state="disabled")
        self.btn_stop.grid(row=3, column=0, padx=20, pady=10, sticky="e") 
        self.btn_start.configure(width=320)
        self.btn_start.grid_configure(column=0, sticky="w")
        self.btn_stop.configure(width=320)
        self.btn_stop.grid_configure(column=0, sticky="e")

        # --- Section 5: Logs ---
        self.textbox_log = ctk.CTkTextbox(self, state="disabled")
        self.textbox_log.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")
        
        self.after(100, self._process_logs)

    def _get_audio_devices(self):
        inputs = {}
        outputs = {}
        try:
            device_info = sd.query_devices()
            for i, d in enumerate(device_info):
                name = f"{i}: {d['name']}"
                if d['max_input_channels'] > 0:
                    inputs[name] = i
                if d['max_output_channels'] > 0:
                    outputs[name] = i
        except Exception as e:
            print(f"Error querying devices: {e}")
        return inputs, outputs

    def _update_input_id(self, choice):
        self.input_device_id.set(self.input_devices[choice])

    def _update_output_id(self, choice):
        self.output_device_id.set(self.output_devices[choice])
        
    def _update_recv_input_id(self, choice):
        self.recv_input_device_id.set(self.input_devices[choice])

    def _update_recv_output_id(self, choice):
        self.recv_output_device_id.set(self.output_devices[choice])

    def log(self, message):
        self.log_queue.put(str(message) + "\n")

    def _process_logs(self):
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            self.textbox_log.configure(state="normal")
            self.textbox_log.insert("end", msg)
            self.textbox_log.see("end")
            self.textbox_log.configure(state="disabled")
        self.after(100, self._process_logs)

    def start_translation(self):
        groq_key = self.api_key_groq.get().strip()
        el_key = self.api_key_elevenlabs.get().strip()
        
        if not groq_key or not el_key:
            self.log("ERROR: API Keys are required!")
            return
            
        keys = {"GROQ_API_KEY": groq_key, "ELEVENLABS_API_KEY": el_key}
        
        # Sender Config
        in_dev = self.input_device_id.get()
        out_dev = self.output_device_id.get()
        
        # Receiver Config
        recv_in = self.recv_input_device_id.get()
        recv_out = self.recv_output_device_id.get()
        
        if in_dev == -1 or out_dev == -1:
             self.log("ERROR: Select valid Sender Devices.")
             return
        if recv_in == -1 or recv_out == -1:
             self.log("ERROR: Select valid Receiver Devices.")
             return

        self.btn_start.configure(state="disabled", fg_color="gray")
        self.btn_stop.configure(state="normal")
        self.entry_groq.configure(state="disabled")
        self.entry_el.configure(state="disabled")
        
        # --- Engine 1: You -> Them (Source -> Target) ---
        self.engine = TranslationEngine(
            api_keys=keys,
            input_device=in_dev,
            output_device=out_dev,
            source_lang=self.opt_src_lang.get(),
            target_lang=self.opt_tgt_lang.get(),
            verbose_callback=lambda m: self.log(f"[You]: {m}"),
            engine_name="SENDER"
        )
        
        # --- Engine 2: Them -> You (Target -> Source) ---
        # Note: Reverse languages
        self.receiver_engine = TranslationEngine(
            api_keys=keys,
            input_device=recv_in,
            output_device=recv_out,
            source_lang=self.opt_tgt_lang.get(), # Input is Their Language
            target_lang=self.opt_src_lang.get(), # Output is Your Language
            verbose_callback=lambda m: self.log(f"[Them]: {m}"),
            engine_name="RECEIVER"
        )
        
        # Run async engines in separate threads
        self.engine_thread = threading.Thread(target=self._run_async_wrapper, args=(self.engine,), daemon=True)
        self.receiver_thread = threading.Thread(target=self._run_async_wrapper, args=(self.receiver_engine,), daemon=True)
        
        self.engine_thread.start()
        self.receiver_thread.start()

    def _run_async_wrapper(self, engine_instance):
        asyncio.run(engine_instance.start())

    def stop_translation(self):
        if self.engine:
            self.engine.stop()
        if hasattr(self, 'receiver_engine') and self.receiver_engine:
            self.receiver_engine.stop()
            
        self.log("Stopping All Engines...")
            
        self.btn_start.configure(state="normal", fg_color="green")
        self.btn_stop.configure(state="disabled")
        self.entry_groq.configure(state="normal")
        self.entry_el.configure(state="normal")

if __name__ == "__main__":
    app = TranslatorApp()
    app.mainloop()
