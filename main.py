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
        self.geometry("700x600")
        
        self.engine = None
        self.engine_thread = None
        self.log_queue = queue.Queue()
        
        # Layout Config
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        # Labels & Vars
        self.api_key_groq = ctk.StringVar(value=os.getenv("GROQ_API_KEY", ""))
        self.api_key_deepgram = ctk.StringVar(value=os.getenv("DEEPGRAM_API_KEY", ""))
        
        # --- Section 1: API Keys ---
        self.frame_api = ctk.CTkFrame(self)
        self.frame_api.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        self.frame_api.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(self.frame_api, text="Groq API Key (STT/LLM):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.entry_groq = ctk.CTkEntry(self.frame_api, textvariable=self.api_key_groq, show="*")
        self.entry_groq.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        ctk.CTkLabel(self.frame_api, text="Deepgram API Key (TTS):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.entry_dg = ctk.CTkEntry(self.frame_api, textvariable=self.api_key_deepgram, show="*")
        self.entry_dg.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        # --- Section 2: Audio Devices ---
        self.frame_devices = ctk.CTkFrame(self)
        self.frame_devices.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.frame_devices.grid_columnconfigure(1, weight=1)
        
        # Get Devices
        self.device_list = self._get_audio_devices()
        self.input_device_id = ctk.IntVar()
        self.output_device_id = ctk.IntVar()
        
        ctk.CTkLabel(self.frame_devices, text="Input Device (Mic):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.opt_input = ctk.CTkOptionMenu(self.frame_devices, values=list(self.device_list.keys()), command=self._update_input_id)
        self.opt_input.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        ctk.CTkLabel(self.frame_devices, text="Output Device (Virt Cable):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.opt_output = ctk.CTkOptionMenu(self.frame_devices, values=list(self.device_list.keys()), command=self._update_output_id)
        self.opt_output.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        
        # Set defaults if possible
        if self.device_list:
            self.opt_input.set(list(self.device_list.keys())[0])
            self.opt_output.set(list(self.device_list.keys())[0])
            self._update_input_id(self.opt_input.get())
            self._update_output_id(self.opt_output.get())

        # --- Section 3: Languages ---
        self.frame_lang = ctk.CTkFrame(self)
        self.frame_lang.grid(row=2, column=0, padx=20, pady=10, sticky="ew")
        self.frame_lang.grid_columnconfigure((0,1,2,3), weight=1)
        
        self.languages = [
            "English", "Urdu", "Hindi", "Spanish", "Japanese", "French", 
            "German", "Chinese", "Arabic", "Russian", "Portuguese", 
            "Italian", "Korean", "Turkish", "Dutch"
        ]
        
        ctk.CTkLabel(self.frame_lang, text="Source Language:").grid(row=0, column=0, padx=10, pady=5)
        self.opt_src_lang = ctk.CTkOptionMenu(self.frame_lang, values=self.languages)
        self.opt_src_lang.grid(row=0, column=1, padx=10, pady=5)
        self.opt_src_lang.set("English")
        
        ctk.CTkLabel(self.frame_lang, text="Target Language:").grid(row=0, column=2, padx=10, pady=5)
        self.opt_tgt_lang = ctk.CTkOptionMenu(self.frame_lang, values=self.languages)
        self.opt_tgt_lang.grid(row=0, column=3, padx=10, pady=5)
        self.opt_tgt_lang.set("Urdu")

        # --- Section 4: Controls ---
        self.btn_start = ctk.CTkButton(self, text="START TRANSLATOR", fg_color="green", height=50, command=self.start_translation)
        self.btn_start.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        
        self.btn_stop = ctk.CTkButton(self, text="STOP", fg_color="red", command=self.stop_translation, state="disabled")
        self.btn_stop.grid(row=3, column=0, padx=20, pady=10, sticky="e") # Overlay or change layout?
        # Let's put Stop next to Start
        self.btn_start.configure(width=320)
        self.btn_start.grid_configure(column=0, sticky="w")
        
        self.btn_stop.configure(width=320)
        self.btn_stop.grid_configure(column=0, sticky="e")

        # --- Section 5: Logs ---
        self.textbox_log = ctk.CTkTextbox(self, state="disabled")
        self.textbox_log.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")
        
        # Start Log Updater
        self.after(100, self._process_logs)

    def _get_audio_devices(self):
        devices = {}
        try:
            device_info = sd.query_devices()
            for i, d in enumerate(device_info):
                # Filter for useful devices (inputs > 0 or outputs > 0)
                # But we need both lists mixed or separate?
                # For simplicity, list all.
                name = f"{i}: {d['name']} ({'In' if d['max_input_channels']>0 else 'Out'})"
                devices[name] = i
        except Exception as e:
            print(f"Error querying devices: {e}")
            devices["No Devices Found"] = -1
        return devices

    def _update_input_id(self, choice):
        self.input_device_id.set(self.device_list[choice])

    def _update_output_id(self, choice):
        self.output_device_id.set(self.device_list[choice])

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
        dg_key = self.api_key_deepgram.get().strip()
        
        if not groq_key or not dg_key:
            self.log("ERROR: API Keys are required!")
            return
            
        keys = {"GROQ_API_KEY": groq_key, "DEEPGRAM_API_KEY": dg_key}
        
        in_dev = self.input_device_id.get()
        out_dev = self.output_device_id.get()
        
        if in_dev == -1 or out_dev == -1:
             self.log("ERROR: Select valid Audio Devices.")
             return

        self.btn_start.configure(state="disabled", fg_color="gray")
        self.btn_stop.configure(state="normal")
        self.entry_groq.configure(state="disabled")
        self.entry_dg.configure(state="disabled")
        
        self.engine = TranslationEngine(
            api_keys=keys,
            input_device=in_dev,
            output_device=out_dev,
            source_lang=self.opt_src_lang.get(),
            target_lang=self.opt_tgt_lang.get(),
            verbose_callback=self.log
        )
        
        # Run async engine in a separate thread
        self.engine_thread = threading.Thread(target=self._run_engine_async, daemon=True)
        self.engine_thread.start()

    def _run_engine_async(self):
        asyncio.run(self.engine.start())

    def stop_translation(self):
        if self.engine:
            self.engine.stop()
            self.log("Stopping...")
            
        self.btn_start.configure(state="normal", fg_color="green")
        self.btn_stop.configure(state="disabled")
        self.entry_groq.configure(state="normal")
        self.entry_dg.configure(state="normal")

if __name__ == "__main__":
    app = TranslatorApp()
    app.mainloop()
