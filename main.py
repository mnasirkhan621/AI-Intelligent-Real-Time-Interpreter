import customtkinter as ctk
import sounddevice as sd
import threading
import asyncio
import logging
import queue
import json
import os
from dotenv import load_dotenv
from translation_engine import TranslationEngine

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

CONFIG_FILE = "config.json"

# Appearance
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class App(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        self.title("Real-Time AI Interpreter (Bi-Directional)")
        self.geometry("700x750")
        
        self.engine = None
        self.engine_thread = None
        self.log_queue = queue.Queue()
        
        # Grid Layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        # Labels & Vars
        self.api_key_groq = ctk.StringVar()
        self.api_key_elevenlabs = ctk.StringVar()
        
        # --- Section 1: API Keys ---
        self.frame_api = ctk.CTkFrame(self)
        self.frame_api.grid(row=0, column=0, padx=20, pady=10, sticky="ew")
        self.frame_api.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(self.frame_api, text="Groq API Key (Llama):").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.entry_groq = ctk.CTkEntry(self.frame_api, textvariable=self.api_key_groq, show="*")
        self.entry_groq.grid(row=0, column=1, padx=10, pady=5, sticky="ew")
        
        ctk.CTkLabel(self.frame_api, text="ElevenLabs Key (Scribe/Turbo):").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.entry_el = ctk.CTkEntry(self.frame_api, textvariable=self.api_key_elevenlabs, show="*")
        self.entry_el.grid(row=1, column=1, padx=10, pady=5, sticky="ew")

        # --- Section 2: Audio Devices (Sender) ---
        self.frame_devices = ctk.CTkFrame(self)
        self.frame_devices.grid(row=1, column=0, padx=20, pady=10, sticky="ew")
        self.frame_devices.grid_columnconfigure(1, weight=1)
        
        # Get Devices
        self.input_devices, self.output_devices = self._get_audio_devices()
        
        self.input_device_id = ctk.IntVar(value=-1)
        self.output_device_id = ctk.IntVar(value=-1)
        self.recv_input_device_id = ctk.IntVar(value=-1)
        self.recv_output_device_id = ctk.IntVar(value=-1)
        
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
        
        # Set defaults (will be overwritten by load_settings if available)
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
        self.frame_controls = ctk.CTkFrame(self)
        self.frame_controls.grid(row=3, column=0, padx=20, pady=10, sticky="ew")
        self.frame_controls.grid_columnconfigure((0,1), weight=1)

        self.btn_start = ctk.CTkButton(self.frame_controls, text="START DUAL TRANSLATION", fg_color="green", height=50, command=self.start_translation)
        self.btn_start.grid(row=0, column=0, padx=(0, 5), pady=10, sticky="ew")
        
        self.btn_stop = ctk.CTkButton(self.frame_controls, text="STOP", fg_color="red", command=self.stop_translation, state="disabled")
        self.btn_stop.grid(row=0, column=1, padx=(5, 0), pady=10, sticky="ew")

        # Progress Bar (VU Meter)
        self.progressbar = ctk.CTkProgressBar(self.frame_controls, orientation="horizontal", mode="determinate")
        self.progressbar.grid(row=1, column=0, columnspan=2, padx=10, pady=(0, 10), sticky="ew")
        self.progressbar.set(0)

        # --- Section 5: Output Area (Chat & Logs) ---
        self.tabview = ctk.CTkTabview(self)
        self.tabview.grid(row=4, column=0, padx=20, pady=10, sticky="nsew")
        
        self.tab_chat = self.tabview.add("Live Chat")
        self.tab_logs = self.tabview.add("Debug Logs")
        
        # Chat Area
        self.tab_chat.grid_columnconfigure(0, weight=1)
        self.tab_chat.grid_rowconfigure(0, weight=1)
        self.chat_frame = ctk.CTkScrollableFrame(self.tab_chat, fg_color="transparent")
        self.chat_frame.grid(row=0, column=0, sticky="nsew")
        self.chat_frame.grid_columnconfigure(0, weight=1)
        
        # Logs Area
        self.tab_logs.grid_columnconfigure(0, weight=1)
        self.tab_logs.grid_rowconfigure(0, weight=1)
        self.textbox_log = ctk.CTkTextbox(self.tab_logs, state="disabled")
        self.textbox_log.grid(row=0, column=0, sticky="nsew")
        
        self.load_settings() # Load saved config
        
        self.after(100, self._process_logs)

    # ... [Device Methods Remain Unchanged] ...

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
        if choice in self.input_devices:
            self.input_device_id.set(self.input_devices[choice])

    def _update_output_id(self, choice):
        if choice in self.output_devices:
            self.output_device_id.set(self.output_devices[choice])
        
    def _update_recv_input_id(self, choice):
        if choice in self.input_devices:
            self.recv_input_device_id.set(self.input_devices[choice])

    def _update_recv_output_id(self, choice):
        if choice in self.output_devices:
            self.recv_output_device_id.set(self.output_devices[choice])
            
    def save_settings(self):
        """Saves current configuration to file."""
        config = {
            "api_key_groq": self.api_key_groq.get(),
            "api_key_elevenlabs": self.api_key_elevenlabs.get(),
            "sender_input": self.opt_input.get(),
            "sender_output": self.opt_output.get(),
            "receiver_input": self.opt_recv_input.get(),
            "receiver_output": self.opt_recv_output.get(),
            "source_lang": self.opt_src_lang.get(),
            "target_lang": self.opt_tgt_lang.get()
        }
        try:
            with open(CONFIG_FILE, "w") as f:
                json.dump(config, f)
            self.log("Settings saved.")
        except Exception as e:
            self.log(f"Error saving settings: {e}")

    def load_settings(self):
        """Loads configuration from file."""
        if not os.path.exists(CONFIG_FILE):
            # Try loading env vars if config doesn't exist
            load_dotenv()
            self.api_key_groq.set(os.getenv("GROQ_API_KEY", ""))
            self.api_key_elevenlabs.set(os.getenv("ELEVENLABS_API_KEY", ""))
            return

        try:
            with open(CONFIG_FILE, "r") as f:
                config = json.load(f)
                
            self.api_key_groq.set(config.get("api_key_groq", ""))
            self.api_key_elevenlabs.set(config.get("api_key_elevenlabs", ""))
            
            # Helper to safely set dropdowns if device exists
            def set_dropdown(opt_menu, key, device_map, callback):
                saved_name = config.get(key)
                if saved_name and saved_name in device_map:
                    opt_menu.set(saved_name)
                    callback(saved_name)
            
            set_dropdown(self.opt_input, "sender_input", self.input_devices, self._update_input_id)
            set_dropdown(self.opt_output, "sender_output", self.output_devices, self._update_output_id)
            set_dropdown(self.opt_recv_input, "receiver_input", self.input_devices, self._update_recv_input_id)
            set_dropdown(self.opt_recv_output, "receiver_output", self.output_devices, self._update_recv_output_id)
            
            self.opt_src_lang.set(config.get("source_lang", "English"))
            self.opt_tgt_lang.set(config.get("target_lang", "Urdu"))
            
            self.log("Settings loaded.")
        except Exception as e:
            self.log(f"Error loading settings: {e}")

    def log(self, message):
        self.log_queue.put(str(message) + "\n")

    def _process_logs(self):
        import re
        while not self.log_queue.empty():
            msg = self.log_queue.get()
            
            # 1. Update Log Console
            self.textbox_log.configure(state="normal")
            self.textbox_log.insert("end", msg)
            self.textbox_log.see("end")
            self.textbox_log.configure(state="disabled")
            
            # 2. Update Chat UI (Parse Translation Logs)
            # Pattern: [SENDER/RECEIVER] Original: ... -> Translated: ...
            match = re.search(r"\[(SENDER|RECEIVER)\] Original: (.*?) -> Translated: (.*)", msg)
            if match:
                role = match.group(1)
                original = match.group(2)
                translated = match.group(3)
                is_sender = (role == "SENDER")
                
                self.add_chat_bubble(original, translated, is_sender)

        self.after(100, self._process_logs)

    def add_chat_bubble(self, original, translated, is_sender):
        # Container Frame
        frame = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        frame.pack(fill="x", pady=5, padx=10)
        
        # Bubble Color & Alignment
        if is_sender:
            color = "#1f6aa5" # Blue-ish for You
            anchor = "e" # East/Right
            align = "right"
            speaker_label = "You"
            col_cfg = (1, 0) # Spacer left, bubble right
        else:
            color = "#2b2b2b" # Dark Gray for Them
            anchor = "w" # West/Left
            align = "left"
            speaker_label = "Them"
            col_cfg = (0, 1) # Bubble left, spacer right
            
        # Grid layout for alignment
        frame.grid_columnconfigure(0, weight=col_cfg[0])
        frame.grid_columnconfigure(1, weight=col_cfg[1])
        
        # Actual Bubble
        bubble = ctk.CTkFrame(frame, fg_color=color, corner_radius=15)
        bubble.grid(row=0, column=0 if not is_sender else 1, sticky=anchor)
        
        # Text
        content = f"{original}\n⬇\n{translated}"
        label = ctk.CTkLabel(bubble, text=content, text_color="white", justify=align, wraplength=400, font=("Arial", 14))
        label.pack(padx=15, pady=10)
        
        # Scroll to bottom
        self.chat_frame._parent_canvas.yview_moveto(1.0)

    def start_translation(self):
        self.save_settings() # Auto-save on start
        
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
        
        # Visualizer Callback
        def update_meter(level):
            self.progressbar.set(level)

        # GLOBAL INTERLOCK (Prevents ANY listening while ANYONE is speaking)
        import threading
        self.speech_interlock = threading.Event()

        # --- Engine 1: You -> Them (Source -> Target) ---
        self.engine = TranslationEngine(
            api_keys=keys,
            input_device=in_dev,
            output_device=out_dev,
            source_lang=self.opt_src_lang.get(),
            target_lang=self.opt_tgt_lang.get(),
            verbose_callback=lambda m: self.log(f"[You]: {m}"),
            volume_callback=update_meter, # Link Visualizer to Sender Mic
            shared_event=self.speech_interlock, # SHARED
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
            volume_callback=None,
            shared_event=self.speech_interlock, # SHARED (If SENDER speaks, RECEIVER deafens)
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
    app = App()
    app.mainloop()
