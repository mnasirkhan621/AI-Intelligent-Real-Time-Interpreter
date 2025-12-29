
# AI Intelligent Real Time Interpreter

A low-latency, real-time bi-directional voice translation app designed for Zoom, Google Meet, and Microsoft Teams.

## Features
- **Real-Time Speech-to-Text (STT)**: Uses Groq (Whisper Large V3).
- **Instant Translation (LLM)**: Uses Groq (Llama 3.1).
- **Text-to-Speech (TTS)**: Uses EdgeTTS for high-speed, multilingual speech generation (Urdu, Hindi, Spanish, etc.).
- **Virtual Audio Routing**: Outputs translated audio to VB-Cable for use as a microphone in meeting apps.
- **Noise Filtering**: Automatically ignores silence and background noise.

## Installation

1.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Setup Virtual Audio Cable**:
    - Install [VB-Cable](https://vb-audio.com/Cable/).
3.  **Configure API Keys**:
    - Create a `.env` file (see `.env.example` or just edit code).
    - Keys needed: `GROQ_API_KEY` (Free), `DEEPGRAM_API_KEY` (Optional, unused in current EdgeTTS version but good to have).

## Usage

```bash
python main.py
```

## How to use in Zoom/Meet
1.  In the App: Select **Output Device** = `CABLE Input`.
2.  In Zoom/Meet: Select **Microphone** = `CABLE Output`.
