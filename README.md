# Real-Time AI Interpreter (Bi-Directional)

A low-latency, real-time voice translation app designed for **Zoom, Google Meet, and Microsoft Teams**. It acts as a "Man-in-the-Middle" interpreter, translating your voice to the other person, and their voice back to you.

## ✨ Features
- **Bi-Directional Translation**: Handles two-way conversation simultaneously.
  - **Sender (You -> Them)**: Your Mic -> AI -> Virtual Cable (Zoom Mic).
  - **Receiver (Them -> You)**: Zoom Audio (Stereo Mix) -> AI -> Your Headphones.
- **Top-Tier Voice Quality**: Powered by **ElevenLabs Turbo v2.5** for ultra-realistic speech.
- **Accurate Recognition**: Uses **ElevenLabs Scribe** (v1) for precise speech-to-text.
- **Fast Translation**: Uses **Groq (Llama 3.1)** for near-instant language translation.
- **Noise Filtering**: Robust filtered gates to ignore clicking, breathing, and background noise.

## 🛠 Prerequisites

### 1. Audio Drivers
- **[VB-Cable](https://vb-audio.com/Cable/)**: Required to "inject" your translated voice into Zoom/Meet.
- **Stereo Mix** (Windows): Required to "hear" the other person from Zoom/Meet.
  - *Enable it via: System Sounds -> Recording -> Right Click "Stereo Mix" -> Enable.*

### 2. Python Dependencies
```bash
pip install -r requirements.txt
```

### 3. API Keys
Create a `.env` file in the root directory:
```env
GROQ_API_KEY=gsk_...
ELEVENLABS_API_KEY=sk_...
```
*   **Groq**: Used for Translation (Llama 3.1).
*   **ElevenLabs**: Used for Speech-to-Text and Text-to-Speech.

## 🚀 How to Use

1.  **Run the App**:
    ```bash
    python main.py
    ```

2.  **Configure [SENDER] (Translating YOU)**:
    -   **Input**: select your **Microphone**.
    -   **Output**: select **CABLE Input (VB-Audio Virtual Cable)**.
    -   *In Zoom/Meet settings, select **CABLE Output** as your Microphone.*

3.  **Configure [RECEIVER] (Translating THEM)**:
    -   **Input**: select **Stereo Mix (Realtek)** (or whatever device captures system audio).
    -   **Output**: select your **Headphones** (so you can hear the English translation).
    -   *Refrain from using Speakers to avoid feedback loops.*

4.  **Select Languages**:
    -   **Your Language**: The language you speak (e.g., English).
    -   **Their Language**: The language they speak (e.g., Urdu).

5.  **Click START DUAL TRANSLATION**:
    -   The logs will show `[SENDER]` when processing your voice.
    -   The logs will show `[RECEIVER]` when processing their voice.

## 💡 Troubleshooting
-   **"Invalid number of channels" crash**: You tried to select a Speaker as an Input. Use the "Inputs" dropdowns strictly for Mics/Stereo Mix.
-   **Not picking up voice?**: Speak clearer. The app has a noise gate to ignore background buzz.
-   **Feedback Loop (Echo)**: Ensure the Receiver Output is set to **Headphones**, not the same speakers that Stereo Mix is listening to.
