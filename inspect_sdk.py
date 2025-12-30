
import asyncio
from elevenlabs.client import AsyncElevenLabs
import os
from dotenv import load_dotenv

load_dotenv()

async def inspect():
    client = AsyncElevenLabs(api_key=os.getenv("ELEVENLABS_API_KEY"))
    print("Methods of text_to_speech client:")
    print(dir(client.text_to_speech))
    
    print("\nDocstring of speech_to_text.convert:")
    print(client.speech_to_text.convert.__doc__)

if __name__ == "__main__":
    asyncio.run(inspect())
