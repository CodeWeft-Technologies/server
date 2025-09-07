import os
import base64
import httpx
from dotenv import load_dotenv

load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
TTS_URL = f"https://texttospeech.googleapis.com/v1/text:synthesize?key={GOOGLE_API_KEY}"

async def google_tts(text: str):
    payload = {
        "input": {"text": text},
        "voice": {"languageCode": "en-US", "name": "en-US-Standard-B", "ssmlGender": "MALE"},
        "audioConfig": {"audioEncoding": "LINEAR16"}
    }
    async with httpx.AsyncClient() as client:
        r = await client.post(TTS_URL, json=payload)
        data = r.json()
        audio = base64.b64decode(data['audioContent'])
        return audio
