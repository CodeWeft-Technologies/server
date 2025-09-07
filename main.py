from fastapi import FastAPI, UploadFile, Form
import asyncio
from asr_vosk import VoskASR
from gpt_handler import GPTHandler
from tts_google import google_tts
import tempfile

app = FastAPI(title="Voice Agent API")

gpt_handler = GPTHandler()

@app.post("/chat/")
async def chat(prompt: str = Form(...)):
    response_text = ""
    async for token in gpt_handler.generate_response(prompt):
        response_text += token + " "
    return {"response": response_text.strip()}

@app.post("/text-to-speech/")
async def text_to_speech(text: str = Form(...)):
    audio_data = await google_tts(text)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.write(audio_data)
    temp_file.close()
    return {"audio_file": temp_file.name}

@app.post("/speech-to-text/")
async def speech_to_text(file: UploadFile):
    # Save uploaded file temporarily
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    temp_file.write(await file.read())
    temp_file.close()

    # Use Vosk for ASR
    import wave
    import json
    import vosk

    wf = wave.open(temp_file.name, "rb")
    model = vosk.Model("models/vosk-model-small-en-us-0.15")
    rec = vosk.KaldiRecognizer(model, wf.getframerate())
    result_text = ""

    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            result_text += res.get("text", "") + " "
    # final result
    res = json.loads(rec.FinalResult())
    result_text += res.get("text", "")
    return {"transcript": result_text.strip()}
