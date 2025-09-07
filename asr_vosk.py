import queue
import sounddevice as sd
import vosk
import json

class VoskASR:
    def __init__(self, model_path="models/vosk-model-small-en-us-0.15"):
        self.model = vosk.Model(model_path)
        self.q = queue.Queue()

    def audio_callback(self, indata, frames, time, status):
        self.q.put(bytes(indata))

    def start_listening(self):
        with sd.RawInputStream(samplerate=16000, blocksize=8000, dtype='int16',
                               channels=1, callback=self.audio_callback):
            rec = vosk.KaldiRecognizer(self.model, 16000)
            print("🎙️ Listening...")
            while True:
                data = self.q.get()
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    yield result.get("text", "")
                else:
                    partial = json.loads(rec.PartialResult())
                    yield partial.get("partial", "")
