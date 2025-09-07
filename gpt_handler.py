import asyncio
from download_model import download_model
from pathlib import Path
from gpt4all import GPT4All

MODEL_PATH = Path("models/gpt4all-lora-quantized.bin")
download_model()

class GPTHandler:
    def __init__(self, model_path=MODEL_PATH):
        self.model = GPT4All(str(model_path))

    async def generate_response(self, prompt: str):
        # streaming tokens
        with self.model.chat_session() as session:
            tokens = self.model.generate(prompt, max_tokens=512)
            for token in tokens.split():
                await asyncio.sleep(0.01)  # simulate streaming
                yield token
