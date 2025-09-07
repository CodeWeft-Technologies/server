import os
import requests
from pathlib import Path

MODEL_URL = "https://gpt4all.io/models/gpt4all-lora-quantized.bin"
MODEL_PATH = Path("models/gpt4all-lora-quantized.bin")
MODEL_PATH.parent.mkdir(exist_ok=True, parents=True)

def download_model():
    if not MODEL_PATH.exists():
        print(f"Downloading GPT4All model from {MODEL_URL}...")
        response = requests.get(MODEL_URL, stream=True)
        with open(MODEL_PATH, "wb") as f:
            total = 0
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                total += len(chunk)
                print(f"\rDownloaded {total / 1024 / 1024:.2f} MB", end="")
        print("\nDownload complete!")
    else:
        print("GPT4All model already exists.")
