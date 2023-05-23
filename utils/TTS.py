import os
import torch
import requests
import urllib.parse
from utils.katakana import katakana_converter

def silero_tts(tts, language, model, speaker):
    device = torch.device('cpu')
    torch.set_num_threads(4)
    local_file = 'model.pt'

    if not os.path.isfile(local_file):
        torch.hub.download_url_to_file(f'https://models.silero.ai/models/tts/{language}/{model}.pt', local_file)

    model = torch.package.PackageImporter(local_file).load_pickle("tts_models", "model")
    model.to(device)

    example_text = "i'm fine thank you and you?"
    sample_rate = 48000

    audio_paths = model.save_wav(text=tts, speaker=speaker, sample_rate=sample_rate)
    # You can further process the audio_paths as per your requirements

if __name__ == "__main__":
    tts_text = "Hello, how are you?"
    language = "en"
    model = "silero_tts"
    speaker = "female"

    silero_tts(tts_text, language, model, speaker)
