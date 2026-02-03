import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from local_config import MODEL_PATH

model = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    device_map="cpu",
    dtype=torch.float32
)

wavs, sr = model.generate_custom_voice(
    text="Hello. Qwen text to speech is finally working.",
    language="English",
    speaker="Ryan",
    instruct="speak fast"
)

sf.write("output.wav", wavs[0], sr)
print("Saved output.wav")
