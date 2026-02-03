import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from local_config import MODEL_PATH

model = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
    device_map="cpu",
    dtype=torch.float32,
)

wavs, sr = model.generate_custom_voice(
    text="this is just a sample for testing instruct。",
    language="English",
    speaker="Vivian",
    instruct="Speak hesitantly and shy"
)

sf.write("output11.wav", wavs[0], sr)
print("Saved output11.wav")

wavs, sr = model.generate_custom_voice(
    text="this is just a sample for testing instruct。",
    language="English",
    speaker="Vivian",
    instruct="Speak enthusiastically and happily"
)

sf.write("output12.wav", wavs[0], sr)
print("Saved output12.wav")
