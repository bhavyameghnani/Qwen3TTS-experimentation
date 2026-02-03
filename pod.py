import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from qwen_tts import Qwen3TTSModel
import os

# =============================
# CONFIG
# =============================
OUTPUT_DIR = "podcast_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HOST_VOICE = "Serena"
GUEST_VOICE = "Uncle_Fu"

HOST_INSTRUCT = "Speak enthusiastically, curious, energetic but clear."
GUEST_INSTRUCT = "Speak calmly, confidently, like an expert explaining concepts."

# =============================
# LOAD MODEL
# =============================
model = Qwen3TTSModel.from_pretrained(
    "/Users/tanayaverma/nomura/models/qwen3-tts-0.6b",
    device_map="cpu",
    dtype=torch.float32,
)

# =============================
# PODCAST SCRIPT
# =============================
PODCAST_SEGMENTS = [
    {
        "role": "host",
        "instruct": (
            "Start with energetic curiosity and a welcoming tone. "
            "Sound excited at the beginning, then slow slightly and become warm and attentive toward the end."
        ),
        "text": (
            "Hey everyone... welcome back to the podcast! "
            "Today, we’re diving into one of the most influential papers in modern machine learning — "
            "Attention Is All You Need. "
            "I’m honestly super excited about this one, because it completely changed how we think about "
            "sequence models. "
            "And joining me today is an expert who’s worked deeply with transformers... welcome!"
        )
    },
    {
        "role": "guest",
        "instruct": (
            "Begin calmly and confidently, like an expert setting context. "
            "Maintain a steady pace, and end with quiet emphasis."
        ),
        "text": (
            "Thanks for having me. "
            "This paper is special because it introduced the Transformer architecture — "
            "an approach that removed recurrence and convolutions entirely... "
            "and instead relied purely on attention mechanisms."
        )
    },
    {
        "role": "host",
        "instruct": (
            "Sound intrigued and slightly amazed at first. "
            "Ask the question with genuine curiosity, then pause briefly before the key point."
        ),
        "text": (
            "That still sounds pretty wild, even today... "
            "Before this paper, most models were using RNNs or CNNs, right? "
            "So what was the core limitation there?"
        )
    },
    {
        "role": "guest",
        "instruct": (
            "Explain thoughtfully and methodically. "
            "Start neutral, then gradually emphasize the problem before ending in a confident resolution."
        ),
        "text": (
            "Exactly. Recurrent models process tokens sequentially, which limits parallelism "
            "and makes training slow. "
            "As sequences get longer, this becomes a serious bottleneck... "
            "The Transformer addresses this by allowing every token to attend to every other token, all at once."
        )
    },
    {
        "role": "host",
        "instruct": (
            "Sound appreciative and intellectually curious. "
            "Slow down slightly when introducing the concept, and invite explanation."
        ),
        "text": (
            "And that’s where self-attention comes in. "
            "I really love how elegant that idea is... "
            "Could you explain scaled dot-product attention, in simple terms?"
        )
    },
    {
        "role": "guest",
        "instruct": (
            "Teach clearly and patiently, like explaining to an attentive audience. "
            "Use gentle emphasis on technical terms, and keep a calm rhythm."
        ),
        "text": (
            "Sure. Each token creates a query, a key, and a value. "
            "We compare queries with keys using dot products, "
            "scale them to keep gradients stable, apply a softmax... "
            "and then compute a weighted sum of values. "
            "This allows the model to focus on the most relevant parts of the sequence."
        )
    },
    {
        "role": "host",
        "instruct": (
            "Sound impressed and slightly animated. "
            "Use a rising tone at the start, then conclude confidently."
        ),
        "text": (
            "And instead of doing this just once... "
            "the model does it multiple times in parallel — "
            "that’s multi-head attention."
        )
    },
    {
        "role": "guest",
        "instruct": (
            "Confirm with confidence and clarity. "
            "Speak smoothly and end with a sense of importance."
        ),
        "text": (
            "Right. Multi-head attention lets the model attend to different representation subspaces "
            "at the same time. "
            "That’s one of the key reasons transformers are so expressive and powerful."
        )
    },
    {
        "role": "host",
        "instruct": (
            "Sound reflective and appreciative. "
            "Slow the pace slightly, ending with warmth and gratitude."
        ),
        "text": (
            "It’s honestly incredible how this single idea reshaped NLP... "
            "vision... and even audio models. "
            "Thanks so much for breaking it down so clearly."
        )
    },
    {
        "role": "guest",
        "instruct": (
            "Close calmly and thoughtfully. "
            "Sound satisfied and reflective."
        ),
        "text": (
            "My pleasure. "
            "This paper really laid the foundation for so much of what we’re building today."
        )
    }
]

  # <-- paste transcript block here

# =============================
# TTS GENERATION
# =============================
audio_segments = []
sample_rate = None

for idx, seg in enumerate(PODCAST_SEGMENTS):
    speaker = HOST_VOICE if seg["role"] == "host" else GUEST_VOICE
    instruct = seg.get("instruct")

    wavs, sr = model.generate_custom_voice(
        text=seg["text"],
        language="English",
        speaker=speaker,
        instruct=instruct,
    )

    audio_segments.append(wavs[0])
    sample_rate = sr

# Concatenate audio
final_audio = np.concatenate(audio_segments)
audio_path = f"{OUTPUT_DIR}/podcast_en.wav"
sf.write(audio_path, final_audio, sample_rate)

print(f"Podcast audio saved to {audio_path}")

