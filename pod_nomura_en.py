import torch
import soundfile as sf
import numpy as np
import matplotlib.pyplot as plt
from moviepy.audio.io.AudioFileClip import AudioFileClip
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from local_config import MODEL_PATH

from qwen_tts import Qwen3TTSModel
import os

# =============================
# CONFIG
# =============================
OUTPUT_DIR = "podcast_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HOST_VOICE = "Ryan"
GUEST_VOICE = "Uncle_Fu"

HOST_INSTRUCT = "Speak enthusiastically, curious, energetic but clear."
GUEST_INSTRUCT = "Speak calmly, confidently, like an expert explaining concepts."

# =============================
# LOAD MODEL
# =============================
model = Qwen3TTSModel.from_pretrained(
    MODEL_PATH,
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
            "Open with bright, welcoming energy and clear enthusiasm. "
            "Sound genuinely excited, curious, and upbeat. "
            "Keep the pace lively at first, then gently slow as you set context and invite listeners in."
        ),
        "text": (
            "Hey everyone, welcome back to the show! "
            "Today’s episode is a special one — we’re unpacking the Nomura Report 2025, "
            "a document that doesn’t just look back at a hundred years of history, "
            "but really lays out how Nomura is thinking about the next decade and beyond. "
            "I’m genuinely excited, because this report is packed with strategy, numbers, and philosophy. "
            "And joining me is someone who’s lived and breathed Nomura for years. "
            "Please welcome Kenji Watanabe."
        )
    },
    {
        "role": "guest",
        "instruct": (
            "Speak with calm authority and steady confidence. "
            "Maintain a measured, unhurried pace. "
            "Sound reflective and experienced, ending with subtle emphasis on long-term perspective."
        ),
        "text": (
            "Thank you, it’s great to be here. "
            "This report is particularly meaningful because it coincides with Nomura’s 100th anniversary. "
            "It’s not just a record of performance — it’s a statement of purpose, "
            "and a clear articulation of how Nomura intends to create sustainable value going forward."
        )
    },
    {
        "role": "host",
        "instruct": (
            "Begin with curiosity and a sense of admiration. "
            "Sound thoughtful, then transition into an exploratory question. "
            "Pause briefly before asking the core question."
        ),
        "text": (
            "That centennial milestone really stood out to me. "
            "The report opens with history, but it doesn’t feel nostalgic — it feels intentional. "
            "So let me ask you this… "
            "why does Nomura place so much emphasis on purpose and long-term value creation right now?"
        )
    },
    {
        "role": "guest",
        "instruct": (
            "Explain in a composed, analytical tone. "
            "Gradually add emphasis as ideas connect. "
            "Conclude with clarity and conviction."
        ),
        "text": (
            "Because financial institutions don’t exist in isolation anymore. "
            "Nomura recognizes that profitability and societal contribution are inseparable. "
            "The Group’s Purpose — creating a better world by harnessing the power of financial markets — "
            "is meant to guide decisions, capital allocation, and behavior. "
            "It’s about earning trust continuously, not episodically."
        )
    },
    {
        "role": "host",
        "instruct": (
            "Sound engaged and impressed. "
            "Slow slightly to underscore importance. "
            "End with an inviting tone that encourages elaboration."
        ),
        "text": (
            "And that purpose seems tightly connected to the 2030 management vision — "
            "Reaching for Sustainable Growth. "
            "The targets are pretty concrete: ROE, income before taxes. "
            "How should listeners interpret these goals?"
        )
    },
    {
        "role": "guest",
        "instruct": (
            "Teach clearly and confidently, as if explaining to an informed listener. "
            "Maintain a calm rhythm with gentle emphasis on key metrics."
        ),
        "text": (
            "The targets are intentionally ambitious but disciplined. "
            "Nomura aims to consistently achieve ROE of 8 to 10 percent or more, "
            "and income before income taxes exceeding 500 billion yen. "
            "What matters is not just hitting those numbers once, "
            "but building earnings stability so they can be sustained across market cycles."
        )
    },
    {
        "role": "host",
        "instruct": (
            "Sound curious and energized by the strategic shift. "
            "Begin lightly, then sharpen focus as the question lands."
        ),
        "text": (
            "One thing I noticed is the repeated emphasis on stable revenues and private markets. "
            "It feels like a deliberate evolution. "
            "What’s driving that shift?"
        )
    },
    {
        "role": "guest",
        "instruct": (
            "Respond with confident assurance and strategic clarity. "
            "End with a sense of momentum and direction."
        ),
        "text": (
            "Volatility management is central here. "
            "By expanding from public to private markets and strengthening businesses like "
            "Wealth Management, Investment Management, and the newly established Banking Division, "
            "Nomura is building a more balanced portfolio. "
            "These areas generate recurring, capital-efficient revenues that support long-term growth."
        )
    },
    {
        "role": "host",
        "instruct": (
            "Shift into a reflective, slightly awed tone. "
            "Slow the pace and sound impressed by scale."
        ),
        "text": (
            "The financial results really back that up. "
            "Record net income, a 10 percent ROE, assets under management at historic highs… "
            "It feels like a turning point year."
        )
    },
    {
        "role": "guest",
        "instruct": (
            "Acknowledge with calm confidence. "
            "Speak as someone who sees progress as part of a longer journey."
        ),
        "text": (
            "Exactly. FY2024/25 showed tangible outcomes from reforms that took years to implement. "
            "Growth across all three core divisions, improved earnings quality, "
            "and stronger global contributions indicate that the foundation for sustainable growth is taking shape."
        )
    },
    {
        "role": "host",
        "instruct": (
            "Sound warmly curious and forward-looking. "
            "End with a question that invites synthesis."
        ),
        "text": (
            "Before we wrap up, one last thing. "
            "If you had to sum up what this report signals about Nomura’s future, "
            "what would you say?"
        )
    },
    {
        "role": "guest",
        "instruct": (
            "Close calmly and thoughtfully. "
            "Sound assured, reflective, and quietly optimistic."
        ),
        "text": (
            "I’d say it signals confidence with humility. "
            "Nomura is honoring its traditions while deliberately reinventing itself from the inside. "
            "With a clear purpose, disciplined targets, and belief in people, "
            "the Group is positioning itself not just to grow — but to remain relevant and trusted for decades to come."
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
audio_path = f"{OUTPUT_DIR}/podcast_nomura_en.wav"
sf.write(audio_path, final_audio, sample_rate)

print(f"Podcast audio saved to {audio_path}")

