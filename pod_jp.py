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

HOST_VOICE = "Ono_Anna"
GUEST_VOICE = "Uncle_Fu"

HOST_INSTRUCT = (
    "Speak with high energy and enthusiasm, like a friendly podcast host. "
    "Sound genuinely curious and engaged, with natural excitement when asking questions. "
    "Use a warm, inviting tone, clear pronunciation, and a lively but not rushed pace. "
    "Add subtle emotional variation to feel human, not robotic."
)
GUEST_INSTRUCT = (
    "Speak in a calm, confident, and composed manner, like a domain expert explaining ideas clearly. "
    "Maintain a steady, controlled pace with precise articulation. "
    "Sound thoughtful and authoritative, but approachable and friendly. "
    "Avoid sounding dramatic; prioritize clarity and confidence."
)


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
        "text": (
            "みなさん、こんにちは！ポッドキャストへようこそ。"
            "今日は、現代の機械学習において最も影響力のある論文のひとつ、"
            "『Attention Is All You Need』について深掘りしていきます。"
            "この論文は、シーケンスモデルの考え方を根本から変えたと言っても過言ではありません。"
            "今日はトランスフォーマーに深く関わってきた専門家をお迎えしています。よろしくお願いします！"
        )
    },
    {
        "role": "guest",
        "text": (
            "お招きいただきありがとうございます。"
            "この論文が画期的だったのは、再帰構造や畳み込みを完全に排除し、"
            "注意機構、つまりアテンションのみに基づいたトランスフォーマーという"
            "新しいアーキテクチャを提案した点です。"
        )
    },
    {
        "role": "host",
        "text": (
            "今聞いても本当に大胆な発想ですよね。"
            "この論文以前は、多くのモデルがRNNやCNNを使っていましたが、"
            "そこにはどんな根本的な課題があったのでしょうか？"
        )
    },
    {
        "role": "guest",
        "text": (
            "そうですね。再帰型モデルはトークンを順番に処理する必要があるため、"
            "並列化が難しく、学習に時間がかかります。"
            "シーケンスが長くなるほど、この問題は深刻になります。"
            "トランスフォーマーは、すべてのトークンが同時に互いを参照できることで、"
            "この制約を解消しました。"
        )
    },
    {
        "role": "host",
        "text": (
            "そこで登場するのがセルフアテンションですね。"
            "このアイデアの美しさ、本当に好きです。"
            "スケールド・ドットプロダクト・アテンションを、"
            "できるだけシンプルに説明してもらえますか？"
        )
    },
    {
        "role": "guest",
        "text": (
            "もちろんです。各トークンはクエリ、キー、バリューという3つのベクトルを生成します。"
            "クエリとキーの内積を計算し、スケーリングしてからソフトマックスを適用します。"
            "その結果を使って、バリューの加重和を求めることで、"
            "文脈の中で重要な情報に集中できるようになります。"
        )
    },
    {
        "role": "host",
        "text": (
            "しかも、それを一度だけでなく、複数同時に行う。"
            "それがマルチヘッド・アテンションなんですよね。"
        )
    },
    {
        "role": "guest",
        "text": (
            "その通りです。マルチヘッド・アテンションによって、"
            "モデルは異なる表現空間に同時に注意を向けることができます。"
            "これがトランスフォーマーの表現力を非常に高くしている理由のひとつです。"
        )
    },
    {
        "role": "host",
        "text": (
            "本当に、このひとつのアイデアが、"
            "自然言語処理だけでなく、画像や音声の分野にまで影響を与えましたよね。"
            "とても分かりやすい解説、ありがとうございました。"
        )
    },
    {
        "role": "guest",
        "text": (
            "こちらこそありがとうございました。"
            "この論文は、今私たちが使っている多くの技術の土台になっています。"
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
    instruct = HOST_INSTRUCT if seg["role"] == "host" else GUEST_INSTRUCT

    wavs, sr = model.generate_custom_voice(
        text=seg["text"],
        language="Japanese",
        speaker=speaker,
        instruct=instruct,
    )

    audio_segments.append(wavs[0])
    sample_rate = sr

# Concatenate audio
final_audio = np.concatenate(audio_segments)
audio_path = f"{OUTPUT_DIR}/podcast_jp.wav"
sf.write(audio_path, final_audio, sample_rate)

print(f"Podcast audio saved to {audio_path}")

# =============================
# AUDIO VISUALIZATION (VIDEO)
# =============================
