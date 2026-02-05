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
GUEST_VOICE = "Dylan"

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
            "明るく、元気で、ワクワクした雰囲気で話し始める。"
            "強い好奇心と前向きな熱意を感じさせ、最初はややテンポよく進める。"
        ),
        "text": (
            "みなさん、こんにちは。ポッドキャストへようこそ。"
            "今回は『ノムラレポート2025』を取り上げます。"
            "100年の歴史だけでなく、これからの10年をどうえがいているのかが詰まっている内容です。"
            "本日は、長年／ムラを知り尽くしている渡辺健司さんをお迎えしています。"
        )
    },
    {
        "role": "guest",
        "instruct": (
            "落ち着きがありつつも前向きなエネルギーを感じさせる話し方。"
            "自信と経験に裏打ちされた、穏やかで安定したトーンを保つ。"
        ),
        "text": (
            "お招きいただきありがとうございます。"
            "このレポートは、ノムラ創立100周年というしめにまとめられました。"
            "単なる業績報告ではなく、将来に向けた価値創造の指針を示しています。"
        )
    },
    {
        "role": "host",
        "instruct": (
            "感心と興味をにじませながら話し、自然な流れで質問へ入る。"
            "相手の話を深掘りしたいという好奇心を強調する。"
        ),
        "text": (
            "確かに、過去を振り返るだけでなく、前を向いた内容ですよね。"
            "なぜ今、ここまでパーパスや長期視点を重視しているのでしょうか？"
        )
    },
    {
        "role": "guest",
        "instruct": (
            "冷静で分析的、かつ説得力のある語り口。"
            "要点をはっきりと伝え、落ち着いた自信を感じさせる。"
        ),
        "text": (
            "金融機関は利益だけを追う存在ではありません。"
            "社会課題と向き合い、頼を積み重ねることが不可欠です。"
            "ノムラのパーパスは、意思決定の軸そのものなのです。"
        )
    },
    {
        "role": "host",
        "instruct": (
            "知的好奇心を前面に出し、少しテンポを落として丁寧に話す。"
            "次のテーマへの期待感を込める。"
        ),
        "text": (
            "その考え方は、2030年に向けた経営ビジョンにも表れていますね。"
            "ROEや、ぜいまえ利益といった目標は、どう受け取るべきでしょうか。"
            "I"
        )
    },
    {
        "role": "guest",
        "instruct": (
            "落ち着いた中にも前向きな力強さを込めて説明する。"
            "数字の意味を噛み砕きながら、自信を持って伝える。"
        ),
        "text": (
            "重要なのは一時的な達成ではありません。"
            "ROE 8から10%以上を安定的に実現し、どんな環境でも持続可能な収益構造を築くことです。"
        )
    },
    {
        "role": "host",
        "instruct": (
            "戦略の変化に対する興奮と関心を込めたトーン。"
            "前向きで探究心のある話し方を意識する。"
        ),
        "text": (
            "レポートでは、安定収益やプライベート市場への注力が目立ちます。"
            "これは大きな転換ですよね。"
        )
    },
    {
        "role": "guest",
        "instruct": (
            "自信に満ち、方向性を明確に示す語り口。"
            "落ち着きと確信を持って簡潔に述べる。"
        ),
        "text": (
            "はい。"
            "ウェルスマネジメント、インベストメント、そして新設のバンキング部門を強化し、収益の安定性と資本効率を高めています。"
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
        language="Japanese",
        speaker=speaker,
        instruct=instruct,
    )

    audio_segments.append(wavs[0])
    sample_rate = sr

# Concatenate audio
final_audio = np.concatenate(audio_segments)
audio_path = f"{OUTPUT_DIR}/podcast_nomura_jp1.wav"
sf.write(audio_path, final_audio, sample_rate)

print(f"Podcast audio saved to {audio_path}")

