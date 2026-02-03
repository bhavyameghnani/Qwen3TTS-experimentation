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
            "明るく前向きで、はっきりとした熱意を持って話し始める。"
            "好奇心とワクワク感を伝え、最初は少しテンポを速く、徐々に落ち着かせる。"
        ),
        "text": (
            "みなさん、こんにちは。ポッドキャストへようこそ。"
            "今回は『ノムラレポート2025』を取り上げます。"
            "100年の歴史だけでなく、これからの10年をどう描いているのかが詰まった内容です。"
            "本日は、長年ノムラを知り尽くしている渡辺健司さんをお迎えしています。"
        )
    },
    {
        "role": "guest",
        "instruct": (
            "落ち着きと自信を持って、ゆっくりと話す。"
            "経験に裏打ちされた視点を感じさせる。"
        ),
        "text": (
            "お招きいただきありがとうございます。"
            "このレポートは、ノムラ創立100周年という節目にまとめられました。"
            "単なる業績報告ではなく、将来に向けた価値創造の指針を示しています。"
        )
    },
    {
        "role": "host",
        "instruct": (
            "感心したトーンで入り、自然に質問へつなげる。"
        ),
        "text": (
            "確かに、過去を振り返るだけでなく前を向いた内容ですよね。"
            "なぜ今、ここまでパーパスや長期視点を重視しているのでしょうか？"
        )
    },
    {
        "role": "guest",
        "instruct": (
            "分析的で整理された語り口。"
            "要点を明確に伝える。"
        ),
        "text": (
            "金融機関は利益だけを追う存在ではありません。"
            "社会課題と向き合い、信頼を積み重ねることが不可欠です。"
            "ノムラのパーパスは、意思決定の軸そのものなのです。"
        )
    },
    {
        "role": "host",
        "instruct": (
            "知的好奇心を込めて、少しゆっくり話す。"
        ),
        "text": (
            "その考え方は、2030年に向けた経営ビジョンにも表れていますね。"
            "ROEや税前利益といった目標は、どう受け取るべきでしょうか。"
        )
    },
    {
        "role": "guest",
        "instruct": (
            "分かりやすく説明し、数字の意味を強調する。"
        ),
        "text": (
            "重要なのは一時的な達成ではありません。"
            "ROE8〜10％以上を安定的に実現し、"
            "どんな環境でも持続可能な収益構造を築くことです。"
        )
    },
    {
        "role": "host",
        "instruct": (
            "戦略の変化に興味を示し、前向きなトーンで。"
        ),
        "text": (
            "レポートでは、安定収益やプライベート市場への注力が目立ちます。"
            "これは大きな転換ですよね。"
        )
    },
    {
        "role": "guest",
        "instruct": (
            "自信を持って、方向性を端的に示す。"
        ),
        "text": (
            "はい。"
            "ウェルス、インベストメント、そして新設のバンキング部門を強化し、"
            "収益の安定性と資本効率を高めています。"
        )
    },
    {
        "role": "host",
        "instruct": (
            "成果に驚きつつ、落ち着いたトーンに。"
        ),
        "text": (
            "実際、過去最高益やROE10％達成など、数字も印象的でした。"
            "一つの節目の年に見えます。"
        )
    },
    {
        "role": "guest",
        "instruct": (
            "冷静に評価し、長期視点を示す。"
        ),
        "text": (
            "これは数年にわたる改革の結果です。"
            "成長の質が高まり、持続的成長への基盤が整いつつあります。"
        )
    },
    {
        "role": "host",
        "instruct": (
            "未来志向で締めくくりの質問を投げる。"
        ),
        "text": (
            "最後に、このレポートが示すノムラの未来を一言で表すと？"
        )
    },
    {
        "role": "guest",
        "instruct": (
            "静かに、前向きな余韻を残して締める。"
        ),
        "text": (
            "伝統を守りながら、内側から進化し続けること。"
            "それが、これからのノムラの姿だと思います。"
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
audio_path = f"{OUTPUT_DIR}/podcast_nomura_jp.wav"
sf.write(audio_path, final_audio, sample_rate)

print(f"Podcast audio saved to {audio_path}")

