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
GUEST_VOICE = "Dylan"

HOST_INSTRUCT = "Speak enthusiastically, curious, energetic but clear."
GUEST_INSTRUCT = "Speak calmly, confidently, like an expert explaining concepts."

def add_pause(wav, sr, seconds=0.3):
    pause = np.zeros(int(sr * seconds))
    return np.concatenate([wav, pause])

def speed_up(wav, factor=1.05):
    indices = np.round(np.arange(0, len(wav), factor))
    indices = indices[indices < len(wav)].astype(int)
    return wav[indices]

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
            "番組の進行役として、明るく端正で洗練されたトーンを保つ。"
            "日本のビジネスラジオを意識し、過度に感情を出さず、"
            "前向きで知的な期待感を軽やかに表現する。"
            "TTSでは中〜やや高めのピッチ、明瞭な発音、"
            "文末は必ずやわらかく下げて安定感を出す。"
            "全体のテンポはやや速めで、会話を前に進める役割を担う。"
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
            "専門家として、重心を低く保ち、落ち着きと余裕を感じさせる話し方。"
            "感情は抑えめにし、経験に裏打ちされた信頼感を優先する。"
            "TTSでは低めのピッチ、やや遅めの話速を設定し、"
            "重要語の前後に自然な間を入れる。"
            "文末は断定しすぎず、静かに着地させる。"
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
            "聞き手の代表として共感を示しつつ、穏やかな好奇心を前面に出す。"
            "問いかけでは語尾をわずかに上げ、対話を促進する。"
            "TTSでは中程度のピッチを維持し、"
            "文と文の間に短いポーズを入れて理解を助ける。"
        ),
        "text": (
            "確かに、過去を振り返るだけでなく、前を向いた内容ですよね。"
            "なぜ今、ここまでパーパスや長期視点を重視しているのでしょうか？"
        )
    },
    {
        "role": "guest",
        "instruct": (
            "分析的かつ理路整然とした語り口を徹底する。"
            "一文一義を意識し、情報の重みを丁寧に伝える。"
            "TTSでは低めのピッチと一定のリズムを保ち、"
            "安心感と説得力を優先する。"
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
            "知的関心を示しながら、進行役として話題を整理する。"
            "説明部分では落ち着き、質問部分で軽く抑揚をつける。"
            "TTSでは中〜やや高めのピッチを維持し、"
            "聞き疲れしない安定したテンポを意識する。"
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
            "静かな自信を保ちつつ、数値の意味を丁寧に説明する。"
            "強調は抑制的に行い、落ち着いた説得力を重視する。"
            "TTSでは一定の話速を保ち、"
            "数値や結論の前後に短い間を入れる。"
        ),
        "text": (
            "重要なのは一時的な達成ではありません。"
            "ROE 8から10%以上を安定的に実現し、どんな環境でも持続可能な収益構造を築くことです。"
        )
    },
    {
        "role": "host",
        "instruct": (
            "戦略の流れを整理し、次の話題へ導く進行役のトーン。"
            "前向きだが落ち着いた期待感を示す。"
            "TTSではやや高めのピッチを保ち、"
            "会話のリズムをコントロールする。"
        ),
        "text": (
            "レポートでは、安定収益やプライベート市場への注力が目立ちます。"
            "これは大きな転換ですよね。"
        )
    },
    {
        "role": "guest",
        "instruct": (
            "結論部分として、簡潔かつ重みのある語り口を意識する。"
            "余計な抑揚を排し、内容そのものの説得力を前に出す。"
            "TTSでは低めのピッチ、短めの文間ポーズで締める。"
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

    wavs, sr = model.generate_custom_voice(
        text=seg["text"],
        language="Japanese",
        speaker=speaker,
        instruct=seg["instruct"],
    )

    wav = wavs[0]

    # Add culturally appropriate pauses
    if seg["role"] == "host":
        wav = add_pause(wav, sr, 0.18)   
    else:
        wav = add_pause(wav, sr, 0.40)   

    audio_segments.append(wav)
    sample_rate = sr


# Concatenate audio
final_audio = np.concatenate(audio_segments)
audio_path = f"{OUTPUT_DIR}/podcast_nomura_jp_anna_dylan.wav"
sf.write(audio_path, final_audio, sample_rate)

print(f"Podcast audio saved to {audio_path}")

