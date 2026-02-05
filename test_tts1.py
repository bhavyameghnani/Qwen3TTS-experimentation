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
    
    language="Japanese",
    speaker="Ono_Anna",
    instruct= (
            "番組の進行役として、明るく端正で洗練されたトーンを保つ。"
            "日本のビジネスラジオを意識し、過度に感情を出さず、"
            "前向きで知的な期待感を軽やかに表現する。"
            "TTSでは中〜やや高めのピッチ、明瞭な発音、"
            "文末は必ずやわらかく下げて安定感を出す。"
            "全体のテンポはやや速めで、会話を前に進める役割を担う。"
        ),
    text= (
            "みなさん、こんにちは。ポッドキャストへようこそ。"
            "今回は『ノムラレポート2025』を取り上げます。"
            "100年の歴史だけでなく、これからの10年をどうえがいているのかが詰まっている内容です。"
            "本日は、長年／ムラを知り尽くしている渡辺健司さんをお迎えしています。"
        )
)

sf.write("output11.wav", wavs[0], sr)
print("Saved output11.wav")

wavs, sr = model.generate_custom_voice(
    language="Japanese",
    speaker="Ono_Anna",
    instruct= (
            "専門家として、重心を低く保ち、落ち着きと余裕を感じさせる話し方。"
            "感情は抑えめにし、経験に裏打ちされた信頼感を優先する。"
            "TTSでは低めのピッチ、やや遅めの話速を設定し、"
            "重要語の前後に自然な間を入れる。"
            "文末は断定しすぎず、静かに着地させる。"
    ),
    text=(
            "お招きいただきありがとうございます。"
            "このレポートは、ノムラ創立100周年というしめにまとめられました。"
            "単なる業績報告ではなく、将来に向けた価値創造の指針を示しています。"
    )
)

sf.write("output12.wav", wavs[0], sr)
print("Saved output12.wav")
