"""
Vision-Language Model推論インターフェース
"""
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoProcessor
try:
    from transformers import Qwen2VLForConditionalGeneration
    MODEL_CLASS = Qwen2VLForConditionalGeneration
except ImportError:
    # Fallback to AutoModel if Qwen2VL is not available
    from transformers import AutoModel
    MODEL_CLASS = AutoModel
from PIL import Image


class VLMInterface:
    """Vision-Language Model推論インターフェース"""

    def __init__(self, model_path: str, device: str = "auto", dtype: str = "bfloat16"):
        """
        Args:
            model_path: ローカルモデルのパス
            device: デバイス指定 ("auto", "cuda:0", "cpu")
            dtype: データ型 ("bfloat16", "float16", "float32")
        """
        self.model_path = Path(model_path)
        self.device = device
        self.dtype = dtype
        self.model = None
        self.processor = None

        self.load_model(str(model_path))

    def load_model(self, model_path: str):
        """
        モデルをメモリにロード

        実装詳細:
        - Qwen2VLForConditionalGeneration または AutoModel を使用
        - device_map="auto" で自動GPU配置
        - torch.bfloat16 または torch.float16
        """
        print(f"モデルを読み込み中: {model_path}")

        # データ型の設定
        torch_dtype = self._get_torch_dtype()

        try:
            import traceback

            # プロセッサー（トークナイザー + 画像プロセッサー）をロード
            print(f"  プロセッサーを読み込み中...")
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True
            )
            print(f"  ✓ プロセッサーの読み込み完了")

            # モデルをロード
            # デバイスマップの設定（CPUモード時は明示的にCPUを指定）
            if self.device == "cpu":
                device_map = {"": "cpu"}
            else:
                device_map = "auto"

            print(f"  モデルクラス: {MODEL_CLASS.__name__}")
            print(f"  デバイスマップ: {device_map}")
            print(f"  モデルを読み込み中...")
            self.model = MODEL_CLASS.from_pretrained(
                model_path,
                torch_dtype=torch_dtype,
                device_map=device_map,
                trust_remote_code=True
            )

            print(f"✓ モデルの読み込みが完了しました")
            print(f"  デバイスマップ: auto")
            print(f"  データ型: {self.dtype}")

            # デバイス情報を表示
            if hasattr(self.model, 'device'):
                print(f"  モデルデバイス: {self.model.device}")
            elif hasattr(self.model, 'hf_device_map'):
                print(f"  デバイスマップ: {self.model.hf_device_map}")

            # CUDA使用状況
            if torch.cuda.is_available():
                print(f"  ✓ CUDA利用可能")
                print(f"  GPU: {torch.cuda.get_device_name(0)}")
                print(f"  GPU メモリ: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                print(f"  ⚠ CUDA利用不可 - CPUで動作中（非常に遅くなります）")

        except Exception as e:
            import traceback
            print(f"✗ モデルの読み込みに失敗しました")
            print(f"  エラー: {e}")
            print(f"\n詳細なエラートレース:")
            traceback.print_exc()
            raise

    def analyze_image_with_prompt(
        self,
        image_path: str,
        prompt_text: str,
        question: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """
        画像とプロンプトを分析

        Args:
            image_path: 分析対象の画像パス
            prompt_text: 元のSDプロンプト
            question: ユーザーの質問
            temperature: 生成温度
            max_tokens: 最大トークン数

        Returns:
            VLMの回答テキスト
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("モデルがロードされていません")

        # 画像を読み込み
        image = Image.open(image_path).convert('RGB')

        # システムメッセージとユーザーメッセージを構築
        conversation = [
            {
                "role": "system",
                "content": "あなたは画像分析の専門家です。Stable Diffusionで生成された画像とそのプロンプトを評価してください。"
            },
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": f"元のプロンプト:\n{prompt_text}\n\n質問: {question}"}
                ]
            }
        ]

        # プロセッサーでテキストと画像を処理
        text_prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text_prompt],
            images=[image],
            return_tensors="pt"
        )

        # GPUに転送（必要な場合）
        if self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available()):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # 推論
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0
            )

        # デコード
        generated_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        # プロンプト部分を削除（応答のみを返す）
        response = generated_text.split("assistant\n")[-1].strip()

        return response

    def chat(
        self,
        message: str,
        image: Optional[Image.Image] = None,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> str:
        """
        シンプルなチャットインターフェース

        Args:
            message: ユーザーメッセージ
            image: PIL Image オブジェクト（オプション）
            temperature: 生成温度
            max_tokens: 最大トークン数

        Returns:
            VLMの回答
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("モデルがロードされていません")

        # メッセージを構築
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": message}
                ]
            }
        ]

        # 画像がある場合は追加
        if image is not None:
            conversation[0]["content"].insert(0, {"type": "image"})
            images = [image]
        else:
            images = None

        # テキストプロンプトを作成
        text_prompt = self.processor.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=True
        )

        # 入力を処理
        inputs = self.processor(
            text=[text_prompt],
            images=images,
            return_tensors="pt"
        )

        # GPUに転送
        if self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available()):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # 推論
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0
            )

        # デコード
        generated_text = self.processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]

        # 応答部分を抽出
        response = generated_text.split("assistant\n")[-1].strip()

        return response

    def unload_model(self):
        """メモリからモデルをアンロード"""
        if self.model is not None:
            del self.model
            self.model = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        # GPUメモリをクリア
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("モデルをアンロードしました")

    def _get_torch_dtype(self):
        """データ型文字列をtorch dtypeに変換"""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)
