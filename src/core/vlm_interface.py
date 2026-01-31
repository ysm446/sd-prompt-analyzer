"""
Vision-Language Model推論インターフェース
"""
from pathlib import Path
from typing import Optional, Generator
from threading import Thread
import torch
import json
from transformers import AutoProcessor, TextIteratorStreamer
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

            # config.jsonからアーキテクチャを判定して適切なモデルクラスを選択
            model_class = self._get_model_class(model_path)

            print(f"  モデルクラス: {model_class.__name__}")
            print(f"  デバイスマップ: {device_map}")
            print(f"  モデルを読み込み中...")
            self.model = model_class.from_pretrained(
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
            return_tensors="pt",
            padding=True
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

    def analyze_image_with_prompt_stream(
        self,
        image_path: str,
        prompt_text: str,
        question: str,
        temperature: float = 0.7,
        max_tokens: int = 512
    ) -> Generator[str, None, None]:
        """
        画像とプロンプトを分析（ストリーミング版）

        Args:
            image_path: 分析対象の画像パス
            prompt_text: 元のSDプロンプト
            question: ユーザーの質問
            temperature: 生成温度
            max_tokens: 最大トークン数

        Yields:
            生成されたテキストの断片
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
            return_tensors="pt",
            padding=True
        )

        # GPUに転送（必要な場合）
        if self.device == "cuda" or (self.device == "auto" and torch.cuda.is_available()):
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        # ストリーマーを作成
        streamer = TextIteratorStreamer(
            self.processor.tokenizer,
            skip_special_tokens=True,
            skip_prompt=True
        )

        # 生成パラメータ
        generation_kwargs = dict(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=temperature,
            do_sample=temperature > 0,
            streamer=streamer
        )

        # 別スレッドで生成を実行
        thread = Thread(target=self.model.generate, kwargs=generation_kwargs)
        thread.start()

        # ストリーマーからトークンを順次yield
        for text in streamer:
            yield text

        thread.join()

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
            return_tensors="pt",
            padding=True
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

    def get_context_length(self) -> int:
        """モデルの最大コンテキスト長を取得"""
        if self.model is None:
            return 0

        # モデルの設定からコンテキスト長を取得
        if hasattr(self.model.config, 'max_position_embeddings'):
            return self.model.config.max_position_embeddings
        elif hasattr(self.model.config, 'max_seq_len'):
            return self.model.config.max_seq_len
        elif hasattr(self.model.config, 'n_positions'):
            return self.model.config.n_positions
        else:
            return 0  # 不明な場合

    def count_tokens(self, text: str) -> int:
        """テキストのトークン数をカウント"""
        if self.processor is None:
            return 0

        tokens = self.processor.tokenizer.encode(text)
        return len(tokens)

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

    def _get_model_class(self, model_path: str):
        """
        config.jsonからアーキテクチャを読み取って適切なモデルクラスを返す

        Args:
            model_path: モデルディレクトリのパス

        Returns:
            モデルクラス
        """
        config_path = Path(model_path) / "config.json"

        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)

                architecture = config.get('architectures', [None])[0]
                print(f"  検出されたアーキテクチャ: {architecture}")

                # アーキテクチャに基づいてモデルクラスを選択
                if architecture == "Qwen2_5_VLForConditionalGeneration":
                    try:
                        from transformers import Qwen2_5_VLForConditionalGeneration
                        return Qwen2_5_VLForConditionalGeneration
                    except ImportError:
                        print(f"  警告: Qwen2_5_VLForConditionalGenerationが見つかりません。AutoModelを使用します。")
                        from transformers import AutoModel
                        return AutoModel

                elif architecture == "Qwen2VLForConditionalGeneration":
                    try:
                        from transformers import Qwen2VLForConditionalGeneration
                        return Qwen2VLForConditionalGeneration
                    except ImportError:
                        print(f"  警告: Qwen2VLForConditionalGenerationが見つかりません。AutoModelを使用します。")
                        from transformers import AutoModel
                        return AutoModel

                else:
                    print(f"  未知のアーキテクチャです。AutoModelを使用します。")
                    from transformers import AutoModel
                    return AutoModel

            except Exception as e:
                print(f"  警告: config.jsonの読み込みに失敗しました: {e}")
                print(f"  AutoModelを使用します。")
                from transformers import AutoModel
                return AutoModel
        else:
            print(f"  警告: config.jsonが見つかりません。AutoModelを使用します。")
            from transformers import AutoModel
            return AutoModel

    def _get_torch_dtype(self):
        """データ型文字列をtorch dtypeに変換"""
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.bfloat16)
