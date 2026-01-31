"""
GGUF形式のVision-Language Model推論インターフェース
llama.cpp (llama-cpp-python) を使用
"""
from pathlib import Path
from typing import Optional, Generator
from PIL import Image
import base64
import io


class VLMInterfaceGGUF:
    """GGUF形式のVision-Language Model推論インターフェース"""

    def __init__(self, model_path: str, n_ctx: int = 4096, n_gpu_layers: int = -1, verbose: bool = False):
        """
        Args:
            model_path: GGUFモデルファイルのパス
            n_ctx: コンテキストサイズ
            n_gpu_layers: GPUに載せるレイヤー数（-1で全て）
            verbose: 詳細ログを表示
        """
        self.model_path = Path(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.verbose = verbose
        self.llm = None

        self.load_model(str(model_path))

    def load_model(self, model_path: str):
        """
        GGUFモデルをメモリにロード

        Args:
            model_path: GGUFファイルのパス
        """
        try:
            from llama_cpp import Llama
            from llama_cpp.llama_chat_format import Llava15ChatHandler
        except ImportError:
            raise ImportError(
                "llama-cpp-python がインストールされていません。\n"
                "以下のコマンドでインストールしてください:\n"
                "pip install llama-cpp-python"
            )

        print(f"GGUFモデルを読み込み中: {model_path}")
        print(f"  コンテキストサイズ: {self.n_ctx}")
        print(f"  GPU レイヤー数: {self.n_gpu_layers}")

        try:
            # モデルファイルが存在するか確認
            if not Path(model_path).exists():
                raise FileNotFoundError(f"モデルファイルが見つかりません: {model_path}")

            # Llama モデルをロード
            # Vision対応モデルの場合、chat_formatは自動検出させる
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=self.verbose,
                n_threads=8,  # CPUスレッド数
            )

            print(f"✓ GGUFモデルの読み込みが完了しました")
            print(f"  モデルパス: {model_path}")

        except Exception as e:
            import traceback
            print(f"✗ GGUFモデルの読み込みに失敗しました")
            print(f"  エラー: {e}")
            print(f"\n詳細なエラートレース:")
            traceback.print_exc()
            raise

    def _image_to_base64_data_uri(self, image: Image.Image) -> str:
        """
        PIL ImageをBase64 Data URIに変換

        Args:
            image: PIL Image

        Returns:
            data:image/jpeg;base64,... 形式の文字列
        """
        buffered = io.BytesIO()
        image.save(buffered, format="JPEG")
        img_str = base64.b64encode(buffered.getvalue()).decode()
        return f"data:image/jpeg;base64,{img_str}"

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
        if self.llm is None:
            raise RuntimeError("モデルがロードされていません")

        # 画像を読み込み
        image = Image.open(image_path).convert('RGB')
        image_data_uri = self._image_to_base64_data_uri(image)

        # メッセージを構築
        messages = [
            {
                "role": "system",
                "content": "あなたは画像分析の専門家です。Stable Diffusionで生成された画像とそのプロンプトを評価してください。"
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                    {"type": "text", "text": f"元のプロンプト:\n{prompt_text}\n\n質問: {question}"}
                ]
            }
        ]

        # 推論実行
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        # 応答を取得
        return response['choices'][0]['message']['content']

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
        if self.llm is None:
            raise RuntimeError("モデルがロードされていません")

        # 画像を読み込み
        image = Image.open(image_path).convert('RGB')
        image_data_uri = self._image_to_base64_data_uri(image)

        # メッセージを構築
        messages = [
            {
                "role": "system",
                "content": "あなたは画像分析の専門家です。Stable Diffusionで生成された画像とそのプロンプトを評価してください。"
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_uri}},
                    {"type": "text", "text": f"元のプロンプト:\n{prompt_text}\n\n質問: {question}"}
                ]
            }
        ]

        # ストリーミング推論
        stream = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True
        )

        # 各チャンクをyield
        for chunk in stream:
            if 'choices' in chunk and len(chunk['choices']) > 0:
                delta = chunk['choices'][0].get('delta', {})
                if 'content' in delta:
                    yield delta['content']

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
        if self.llm is None:
            raise RuntimeError("モデルがロードされていません")

        # メッセージを構築
        if image is not None:
            image_data_uri = self._image_to_base64_data_uri(image)
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_data_uri}},
                        {"type": "text", "text": message}
                    ]
                }
            ]
        else:
            messages = [
                {
                    "role": "user",
                    "content": message
                }
            ]

        # 推論実行
        response = self.llm.create_chat_completion(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response['choices'][0]['message']['content']

    def get_context_length(self) -> int:
        """モデルの最大コンテキスト長を取得"""
        return self.n_ctx

    def count_tokens(self, text: str) -> int:
        """テキストのトークン数をカウント"""
        if self.llm is None:
            return 0

        tokens = self.llm.tokenize(text.encode('utf-8'))
        return len(tokens)

    def unload_model(self):
        """メモリからモデルをアンロード"""
        if self.llm is not None:
            del self.llm
            self.llm = None

        print("GGUFモデルをアンロードしました")
