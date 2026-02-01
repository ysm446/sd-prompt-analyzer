"""
VLMモデルのダウンロード・管理
"""
from pathlib import Path
from typing import List, Dict, Optional
import os
from huggingface_hub import snapshot_download


class ModelManager:
    """VLMモデルのダウンロード・管理クラス"""

    def __init__(self, models_dir: str = "./models"):
        """
        Args:
            models_dir: モデル保存ディレクトリ
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def download_model(
        self,
        repo_id: str,
        local_name: str = None,
        force_download: bool = False
    ) -> str:
        """
        Hugging Faceからモデルをダウンロード

        Args:
            repo_id: HF repo ID (例: "Qwen/Qwen2-VL-7B-Instruct")
            local_name: ローカル保存名（未指定ならrepo名を使用）
            force_download: 既存モデルを上書き

        Returns:
            ダウンロードしたモデルのパス
        """
        if local_name is None:
            # repo IDからローカル名を生成 (例: "Qwen/Qwen2-VL-7B" -> "qwen2-vl-7b")
            local_name = repo_id.split('/')[-1].lower()

        local_path = self.models_dir / local_name

        # 既存モデルのチェック
        if local_path.exists() and not force_download:
            print(f"モデルは既にダウンロード済みです: {local_path}")
            return str(local_path)

        print(f"モデルをダウンロード中: {repo_id}")
        print(f"保存先: {local_path}")

        try:
            # Hugging Faceからモデルをダウンロード
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=str(local_path),
                local_dir_use_symlinks=False,
                resume_download=True
            )

            print(f"✓ ダウンロード完了: {downloaded_path}")
            return str(local_path)

        except Exception as e:
            print(f"✗ ダウンロード失敗: {e}")
            raise

    def list_local_models(self) -> List[Dict]:
        """
        ローカルに保存済みのモデル一覧

        Returns:
            [
                {
                    'name': 'qwen2-vl-7b',
                    'path': './models/qwen2-vl-7b',
                    'size': '14.5 GB',
                    'files': ['config.json', 'model.safetensors', ...]
                },
                ...
            ]
        """
        models = []

        if not self.models_dir.exists():
            return models

        for model_dir in self.models_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith('.'):
                continue

            # モデルディレクトリのサイズを計算
            total_size = sum(
                f.stat().st_size
                for f in model_dir.rglob('*')
                if f.is_file()
            )

            # ファイル一覧
            files = [f.name for f in model_dir.iterdir() if f.is_file()]

            models.append({
                'name': model_dir.name,
                'path': str(model_dir),
                'size': self._format_size(total_size),
                'files': files
            })

        return models

    def validate_model(self, model_path: str) -> bool:
        """
        モデルが正しく配置されているか検証

        Transformers形式の必須ファイル:
        - config.json
        - tokenizer.json / tokenizer_config.json
        - model.safetensors (または分割ファイル)
        - preprocessor_config.json (VLMの場合)

        Args:
            model_path: モデルディレクトリのパス

        Returns:
            True if valid
        """
        model_path = Path(model_path)

        if not model_path.exists() or not model_path.is_dir():
            print(f"モデルディレクトリが見つかりません: {model_path}")
            return False

        # 必須ファイルのチェック
        required_files = [
            'config.json',
        ]

        # トークナイザー関連（どちらかが存在すればOK）
        tokenizer_files = ['tokenizer.json', 'tokenizer_config.json']

        # モデルファイル（どれかが存在すればOK）
        model_files = [
            'model.safetensors',
            'pytorch_model.bin',
        ]

        # 必須ファイルの確認
        for required_file in required_files:
            if not (model_path / required_file).exists():
                print(f"必須ファイルが見つかりません: {required_file}")
                return False

        # トークナイザーファイルの確認
        if not any((model_path / f).exists() for f in tokenizer_files):
            print(f"トークナイザーファイルが見つかりません")
            return False

        # モデルファイルの確認（分割ファイルも考慮）
        has_model_file = any((model_path / f).exists() for f in model_files)
        has_split_files = any(model_path.glob('*.safetensors'))

        if not (has_model_file or has_split_files):
            print(f"モデルファイルが見つかりません")
            return False

        return True

    @staticmethod
    def _format_size(size_bytes: int) -> str:
        """
        バイト数を人間が読みやすい形式に変換

        Args:
            size_bytes: バイト数

        Returns:
            フォーマットされたサイズ文字列（例: "14.5 GB"）
        """
        for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
            if size_bytes < 1024.0:
                return f"{size_bytes:.1f} {unit}"
            size_bytes /= 1024.0
        return f"{size_bytes:.1f} PB"
