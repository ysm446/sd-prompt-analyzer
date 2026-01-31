"""
設定ファイル読み込みユーティリティ
"""
import yaml
from pathlib import Path
from typing import Dict, Any


class ConfigLoader:
    """設定ファイルを読み込むクラス"""

    def __init__(self, config_dir: str = "./config"):
        """
        Args:
            config_dir: 設定ファイルのディレクトリパス
        """
        self.config_dir = Path(config_dir)

    def load_settings(self) -> Dict[str, Any]:
        """
        settings.yamlを読み込み

        Returns:
            設定内容の辞書
        """
        settings_path = self.config_dir / "settings.yaml"

        if not settings_path.exists():
            raise FileNotFoundError(f"設定ファイルが見つかりません: {settings_path}")

        with open(settings_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)

        return config

    def load_model_presets(self) -> Dict[str, Any]:
        """
        model_presets.yamlを読み込み

        Returns:
            モデルプリセットの辞書
        """
        presets_path = self.config_dir / "model_presets.yaml"

        if not presets_path.exists():
            raise FileNotFoundError(f"モデルプリセットファイルが見つかりません: {presets_path}")

        with open(presets_path, 'r', encoding='utf-8') as f:
            presets = yaml.safe_load(f)

        return presets.get('presets', {})

    def save_settings(self, config: Dict[str, Any]) -> None:
        """
        設定をsettings.yamlに保存

        Args:
            config: 保存する設定内容
        """
        settings_path = self.config_dir / "settings.yaml"

        with open(settings_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, allow_unicode=True, default_flow_style=False)
