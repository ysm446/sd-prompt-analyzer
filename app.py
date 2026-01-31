"""
SD Prompt Analyzer - メインエントリーポイント
"""
import os
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.utils.config_loader import ConfigLoader
from src.ui.gradio_app import PromptAnalyzerUI


def main():
    """アプリケーションのメインエントリーポイント"""

    # 設定ファイルを読み込み
    config_loader = ConfigLoader()
    config = config_loader.load_settings()

    print(f"=== {config['app']['name']} v{config['app']['version']} ===")
    print(f"Server Port: {config['ui']['server_port']}")
    print(f"Models Directory: {config['paths']['models_dir']}")
    print(f"Image Folder: {config['paths']['image_folder']}")
    print("-" * 50)

    # UIを作成して起動
    ui = PromptAnalyzerUI(config)
    interface = ui.create_interface()

    interface.launch(
        server_port=config['ui']['server_port'],
        share=config['ui']['share'],
        inbrowser=True
    )


if __name__ == "__main__":
    main()
