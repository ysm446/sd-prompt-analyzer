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

    # ポートが使用中の場合は次の空きポートを探す
    try:
        interface.launch(
            server_port=config['ui']['server_port'],
            share=config['ui']['share'],
            inbrowser=True,
            theme=config['ui']['theme']
        )
    except OSError as e:
        if "Cannot find empty port" in str(e):
            print(f"\nポート {config['ui']['server_port']} は使用中です。")
            print("別のポートで起動します...")
            interface.launch(
                share=config['ui']['share'],
                inbrowser=True,
                theme=config['ui']['theme']
            )
        else:
            raise


if __name__ == "__main__":
    main()
