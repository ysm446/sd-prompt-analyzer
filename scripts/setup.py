"""
初期セットアップスクリプト
プロジェクトの初期設定を行います
"""
import os
import sys
from pathlib import Path

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def create_directories():
    """必要なディレクトリを作成"""
    directories = [
        "data/sd_outputs",
        "data/database",
        "data/downloads",
        "models",
    ]

    print("ディレクトリを作成中...")
    for directory in directories:
        dir_path = project_root / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")


def check_dependencies():
    """依存関係がインストールされているか確認"""
    print("\n依存関係を確認中...")
    required_packages = [
        "gradio",
        "transformers",
        "torch",
        "PIL",
        "yaml",
        "huggingface_hub"
    ]

    missing_packages = []
    for package in required_packages:
        try:
            if package == "PIL":
                __import__("PIL")
            elif package == "yaml":
                __import__("yaml")
            else:
                __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (未インストール)")
            missing_packages.append(package)

    if missing_packages:
        print(f"\n警告: 以下のパッケージがインストールされていません:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\n以下のコマンドで依存関係をインストールしてください:")
        print("  pip install -r requirements.txt")
        return False

    return True


def main():
    """セットアップのメイン処理"""
    print("=== SD Prompt Analyzer セットアップ ===\n")

    # ディレクトリ作成
    create_directories()

    # 依存関係チェック
    dependencies_ok = check_dependencies()

    print("\n" + "=" * 50)
    if dependencies_ok:
        print("✓ セットアップが完了しました！")
        print("\nアプリケーションを起動するには:")
        print("  python app.py")
    else:
        print("! セットアップが完了しましたが、依存関係のインストールが必要です")
        print("\n次のステップ:")
        print("  1. pip install -r requirements.txt")
        print("  2. python app.py")


if __name__ == "__main__":
    main()
