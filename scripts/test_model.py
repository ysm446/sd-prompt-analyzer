"""
モデル動作テストスクリプト
ローカルVLMモデルが正しく動作するかテストします
"""
import os
import sys
from pathlib import Path
import argparse

# プロジェクトルートをPythonパスに追加
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_model_loading(model_path: str):
    """モデルの読み込みテスト"""
    print(f"モデルパス: {model_path}")
    print("モデルを読み込んでいます...")

    try:
        from src.core.vlm_interface import VLMInterface

        vlm = VLMInterface(model_path)
        print("✓ モデルの読み込みに成功しました")

        # メモリから解放
        vlm.unload_model()
        print("✓ モデルをアンロードしました")
        return True

    except Exception as e:
        print(f"✗ エラー: {e}")
        return False


def test_model_validation(model_path: str):
    """モデルファイルの検証テスト"""
    print(f"\nモデルファイルの検証中: {model_path}")

    try:
        from src.core.model_manager import ModelManager

        manager = ModelManager()
        is_valid = manager.validate_model(model_path)

        if is_valid:
            print("✓ モデルファイルは正常です")
        else:
            print("✗ モデルファイルに問題があります")

        return is_valid

    except Exception as e:
        print(f"✗ エラー: {e}")
        return False


def main():
    """テストのメイン処理"""
    parser = argparse.ArgumentParser(description="VLMモデルの動作テスト")
    parser.add_argument(
        "--model",
        type=str,
        default="./models/qwen2-vl-7b",
        help="テストするモデルのパス"
    )
    parser.add_argument(
        "--skip-loading",
        action="store_true",
        help="モデル読み込みテストをスキップ（検証のみ）"
    )

    args = parser.parse_args()

    print("=== VLMモデル動作テスト ===\n")

    # モデルファイル検証
    validation_ok = test_model_validation(args.model)

    if not validation_ok:
        print("\n! モデルファイルの検証に失敗しました")
        sys.exit(1)

    # モデル読み込みテスト
    if not args.skip_loading:
        print("\n" + "-" * 50)
        loading_ok = test_model_loading(args.model)

        if not loading_ok:
            print("\n! モデルの読み込みに失敗しました")
            sys.exit(1)

    print("\n" + "=" * 50)
    print("✓ すべてのテストが成功しました！")


if __name__ == "__main__":
    main()
