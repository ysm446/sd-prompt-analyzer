"""
画像処理補助ユーティリティ
"""
from pathlib import Path
from typing import List, Tuple
from PIL import Image


def get_image_files(directory: str, extensions: List[str] = None) -> List[Path]:
    """
    指定されたディレクトリから画像ファイル一覧を取得

    Args:
        directory: 検索対象ディレクトリ
        extensions: 対象とする拡張子のリスト（デフォルト: ['.png', '.jpg', '.jpeg']）

    Returns:
        画像ファイルのPathオブジェクトのリスト
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.webp']

    directory_path = Path(directory)

    if not directory_path.exists():
        return []

    # 大文字小文字を区別しないファイル検索
    image_files_set = set()
    for ext in extensions:
        # 小文字と大文字の両方でマッチング
        image_files_set.update(directory_path.glob(f"*{ext}"))
        image_files_set.update(directory_path.glob(f"*{ext.upper()}"))

    # Pathオブジェクトのリストに変換してソート
    return sorted(list(image_files_set))


def resize_image(image: Image.Image, max_size: Tuple[int, int] = (1024, 1024)) -> Image.Image:
    """
    画像をリサイズ（アスペクト比を維持）

    Args:
        image: PIL Image オブジェクト
        max_size: 最大サイズ (width, height)

    Returns:
        リサイズされた画像
    """
    image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return image


def get_image_info(image_path: Path) -> dict:
    """
    画像の基本情報を取得

    Args:
        image_path: 画像ファイルのパス

    Returns:
        画像情報の辞書 {
            'path': str,
            'filename': str,
            'size': tuple,
            'mode': str,
            'format': str
        }
    """
    with Image.open(image_path) as img:
        return {
            'path': str(image_path),
            'filename': image_path.name,
            'size': img.size,
            'mode': img.mode,
            'format': img.format
        }
