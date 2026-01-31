"""
CUDA環境チェックスクリプト
"""
import subprocess
import sys

print("=" * 60)
print("CUDA環境チェック")
print("=" * 60)
print()

# 1. nvidia-smi でドライバー情報を確認
print("1. NVIDIA ドライバー & CUDA ドライバーバージョン")
print("-" * 60)
try:
    result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
    if result.returncode == 0:
        # CUDAバージョンの行を抽出
        for line in result.stdout.split('\n'):
            if 'CUDA Version' in line:
                print(f"  {line.strip()}")
                break
        # GPUの最初の行を抽出
        lines = result.stdout.split('\n')
        for i, line in enumerate(lines):
            if '|' in line and 'N/A' in line and i < 10:
                print(f"  GPU: {line.strip()}")
                break
    else:
        print("  ✗ nvidia-smi が実行できません")
except FileNotFoundError:
    print("  ✗ nvidia-smi が見つかりません（NVIDIAドライバー未インストール）")

print()

# 2. nvcc でCUDA Toolkitバージョンを確認
print("2. CUDA Toolkit バージョン")
print("-" * 60)
try:
    result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        # バージョン情報の行を抽出
        for line in result.stdout.split('\n'):
            if 'release' in line.lower():
                print(f"  {line.strip()}")
    else:
        print("  ✗ CUDA Toolkit が見つかりません")
except FileNotFoundError:
    print("  ⚠ CUDA Toolkit が見つかりません（インストールされていない可能性）")
    print("  （注: PyTorchを使用する場合、CUDA Toolkitは必須ではありません）")

print()

# 3. PyTorch の CUDA バージョンを確認
print("3. PyTorch CUDA バージョン")
print("-" * 60)
try:
    import torch
    print(f"  PyTorch バージョン: {torch.__version__}")
    print(f"  PyTorch CUDA バージョン: {torch.version.cuda}")
    print(f"  CUDA 利用可能: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU 数: {torch.cuda.device_count()}")
        print(f"  現在のGPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("  ✗ PyTorch がインストールされていません")

print()
print("=" * 60)
print("推奨事項:")
print("- ドライバーCUDAバージョンとPyTorch CUDAバージョンが近いことを確認")
print("- PyTorchは通常、ドライバーがサポートする範囲内のCUDAバージョンで動作")
print("=" * 60)
