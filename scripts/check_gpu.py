"""
GPU使用状況チェックスクリプト
"""
import sys

print("=" * 60)
print("GPU環境チェック")
print("=" * 60)
print()

# PyTorchのインポート
try:
    import torch
    print("✓ PyTorch インストール済み")
    print(f"  バージョン: {torch.__version__}")
except ImportError:
    print("✗ PyTorch がインストールされていません")
    sys.exit(1)

print()

# CUDAの確認
cuda_available = torch.cuda.is_available()
print(f"CUDA利用可能: {cuda_available}")

if cuda_available:
    print(f"✓ GPUが利用可能です")
    print(f"  CUDAバージョン: {torch.version.cuda}")
    print(f"  GPU数: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\n  GPU {i}:")
        print(f"    名前: {torch.cuda.get_device_name(i)}")
        print(f"    メモリ: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

        # 現在のメモリ使用状況
        allocated = torch.cuda.memory_allocated(i) / 1024**3
        reserved = torch.cuda.memory_reserved(i) / 1024**3
        print(f"    割り当て済み: {allocated:.2f} GB")
        print(f"    予約済み: {reserved:.2f} GB")

    print()
    print("推奨: モデルはGPUで実行されます")

else:
    print("✗ GPUが利用できません（CPUモードで動作）")
    print()
    print("原因の可能性:")
    print("  1. NVIDIA GPUがインストールされていない")
    print("  2. CUDAドライバーがインストールされていない")
    print("  3. PyTorchがCPU版でインストールされている")
    print()
    print("解決方法:")
    print("  PyTorchをCUDA版で再インストール:")
    print("  pip uninstall torch torchvision")
    print("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

print()
print("=" * 60)
