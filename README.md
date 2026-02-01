# SD Prompt Analyzer

Stable Diffusionで生成された画像とそのプロンプトを、Vision-Language Model（VLM）を使って評価・分析する対話型アプリケーション

## 主要機能

- PNG画像からメタデータ（プロンプト情報）を自動抽出
- VLMを使った画像とプロンプトの評価・分析
- ローカルVLMモデルとの対話インターフェース
- Hugging Faceからのモデル自動ダウンロード
- 複数モデルの切り替え機能

## 技術スタック

- **Python 3.10+**
- **Gradio 4.0+** - ユーザーインターフェース
- **Transformers** - VLMモデル推論
- **Pillow** - 画像処理
- **PyTorch** - 深層学習フレームワーク

## 対応モデル

- **Qwen2.5-VL系** (3B, 7B)
- **Qwen3-VL系** (4B, 8B) - 最新世代
- **Huihui-AI abliterated** (試験用)

## プロジェクト構成

```
sd-prompt-analyzer/
├── data/
│   ├── sd_outputs/          # SD生成画像の保存先
│   ├── database/            # メタデータDB
│   └── downloads/           # ダウンロードした画像等
├── models/                  # ローカルVLMモデル保存先
├── src/
│   ├── core/                # コアモジュール
│   │   ├── image_parser.py
│   │   ├── model_manager.py
│   │   ├── vlm_interface.py
│   │   └── vlm_interface_gguf.py  # GGUF形式対応
│   ├── ui/                  # UIモジュール
│   │   └── gradio_app.py
│   └── utils/               # ユーティリティ
│       └── config_loader.py
├── config/                  # 設定ファイル
│   ├── settings.yaml
│   └── model_presets.yaml
├── scripts/                 # スクリプト
│   ├── setup.py
│   ├── test_model.py
│   ├── check_cuda.py
│   └── check_gpu.py
├── requirements.txt
├── start.bat                # Windows起動スクリプト
└── app.py                   # エントリーポイント
```

## セットアップ方法

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd sd-prompt-analyzer
```

### 2. Conda環境の作成（推奨）

```bash
# Conda環境を作成
conda create -n sd-prompt-analyzer python=3.10 -y

# 環境をアクティベート
conda activate sd-prompt-analyzer
```

### 3. 依存関係のインストール

```bash
pip install -r requirements.txt
```

### 4. 初期セットアップ

```bash
python scripts/setup.py
```

## 使用方法

### アプリケーションの起動

```bash
# Conda環境をアクティベート
conda activate sd-prompt-analyzer

# アプリケーションを起動
python app.py
```

ブラウザで `http://localhost:7860` を開く

### 基本的な使い方

1. **モデル管理タブ**
   - プリセットから使用したいモデルを選択
   - ダウンロードボタンをクリック
   - ダウンロード完了後、モデルをロード

2. **画像分析タブ**
   - 画像フォルダのパスを入力
   - 「フォルダを読み込み」ボタンをクリック
   - 画像とプロンプト情報が表示される
   - チャット欄で質問を入力して分析

3. **設定タブ**
   - Temperature、Max Tokens等のパラメータを調整

## モデルのダウンロード

### UI上でダウンロード

1. 「モデル管理」タブを開く
2. プリセットを選択（例: qwen3-vl-4b）
3. 「ダウンロード開始」ボタンをクリック

### プリセット一覧

**Qwen2.5-VL シリーズ**
- **qwen2.5-vl-3b**: 軽量版（VRAM ~5GB）- 低メモリ環境向け
- **qwen2.5-vl-7b**: 高品質（VRAM ~8-10GB）

**Qwen3-VL シリーズ（最新世代）**
- **qwen3-vl-4b**: バランス型（VRAM ~6GB）- 推奨
- **qwen3-vl-8b**: 高性能（VRAM ~10-12GB）

**試験用**
- **huihui-qwen3-vl-4b-abliterated**: フィルタ除去版 4B
- **huihui-qwen3-vl-8b-abliterated**: フィルタ除去版 8B

## 設定ファイル

### config/settings.yaml

アプリケーションの基本設定を管理

```yaml
app:
  name: "SD Prompt Analyzer"
  version: "0.1.0"

paths:
  image_folder: "./data/sd_outputs"
  models_dir: "./models"

model:
  default: "qwen3-vl-4b"
  device: "auto"
  dtype: "bfloat16"

inference:
  temperature: 0.7
  max_tokens: 512
  top_p: 0.9

ui:
  theme: "soft"
  share: false
  server_port: 7860
```

### config/model_presets.yaml

モデルプリセットの定義

## トラブルシューティング

### モデルのロードに失敗する

- GPUメモリが不足している可能性があります
- より小さいモデル（3B/4Bモデル）を試してください
- `config/settings.yaml`の`device`を`cpu`に変更してください（遅くなります）

### 画像にメタデータがない

- Stable Diffusionで生成された画像でない可能性があります
- PNG形式で保存されているか確認してください
- A1111 WebUIやComfyUIで生成された画像は対応しています

### ダウンロードが進まない

- インターネット接続を確認してください
- Hugging Faceへのアクセスが制限されていないか確認してください

## システム要件

### 最小要件

- Python 3.10以上
- RAM: 16GB以上
- ストレージ: 20GB以上（モデル保存用）

### 推奨要件

- Python 3.10以上
- GPU: NVIDIA GPU（VRAM 12GB以上）
- RAM: 32GB以上
- ストレージ: 50GB以上

## ライセンス

MIT License

## 開発ロードマップ

### Phase 1: 基本機能（完了）
- プロジェクト構造作成
- 画像パーサー実装
- VLM推論インターフェース実装
- 基本的なGradio UI

### Phase 2: モデル管理（完了）
- モデルマネージャー実装
- HFダウンロード機能
- UIにモデル管理タブ追加

### Phase 3: UI改善（進行中）
- 画像ナビゲーション強化
- チャット履歴機能
- エラーハンドリング

### Phase 4: 将来的な拡張
- データベース機能
- バッチ処理
- プロンプト改善提案

## 貢献

バグ報告や機能提案は、GitHubのIssuesでお願いします。

## サポート

問題が発生した場合は、以下を確認してください：

1. 依存関係が正しくインストールされているか
2. Conda環境が正しくアクティベートされているか
3. モデルファイルが正しくダウンロードされているか

詳細なログを確認する場合は、ターミナルの出力を確認してください。
