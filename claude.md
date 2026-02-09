# SD Prompt Analyzer

Stable DiffusionやComfyUIで生成された画像のメタデータ（プロンプト）を、Vision-Language Model（VLM）を使って対話的に評価・分析するWebアプリケーション。

## 技術スタック

- **Python 3.10+**
- **Gradio 4.0+** - WebUIフレームワーク
- **PyTorch 2.0+** - 深層学習フレームワーク
- **Transformers 4.40+** - VLMモデル推論
- **Qwen VL** - 主要なVLMモデル（Qwen2-VL, Qwen2.5-VL, Qwen3-VL対応）

## ディレクトリ構造

```
sd-prompt-analyzer/
├── app.py                      # エントリーポイント
├── config/
│   ├── settings.yaml           # アプリケーション設定
│   └── model_presets.yaml      # VLMモデルプリセット定義
├── src/
│   ├── core/
│   │   ├── image_parser.py     # PNGメタデータ抽出
│   │   ├── model_manager.py    # VLMダウンロード・管理
│   │   ├── vlm_interface.py    # Transformers形式VLM推論
│   │   └── vlm_interface_gguf.py # GGUF形式VLM推論
│   ├── ui/
│   │   └── gradio_app.py       # Gradio UIメインモジュール
│   └── utils/
│       └── config_loader.py    # YAML設定読み込み
├── data/
│   ├── sd_outputs/             # SD生成画像保存先
│   └── database/               # メタデータDB
├── models/                     # ローカルVLMモデル保存先
└── scripts/                    # セットアップ・テストスクリプト
```

## 主要コンポーネント

| モジュール | 役割 |
|-----------|------|
| `app.py` | ConfigLoaderで設定を読み込み、PromptAnalyzerUIを起動 |
| `image_parser.py` | PNG画像からプロンプト、ネガティブプロンプト、設定パラメータを抽出（SD/A1111形式とComfyUI形式の両方に対応） |
| `model_manager.py` | Hugging Faceからのモデルダウンロード、ローカルモデル管理 |
| `vlm_interface.py` | Transformers形式VLMのモデルロード・推論 |
| `vlm_interface_gguf.py` | GGUF形式（llama.cpp対応）VLMの推論 |
| `gradio_app.py` | 3タブ構成のWebUI（画像分析、モデル管理、設定） |

## 対応している画像形式

- **Stable Diffusion (A1111形式)**: `parameters`キーにプロンプト情報が保存されているPNG画像
- **ComfyUI形式**: `prompt`または`workflow`キーにワークフローJSONが保存されているPNG画像
  - ワークフロー内のCLIPTextEncodeノードからプロンプトを自動抽出
  - KSamplerノードから設定情報（steps, CFG, samplerなど）を抽出
  - 複数のプロンプトノードがある場合は`---`区切りで結合

## コーディング規約

- 言語: 日本語コメント可、変数名・関数名は英語
- 型ヒント: 必須
- docstring: Google スタイル
- インデント: スペース4つ
- 最大行長: 100文字

## 起動方法

```bash
python app.py
# ブラウザで http://localhost:7860 にアクセス
```

## 設定ファイル

- `config/settings.yaml`: モデル設定、推論パラメータ、UIテーマなど
- `config/model_presets.yaml`: VLMモデルのHugging FaceリポジトリIDとローカル保存名

## 開発時の注意

- VLMモデルは大容量（数GB〜数十GB）のため、`models/` ディレクトリはgit管理対象外
- GPU（CUDA）推奨、CPU動作も可能だが低速
- 新しいモデルプリセット追加時は `config/model_presets.yaml` を編集
