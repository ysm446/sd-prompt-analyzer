"""
Gradio UI実装
"""
import gradio as gr
import json
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from src.core.image_parser import ImageParser
from src.core.model_manager import ModelManager
from src.core.vlm_interface import VLMInterface
from src.utils.config_loader import ConfigLoader

# GGUF対応（llama-cpp-pythonがインストールされている場合のみ）
try:
    from src.core.vlm_interface_gguf import VLMInterfaceGGUF
    GGUF_AVAILABLE = True
except ImportError:
    GGUF_AVAILABLE = False
    print("Warning: llama-cpp-python is not installed. GGUF models will not be available.")


class PromptAnalyzerUI:
    """メインUIクラス"""

    def __init__(self, config: Dict):
        """
        Args:
            config: settings.yamlから読み込んだ設定
        """
        self.config = config
        self.model_manager = ModelManager(config['paths']['models_dir'])
        self.current_vlm: Optional[VLMInterface] = None
        self.current_image_path: Optional[str] = None
        self.current_metadata: Optional[Dict] = None
        self.selected_model_path: Optional[str] = None  # 選択されているモデルのパス
        self.last_model_cache_file = Path(".last_model_cache.json")

        # モデルプリセットを読み込み
        config_loader = ConfigLoader()
        self.model_presets = config_loader.load_model_presets()

    def create_interface(self) -> gr.Blocks:
        """
        Gradio UIを構築

        UI構成:
        - タブ1: 画像分析
        - タブ2: モデル管理
        - タブ3: 設定
        """
        # カスタムCSS（フォント変更）
        custom_css = """
        * {
            font-family: "Segoe UI", "Yu Gothic", "Meiryo", Arial, sans-serif !important;
        }
        """

        # キャッシュから推論設定を読み込み（なければconfigのデフォルト値を使用）
        cached_settings = self.load_inference_settings()
        initial_temperature = cached_settings.get('temperature', self.config['inference']['temperature'])
        initial_max_tokens = cached_settings.get('max_tokens', self.config['inference']['max_tokens'])
        initial_top_p = cached_settings.get('top_p', self.config['inference']['top_p'])

        with gr.Blocks(title="SD Prompt Analyzer", css=custom_css) as interface:
            gr.Markdown("# SD Prompt Analyzer")
            gr.Markdown("Stable Diffusion画像のプロンプトを分析するツール")

            with gr.Tabs():
                # タブ1: 画像分析
                with gr.Tab("画像分析"):
                    with gr.Row():
                        # 左側: 画像表示
                        with gr.Column(scale=1):
                            image_display = gr.Image(
                                label="ここに画像をドロップ",
                                type="filepath",
                                sources=["upload"],
                                height=400
                            )

                            # プロンプト情報表示
                            with gr.Accordion("プロンプト情報", open=True):
                                prompt_display = gr.Textbox(
                                    label="Prompt",
                                    lines=3,
                                    interactive=False
                                )
                                negative_prompt_display = gr.Textbox(
                                    label="Negative Prompt",
                                    lines=2,
                                    interactive=False
                                )
                                settings_display = gr.Code(
                                    label="Settings",
                                    language="json",
                                    interactive=False,
                                    lines=5
                                )

                        # 右側: チャット
                        with gr.Column(scale=1):
                            chatbot = gr.Chatbot(label="AI分析", height=500)
                            clear_btn = gr.Button("🗑️ チャット履歴をクリア", size="sm", variant="secondary")
                            context_info = gr.Markdown(
                                value="<small style='color: gray;'>--</small>",
                                elem_id="context-info"
                            )

                            # 質問プリセットボタン
                            gr.Markdown("### クイック質問")
                            with gr.Row():
                                preset_btn_1 = gr.Button("📸 この画像について説明", size="sm")
                                preset_btn_2 = gr.Button("✅ プロンプトとの一致確認", size="sm")
                            with gr.Row():
                                preset_btn_3 = gr.Button("✨ プロンプト改善案", size="sm")
                                preset_btn_4 = gr.Button("📝 詳細プロンプト提案", size="sm")

                            user_input = gr.Textbox(
                                label="質問を入力",
                                placeholder="または、上のボタンから質問を選択。Enterで送信",
                                lines=1,
                                max_lines=1
                            )
                            submit_btn = gr.Button("送信", variant="primary")

                            # モデル選択
                            model_dropdown = gr.Dropdown(
                                label="使用するモデル",
                                choices=[],
                                value=None,
                                interactive=True
                            )
                            load_model_btn = gr.Button("モデルをロード")
                            model_status = gr.Textbox(
                                label="モデル状態",
                                value="モデル未ロード",
                                interactive=False
                            )

                # タブ2: モデル管理
                with gr.Tab("モデル管理"):
                    gr.Markdown("### ローカルモデル")
                    refresh_models_btn = gr.Button("モデル一覧を更新")
                    local_models_display = gr.DataFrame(
                        headers=["モデル名", "パス", "サイズ"],
                        datatype=["str", "str", "str"],
                        label="保存済みモデル"
                    )

                    gr.Markdown("### モデルをダウンロード")
                    with gr.Row():
                        with gr.Column():
                            preset_dropdown = gr.Dropdown(
                                label="プリセット",
                                choices=list(self.model_presets.keys()),
                                value=None
                            )
                            repo_id_input = gr.Textbox(
                                label="Repository ID",
                                placeholder="Qwen/Qwen2-VL-7B-Instruct",
                                value=""
                            )
                            local_name_input = gr.Textbox(
                                label="ローカル保存名",
                                placeholder="qwen2-vl-7b",
                                value=""
                            )
                            download_btn = gr.Button("ダウンロード開始", variant="primary")

                        with gr.Column():
                            preset_info = gr.Markdown("プリセットを選択すると詳細が表示されます")
                            download_status = gr.Textbox(
                                label="ダウンロード状態",
                                value="",
                                interactive=False,
                                lines=5
                            )

                # タブ3: 設定
                with gr.Tab("設定"):
                    with gr.Row():
                        with gr.Column():
                            temperature_slider = gr.Slider(
                                label="Temperature",
                                info="ランダム性を制御（低い値=正確、高い値=創造的）。画像分析では0.1～0.3を推奨",
                                minimum=0.0,
                                maximum=2.0,
                                value=initial_temperature,
                                step=0.1
                            )
                            max_tokens_slider = gr.Slider(
                                label="Max Tokens",
                                info="生成する最大トークン数（文章の長さ）",
                                minimum=64,
                                maximum=2048,
                                value=initial_max_tokens,
                                step=64
                            )
                            top_p_slider = gr.Slider(
                                label="Top P",
                                info="語彙の多様性を制御。0.9前後を推奨",
                                minimum=0.0,
                                maximum=1.0,
                                value=initial_top_p,
                                step=0.05
                            )

            # イベントハンドラー
            # 画像アップロード（changeイベントで処理）
            image_display.change(
                fn=self.on_image_upload,
                inputs=[image_display],
                outputs=[prompt_display, negative_prompt_display, settings_display]
            )

            # チャット
            submit_btn.click(
                fn=self.chat_with_image,
                inputs=[user_input, chatbot, temperature_slider, max_tokens_slider],
                outputs=[chatbot, user_input, context_info, model_status]
            )

            # Enterキーでも送信
            user_input.submit(
                fn=self.chat_with_image,
                inputs=[user_input, chatbot, temperature_slider, max_tokens_slider],
                outputs=[chatbot, user_input, context_info, model_status]
            )

            # 質問プリセットボタン
            preset_btn_1.click(
                fn=self.preset_question_1,
                inputs=[chatbot, temperature_slider, max_tokens_slider],
                outputs=[chatbot, user_input, context_info, model_status]
            )

            preset_btn_2.click(
                fn=self.preset_question_2,
                inputs=[chatbot, temperature_slider, max_tokens_slider],
                outputs=[chatbot, user_input, context_info, model_status]
            )

            preset_btn_3.click(
                fn=self.preset_question_3,
                inputs=[chatbot, temperature_slider, max_tokens_slider],
                outputs=[chatbot, user_input, context_info, model_status]
            )

            preset_btn_4.click(
                fn=self.preset_question_4,
                inputs=[chatbot, temperature_slider, max_tokens_slider],
                outputs=[chatbot, user_input, context_info, model_status]
            )

            clear_btn.click(
                fn=lambda: [],
                outputs=[chatbot]
            )

            # モデル管理
            refresh_models_btn.click(
                fn=self.refresh_local_models,
                outputs=[local_models_display, model_dropdown]
            )

            # モデルドロップダウンの変更時に選択を保存
            def save_selected_model(path):
                self.selected_model_path = path
                self.save_last_model_path(path) if path else None

            model_dropdown.change(
                fn=save_selected_model,
                inputs=[model_dropdown],
                outputs=[]
            )

            load_model_btn.click(
                fn=self.load_vlm_model,
                inputs=[model_dropdown],
                outputs=[model_status, context_info]
            )

            preset_dropdown.change(
                fn=self.update_preset_info,
                inputs=[preset_dropdown],
                outputs=[preset_info, repo_id_input, local_name_input]
            )

            download_btn.click(
                fn=self.download_model,
                inputs=[repo_id_input, local_name_input],
                outputs=[download_status]
            )

            # 推論設定の変更時にキャッシュを更新
            def on_settings_change(temp, tokens, top_p):
                self.save_inference_settings(temp, tokens, top_p)

            temperature_slider.change(
                fn=on_settings_change,
                inputs=[temperature_slider, max_tokens_slider, top_p_slider],
                outputs=[]
            )
            max_tokens_slider.change(
                fn=on_settings_change,
                inputs=[temperature_slider, max_tokens_slider, top_p_slider],
                outputs=[]
            )
            top_p_slider.change(
                fn=on_settings_change,
                inputs=[temperature_slider, max_tokens_slider, top_p_slider],
                outputs=[]
            )

            # 初期ロード
            interface.load(
                fn=self.refresh_local_models,
                outputs=[local_models_display, model_dropdown]
            )

        return interface

    def on_image_upload(self, image_path: str) -> Tuple:
        """画像がアップロードされたときの処理"""
        try:
            # 画像パスがNoneまたは空の場合はクリア
            if not image_path:
                self.current_image_path = None
                self.current_metadata = None
                return "", "", "{}"

            # 画像パスを保存
            self.current_image_path = image_path

            # メタデータを抽出
            self.current_metadata = ImageParser.extract_metadata(image_path)

            # SettingsをJSON文字列に変換
            settings_json = json.dumps(self.current_metadata['settings'], indent=2, ensure_ascii=False)

            return (
                self.current_metadata['prompt'],
                self.current_metadata['negative_prompt'],
                settings_json
            )
        except Exception as e:
            print(f"画像読み込みエラー: {e}")
            import traceback
            traceback.print_exc()
            # エラーが発生した場合も状態をクリア
            self.current_image_path = None
            self.current_metadata = None
            return "画像の読み込みに失敗しました。もう一度ドロップしてください。", "", "{}"

    def preset_question_1(self, history: List, temperature: float, max_tokens: int):
        """プリセット質問1: この画像について説明"""
        for result in self.chat_with_image("この画像について説明してください", history, temperature, max_tokens):
            yield result

    def preset_question_2(self, history: List, temperature: float, max_tokens: int):
        """プリセット質問2: プロンプトとの一致確認"""
        for result in self.chat_with_image("この画像とプロンプトは一致していますか?", history, temperature, max_tokens):
            yield result

    def preset_question_3(self, history: List, temperature: float, max_tokens: int):
        """プリセット質問3: プロンプト改善案"""
        for result in self.chat_with_image("改善したプロンプトを書いてください", history, temperature, max_tokens):
            yield result

    def preset_question_4(self, history: List, temperature: float, max_tokens: int):
        """プリセット質問4: 詳細プロンプト提案"""
        for result in self.chat_with_image("より詳細なプロンプトを提案してください", history, temperature, max_tokens):
            yield result

    def _get_model_status(self) -> str:
        """現在のモデル状態を取得"""
        if self.current_vlm is None:
            return "モデル未ロード"
        if self.selected_model_path:
            return f"✓ モデルロード済み: {Path(self.selected_model_path).name}"
        return "モデルロード済み"

    def chat_with_image(
        self,
        message: str,
        history: List,
        temperature: float,
        max_tokens: int
    ):
        """画像について質問（ストリーミング対応）"""
        max_tokens_int = int(max_tokens)

        if not message:
            yield history, "", self._get_context_info(history), self._get_model_status()
            return

        # モデルが未ロードで、モデルが選択されている場合は自動ロード
        if self.current_vlm is None and self.selected_model_path:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "モデルをロード中..."})
            yield history, "", "<small style='color: gray;'>モデルをロード中...</small>", "モデルをロード中..."

            # モデルをロード
            status, context = self.load_vlm_model(self.selected_model_path)

            if "✓" not in status:
                # ロード失敗
                history[-1]["content"] = f"エラー: モデルのロードに失敗しました\n{status}"
                yield history, "", "<small style='color: gray;'>--</small>", status
                return

            # ロード成功、メッセージを削除して再実行
            history.pop()
            history.pop()

        if self.current_vlm is None:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "エラー: モデルを選択してください"})
            yield history, "", "<small style='color: gray;'>--</small>", "エラー: モデルを選択してください"
            return

        if not self.current_image_path or self.current_metadata is None:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "エラー: 画像が読み込まれていません"})
            yield history, "", self._get_context_info(history), self._get_model_status()
            return

        # 現在の画像パス
        prompt_text = self.current_metadata['prompt']

        # ユーザーメッセージを先に追加
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": ""})

        try:
            # VLMでストリーミング分析
            response = ""
            for chunk in self.current_vlm.analyze_image_with_prompt_stream(
                image_path=self.current_image_path,
                prompt_text=prompt_text,
                question=message,
                temperature=temperature,
                max_tokens=max_tokens_int
            ):
                response += chunk
                history[-1]["content"] = response
                yield history, "", self._get_context_info(history), self._get_model_status()

        except Exception as e:
            history[-1]["content"] = f"エラー: {str(e)}"
            yield history, "", self._get_context_info(history), self._get_model_status()

    def _get_context_info(self, history: List) -> str:
        """コンテキスト情報を取得（Markdown形式）"""
        if self.current_vlm is None:
            return "<small style='color: gray;'>--</small>"

        # 履歴のテキストを結合してトークン数を計算
        total_text = ""
        for msg in history:
            if isinstance(msg, dict) and "content" in msg:
                content = msg["content"]
                # contentがリストの場合はテキスト部分のみ抽出
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and item.get("type") == "text":
                            total_text += item.get("text", "") + "\n"
                        elif isinstance(item, str):
                            total_text += item + "\n"
                elif isinstance(content, str):
                    total_text += content + "\n"

        used_tokens = self.current_vlm.count_tokens(total_text)
        context_length = self.current_vlm.get_context_length()

        if context_length > 0:
            return f"<small style='color: gray;'>📊 CONTEXT: {used_tokens:,} / {context_length:,}</small>"
        else:
            return f"<small style='color: gray;'>📊 CONTEXT: {used_tokens:,}</small>"

    def refresh_local_models(self) -> Tuple:
        """ローカルモデル一覧を更新"""
        models = self.model_manager.list_local_models()

        # DataFrameデータを作成
        df_data = [[m['name'], m['path'], m['size']] for m in models]

        # ドロップダウン用の選択肢
        choices = [m['path'] for m in models]

        # 前回使用したモデルを読み込み
        last_model_path = self.load_last_model_path()

        # 前回のモデルがまだ存在する場合は初期値に設定
        if last_model_path and last_model_path in choices:
            self.selected_model_path = last_model_path
            return df_data, gr.Dropdown(choices=choices, value=last_model_path)

        return df_data, gr.Dropdown(choices=choices)

    def load_vlm_model(self, model_path: str) -> Tuple[str, str]:
        """VLMモデルをロード（GGUF/Transformers自動判定）"""
        if not model_path:
            return "エラー: モデルが選択されていません", "<small style='color: gray;'>--</small>"

        # 選択されたモデルパスを保存
        self.selected_model_path = model_path

        try:
            # 既存モデルをアンロード
            if self.current_vlm is not None:
                self.current_vlm.unload_model()

            # GGUFかTransformersかを判定
            model_path_obj = Path(model_path)
            is_gguf = False

            if model_path_obj.is_file() and model_path_obj.suffix == '.gguf':
                is_gguf = True
            elif model_path_obj.is_dir():
                # ディレクトリ内にGGUFファイルがあるか確認
                gguf_files = list(model_path_obj.glob('*.gguf'))
                if gguf_files:
                    is_gguf = True
                    model_path = str(gguf_files[0])  # 最初のGGUFファイルを使用

            # モデルタイプに応じてロード
            if is_gguf:
                # GGUFモデルをロード
                if not GGUF_AVAILABLE:
                    return "✗ エラー: llama-cpp-pythonがインストールされていません。GGUFモデルを使用するには、llama-cpp-pythonをインストールしてください。", "<small style='color: gray;'>--</small>"

                gguf_config = self.config.get('gguf', {})
                self.current_vlm = VLMInterfaceGGUF(
                    model_path=model_path,
                    n_ctx=gguf_config.get('n_ctx', 4096),
                    n_gpu_layers=gguf_config.get('n_gpu_layers', -1),
                    verbose=gguf_config.get('verbose', False)
                )
                model_type_label = "GGUF"
            else:
                # Transformersモデルをロード
                self.current_vlm = VLMInterface(
                    model_path=model_path,
                    device=self.config['model']['device'],
                    dtype=self.config['model']['dtype']
                )
                model_type_label = "Transformers"

            # コンテキスト長を取得
            context_length = self.current_vlm.get_context_length()
            if context_length > 0:
                context_info = f"<small style='color: gray;'>📊 CONTEXT: 0 / {context_length:,}</small>"
            else:
                context_info = "<small style='color: gray;'>📊 CONTEXT: 0</small>"

            # 最後に使用したモデルとして保存
            self.save_last_model_path(model_path)

            return f"✓ モデルをロードしました [{model_type_label}]: {Path(model_path).name}", context_info

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return f"✗ エラー: {str(e)}\n\n詳細:\n{error_detail}", "<small style='color: gray;'>--</small>"

    def update_preset_info(self, preset_name: str) -> Tuple:
        """プリセット情報を表示"""
        if not preset_name or preset_name not in self.model_presets:
            return "プリセットを選択すると詳細が表示されます", "", ""

        preset = self.model_presets[preset_name]

        # モデルタイプを取得（GGUFかどうか）
        model_type = preset.get('model_type', 'transformers')
        model_type_label = "GGUF" if model_type == 'gguf' else "Transformers"

        info_md = f"""
### {preset_name}

**モデルタイプ**: {model_type_label}
**説明**: {preset['description']}
**推奨用途**: {preset['recommended_for']}
**Repository ID**: `{preset['repo_id']}`
"""

        # GGUFの場合はファイル名も表示
        if 'filename' in preset:
            info_md += f"\n**ファイル名**: `{preset['filename']}`"

        return info_md, preset['repo_id'], preset['local_name']

    def save_last_model_path(self, model_path: str):
        """最後に使用したモデルのパスを保存（settings含む）"""
        try:
            # 既存のデータを読み込み
            data = {}
            if self.last_model_cache_file.exists():
                try:
                    data = json.loads(self.last_model_cache_file.read_text(encoding='utf-8'))
                except:
                    pass

            # モデルパスを更新
            data["last_model"] = model_path

            self.last_model_cache_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        except Exception as e:
            print(f"警告: モデルパスの保存に失敗しました: {e}")

    def save_inference_settings(self, temperature: float, max_tokens: int, top_p: float):
        """推論設定を保存"""
        try:
            # 既存のデータを読み込み
            data = {}
            if self.last_model_cache_file.exists():
                try:
                    data = json.loads(self.last_model_cache_file.read_text(encoding='utf-8'))
                except:
                    pass

            # 設定を更新
            data["inference_settings"] = {
                "temperature": temperature,
                "max_tokens": int(max_tokens),
                "top_p": top_p
            }

            self.last_model_cache_file.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
        except Exception as e:
            print(f"警告: 推論設定の保存に失敗しました: {e}")

    def load_last_model_path(self) -> Optional[str]:
        """最後に使用したモデルのパスを読み込み"""
        try:
            if self.last_model_cache_file.exists():
                data = json.loads(self.last_model_cache_file.read_text(encoding='utf-8'))
                return data.get("last_model")
        except Exception as e:
            print(f"警告: モデルパスの読み込みに失敗しました: {e}")
        return None

    def load_inference_settings(self) -> dict:
        """推論設定を読み込み"""
        try:
            if self.last_model_cache_file.exists():
                data = json.loads(self.last_model_cache_file.read_text(encoding='utf-8'))
                return data.get("inference_settings", {})
        except Exception as e:
            print(f"警告: 推論設定の読み込みに失敗しました: {e}")
        return {}

    def download_model(self, repo_id: str, local_name: str) -> str:
        """モデルをダウンロード（GGUF対応）"""
        if not repo_id:
            return "エラー: Repository IDを入力してください"

        try:
            # 選択されたプリセットから情報を取得
            filename = None
            for preset_name, preset in self.model_presets.items():
                if preset['repo_id'] == repo_id and preset['local_name'] == local_name:
                    filename = preset.get('filename', None)
                    break

            # ダウンロード実行
            downloaded_path = self.model_manager.download_model(
                repo_id=repo_id,
                local_name=local_name if local_name else None,
                filename=filename
            )

            if filename:
                return f"✓ GGUFダウンロード完了\n保存先: {downloaded_path}"
            else:
                return f"✓ ダウンロード完了\n保存先: {downloaded_path}"

        except Exception as e:
            import traceback
            error_detail = traceback.format_exc()
            return f"✗ ダウンロード失敗\nエラー: {str(e)}\n\n詳細:\n{error_detail}"
