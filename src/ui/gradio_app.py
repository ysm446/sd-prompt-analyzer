"""
Gradio UI実装
"""
import gradio as gr
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from PIL import Image

from src.core.image_parser import ImageParser
from src.core.model_manager import ModelManager
from src.core.vlm_interface import VLMInterface
from src.utils.image_utils import get_image_files
from src.utils.config_loader import ConfigLoader


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
        self.image_list: List[Path] = []
        self.current_index: int = 0
        self.current_metadata: Optional[Dict] = None

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
        with gr.Blocks(title="SD Prompt Analyzer") as interface:
            gr.Markdown("# SD Prompt Analyzer")
            gr.Markdown("Stable Diffusion画像のプロンプトを分析するツール")

            with gr.Tabs():
                # タブ1: 画像分析
                with gr.Tab("画像分析"):
                    with gr.Row():
                        # 左側: 画像表示
                        with gr.Column(scale=1):
                            image_display = gr.Image(label="画像", type="pil", height=400)

                            with gr.Row():
                                prev_btn = gr.Button("← 前へ", size="sm")
                                next_btn = gr.Button("次へ →", size="sm")

                            folder_path = gr.Textbox(
                                label="画像フォルダ",
                                value=self.config['paths']['image_folder'],
                                placeholder="./data/sd_outputs"
                            )
                            load_folder_btn = gr.Button("フォルダを読み込み", variant="primary")

                            image_counter = gr.Textbox(
                                label="画像番号",
                                value="0 / 0",
                                interactive=False
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
                                settings_display = gr.JSON(label="Settings", value={})

                        # 右側: チャット
                        with gr.Column(scale=1):
                            chatbot = gr.Chatbot(label="AI分析", height=500)
                            user_input = gr.Textbox(
                                label="質問を入力",
                                placeholder="この画像とプロンプトは一致していますか？",
                                lines=2
                            )
                            submit_btn = gr.Button("送信", variant="primary")
                            clear_btn = gr.Button("チャット履歴をクリア")

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
                                minimum=0.0,
                                maximum=2.0,
                                value=self.config['inference']['temperature'],
                                step=0.1
                            )
                            max_tokens_slider = gr.Slider(
                                label="Max Tokens",
                                minimum=64,
                                maximum=2048,
                                value=self.config['inference']['max_tokens'],
                                step=64
                            )
                            top_p_slider = gr.Slider(
                                label="Top P",
                                minimum=0.0,
                                maximum=1.0,
                                value=self.config['inference']['top_p'],
                                step=0.05
                            )

            # イベントハンドラー
            # フォルダ読み込み
            load_folder_btn.click(
                fn=self.load_image_folder,
                inputs=[folder_path],
                outputs=[image_display, image_counter, prompt_display,
                         negative_prompt_display, settings_display]
            )

            # 画像ナビゲーション
            next_btn.click(
                fn=self.next_image,
                outputs=[image_display, image_counter, prompt_display,
                         negative_prompt_display, settings_display]
            )

            prev_btn.click(
                fn=self.prev_image,
                outputs=[image_display, image_counter, prompt_display,
                         negative_prompt_display, settings_display]
            )

            # チャット
            submit_btn.click(
                fn=self.chat_with_image,
                inputs=[user_input, chatbot, temperature_slider, max_tokens_slider],
                outputs=[chatbot, user_input]
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

            load_model_btn.click(
                fn=self.load_vlm_model,
                inputs=[model_dropdown],
                outputs=[model_status]
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

            # 初期ロード
            interface.load(
                fn=self.refresh_local_models,
                outputs=[local_models_display, model_dropdown]
            )

        return interface

    def load_image_folder(self, folder_path: str) -> Tuple:
        """画像フォルダをスキャン"""
        self.image_list = get_image_files(folder_path)
        self.current_index = 0

        if not self.image_list:
            return None, "0 / 0", "", "", {}

        return self._get_current_image_data()

    def next_image(self) -> Tuple:
        """次の画像に移動"""
        if not self.image_list:
            return None, "0 / 0", "", "", {}

        self.current_index = (self.current_index + 1) % len(self.image_list)
        return self._get_current_image_data()

    def prev_image(self) -> Tuple:
        """前の画像に移動"""
        if not self.image_list:
            return None, "0 / 0", "", "", {}

        self.current_index = (self.current_index - 1) % len(self.image_list)
        return self._get_current_image_data()

    def _get_current_image_data(self) -> Tuple:
        """現在の画像とメタデータを取得"""
        if not self.image_list:
            return None, "0 / 0", "", "", {}

        current_image_path = self.image_list[self.current_index]

        # 画像を読み込み
        image = Image.open(current_image_path)

        # メタデータを抽出
        self.current_metadata = ImageParser.extract_metadata(str(current_image_path))

        # カウンター
        counter = f"{self.current_index + 1} / {len(self.image_list)}"

        return (
            image,
            counter,
            self.current_metadata['prompt'],
            self.current_metadata['negative_prompt'],
            self.current_metadata['settings']
        )

    def chat_with_image(
        self,
        message: str,
        history: List,
        temperature: float,
        max_tokens: int
    ) -> Tuple:
        """画像について質問"""
        if not message:
            return history, ""

        if self.current_vlm is None:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "エラー: モデルがロードされていません"})
            return history, ""

        if not self.image_list or self.current_metadata is None:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "エラー: 画像が読み込まれていません"})
            return history, ""

        # 現在の画像パス
        current_image_path = str(self.image_list[self.current_index])
        prompt_text = self.current_metadata['prompt']

        try:
            # VLMで分析
            response = self.current_vlm.analyze_image_with_prompt(
                image_path=current_image_path,
                prompt_text=prompt_text,
                question=message,
                temperature=temperature,
                max_tokens=int(max_tokens)
            )

            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})

        except Exception as e:
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": f"エラー: {str(e)}"})

        return history, ""

    def refresh_local_models(self) -> Tuple:
        """ローカルモデル一覧を更新"""
        models = self.model_manager.list_local_models()

        # DataFrameデータを作成
        df_data = [[m['name'], m['path'], m['size']] for m in models]

        # ドロップダウン用の選択肢
        choices = [m['path'] for m in models]

        return df_data, gr.Dropdown(choices=choices)

    def load_vlm_model(self, model_path: str) -> str:
        """VLMモデルをロード"""
        if not model_path:
            return "エラー: モデルが選択されていません"

        try:
            # 既存モデルをアンロード
            if self.current_vlm is not None:
                self.current_vlm.unload_model()

            # 新しいモデルをロード
            self.current_vlm = VLMInterface(
                model_path=model_path,
                device=self.config['model']['device'],
                dtype=self.config['model']['dtype']
            )

            return f"✓ モデルをロードしました: {Path(model_path).name}"

        except Exception as e:
            return f"✗ エラー: {str(e)}"

    def update_preset_info(self, preset_name: str) -> Tuple:
        """プリセット情報を表示"""
        if not preset_name or preset_name not in self.model_presets:
            return "プリセットを選択すると詳細が表示されます", "", ""

        preset = self.model_presets[preset_name]

        info_md = f"""
### {preset_name}

**説明**: {preset['description']}
**推奨用途**: {preset['recommended_for']}
**Repository ID**: `{preset['repo_id']}`
"""

        return info_md, preset['repo_id'], preset['local_name']

    def download_model(self, repo_id: str, local_name: str) -> str:
        """モデルをダウンロード"""
        if not repo_id:
            return "エラー: Repository IDを入力してください"

        try:
            downloaded_path = self.model_manager.download_model(
                repo_id=repo_id,
                local_name=local_name if local_name else None
            )

            return f"✓ ダウンロード完了\n保存先: {downloaded_path}"

        except Exception as e:
            return f"✗ ダウンロード失敗\nエラー: {str(e)}"
