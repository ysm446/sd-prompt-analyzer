"""
PNG画像からStable Diffusionのメタデータを抽出
"""
from pathlib import Path
from typing import Dict, Optional
from PIL import Image
import re
import json


class ImageParser:
    """PNG画像のメタデータ抽出クラス"""

    @staticmethod
    def extract_metadata(image_path: str) -> Dict:
        """
        PNGのメタデータを抽出

        Args:
            image_path: 画像ファイルパス

        Returns:
            {
                'path': str,
                'filename': str,
                'size': tuple,
                'prompt': str,
                'negative_prompt': str,
                'settings': dict  # steps, CFG, sampler等
            }
        """
        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(f"画像ファイルが見つかりません: {image_path}")

        with Image.open(image_path) as img:
            # 基本情報
            metadata = {
                'path': str(image_path),
                'filename': image_path.name,
                'size': img.size,
                'prompt': '',
                'negative_prompt': '',
                'settings': {}
            }

            # PNG infoからパラメータを取得
            if hasattr(img, 'info') and img.info:
                # 一般的なキー: 'parameters', 'Description', 'UserComment'
                params_text = None

                for key in ['parameters', 'Parameters', 'Description', 'UserComment']:
                    if key in img.info:
                        params_text = img.info[key]
                        break

                if params_text:
                    parsed = ImageParser.parse_parameters(params_text)
                    metadata.update(parsed)
                else:
                    # 通常のSDメタデータが見つからない場合、ComfyUIフォーマットを試す
                    comfy_data = ImageParser.parse_comfyui_metadata(img.info)
                    if comfy_data:
                        metadata.update(comfy_data)

        return metadata

    @staticmethod
    def parse_parameters(params_text: str) -> Dict:
        """
        parametersテキストをパース

        Args:
            params_text: PNG info の 'parameters' フィールド

        Returns:
            {
                'prompt': str,
                'negative_prompt': str,
                'settings': dict
            }

        例:
            入力: "masterpiece, 1girl\nNegative prompt: bad hands\nSteps: 28, Sampler: DPM++ 2M"
            出力: {
                'prompt': 'masterpiece, 1girl',
                'negative_prompt': 'bad hands',
                'settings': {'steps': 28, 'sampler': 'DPM++ 2M'}
            }
        """
        result = {
            'prompt': '',
            'negative_prompt': '',
            'settings': {}
        }

        if not params_text:
            return result

        # 行ごとに分割
        lines = params_text.strip().split('\n')

        # プロンプト（最初の行、Negative promptより前）
        prompt_lines = []
        settings_line = None

        for i, line in enumerate(lines):
            if line.lower().startswith('negative prompt:'):
                # Negative promptが見つかった
                negative_part = line.split(':', 1)[1].strip()
                result['negative_prompt'] = negative_part
            elif re.match(r'^\s*(Steps|Seed|Size|Model|CFG scale|Sampler)', line, re.IGNORECASE):
                # 設定行
                settings_line = line
                break
            else:
                # プロンプト行
                prompt_lines.append(line)

        result['prompt'] = '\n'.join(prompt_lines).strip()

        # 設定のパース
        if settings_line:
            result['settings'] = ImageParser._parse_settings_line(settings_line)

        return result

    @staticmethod
    def _parse_settings_line(settings_line: str) -> Dict:
        """
        設定行をパース

        Args:
            settings_line: "Steps: 28, Sampler: DPM++ 2M, CFG scale: 7"

        Returns:
            {'steps': 28, 'sampler': 'DPM++ 2M', 'cfg_scale': 7}
        """
        settings = {}

        # カンマで分割
        parts = settings_line.split(',')

        for part in parts:
            part = part.strip()
            if ':' not in part:
                continue

            key, value = part.split(':', 1)
            key = key.strip().lower().replace(' ', '_')
            value = value.strip()

            # 数値に変換を試みる
            try:
                if '.' in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass  # 文字列のまま

            settings[key] = value

        return settings

    @staticmethod
    def parse_comfyui_metadata(png_info: Dict) -> Optional[Dict]:
        """
        ComfyUIのメタデータをパース

        Args:
            png_info: PIL Imageのinfoディクショナリ

        Returns:
            {
                'prompt': str,
                'negative_prompt': str,
                'settings': dict
            }
            または None（ComfyUIメタデータが見つからない場合）
        """
        result = {
            'prompt': '',
            'negative_prompt': '',
            'settings': {}
        }

        # ComfyUIは 'prompt' または 'workflow' キーにJSONを保存
        workflow_json = None

        for key in ['prompt', 'workflow']:
            if key in png_info:
                try:
                    workflow_json = json.loads(png_info[key])
                    break
                except (json.JSONDecodeError, TypeError):
                    continue

        if not workflow_json:
            return None

        # ワークフローからプロンプトとネガティブプロンプトを抽出
        prompts = ImageParser._extract_comfyui_prompts(workflow_json)

        if prompts['positive']:
            result['prompt'] = '\n\n---\n\n'.join(prompts['positive'])
        if prompts['negative']:
            result['negative_prompt'] = '\n\n---\n\n'.join(prompts['negative'])

        # 設定情報を抽出
        settings = ImageParser._extract_comfyui_settings(workflow_json)
        if settings:
            result['settings'] = settings

        # プロンプトが見つかった場合のみ結果を返す
        if result['prompt'] or result['negative_prompt']:
            result['settings']['source'] = 'ComfyUI'
            return result

        return None

    @staticmethod
    def _extract_comfyui_prompts(workflow: Dict) -> Dict:
        """
        ComfyUIワークフローからプロンプトを抽出

        Args:
            workflow: ComfyUIのワークフローJSON

        Returns:
            {
                'positive': [プロンプト文字列のリスト],
                'negative': [ネガティブプロンプト文字列のリスト]
            }
        """
        positive_prompts = []
        negative_prompts = []

        # ワークフローの各ノードを確認
        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue

            # class_typeでノードタイプを判定
            class_type = node_data.get('class_type', '')

            # プロンプト関連のノードを検出
            # CLIPTextEncode系のノード
            if 'CLIPTextEncode' in class_type or 'Text' in class_type:
                inputs = node_data.get('inputs', {})

                # テキストフィールドを探す
                text = inputs.get('text', '')
                if text and isinstance(text, str):
                    # ノードのタイトルやclass_typeから正負を判定
                    node_title = node_data.get('_meta', {}).get('title', '').lower()

                    # ネガティブプロンプトの判定
                    is_negative = (
                        'negative' in node_title or
                        'negative' in class_type.lower() or
                        node_id.endswith('_negative') or
                        # inputsにconditioning_toがある場合、接続先から判定も可能
                        any('negative' in str(v).lower() for v in inputs.values() if isinstance(v, (str, list)))
                    )

                    if is_negative:
                        negative_prompts.append(text)
                    else:
                        positive_prompts.append(text)

        return {
            'positive': positive_prompts,
            'negative': negative_prompts
        }

    @staticmethod
    def _extract_comfyui_settings(workflow: Dict) -> Dict:
        """
        ComfyUIワークフローから設定情報を抽出

        Args:
            workflow: ComfyUIのワークフローJSON

        Returns:
            設定情報のディクショナリ
        """
        settings = {}

        for node_id, node_data in workflow.items():
            if not isinstance(node_data, dict):
                continue

            class_type = node_data.get('class_type', '')
            inputs = node_data.get('inputs', {})

            # KSampler系のノードから設定を抽出
            if 'KSampler' in class_type or 'Sampler' in class_type:
                if 'steps' in inputs:
                    settings['steps'] = inputs['steps']
                if 'cfg' in inputs:
                    settings['cfg_scale'] = inputs['cfg']
                if 'sampler_name' in inputs:
                    settings['sampler'] = inputs['sampler_name']
                if 'scheduler' in inputs:
                    settings['scheduler'] = inputs['scheduler']
                if 'seed' in inputs:
                    settings['seed'] = inputs['seed']
                if 'denoise' in inputs:
                    settings['denoise'] = inputs['denoise']

            # CheckpointLoaderノードからモデル情報を抽出
            elif 'CheckpointLoader' in class_type:
                if 'ckpt_name' in inputs:
                    settings['model'] = inputs['ckpt_name']

            # EmptyLatentImageノードから画像サイズを抽出
            elif 'EmptyLatentImage' in class_type or 'LatentImage' in class_type:
                if 'width' in inputs and 'height' in inputs:
                    settings['size'] = f"{inputs['width']}x{inputs['height']}"

        return settings
