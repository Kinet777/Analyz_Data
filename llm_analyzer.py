import json
import os

from PIL import Image


class LLMAnalyzer:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        self.model = None

        if self.api_key:
            try:
                import google.generativeai as genai

                genai.configure(api_key=self.api_key)
                self.model = genai.GenerativeModel(self.model_name)
                print(f"Инициализация LLMAnalyzer через Gemini API ({self.model_name})...")
            except Exception as exc:
                print(f"Gemini API недоступен, включен локальный fallback: {exc}")

        if not self.model:
            print("Инициализация локального эвристического LLM fallback...")

        # Маппинг объектов YOLO (COCO) в подклассы linza
        self.yolo_mapping = {
            'wine glass': 'ALCOHOL',
            'bottle': 'ALCOHOL',
            'cup': 'ALCOHOL',
            'knife': 'VIOLENCE',
            'gun': 'TERROR',
            'baseball bat': 'VIOLENCE',
            'scissors': 'VIOLENCE'
        }
        
        # Маппинг сцен ResNet (ImageNet)
        self.resnet_mapping = {
            'syringe': 'DRUGS',
            'pill bottle': 'DRUGS',
            'lighter': 'SMOKING',
            'match': 'SMOKING',
            'bikini': 'NUDE',
            'swimming trunks': 'NUDE',
            'brassiere': 'NUDE',
            'military uniform': 'EXTREMISM',
            'assault rifle': 'TERROR'
        }
        
        # Маппинг текста (OCR)
        self.text_mapping = {
            'казино': 'LUDOMANIA',
            'bet': 'LUDOMANIA',
            'ставки': 'LUDOMANIA',
            '1xbet': 'LUDOMANIA',
            'убью': 'VIOLENCE',
            'смерть': 'SUICIDE',
            'наркотик': 'DRUGS',
            'соль': 'DRUGS',
            'мефедрон': 'DRUGS',
            'лгбт': 'LGBT',
            'гей': 'LGBT',
            'война': 'ANTIWAR',
            'нет войне': 'ANTIWAR'
        }

    def _analyze_locally(self, v_results):
        detected_subclasses = set()

        if not v_results:
            return list(detected_subclasses)

        # 1. Проверяем объекты YOLO
        for obj in v_results.get("yolo_objects", []):
            if obj in self.yolo_mapping:
                detected_subclasses.add(self.yolo_mapping[obj])

        # 2. Проверяем сцены ResNet
        for scene in v_results.get("resnet_scene", []):
            scene_lower = scene.lower()
            for key, subclass in self.resnet_mapping.items():
                if key in scene_lower:
                    detected_subclasses.add(subclass)

        # 3. Проверяем текст OCR
        for text in v_results.get("text", []):
            text_lower = text.lower()
            for key, subclass in self.text_mapping.items():
                if key in text_lower:
                    detected_subclasses.add(subclass)

        return sorted(detected_subclasses)

    def _analyze_with_gemini(self, frame_path, v_results):
        prompt = (
            "Проанализируй кадр видео и результаты OCR/YOLO/ResNet для задачи "
            "классификации потенциально деструктивного контента. Верни только JSON "
            "в формате {\"subclasses\": [\"ALCOHOL\"], \"comment\": \"...\"}. "
            "Допустимые subclasses: ALCOHOL, DRUGS, SMOKING, VIOLENCE, TERROR, "
            "EXTREMISM, SUICIDE, LUDOMANIA, LGBT, ANTIWAR, OBSCENE_LANGUAGE, NUDE. "
            f"Результаты классических моделей: {json.dumps(v_results, ensure_ascii=False)}"
        )
        image = Image.open(frame_path).convert("RGB")
        response = self.model.generate_content([prompt, image])
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json\n", "", 1).replace("JSON\n", "", 1)

        data = json.loads(text)
        subclasses = data.get("subclasses", [])
        return sorted({str(item).upper() for item in subclasses if item})

    def analyze_frame(self, frame_path, v_results=None):
        """Returns destructive-content subclasses detected by Gemini or local fallback."""
        if self.model:
            try:
                return self._analyze_with_gemini(frame_path, v_results or {})
            except Exception as exc:
                print(f"Ошибка Gemini для кадра {frame_path}, используется fallback: {exc}")

        return self._analyze_locally(v_results or {})
