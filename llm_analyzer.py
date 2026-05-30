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

        # Маппинг объектов COCO из YOLO в подклассы linza.
        self.object_mapping = {
            'wine glass': 'ALCOHOL',
            'bottle': 'ALCOHOL',
            'cup': 'ALCOHOL',
            'knife': 'VIOLENCE',
            'gun': 'TERROR',
            'baseball bat': 'VIOLENCE',
            'scissors': 'VIOLENCE'
        }
        
        # Маппинг ImageNet-классов визуального классификатора в подклассы linza.
        self.image_classifier_mapping = {
            'syringe': 'DRUGS',
            'pill bottle': 'DRUGS',
            'medicine chest': 'DRUGS',
            'lighter': 'SMOKING',
            'match': 'SMOKING',
            'cigarette': 'SMOKING',
            'bikini': 'NUDE',
            'swimming trunks': 'NUDE',
            'brassiere': 'NUDE',
            'military uniform': 'EXTREMISM',
            'assault rifle': 'TERROR',
            'rifle': 'TERROR',
            'revolver': 'TERROR'
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

        # 1. Проверяем объекты YOLO. Новые результаты содержат confidence,
        # старый список yolo_objects оставлен как совместимый fallback.
        object_labels = []
        for detection in v_results.get("object_detections", []):
            if isinstance(detection, dict) and detection.get("confidence", 0) >= 0.45:
                object_labels.append(detection.get("label", ""))
        object_labels.extend(v_results.get("yolo_objects", []))

        for obj in object_labels:
            if obj in self.object_mapping:
                detected_subclasses.add(self.object_mapping[obj])

        # 2. Проверяем общий классификатор изображения.
        classifier_labels = []
        for prediction in v_results.get("image_classifier_predictions", []):
            if isinstance(prediction, dict) and prediction.get("confidence", 0) >= 0.05:
                classifier_labels.append(prediction.get("label", ""))
        classifier_labels.extend(v_results.get("image_classifier_labels", []))

        for label in classifier_labels:
            label_lower = label.lower()
            for key, subclass in self.image_classifier_mapping.items():
                if key in label_lower:
                    detected_subclasses.add(subclass)

        # 3. Проверяем текст OCR
        for text in v_results.get("text", []):
            text_lower = text.lower()
            for key, subclass in self.text_mapping.items():
                if key in text_lower:
                    detected_subclasses.add(subclass)

        return sorted(detected_subclasses)

    def _build_prompt(self, v_results):
        """Builds a strict moderation prompt for one video frame."""
        model_context = json.dumps(v_results, ensure_ascii=False, indent=2)
        return f"""
Ты являешься модулем смыслового анализа кадра в системе видеоаналитики для выявления потенциально деструктивного контента.

Твоя задача: проанализировать один кадр видео вместе с результатами классических моделей:
- OCR: текст, найденный на кадре;
- YOLOv8: объекты, найденные на кадре;
- EfficientNet/ImageNet-классификатор: вероятные визуальные категории кадра.

Нужно определить, есть ли на кадре признаки запрещенного, опасного или требующего модерации контента.
Анализируй именно видимый кадр и предоставленные результаты моделей. Не придумывай факты, которых нет на изображении или в OCR/YOLO/классификаторе изображения.

Классы и подклассы:
1. DRUGS:
   - ALCOHOL: алкогольные напитки, бутылки/бокалы в контексте употребления или рекламы алкоголя.
   - SMOKING: сигареты, вейпы, кальяны, зажигалки в контексте курения.
   - DRUGS: наркотики, шприцы, таблетки, порошки, упаковки, сленг и явная реклама наркотиков.
   - DRUGS2KIDS: наркотики, явно адресованные детям или подросткам.

2. DEVIANT:
   - VANDALISM: порча имущества, граффити как акт повреждения, разрушение объектов.
   - VIOLENCE: драки, оружие ближнего боя, кровь, избиение, угроза физического вреда.
   - SUICIDE: самоповреждение, суицидальные сцены, явные призывы к суициду.
   - KIDSSUICIDE: суицидальный контент с участием или адресацией детям.
   - OBSCENE_LANGUAGE: мат, грубая нецензурная лексика в OCR или очевидном контексте.

3. TERRORISM:
   - TERROR: оружие, взрывчатка, террористические угрозы, сцены подготовки нападения.
   - EXTREMISM: экстремистская символика, призывы, униформа/атрибутика в соответствующем контексте.
   - TERRORCONTENT: пропаганда, инструкции или материалы террористической направленности.

4. SEX:
   - NUDE: обнаженность или выраженно сексуализированная нагота.
   - SEX: сексуальные действия или явный сексуальный контент.
   - KIDSPORN: сексуальный контент с участием несовершеннолетних. Отмечай только при явных признаках.

5. ANTITRADITIONAL:
   - LGBT: явная тематическая маркировка или пропагандистский контент по теме LGBT.
   - CHILDFREE: явная агитация против рождения детей или семейных ценностей.

6. ANTIPATRIOTIC:
   - INOAGENT: маркировка или упоминание иностранного агента.
   - INOAGENTCONTENT: контент, явно связанный с иностранными агентами.
   - ANTIWAR: антивоенные лозунги, призывы или символика.

7. LUDOMANIA:
   - LUDOMANIA: казино, ставки, букмекерские сервисы, игровые автоматы, промокоды азартных игр.

Правила принятия решения:
- Возвращай подкласс только если есть достаточно сильный визуальный или текстовый признак.
- OCR важен: если на кадре найден текст "наркотики", "ставки", "казино", "убью", "бомба" и т.п., учитывай его даже если визуальные модели не уверены.
- YOLO и классификатор изображения являются подсказками, но могут ошибаться. Проверяй их по изображению и общему контексту.
- Не отмечай ALCOHOL только из-за обычной бутылки воды, стакана или чашки без алкогольного контекста.
- Не отмечай VIOLENCE только из-за человека, спорта или динамичной позы без явной драки/оружия/крови/угрозы.
- Не отмечай TERROR только из-за военной формы без оружия, угроз, символики или террористического контекста.
- Не отмечай NUDE/SEX при обычной пляжной, спортивной или медицинской сцене без явной сексуализации.
- Если кадр нейтральный, верни пустой массив subclasses.
- Если есть сомнение, лучше не добавлять подкласс, но укажи сомнение в поле comment.

Верни строго один JSON-объект без Markdown, без пояснений до или после JSON.
Схема ответа:
{{
  "subclasses": ["DRUGS"],
  "confidence": 0.0,
  "evidence": [
    {{
      "subclass": "DRUGS",
      "source": "ocr|yolo|classifier|image|combined",
      "reason": "краткое объяснение признака"
    }}
  ],
  "comment": "короткий общий комментарий на русском"
}}

Требования к JSON:
- subclasses: массив только из допустимых подклассов.
- confidence: число от 0 до 1 для общей уверенности.
- evidence: массив аргументов; для каждого найденного подкласса желательно указать источник.
- comment: одна короткая фраза.
- Не используй неизвестные подклассы.
- Не возвращай trailing comma.

Допустимые подклассы:
ALCOHOL, SMOKING, DRUGS, DRUGS2KIDS,
VANDALISM, VIOLENCE, SUICIDE, KIDSSUICIDE, OBSCENE_LANGUAGE,
TERROR, EXTREMISM, TERRORCONTENT,
NUDE, SEX, KIDSPORN,
LGBT, CHILDFREE,
INOAGENT, INOAGENTCONTENT, ANTIWAR,
LUDOMANIA.

Результаты классических моделей для этого кадра:
{model_context}
""".strip()

    def _analyze_with_gemini(self, frame_path, v_results):
        prompt = self._build_prompt(v_results)
        image = Image.open(frame_path).convert("RGB")
        response = self.model.generate_content([prompt, image])
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = text.strip("`")
            text = text.replace("json\n", "", 1).replace("JSON\n", "", 1)

        data = json.loads(text)
        subclasses = data.get("subclasses", [])
        allowed_subclasses = {
            "ALCOHOL", "SMOKING", "DRUGS", "DRUGS2KIDS",
            "VANDALISM", "VIOLENCE", "SUICIDE", "KIDSSUICIDE", "OBSCENE_LANGUAGE",
            "TERROR", "EXTREMISM", "TERRORCONTENT",
            "NUDE", "SEX", "KIDSPORN",
            "LGBT", "CHILDFREE",
            "INOAGENT", "INOAGENTCONTENT", "ANTIWAR",
            "LUDOMANIA",
        }
        return sorted({
            str(item).upper()
            for item in subclasses
            if str(item).upper() in allowed_subclasses
        })

    def analyze_frame(self, frame_path, v_results=None):
        """Returns destructive-content subclasses detected by Gemini or local fallback."""
        if self.model:
            try:
                return self._analyze_with_gemini(frame_path, v_results or {})
            except Exception as exc:
                print(f"Ошибка Gemini для кадра {frame_path}, используется fallback: {exc}")

        return self._analyze_locally(v_results or {})
