import whisper

class AudioAnalyzer:
    def __init__(self, model_size="base"):
        print(f"Загрузка модели Whisper ({model_size})...")
        self.model = whisper.load_model(model_size)
        
    def transcribe(self, audio_path):
        """
        Транскрибирует аудио в текст.
        Возвращает список сегментов: [{'start': 0.0, 'end': 5.0, 'text': 'текст'}, ...]
        """
        print("Начало транскрибации аудио...")
        if not audio_path:
            return []
            
        result = self.model.transcribe(audio_path)
        
        segments = []
        for segment in result["segments"]:
            segments.append({
                "start": segment["start"],
                "end": segment["end"],
                "text": segment["text"].strip()
            })
            
        print("Транскрибация завершена.")
        return segments

    def find_trigger_words(self, segments, trigger_words):
        """Ищет триггерные (деструктивные) слова в тексте."""
        detections = []
        for segment in segments:
            text_lower = segment["text"].lower()
            for word in trigger_words:
                if word.lower() in text_lower:
                    detections.append({
                        "time": segment["start"],
                        "word": word,
                        "context": segment["text"]
                    })
        return detections
