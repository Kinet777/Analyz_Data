import json
from collections import defaultdict

class PostProcessor:
    def __init__(self):
        pass

    def deduplicate_text(self, frames_data):
        """Дедубликация одинакового текста, идущего подряд на соседних кадрах."""
        last_seen_texts = set()
        deduplicated = defaultdict(list)
        
        sorted_times = sorted(frames_data.keys())
        for time_sec in sorted_times:
            current_texts = set(frames_data[time_sec].get("text", []))
            new_texts = current_texts - last_seen_texts
            if new_texts:
                deduplicated[time_sec] = list(new_texts)
            last_seen_texts = current_texts
            
        return deduplicated

    def merge_detections(self, frames_data, window_sec=3.0):
        """Склеивает объекты, обнаруженные на соседних кадрах."""
        merged = defaultdict(list)
        for time_sec, data in frames_data.items():
            objects = data.get("object_detections") or data.get("yolo_objects", [])
            for obj in objects:
                if isinstance(obj, dict) and "label" in obj:
                    merged[obj["label"]].append(time_sec)
                elif isinstance(obj, str):
                    merged[obj].append(time_sec)
        return dict(merged)

    def _format_time(self, seconds):
        import datetime
        return str(datetime.timedelta(seconds=int(seconds))).zfill(8)

    def generate_report(self, source_info, frames_data, audio_alerts, deduplicated_text, merged_objects, output_file="report.json"):
        """Формирует итоговый сводный отчет в формате TIME_BASED_REPORT."""
        
        detections = []
        fps = source_info.get("fps", 30)
        
        # Сначала склеиваем временные интервалы по классам из frames_data (LLM analysis list)
        subclass_timeline = defaultdict(list)
        sorted_times = sorted(frames_data.keys())
        
        for time_sec in sorted_times:
            # Наш llm_analyzer теперь возвращает список найденных подклассов (subclasses)
            subclasses = frames_data[time_sec].get("llm_analysis", [])
            for sub in subclasses:
                subclass_timeline[sub.lower()].append(time_sec)
                
        # Функция для склейки интервалов (окно 3 секунды)
        def merge_times(times, window=3.0):
            if not times: return []
            intervals = []
            start = last = times[0]
            for t in times[1:]:
                if t - last <= window:
                    last = t
                else:
                    intervals.append((start, last))
                    start = last = t
            intervals.append((start, last))
            return intervals

        # 1. Добавляем видео-детекции (subclasses из кадра)
        for sub, times in subclass_timeline.items():
            intervals = merge_times(times)
            for start_sec, end_sec in intervals:
                # В случае единичного кадра интервал может быть 0, добавим 1 сек для красоты
                if end_sec == start_sec: end_sec += 1
                
                start_frame = int(start_sec * fps)
                end_frame = int(end_sec * fps)
                s_time = self._format_time(start_sec)
                e_time = self._format_time(end_sec)
                
                detections.append({
                    "startFrame": start_frame,
                    "endFrame": end_frame,
                    "start_time": s_time,
                    "end_time": e_time,
                    "time_interval": f"{s_time} - {e_time}",
                    "subclass": sub,
                    "confidence": 0.95, # Хардкод для эвристики
                    "type": "video"
                })

        # 2. Обрабатываем аудио-алерты
        # Нам нужно маппить слова в подклассы:
        audio_mapping = {
            "оружие": "violence", "убить": "violence", "бомба": "terror",
            "взрыв": "terror", "кровь": "violence", "драка": "violence",
            "сука": "obscene_language", "бля": "obscene_language", "черт": "obscene_language"
        }
        
        for alert in audio_alerts:
            t = alert['time']
            word = alert['word']
            subclass = "obscene_language"
            for k, v in audio_mapping.items():
                if k in word:
                    subclass = v
                    break
                    
            start_sec = t
            end_sec = t + 2 # Дадим 2 секунды на слово
            s_time = self._format_time(start_sec)
            e_time = self._format_time(end_sec)
            
            detections.append({
                "startFrame": int(start_sec * fps),
                "endFrame": int(end_sec * fps),
                "start_time": s_time,
                "end_time": e_time,
                "time_interval": f"{s_time} - {e_time}",
                "subclass": subclass,
                "confidence": 0.98,
                "type": "audio"
            })

        report = {
            "report_type": "TIME_BASED_REPORT",
            "source_info": source_info,
            "detections": detections,
            "postprocessing": {
                "deduplicated_text": {str(k): v for k, v in deduplicated_text.items()},
                "merged_object_detections": merged_objects,
            },
            "raw_summary": {
                "frames_analyzed": len(frames_data),
                "audio_alerts": audio_alerts,
            },
            "sourceInfo": {
                "frameCount": source_info.get("frameCount", 0),
                "fps": fps,
                "video_duration_seconds": source_info.get("video_duration_seconds", 0),
                "video_duration_formatted": source_info.get("video_duration_formatted", ""),
                "analysis_timestamp": source_info.get("analysis_timestamp", "")
            }
        }
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=4)
            
        print(f"JSON Отчет успешно сохранен в {output_file}")
        
        # Обновленный Markdown отчет для удобства чтения
        md_file = output_file.replace(".json", ".md")
        with open(md_file, "w", encoding="utf-8") as f:
            f.write("# Итоговый Отчет Анализа Видео\n\n")
            f.write(f"**Длительность**: {source_info.get('video_duration_formatted', '')} | **FPS**: {fps}\n\n")
            
            f.write("## Выявленные нарушения (Detections)\n")
            if detections:
                for d in sorted(detections, key=lambda x: x['startFrame']):
                    f.write(f"- **{d['time_interval']}**: [{d['type'].upper()}] {d['subclass']} (Уверенность: {d['confidence']:.2f})\n")
            else:
                f.write("- Нарушения не обнаружены.\n")

            f.write("\n## Постобработка\n")
            f.write(f"- Проанализировано кадров: {len(frames_data)}\n")
            f.write(f"- Дедублицированных текстовых фрагментов: {sum(len(v) for v in deduplicated_text.values())}\n")
            f.write(f"- Склеенных групп объектных детекций: {len(merged_objects)}\n")

        print(f"Markdown отчет сохранен в {md_file}")
        return report
