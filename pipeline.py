import datetime
from functools import lru_cache
import json
import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import cv2

from audio_analyzer import AudioAnalyzer
from llm_analyzer import LLMAnalyzer
from postprocessor import PostProcessor
from video_utils import download_video, extract_audio, extract_frames
from vision_analyzer import VisionAnalyzer


@lru_cache(maxsize=4)
def _get_audio_analyzer(model_size):
    return AudioAnalyzer(model_size=model_size)


@lru_cache(maxsize=1)
def _get_vision_analyzer():
    return VisionAnalyzer()


@lru_cache(maxsize=1)
def _get_llm_analyzer():
    return LLMAnalyzer()


@lru_cache(maxsize=1)
def _get_postprocessor():
    return PostProcessor()


def _get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = frame_count / fps if fps > 0 else 0
    cap.release()

    return {
        "frameCount": frame_count,
        "fps": fps,
        "video_duration_seconds": duration_seconds,
        "video_duration_formatted": str(datetime.timedelta(seconds=int(duration_seconds))),
        "analysis_timestamp": datetime.datetime.now().isoformat(),
    }


def run_analysis(
    video,
    is_url=False,
    fps=1.0,
    output_dir="results",
    whisper_model="base",
    download_quality="best[height<=480]/worst",
    analyze_audio=True,
    max_frames=None,
):
    """Runs the full course-work video analysis pipeline and returns the report."""
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")

    video_path = video
    if is_url:
        print(f"Загрузка видео по URL: {video} ...")
        video_path = download_video(
            video,
            quality=download_quality,
            output=os.path.join(output_dir, "downloaded.mp4"),
        )

    audio_path = os.path.join(output_dir, "audio.wav")
    extracted_audio = extract_audio(video_path, audio_path) if analyze_audio else None

    print("Нарезка видео на кадры...")
    frames = extract_frames(video_path, frames_dir, frame_rate=fps, max_frames=max_frames)
    if max_frames is not None:
        print(f"Быстрый режим: анализируются первые {len(frames)} кадров.")
    source_info = _get_video_info(video_path)
    source_info["requested_fps"] = fps
    source_info["frames_extracted"] = len(frames)
    source_info["max_frames"] = max_frames
    source_info["audio_analysis_enabled"] = analyze_audio

    vision_analyzer = _get_vision_analyzer()
    llm_analyzer = _get_llm_analyzer()
    postprocessor = _get_postprocessor()

    audio_segments = []
    audio_alerts = []
    if extracted_audio:
        audio_analyzer = _get_audio_analyzer(whisper_model)
        audio_segments = audio_analyzer.transcribe(extracted_audio)
        trigger_words = ["оружие", "убить", "бомба", "взрыв", "кровь", "драка", "сука", "бля", "черт"]
        audio_alerts = audio_analyzer.find_trigger_words(audio_segments, trigger_words)

    frames_data = {}
    print(f"Начало анализа {len(frames)} кадров...")

    for time_sec, frame_path in frames:
        print(f"Анализ кадра на {time_sec:.1f} секунде...")
        v_results = vision_analyzer.analyze_frame(frame_path)
        llm_res = llm_analyzer.analyze_frame(frame_path, v_results=v_results)

        frames_data[time_sec] = {
            "text": v_results["text"],
            "ocr_detections": v_results["ocr_detections"],
            "yolo_objects": v_results["yolo_objects"],
            "object_detections": v_results["object_detections"],
            "image_classifier_labels": v_results["image_classifier_labels"],
            "image_classifier_predictions": v_results["image_classifier_predictions"],
            "vision_models": v_results["vision_models"],
            "llm_analysis": llm_res,
        }

    print("Постобработка результатов...")
    deduplicated_text = postprocessor.deduplicate_text(frames_data)
    merged_objects = postprocessor.merge_detections(frames_data, window_sec=3.0)

    report_json = os.path.join(output_dir, "report.json")
    report = postprocessor.generate_report(
        source_info,
        frames_data,
        audio_alerts,
        deduplicated_text,
        merged_objects,
        report_json,
    )

    report_md = report_json.replace(".json", ".md")
    report["output_files"] = {
        "json": report_json,
        "markdown": report_md,
        "frames_dir": frames_dir,
        "audio": audio_path if extracted_audio else None,
    }

    with open(report_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=4)

    print("Пайплайн успешно завершен! Проверьте папку", output_dir)
    return report
