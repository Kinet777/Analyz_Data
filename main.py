import os
import argparse
from pipeline import run_analysis

def main():
    parser = argparse.ArgumentParser(description="Пайплайн Видеоаналитики (ПЗ 1-8)")
    parser.add_argument("--video", type=str, required=True, help="Путь к локальному видео или URL (youtube/rutube)")
    parser.add_argument("--is_url", action="store_true", help="Флаг, если --video это URL")
    parser.add_argument("--fps", type=float, default=1.0, help="Частота извлечения кадров (кадров в секунду)")
    parser.add_argument("--output", type=str, default="results", help="Папка для сохранения результатов")
    parser.add_argument("--whisper_model", type=str, default="base", help="Размер модели Whisper: tiny, base, small, medium, large")
    parser.add_argument("--no_audio", action="store_true", help="Не извлекать и не анализировать аудио")
    parser.add_argument("--max_frames", type=int, default=None, help="Ограничить число кадров для быстрого теста")
    parser.add_argument(
        "--download_quality",
        type=str,
        default="best[height<=480]/worst",
        help="Качество загрузки для URL в формате yt-dlp",
    )
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    run_analysis(
        video=args.video,
        is_url=args.is_url,
        fps=args.fps,
        output_dir=args.output,
        whisper_model=args.whisper_model,
        download_quality=args.download_quality,
        analyze_audio=not args.no_audio,
        max_frames=args.max_frames,
    )

if __name__ == "__main__":
    main()
