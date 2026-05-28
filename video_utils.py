import os
import cv2
import yt_dlp
import requests
from tqdm import tqdm
from moviepy.editor import VideoFileClip

def download_video(url, quality="best", output="video.mp4"):
    """Скачивает видео по URL с помощью yt-dlp."""
    if os.path.exists(output):
        try:
            os.remove(output)
        except OSError:
            pass
            
    ydl_opts = {
        "outtmpl": output,
        "format": quality,
        "quiet": False,
        "overwrites": True
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
    return output

def extract_frames(video_path, output_folder, frame_rate=2):
    """Нарезает видео на кадры с заданной частотой кадров в секунду."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        for filename in os.listdir(output_folder):
            file_path = os.path.join(output_folder, filename)
            if os.path.isfile(file_path):
                try:
                    os.remove(file_path)
                except OSError:
                    pass
    
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 30 # Фолбэк, если fps не определился
    
    frame_interval = int(fps / frame_rate) if fps > frame_rate else 1
    
    count = 0
    saved = 0
    frames_paths = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if count % frame_interval == 0:
            # Вычисляем таймкод в секундах
            time_sec = count / fps
            filename = os.path.join(output_folder, f"frame_{time_sec:.2f}.jpg")
            cv2.imwrite(filename, frame)
            frames_paths.append((time_sec, filename))
            saved += 1
            
        count += 1
        
    cap.release()
    print(f"Извлечено кадров: {saved}")
    return frames_paths

def extract_audio(video_path, output_audio="audio.wav"):
    """Извлекает аудиодорожку из видео."""
    try:
        video = VideoFileClip(video_path)
        if video.audio is not None:
            video.audio.write_audiofile(output_audio, codec='pcm_s16le', verbose=False, logger=None)
            print(f"Аудио успешно извлечено в {output_audio}")
            return output_audio
        else:
            print("В видео нет аудиодорожки.")
            return None
    except Exception as e:
        print(f"Ошибка при извлечении аудио: {e}")
        return None
