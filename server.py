import os
import shutil
import uuid
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from pipeline import run_analysis


app = FastAPI(
    title="Video Analysis Course Pipeline",
    description="HTTP API для курсовой: загрузка видео, распознавание объектов/текста/аудио, постобработка и JSON-отчет.",
    version="1.0.0",
)


class UrlAnalyzeRequest(BaseModel):
    url: str = Field(..., description="URL видео для yt-dlp")
    fps: float = Field(1.0, gt=0, le=10)
    whisper_model: str = "base"
    analyze_audio: bool = Field(
        True,
        description="Whether to extract and transcribe audio with Whisper.",
    )
    max_frames: int | None = Field(
        None,
        gt=0,
        description="Optional limit for the number of extracted frames to analyze.",
    )
    download_quality: str = Field(
        "best[height<=480]/worst",
        description="yt-dlp format selector. Lower quality downloads faster.",
    )


@app.get("/")
def root():
    return {
        "service": "video-analysis-pipeline",
        "status": "ok",
        "endpoints": {
            "health": "GET /health",
            "upload_video": "POST /analyze multipart/form-data: video=@file.mp4",
            "analyze_url": "POST /analyze-url JSON: {\"url\":\"...\"}",
        },
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
def analyze_uploaded_video(
    video: UploadFile = File(...),
    fps: float = Form(1.0),
    whisper_model: str = Form("base"),
    analyze_audio: bool = Form(True),
    max_frames: int | None = Form(None),
):
    if not video.filename:
        raise HTTPException(status_code=400, detail="Файл видео не передан")

    run_id = uuid.uuid4().hex[:12]
    output_dir = Path("results") / f"api_{run_id}"
    upload_dir = output_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    suffix = Path(video.filename).suffix or ".mp4"
    video_path = upload_dir / f"input{suffix}"

    with video_path.open("wb") as f:
        shutil.copyfileobj(video.file, f)

    try:
        report = run_analysis(
            video=str(video_path),
            is_url=False,
            fps=fps,
            output_dir=str(output_dir),
            whisper_model=whisper_model,
            analyze_audio=analyze_audio,
            max_frames=max_frames,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа видео: {exc}") from exc

    return JSONResponse(report)


@app.post("/analyze-url")
def analyze_video_url(payload: UrlAnalyzeRequest):
    run_id = uuid.uuid4().hex[:12]
    output_dir = os.path.join("results", f"api_{run_id}")

    try:
        report = run_analysis(
            video=payload.url,
            is_url=True,
            fps=payload.fps,
            output_dir=output_dir,
            whisper_model=payload.whisper_model,
            download_quality=payload.download_quality,
            analyze_audio=payload.analyze_audio,
            max_frames=payload.max_frames,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ошибка анализа URL: {exc}") from exc

    return JSONResponse(report)
