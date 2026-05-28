# Курсовая работа: пайплайн видеоаналитики

Проект собирает все практические задания в один прикладной сценарий: пользователь загружает видеофайл или передает ссылку, система нарезает видео на кадры, распознает текст, объекты и аудио, выполняет постобработку результатов и формирует итоговый отчет в `JSON` и `Markdown`.

Прикладная задача: выявление и классификация потенциально деструктивного контента в видео.

## Соответствие ПЗ

| Требование | Где реализовано |
| --- | --- |
| ПЗ 1: обработка изображений на OpenCV | `video_utils.py`, `vision_analyzer.py`: чтение кадров через OpenCV, подготовка изображений к анализу |
| ПЗ 2: нарезка видео на изображения | `extract_frames()` в `video_utils.py` |
| ПЗ 3: распознавание текста из потока изображений | EasyOCR в `VisionAnalyzer.analyze_frame()` |
| ПЗ 4: извлечение звука и распознавание аудиоряда Whisper | `extract_audio()` и `AudioAnalyzer` |
| ПЗ 5: распознавание объектов на видео YOLO | YOLOv8 в `VisionAnalyzer` |
| ПЗ 6: распознавание объектов/сцен ResNet | ResNet50 в `VisionAnalyzer` |
| ПЗ 7: распознавание объектов/смысла с помощью LLM | `LLMAnalyzer`: Gemini API при наличии `GEMINI_API_KEY`, локальный fallback без ключа |
| ПЗ 8: обработка результатов | `PostProcessor`: дедубликация OCR-текста, склейка временных детекций объектов и классов |
| Курсовая: единый цикл | `pipeline.py`, CLI `main.py`, HTTP API `server.py` |

Итоговый сценарий соответствует формулировке: **загрузили файл -> распознали объекты -> распознали текст -> обработали результаты -> сформировали итоговый отчет**.

## Структура проекта

- `main.py` - запуск пайплайна из терминала.
- `server.py` - FastAPI-сервер для развертывания на ВМ и приема видео через HTTP.
- `pipeline.py` - единый сценарий анализа, общий для CLI и API.
- `video_utils.py` - скачивание видео, нарезка на кадры, извлечение аудио.
- `vision_analyzer.py` - EasyOCR, YOLOv8, ResNet50.
- `audio_analyzer.py` - Whisper и поиск триггерных слов.
- `llm_analyzer.py` - Gemini API или локальная эвристическая классификация.
- `postprocessor.py` - дедубликация текста, склейка детекций, генерация отчетов.
- `results/` - результаты запусков: `report.json`, `report.md`, кадры и аудио.

## Что получается на выходе

После анализа формируются:

- `report.json` - машинно-читаемый отчет.
- `report.md` - текстовый отчет для просмотра человеком.
- `frames/` - извлеченные кадры.
- `audio.wav` - извлеченная аудиодорожка, если она есть в видео.

Основные поля JSON:

- `source_info` - FPS, длительность, число кадров, время анализа.
- `detections` - найденные нарушения с таймкодами, типом источника (`video` или `audio`) и подклассом.
- `postprocessing.deduplicated_text` - OCR-текст после дедубликации.
- `postprocessing.merged_yolo_objects` - склеенные группы объектов YOLO.
- `output_files` - пути к созданным файлам отчета.

## Локальный запуск

### 1. Установка системных зависимостей

Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y python3-pip python3-venv ffmpeg libsm6 libxext6
```

### 2. Установка Python-зависимостей

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Gemini API для ПЗ 7

Если есть ключ Gemini, укажите его:

```bash
export GEMINI_API_KEY="ВАШ_КЛЮЧ"
```

Можно выбрать модель:

```bash
export GEMINI_MODEL="gemini-1.5-flash"
```

Если ключ не указан, проект все равно работает: `LLMAnalyzer` использует локальный fallback на основе результатов YOLO, ResNet и OCR.

### 4. Запуск анализа из терминала

Локальный файл:

```bash
python main.py --video "./test.mp4" --fps 1 --output results
```

Видео по ссылке:

```bash
python main.py --video "https://example.com/video" --is_url --fps 1 --output results
```

Для слабой ВМ можно ускорить тест:

```bash
python main.py --video "./test.mp4" --fps 0.5 --whisper_model tiny
```

## HTTP API

Запуск сервера:

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Проверка:

```bash
curl http://127.0.0.1:8000/health
```

Анализ загруженного файла:

```bash
curl -X POST "http://127.0.0.1:8000/analyze" \
  -F "video=@./test.mp4" \
  -F "fps=1" \
  -F "whisper_model=tiny"
```

Анализ видео по URL:

```bash
curl -X POST "http://127.0.0.1:8000/analyze-url" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/video","fps":1,"whisper_model":"tiny","download_quality":"best[height<=480]/worst"}'
```

Для быстрого анализа по ссылке сервер по умолчанию скачивает версию не выше 480p.
Если нужна максимальная четкость кадров, можно передать `"download_quality":"best"`,
но загрузка и анализ будут дольше.

Быстрая проверка API без долгого Whisper и без анализа всего видео:

```bash
curl -X POST "http://127.0.0.1:8000/analyze-url" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://rutube.ru/video/de671c5c7bdceba541ddc7d0d24c9b9c/","fps":0.2,"whisper_model":"tiny","analyze_audio":false,"max_frames":10,"download_quality":"worst"}'
```

Параметр `max_frames` ограничивает количество кадров, которые пойдут в OCR/YOLO/ResNet/LLM.
Параметр `analyze_audio:false` отключает извлечение аудио и Whisper, что особенно полезно для
быстрой демонстрации на слабой ВМ.

Ответ сервера - это JSON-отчет. То есть для сдачи развернутое решение может работать именно так: вы отправляете запрос на сервер, а сервер возвращает результат в терминал или любому клиентскому приложению.

## Развертывание на Timeweb Cloud

### 1. Создать сервер

Рекомендуемые параметры:

- ОС: Ubuntu 22.04 или Ubuntu 24.04.
- RAM: минимум 4 ГБ, лучше 8 ГБ.
- CPU: минимум 2 ядра, лучше 4.
- Диск: от 20 ГБ.

### 2. Подключиться по SSH

```bash
ssh root@IP_АДРЕС_СЕРВЕРА
```

### 3. Подготовить систему

```bash
apt update && apt upgrade -y
apt install -y python3-pip python3-venv git ffmpeg libsm6 libxext6
```

### 4. Загрузить проект

```bash
git clone https://github.com/ВАШ_ЛОГИН/ВАШ_РЕПОЗИТОРИЙ.git
cd ВАШ_РЕПОЗИТОРИЙ
```

### 5. Установить зависимости

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Первый запуск может быть долгим: Whisper, EasyOCR, YOLO и ResNet скачивают веса моделей.

### 6. Создать файл окружения

```bash
nano /root/video-analysis.env
```

Пример содержимого:

```bash
GEMINI_API_KEY=ВАШ_КЛЮЧ
GEMINI_MODEL=gemini-1.5-flash
```

Если Gemini не нужен для демонстрационного запуска, файл можно оставить без ключа или не подключать.

### 7. Запустить сервер вручную

```bash
source venv/bin/activate
set -a
source /root/video-analysis.env
set +a
uvicorn server:app --host 0.0.0.0 --port 8000
```

С локального компьютера:

```bash
curl http://IP_АДРЕС_СЕРВЕРА:8000/health
```

### 8. Настроить автозапуск через systemd

Создайте service-файл:

```bash
nano /etc/systemd/system/video-analysis.service
```

Вставьте, заменив путь `/root/ВАШ_РЕПОЗИТОРИЙ` на реальную папку проекта.
Например, если проект лежит в `/root/Analyz_Data`, используйте именно этот путь:

```ini
[Unit]
Description=Video Analysis Course API
After=network.target

[Service]
WorkingDirectory=/root/Analyz_Data
EnvironmentFile=-/root/video-analysis.env
ExecStart=/root/Analyz_Data/venv/bin/uvicorn server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Если `systemctl status video-analysis` показывает `status=200/CHDIR`, значит
`WorkingDirectory` указывает на папку, которой нет на сервере. Проверьте путь командой
`pwd` внутри папки проекта и вставьте получившееся значение в `WorkingDirectory`.

Запустите сервис:

```bash
systemctl daemon-reload
systemctl enable video-analysis
systemctl start video-analysis
systemctl status video-analysis
```

Логи:

```bash
journalctl -u video-analysis -f
```

### 9. Проверить API на ВМ

```bash
curl -X POST "http://IP_АДРЕС_СЕРВЕРА:8000/analyze" \
  -F "video=@./test.mp4" \
  -F "fps=0.5" \
  -F "whisper_model=tiny"
```

Если запрос вернул JSON с `report_type`, `source_info`, `detections` и `postprocessing`, развертывание работает.

## GitHub

Перед отправкой в GitHub проверьте, что в репозиторий не попадают тяжелые результаты и видео. Для этого добавлен `.gitignore`.

```bash
git init
git add .
git commit -m "Course video analysis pipeline"
git branch -M main
git remote add origin https://github.com/ВАШ_ЛОГИН/ВАШ_РЕПОЗИТОРИЙ.git
git push -u origin main
```

## Итоговая оценка готовности

Проект покрывает все пункты ПЗ и требование курсовой про единый рабочий сценарий.

- CLI: `python main.py --video ./test.mp4 --fps 1`.
- Развернутый API: `curl -X POST http://IP:8000/analyze -F video=@./test.mp4`.
