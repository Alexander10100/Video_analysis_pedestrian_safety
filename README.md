# Система видеоаналитики безопасности пешеходов

[![CI](https://github.com/Alexander10100/Video_analysis_pedestrian_safety/actions/workflows/ci-cd.yml/badge.svg)](https://github.com/Alexander10100/Video_analysis_pedestrian_safety/actions/workflows/ci-cd.yml)
![Python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![Poetry](https://img.shields.io/badge/deps-poetry-60A5FA)
![Ruff](https://img.shields.io/badge/lint-ruff-D7FF64)
![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen)

Система реального времени для детекции людей на видеопотоке: определяет нарушения ПДД,
классифицирует возраст (взрослый / ребёнок) и строит тепловую карту нарушений.

📄 **[Model Card](docs/MODEL_CARD.md)**: цель, метрики, ограничения, fairness, версии данных.
🔁 **[BPMN retraining-цикла](docs/bpmn/README.md)**: дрифт → переобучение → аукцион по AUC → валидация → деплой.
🛠 **[Как вносить изменения](CONTRIBUTING.md)**: окружение, линтеры, pre-commit, ветки.

---

## Возможности

- **Детекция людей**: YOLOv8 с трекингом объектов (модели n / s / m / l / x)
- **Нарушения ПДД**: переход дороги вне пешеходного перехода, проход на красный свет
- **Классификация возраста**: разделение на взрослых и детей
- **Зоны разметки**: настраиваемые полигоны дорог и пешеходных переходов для каждой камеры
- **Тепловая карта**: веб-интерфейс со сводной статистикой нарушений по всем камерам
- **Карта камер**: расположение камер на Яндекс.Картах
- **Источники**: HLS/RTSP-потоки через FFmpeg и локальные видеофайлы

---

## Архитектура

Три независимых веб-сервиса и набор CLI-пайплайнов поверх общего доменного ядра:

| Сервис | Команда | Порт | Назначение |
|---|---|---|---|
| Стрим-детекция | `psafety-stream` | 5000 | Детекция, трекинг, разметка зон, визуализация потока |
| Тепловая карта | `psafety-heatmap` | 5055 | Тепловая карта нарушений по камерам |
| Карта камер | `psafety-camera-map` | 5060 | Географическая карта камер |

| CLI | Назначение |
|---|---|
| `psafety-offline` | Офлайн-обработка видео: аннотированное видео и отчёт |
| `psafety-batch` | Пакетная обработка папок с видео и сводный отчёт |
| `psafety-report` | JSON-отчёт по журналу нарушений за период |
| `psafety-calibrate` | Калибровка классификатора возраста для камеры |

## Структура проекта

```
.
├── src/pedestrian_safety/        # Python-пакет (src-layout)
│   ├── config.py                 # настройки из переменных окружения / .env
│   ├── logging_config.py         # единое логирование (текст или JSON)
│   ├── core/                     # доменная логика, без Flask и YOLO
│   │   ├── zone_manager.py       #   зоны разметки
│   │   ├── traffic_light.py      #   определение сигнала светофора
│   │   ├── violation_detector.py #   правила нарушений + отрисовка
│   │   └── age_classifier.py     #   взрослый/ребёнок, EMA, трекер
│   ├── services/                 # веб-сервисы (Flask) + templates/
│   └── pipelines/                # офлайн, батч, отчёты, калибровка
├── tests/                        # pytest (190 тестов)
├── docs/
│   ├── MODEL_CARD.md
│   └── bpmn/                     # retraining.bpmn + svg + описание
├── zones.json                    # разметка зон по камерам
├── calibrations/<camera_id>.json # калибровки классификатора возраста
├── heatmap_assets/               # фон тепловой карты
├── pyproject.toml                # зависимости и настройки инструментов
├── poetry.lock                   # зафиксированные версии
├── poetry.toml                   # .venv внутри проекта
├── requirements.txt              # экспорт из poetry.lock (для Docker)
├── .pre-commit-config.yaml
├── Dockerfile / docker-compose.yml
└── .github/workflows/ci-cd.yml
```

---

## Быстрый старт

### 1. Установить Poetry (один раз)

```bash
pipx install poetry          # или: pip install --user poetry
poetry --version             # нужна версия 2.x
```

### 2. Создать виртуальное окружение и поставить зависимости

```bash
git clone https://github.com/Alexander10100/Video_analysis_pedestrian_safety.git
cd Video_analysis_pedestrian_safety
poetry install               # создаст ./.venv и поставит всё из poetry.lock
```

Окружение создаётся **внутри проекта** в папке `.venv` (настройка в `poetry.toml`).
В git оно не попадает (`.gitignore`), зато воспроизводится у всех одинаково по
`poetry.lock`. IDE (VS Code, PyCharm) подхватывают `.venv` автоматически.

```bash
poetry env activate          # показать команду активации
# Windows (PowerShell):  .\.venv\Scripts\Activate.ps1
# Linux / macOS:         source .venv/bin/activate
```

### 3. Запуск

```bash
# стрим из папки с видео
poetry run psafety-stream --folder video/cam1 --camera cam1

# HLS/RTSP-поток
poetry run psafety-stream --url "https://…/stream.m3u8" --camera cam1

poetry run psafety-heatmap      # http://localhost:5055
poetry run psafety-camera-map   # http://localhost:5060

# отчёт за последние сутки
poetry run psafety-report --camera cam1 --last 24h --output reports/cam1.json

# калибровка возраста для новой камеры
poetry run psafety-calibrate --video video/cam1/day.mp4 --camera cam1
```

Открыть браузер: <http://localhost:5000>. Подробности по параметрам: `--help` у любой команды.

---

## Конфигурация

Настройки читаются из переменных окружения (или файла `.env` в рабочей папке,
см. [`.env.example`](.env.example)):

| Переменная | По умолчанию | Описание |
|---|---|---|
| `PSAFETY_DATA_DIR` | текущая папка | Где лежат `zones.json`, `calibrations/`, `video/`, `reports/` |
| `PSAFETY_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING` / `ERROR` |
| `PSAFETY_LOG_JSON` | `0` | `1`: логи одной JSON-строкой (для Docker / ELK) |
| `FFMPEG_PATH` | `ffmpeg` из `PATH` | На Windows также ищется `ffmpeg/bin/ffmpeg.exe` |
| `YANDEX_MAPS_API_KEY` | — | Ключ для карты камер |
| `PORT` | 5055 / 5060 | Порт тепловой карты и карты камер |

---

## Docker

Все три сервиса сразу:

```bash
docker compose up -d
```

Только тепловая карта из готового образа Docker Hub:

```bash
docker pull ancici/video-analysis-pedestrian-safety-heatmap:latest
docker run -d --name heatmap -p 5055:5055 \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/reports_batches:/app/reports_batches \
  -v $(pwd)/heatmap_assets:/app/heatmap_assets \
  -v $(pwd)/heatmap_camera_positions.json:/app/heatmap_camera_positions.json \
  ancici/video-analysis-pedestrian-safety-heatmap:latest
docker logs -f heatmap      # логи;  docker rm -f heatmap — удалить
```

Перед первым запуском через compose создайте файлы, которые монтируются как volume:

```bash
mkdir -p calibrations video reports reports_batches heatmap_assets/backgrounds
touch violations_log.jsonl camera_points.json
```

Образ собирается из `requirements.txt`, который генерируется из `poetry.lock`, поэтому
версии в контейнере совпадают с локальными. PyTorch ставится в CPU-сборке той же
версии. Push в `main_branch` собирает и публикует образы в Docker Hub (см. CI).

---

## Разработка

```bash
poetry install                                          # + dev-зависимости
poetry run pre-commit install -t pre-commit -t pre-push # хуки git

poetry run ruff check .          # линтер
poetry run ruff format .         # форматирование
poetry run mypy                  # типы
poetry run pytest --cov          # тесты + покрытие
poetry run pre-commit run --all-files
```

| Инструмент | Что проверяет | Конфиг |
|---|---|---|
| **ruff** | PEP 8, импорты, баги (bugbear), упрощения, именование | `[tool.ruff]` в `pyproject.toml` |
| **ruff format** | единый стиль кода | `[tool.ruff.format]` |
| **mypy** | статическая типизация `src/` | `[tool.mypy]` |
| **pytest + coverage** | 190 тестов | `[tool.pytest.ini_options]` |
| **pre-commit** | всё перечисленное + гигиена файлов, `poetry check`, синхронность `requirements.txt`, запрет прямых коммитов в `main_branch`, тесты перед push | `.pre-commit-config.yaml` |

### CI/CD (GitHub Actions)

1. **lint**: `pre-commit run --all-files`.
2. **test**: `poetry install` → `mypy` → `pytest --cov` на Python 3.10 и 3.11.
3. **docker**: только для `main_branch` после зелёных lint и test; сборка и push трёх образов.

---

## Зоны разметки

Зоны настраиваются через веб-интерфейс детекции (http://localhost:5000) и сохраняются в `zones.json`:

```json
{
  "cameras": {
    "cam_01": {
      "label": "Перекрёсток ул. Ленина",
      "zones": [
        {
          "id": "uuid",
          "label": "Дорога №1",
          "type": "road",
          "polygon": [[0.1, 0.4], [0.9, 0.4], [0.9, 0.9], [0.1, 0.9]],
          "color": [0, 80, 220]
        },
        {
          "id": "uuid",
          "label": "Переход",
          "type": "crosswalk",
          "polygon": [[0.3, 0.5], [0.6, 0.5], [0.6, 0.7], [0.3, 0.7]],
          "has_light": true,
          "light_type": "pedestrian",
          "traffic_light_roi": [0.62, 0.3, 0.08, 0.15],
          "color": [220, 160, 0]
        }
      ]
    }
  }
}
```

Координаты нормализованы: `0.0` — левый/верхний край кадра, `1.0` — правый/нижний.
`light_type`: `pedestrian` — пешеходный светофор; `vehicle` — автомобильный, у которого
красный для машин означает «можно» для пешеходов.

## Калибровка классификатора возраста

Классификатор калибруется под каждую камеру отдельно, перед боевым использованием:

```bash
poetry run psafety-calibrate --video cam1.mp4 --camera cam_01
```

Результат сохраняется в `calibrations/cam_01.json` и подхватывается при запуске с
`--camera cam_01`. Без калибровки все люди считаются взрослыми (подробнее в [Model Card](docs/MODEL_CARD.md)).

## Отчёты о нарушениях

Нарушения пишутся в реальном времени в `violations_log.jsonl`, по одному событию на строку:

```json
{
  "timestamp": "2024-01-15T10:23:45.123456",
  "track_id": 42,
  "camera_id": "cam_01",
  "violation_type": "red_light",
  "zone_label": "Переход",
  "note": "",
  "age_label": "adult",
  "person_conf": 0.87,
  "age_conf": 0.92
}
```

Типы нарушений: `road_trespass` (выход на проезжую часть вне перехода) и
`red_light` (переход на красный). Сводные отчёты — `psafety-report` и тепловая
карта (http://localhost:5055).

---

## Changelog

### 0.2.0: production-рефакторинг
- src-layout пакет `pedestrian_safety`, консольные команды `psafety-*`
- Poetry (`pyproject.toml` + `poetry.lock`), окружение в `.venv`
- ruff, mypy, pre-commit, CI на Poetry
- конфигурация через переменные окружения, `logging` вместо `print`
- исправлено: потеря каждого второго кадра в стриме; падение HLS/RTSP на Linux/Docker
  (`FFMPEG_PATH`); двойная запись отчёта; дублирование логики отчётов
- документация: Model Card, BPMN retraining-цикла
