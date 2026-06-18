# Система видеоаналитики безопасности пешеходов

Система реального времени для детекции людей на видеопотоке с определением нарушений ПДД, классификацией возраста (взрослый/ребёнок) и визуализацией тепловой карты нарушений.

---

## Возможности

- **Детекция людей** — YOLOv8 с трекингом объектов (модели n / s / m / l / x)
- **Нарушения ПДД** — переход дороги вне пешеходного перехода, проход на красный свет
- **Классификация возраста** — разделение на взрослых и детей с temporal-сглаживанием
- **Зоны разметки** — настраиваемые полигоны дорог и пешеходных переходов на камеру
- **Тепловая карта** — веб-интерфейс сводной статистики нарушений по всем камерам
- **Карта камер** — географическое расположение камер на Яндекс.Картах
- **Источники** — HLS/RTSP потоки через FFmpeg и локальные видеофайлы

---

## Архитектура

Система состоит из трёх независимых сервисов:

| Сервис | Порт | Назначение |
|---|---|---|
| `stream_detect` | 5000 | Детекция, трекинг, визуализация потока |
| `heatmap` | 5055 | Тепловая карта нарушений по камерам |
| `camera_map` | 5060 | Географическая карта камер |

---

## Структура файлов

```
.
├── zones.json                    # Зоны разметки по камерам
├── calibrations/                 # Калибровки классификатора возраста
│   └── <camera_id>.json
├── video/                        # Локальные видеофайлы
│   └── <camera_id>/
│       └── *.mp4
├── violations_log.jsonl          # Лог нарушений (JSONL)
├── reports/                      # Отчёты по нарушениям
├── reports_batches/              # Пакетные отчёты
├── heatmap_assets/               # Фоновые изображения для тепловой карты
│   └── backgrounds/
├── heatmap_camera_positions.json # Позиции камер на тепловой карте
└── camera_points.json            # Географические координаты камер
```

Перед первым запуском создайте недостающие файлы и папки:

```bash
mkdir -p calibrations video reports reports_batches heatmap_assets/backgrounds
touch violations_log.jsonl heatmap_camera_positions.json camera_points.json
echo '{"cameras":{}}' > zones.json
```

---

## Запуск в Docker

### Предварительные требования

- Docker Engine 24+ и Docker Compose v2
- Образы опубликованы на Docker Hub или собраны локально (см. ниже)

### Переменные окружения

Создайте файл `.env` в корне проекта:

```dotenv
# Имя пользователя на Docker Hub (для готовых образов)
# Если собираете локально — оставьте пустым или укажите "local"
DOCKERHUB_USERNAME=your_dockerhub_username

# Тег образа (по умолчанию: latest)
IMAGE_TAG=latest

# API-ключ Яндекс.Карт для сервиса camera_map (необязательно)
YANDEX_MAPS_API_KEY=your_yandex_api_key
```

### Файл docker-compose.yml

```yaml
services:
  stream_detect:
    image: ${DOCKERHUB_USERNAME:-local}/video-analysis-pedestrian-safety-stream-detect:${IMAGE_TAG:-latest}
    # Для HLS/RTSP потока замените на:
    # command: python stream_detect.py --url "https://..."
    command: python stream_detect.py --folder /app/video
    ports:
      - "5000:5000"
    volumes:
      - ./zones.json:/app/zones.json
      - ./calibrations:/app/calibrations
      - ./video:/app/video
      - ./violations_log.jsonl:/app/violations_log.jsonl
      - ./reports:/app/reports
      - ./heatmap_assets:/app/heatmap_assets

  heatmap:
    image: ${DOCKERHUB_USERNAME:-local}/video-analysis-pedestrian-safety-heatmap:${IMAGE_TAG:-latest}
    command: python heatmap_web.py
    ports:
      - "5055:5055"
    volumes:
      - ./reports:/app/reports
      - ./reports_batches:/app/reports_batches
      - ./heatmap_assets:/app/heatmap_assets
      - ./heatmap_camera_positions.json:/app/heatmap_camera_positions.json

  camera_map:
    image: ${DOCKERHUB_USERNAME:-local}/video-analysis-pedestrian-safety-camera-map:${IMAGE_TAG:-latest}
    command: python camera_map_web.py
    ports:
      - "5060:5060"
    environment:
      - YANDEX_MAPS_API_KEY=${YANDEX_MAPS_API_KEY:-}
    volumes:
      - ./zones.json:/app/zones.json
      - ./video:/app/video
      - ./camera_points.json:/app/camera_points.json
```

### Сборка образов локально

Если готовых образов на Docker Hub нет, соберите их из исходников. Проект использует многоэтапный `Dockerfile` с тремя финальными стейджами:

```bash
# stream_detect
docker build --target stream_detect \
  -t local/video-analysis-pedestrian-safety-stream-detect:latest .

# heatmap
docker build --target heatmap \
  -t local/video-analysis-pedestrian-safety-heatmap:latest .

# camera_map
docker build --target camera_map \
  -t local/video-analysis-pedestrian-safety-camera-map:latest .
```

> **Первая сборка занимает 10–20 минут** — скачиваются PyTorch (~700 МБ CPU-версия) и веса YOLOv8 n/s/m. Повторные сборки используют кэш слоёв.

### Запуск всех сервисов

```bash
docker compose up -d
```

Проверьте, что все три контейнера запустились:

```bash
docker compose ps
```

Откройте интерфейсы в браузере:

| Сервис | URL |
|---|---|
| Детекция потока | http://localhost:5000 |
| Тепловая карта | http://localhost:5055 |
| Карта камер | http://localhost:5060 |

### Запуск отдельного сервиса

```bash
# Только детекция потока
docker compose up -d stream_detect

# Только тепловая карта
docker compose up -d heatmap
```

### Подключение к HLS/RTSP потоку

По умолчанию `stream_detect` читает видеофайлы из папки `./video`. Чтобы переключиться на живой поток, измените `command` в `docker-compose.yml`:

```yaml
services:
  stream_detect:
    command: python stream_detect.py --url "rtsp://camera.example.com/stream"
```

Или передайте дополнительные параметры:

```yaml
    command: >
      python stream_detect.py
      --url "https://example.com/hls/stream.m3u8"
      --model s
      --conf 0.40
      --fpm 120
      --camera cam_01
```

| Параметр | По умолчанию | Описание |
|---|---|---|
| `--url` | — | URL HLS/RTSP потока |
| `--folder` | — | Путь к папке с видеофайлами |
| `--model` | `m` | Размер модели: n / s / m / l / x |
| `--conf` | `0.45` | Порог уверенности детекции |
| `--imgsz` | `640` | Размер входного изображения |
| `--fpm` | `60` | Лимит детекций в минуту |
| `--camera` | — | ID камеры для автозагрузки зон |
| `--port` | `5000` | Порт веб-сервера |

### Просмотр логов

```bash
# Все сервисы
docker compose logs -f

# Только детекция
docker compose logs -f stream_detect
```

### Остановка

```bash
docker compose down
```

---

## Зоны разметки

Зоны настраиваются через веб-интерфейс детекции (http://localhost:5000) и сохраняются в `zones.json`. Структура файла:

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

Координаты полигонов нормализованы: `0.0` — левый/верхний край кадра, `1.0` — правый/нижний.

---

## Калибровка классификатора возраста

Для точной классификации взрослый/ребёнок рекомендуется откалибровать классификатор под каждую камеру. Калибровка запускается отдельно перед боевым использованием:

```bash
python calibrate_camera.py --video cam1.mp4 --camera cam_01
```

Результат сохраняется в `calibrations/cam_01.json` и автоматически подхватывается при следующем запуске с `--camera cam_01`.

---

## Отчёты о нарушениях

Нарушения записываются в реальном времени в `violations_log.jsonl`. Каждая строка — отдельное событие:

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

Типы нарушений:

- `road_trespass` — выход на проезжую часть вне перехода
- `red_light` — переход на красный сигнал светофора

Сводные отчёты по камерам доступны через тепловую карту (http://localhost:5055).
