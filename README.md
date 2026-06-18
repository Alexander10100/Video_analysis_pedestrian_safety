# Система видеоаналитики безопасности пешеходов

Система реального времени для детекции людей на видеопотоке с определением нарушений ПДД, классификацией возраста (взрослый/ребёнок) и визуализацией тепловой карты нарушений.

---

## Возможности

- **Детекция людей** — YOLOv8 с трекингом объектов (модели n / s / m / l / x)
- **Нарушения ПДД** — переход дороги вне пешеходного перехода, проход на красный свет
- **Классификация возраста** — разделение на взрослых и детей
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

## Запуск через Docker

### 1. Установить Docker

Проверьте, что Docker установлен:

```bash
docker --version
```

Если Docker отсутствует, установите его согласно официальной документации.

---

### 2. Скачать образ из Docker Hub

```bash
docker pull ancici/video-analysis-pedestrian-safety-heatmap:latest
```

---

### 3. Подготовить директории

Создайте необходимые каталоги:

```bash
mkdir -p reports
mkdir -p reports_batches
mkdir -p heatmap_assets
```

Создайте файл настроек:

```bash
touch heatmap_camera_positions.json
```

---

### 4. Запустить контейнер

```bash
docker run -d \
  --name heatmap \
  -p 5055:5055 \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/reports_batches:/app/reports_batches \
  -v $(pwd)/heatmap_assets:/app/heatmap_assets \
  -v $(pwd)/heatmap_camera_positions.json:/app/heatmap_camera_positions.json \
  ancici/video-analysis-pedestrian-safety-heatmap:latest
```

---

### 5. Проверить запуск

Проверить работающий контейнер:

```bash
docker ps
```

Посмотреть логи:

```bash
docker logs -f heatmap
```

---

### 6. Открыть веб-интерфейс

Если приложение запущено локально:

```
http://localhost:5055
```

Если приложение работает на удалённом сервере:

```
http://<SERVER_IP>:5055
```

Например:

```
http://192.168.1.100:5055
```

---

### Управление контейнером

Остановить:

```bash
docker stop heatmap
```

Запустить снова:

```bash
docker start heatmap
```

Перезапустить:

```bash
docker restart heatmap
```

Удалить контейнер:

```bash
docker rm -f heatmap
```

Обновить до новой версии образа:

```bash
docker pull ancici/video-analysis-pedestrian-safety-heatmap:latest

docker rm -f heatmap

docker run -d \
  --name heatmap \
  -p 5055:5055 \
  -v $(pwd)/reports:/app/reports \
  -v $(pwd)/reports_batches:/app/reports_batches \
  -v $(pwd)/heatmap_assets:/app/heatmap_assets \
  -v $(pwd)/heatmap_camera_positions.json:/app/heatmap_camera_positions.json \
  ancici/video-analysis-pedestrian-safety-heatmap:latest
```


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
