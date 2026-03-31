# Детекция людей
## Демо
## Быстрый старт

### 1. Установка зависимостей

### 2. Запуск с HLS-потоком

```bash

python stream_detect.py --url "https://video2.interra.ru/glaz.naroda.112-dea00a5e37/tracks-v1/mono.ts.m3u8?token=3.9CzUU5u-AAAAAAAAAEsAAAAAAAAAAPASQXYIWbtJQP53vHF_eoWlKNLd"

```

### 3. Запуск с папки видео

```bash

python stream_detect.py --folder ./videos

```

## Параметры запуска

```bash

python stream_detect.py --url "..." --model s --conf 0.40 --fpm 30 --port 5000

```

| Параметр | Описание | Значения | По умолчанию |

|----------|----------|----------|--------------|

| `--url` | HLS/RTSP URL | строка | - |

| `--folder` | Папка с видео | путь | - |

| `--model` | Размер YOLOv8 | `n,s,m,l,x` | `m` |

| `--conf` | Порог уверенности | `0.1-0.9` | `0.45` |

| `--imgsz` | Размер входа модели | `320-1920` | `640` |

| `--fpm` | Детекций в минуту | `1-300` | `60` |

| `--port` | Порт Flask | `1024-65535` | `5000` |