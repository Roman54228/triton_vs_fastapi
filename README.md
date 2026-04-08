# Triton vs FastAPI: сравнение инференс-серверов

Проект для сравнения **NVIDIA Triton Inference Server** и **FastAPI**

Включает:
- Triton с Python Backend и dynamic batching
- FastAPI с HTTP + gRPC эндпоинтами
- Prometheus + Grafana для мониторинга
- Бенчмарк-скрипт для нагрузочного тестирования

## Структура проекта

```
.
├── triton/                  # Triton: Dockerfile + model_repository
├── fastapi_app/             # FastAPI: Dockerfile + приложение
├── grafana/                 # Дашборды и provisioning для Grafana
├── prometheus.yml           # Конфигурация Prometheus
├── docker-compose.yml       # Запуск всего стека (GPU)
├── docker-compose.cpu.yml   # Запуск без GPU (для отладки)
├── export_models.py         # Экспорт моделей в ONNX
├── benchmark.py             # Нагрузочное тестирование
└── README.md
```

### Быстрая проверка GPU в Docker

```bash
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi
```

Если видите таблицу с GPU — всё ок.

## Запуск на сервере с GPU

### 1. Клонировать репозиторий

```bash
git clone https://github.com/Roman54228/triton_vs_fastapi
cd triton_vs_fastapi
```

### 2. Подготовить тестовое изображение

```bash
mkdir sample_data
# Положить тестовую картинку (любое JPEG-изображение)
cp /path/to/any/image.jpg sample_data/test.jpg
# Также скопировать для Triton model_repository
cp sample_data/test.jpg triton/model_repository/test.jpg
```

### 3. Запустить весь стек

```bash
docker compose up -d --build
```

Это поднимет 4 сервиса:

| Сервис     | Порт(ы)                          | Описание                    |
|------------|----------------------------------|-----------------------------|
| triton     | 8000 (HTTP), 8001 (gRPC), 8002 (metrics) | NVIDIA Triton Inference Server |
| fastapi    | 9000 (HTTP), 50051 (gRPC)       | FastAPI inference server     |
| prometheus | 9090                             | Сбор метрик                  |
| grafana    | 3000                             | Визуализация метрик          |

### 4. Проверить, что всё поднялось

```bash
# Статус контейнеров
docker compose ps

# Логи (если что-то не стартует)
docker compose logs triton
docker compose logs fastapi

# Проверить готовность Triton
curl -s http://localhost:8000/v2/health/ready
# Ожидаемый ответ: (пустой ответ с кодом 200)

# Проверить готовность FastAPI
curl -s http://localhost:9000/health
# Ожидаемый ответ: {"status":"healthy"}
```

## Примеры запросов

### Запрос к Triton (HTTP)

```bash
curl -X POST http://localhost:8000/v2/models/pipeline/infer \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": [
      {
        "name": "DUMMY",
        "shape": [1],
        "datatype": "FP32",
        "data": [0.0]
      }
    ],
    "outputs": [
      {"name": "DETECTIONS_YOLO1"},
      {"name": "DETECTIONS_YOLO2"}
    ]
  }'
```

### Запрос к FastAPI (HTTP)

```bash
curl -X POST http://localhost:9000/predict \
  -F "file=@sample_data/test.jpg"
```

### Запрос к FastAPI (gRPC) через клиент

```bash
# Из контейнера FastAPI
docker compose exec fastapi python3.11 -c "
from app.proto import inference_pb2, inference_pb2_grpc
import grpc
channel = grpc.insecure_channel('localhost:50051')
stub = inference_pb2_grpc.InferenceServiceStub(channel)
with open('/app/sample_data/test.jpg', 'rb') as f:
    resp = stub.Predict(inference_pb2.PredictRequest(image_data=f.read()))
print(f'Detections: {len(resp.detections)}')
for d in resp.detections:
    print(f'  box=({d.x1:.0f},{d.y1:.0f},{d.x2:.0f},{d.y2:.0f}) conf={d.confidence:.3f} label={d.label}')
"
```

## Бенчмарк

Скрипт `benchmark.py` отправляет параллельные запросы к обоим серверам и сравнивает latency.

### Установить зависимости (локально или на сервере)

```bash
pip install requests numpy
```

### Запуск

```bash
# Сравнить оба сервера: 50 запросов, 10 воркеров
python benchmark.py --target both -n 50 -w 10 --image sample_data/test.jpg

# Только Triton
python benchmark.py --target triton -n 100 -w 20

# Только FastAPI
python benchmark.py --target fastapi -n 100 -w 20 --image sample_data/test.jpg

# С кастомными URL (если запускаете удалённо)
python benchmark.py --target both \
  --triton-url http://<server-ip>:8000 \
  --fastapi-url http://<server-ip>:9000 \
  --image sample_data/test.jpg \
  -n 100 -w 20
```

## Проброс портов и просмотр Grafana

Если сервисы запущены на удалённом сервере, а вы хотите открыть Grafana и Prometheus у себя в браузере — пробросьте порты через SSH.

### Проброс портов

```bash
# Один порт (только Grafana)
ssh -L 3000:localhost:3000 user@<server-ip>

# Все нужные порты одной командой
ssh -L 3000:localhost:3000 \
    -L 9090:localhost:9090 \
    -L 8000:localhost:8000 \
    -L 9000:localhost:9000 \
    user@<server-ip>
```

Флаг `-L` пробрасывает порт: `-L <локальный_порт>:localhost:<удалённый_порт>`.

После подключения порты доступны на вашей машине, как будто сервисы запущены локально.

### Открыть Grafana в браузере

1. Перейти по адресу: **http://localhost:3000**
2. Логин: `admin` / Пароль: `admin`
3. Дашборд уже преднастроен — открыть **Dashboards** в боковом меню
4. Выбрать дашборд **Triton vs FastAPI**

Дашборд показывает:
- GPU Utilization
- Метрики Triton (request rate, latency, queue time)
- Метрики FastAPI (request rate, latency)

### Открыть Prometheus (опционально)

**http://localhost:9090** — можно проверить, что таргеты `triton` и `fastapi` в статусе UP:
перейти в **Status → Targets**.

## Остановка

```bash
docker compose down
```

## Запуск без GPU (для отладки)

```bash
docker compose -f docker-compose.cpu.yml up -d --build
```

В этом режиме запускаются только Triton и FastAPI без GPU (инференс на CPU, будет медленнее).
