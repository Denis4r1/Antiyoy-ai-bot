import os
import time
import psutil
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
    multiprocess,
    CollectorRegistry,
)
from fastapi import Response
from fastapi.responses import PlainTextResponse
from ..config import PROMETHEUS_MULTIPROC_DIR
from .logging_config import get_logger

logger = get_logger("metrics")

# Создаем директорию для multiprocess метрик
os.makedirs(PROMETHEUS_MULTIPROC_DIR, exist_ok=True)
os.environ["PROMETHEUS_MULTIPROC_DIR"] = PROMETHEUS_MULTIPROC_DIR

# Метрики - счетчики
rooms_created_total = Counter("gameserver_rooms_created_total", "Total number of rooms created")

games_started_total = Counter("gameserver_games_started_total", "Total number of games started")

websocket_connections_total = Counter(
    "gameserver_websocket_connections_total", "Total WebSocket connections", ["type"]  # lobby, game
)

websocket_disconnections_total = Counter(
    "gameserver_websocket_disconnections_total", "Total WebSocket disconnections", ["type"]  # lobby, game
)

api_requests_total = Counter("gameserver_api_requests_total", "Total API requests", ["method", "endpoint", "status"])

errors_total = Counter("gameserver_errors_total", "Total errors", ["type"])  # websocket, api, game

# Метрики - измерения
active_rooms = Gauge("gameserver_active_rooms", "Number of active rooms")

active_games = Gauge("gameserver_active_games", "Number of active games")

lobby_connections = Gauge("gameserver_lobby_connections", "Number of active lobby WebSocket connections")

game_connections = Gauge("gameserver_game_connections", "Number of active game WebSocket connections")

memory_usage_bytes = Gauge("gameserver_memory_usage_bytes", "Memory usage in bytes", ["type"])  # rss, vms

# Метрики - гистограммы
request_duration = Histogram(
    "gameserver_request_duration_seconds", "Request duration in seconds", ["method", "endpoint"]
)

game_duration = Histogram(
    "gameserver_game_duration_seconds",
    "Game duration in seconds",
    buckets=[60, 300, 600, 1200, 1800, 3600, float("inf")],
)

lobby_wait_time = Histogram(
    "gameserver_lobby_wait_time_seconds",
    "Time spent waiting in lobby",
    buckets=[5, 10, 30, 60, 120, 300, float("inf")],
)


class GameServerMetrics:
    def __init__(self):
        self.process = psutil.Process()
        self.start_time = time.time()

    def update_system_metrics(self):
        """Обновление системных метрик"""
        try:
            memory_info = self.process.memory_info()
            memory_usage_bytes.labels(type="rss").set(memory_info.rss)
            memory_usage_bytes.labels(type="vms").set(memory_info.vms)
        except Exception as e:
            logger.warning(f"Failed to update system metrics: {e}")

    def update_game_metrics(self, rooms_dict, lobby_manager, game_manager):
        """Обновление игровых метрик"""
        try:
            # Активные комнаты
            active_rooms.set(len(rooms_dict))

            # Активные игры
            games_count = sum(1 for room in rooms_dict.values() if room.game_controller is not None)
            active_games.set(games_count)

            # Подключения в лобби
            lobby_count = sum(len(connections) for connections in lobby_manager.active_connections.values())
            lobby_connections.set(lobby_count)

            # Подключения в играх
            game_count = sum(len(connections) for connections in game_manager.game_connections.values())
            game_connections.set(game_count)

        except Exception as e:
            logger.warning(f"Failed to update game metrics: {e}")


# Глобальный экземпляр
metrics = GameServerMetrics()


def get_metrics_response() -> Response:
    """Получить ответ с метриками для Prometheus"""
    try:
        # Обновляем системные метрики перед отдачей
        metrics.update_system_metrics()

        # Для multiprocess режима используем специальный registry
        registry = CollectorRegistry()
        multiprocess.MultiProcessCollector(registry)

        data = generate_latest(registry)
        return PlainTextResponse(content=data.decode("utf-8"), headers={"Content-Type": CONTENT_TYPE_LATEST})
    except Exception as e:
        logger.error(f"Failed to generate metrics: {e}")
        return PlainTextResponse(content="# Error generating metrics\n", status_code=500)


# Middleware для автоматического сбора метрик
async def metrics_middleware(request, call_next):
    """Middleware для сбора метрик по запросам"""
    start_time = time.time()

    response = await call_next(request)

    # Записываем метрики
    duration = time.time() - start_time
    method = request.method
    path = request.url.path
    status = response.status_code

    # Упрощаем путь для метрик (убираем динамические части)
    endpoint = simplify_endpoint(path)

    api_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()

    request_duration.labels(method=method, endpoint=endpoint).observe(duration)

    return response


def simplify_endpoint(path: str) -> str:
    """Упрощение пути для метрик (замена ID на placeholder)"""
    parts = path.split("/")
    simplified = []

    for part in parts:
        if not part:
            continue
        # Если часть выглядит как ID (6+ символов hex), заменяем на placeholder
        if len(part) >= 6 and all(c in "0123456789ABCDEFabcdef" for c in part):
            simplified.append("{id}")
        else:
            simplified.append(part)

    return "/" + "/".join(simplified) if simplified else "/"
