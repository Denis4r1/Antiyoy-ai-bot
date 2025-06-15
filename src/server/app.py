import os
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from .config import STATIC_DIR, CORS_ORIGINS, DEBUG
from .models.managers import LobbyConnectionManager, GameConnectionManager
from .services.game_service import init_game_service
from .handlers.api import create_api_router
from .handlers.websockets import create_websocket_router
from .monitoring.logging_config import setup_logging, get_logger
from .monitoring.metrics import get_metrics_response, metrics_middleware, metrics

# Настройка логирования
setup_logging()
logger = get_logger("app")

# Глобальные менеджеры
lobby_manager = LobbyConnectionManager()
game_manager = GameConnectionManager()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    logger.info("Starting game server...")

    # Инициализация сервисов
    init_game_service(lobby_manager, game_manager)

    # Создаем фоновую задачу для обновления метрик
    async def update_metrics_task():
        while True:
            try:
                from .services.game_service import get_game_service

                game_service = get_game_service()
                if game_service:
                    metrics.update_game_metrics(
                        game_service.rooms, lobby_manager, game_manager
                    )
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
            await asyncio.sleep(30)  # Обновляем каждые 30 секунд

    metrics_task = asyncio.create_task(update_metrics_task())

    logger.info("Game server started successfully")

    yield

    # Завершение работы
    logger.info("Shutting down game server...")
    metrics_task.cancel()
    try:
        await metrics_task
    except asyncio.CancelledError:
        pass
    logger.info("Game server shutdown complete")


def create_app() -> FastAPI:
    """Создать и настроить FastAPI приложение"""

    app = FastAPI(
        title="Game Server",
        description="WebSocket-based multiplayer game server",
        version="1.0.0",
        debug=DEBUG,
        lifespan=lifespan,
    )

    # Middleware для метрик (должен быть первым)
    app.middleware("http")(metrics_middleware)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=CORS_ORIGINS,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Статические файлы
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # API routes
    api_router = create_api_router(game_manager)
    app.include_router(api_router)

    # WebSocket routes
    ws_router = create_websocket_router(lobby_manager, game_manager)
    app.include_router(ws_router)

    # Метрики Prometheus
    @app.get("/metrics")
    async def get_metrics():
        return get_metrics_response()

    # Логирование старта
    logger.info(f"FastAPI app created with debug={DEBUG}")

    return app


# Создаем приложение
app = create_app()

if __name__ == "__main__":
    import uvicorn

    # Для разработки
    uvicorn.run(
        "src.server.app:app", host="0.0.0.0", port=8000, reload=DEBUG, log_level="info"
    )
