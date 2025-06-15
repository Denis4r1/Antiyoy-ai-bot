from fastapi import APIRouter, HTTPException, Query, Depends
from fastapi.responses import HTMLResponse
from ..services.game_service import get_game_service
from ..config import TEMPLATES_DIR
from ..monitoring.logging_config import get_logger
from ..monitoring.metrics import rooms_created_total, api_requests_total, errors_total

logger = get_logger("api")


def create_api_router(game_manager=None):
    """Создать API роутер с зависимостями"""
    router = APIRouter()

    @router.get("/")
    async def get_index():
        """Главная страница - лобби"""
        try:
            with open(TEMPLATES_DIR / "newlobby.html", "r", encoding="utf-8") as f:
                content = f.read()
            return HTMLResponse(content)
        except Exception as e:
            logger.error(f"Failed to serve lobby page: {e}")
            errors_total.labels(type="api").inc()
            raise HTTPException(status_code=500, detail="Failed to load lobby page")

    @router.get("/game/{room_id}")
    async def get_game_page(room_id: str, token: str = Query(None)):
        """Страница игры"""
        try:
            # Проверяем существование комнаты
            game_service = get_game_service()
            room = game_service.get_room(room_id)

            if not room:
                logger.warning(f"Attempt to access non-existent game room: {room_id}")
                raise HTTPException(status_code=404, detail="Комната не найдена")

            # Проверяем что игра началась
            if not room.has_game_started():
                logger.warning(
                    f"Attempt to access game page for room {room_id} before game started"
                )
                raise HTTPException(status_code=400, detail="Игра еще не началась")

            # Проверяем токен если игра началась
            if room.has_game_started() and token and token not in room.users:
                logger.warning(f"Invalid token {token[:8]}... for room {room_id}")
                raise HTTPException(status_code=403, detail="Недействительный токен")

            # Проверяем токен если передан
            if token and token not in room.users:
                logger.warning(f"Invalid token {token[:8]}... for room {room_id}")
                raise HTTPException(status_code=403, detail="Недействительный токен")

            with open(TEMPLATES_DIR / "gamepage.html", "r", encoding="utf-8") as f:
                content = f.read()
            return HTMLResponse(content)

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to serve game page for room {room_id}: {e}")
            errors_total.labels(type="api").inc()
            raise HTTPException(status_code=500, detail="Failed to load game page")

    @router.post("/create_room")
    def create_room():
        """Создать новую комнату"""
        try:
            game_service = get_game_service()
            room_id = game_service.create_room()

            rooms_created_total.inc()
            logger.info(f"API: Created room {room_id}")

            return {"room_id": room_id}

        except Exception as e:
            logger.error(f"Failed to create room: {e}")
            errors_total.labels(type="api").inc()
            raise HTTPException(status_code=500, detail="Failed to create room")

    @router.post("/join_room")
    def join_room(
        room_id: str = Query(..., description="ID комнаты"),
        token: str = Query(..., description="Токен пользователя"),
        name: str = Query(..., description="Имя пользователя"),
    ):
        """Присоединиться к комнате"""
        try:
            game_service = get_game_service()
            room = game_service.get_room(room_id)

            if not room:
                logger.warning(f"Attempt to join non-existent room: {room_id}")
                raise HTTPException(status_code=404, detail="Комната не найдена")

            # Проверяем что игра не началась
            if room.has_game_started():
                logger.warning(f"Attempt to join room {room_id} after game started")
                raise HTTPException(status_code=400, detail="Игра уже началась")

            # Проверяем уникальность имени
            for user_data in room.users.values():
                if user_data["name"] == name and user_data["token"] != token:
                    logger.warning(f"Name '{name}' already taken in room {room_id}")
                    raise HTTPException(status_code=400, detail="Имя занято")

            # Добавляем пользователя
            if token in room.users:
                # Обновляем имя если нужно
                room.users[token]["name"] = name
                logger.info(f"API: User {name} reconnected to room {room_id}")
            else:
                # Добавляем нового пользователя
                room.add_user(token, name)

            logger.info(
                f"API: User {name} (token: {token[:8]}...) joined room {room_id}"
            )

            return {
                "detail": "Вы успешно подключились к комнате",
                "room_id": room_id,
                "players_count": room.get_user_count(),
            }

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Failed to join room {room_id}: {e}")
            errors_total.labels(type="api").inc()
            raise HTTPException(status_code=500, detail="Failed to join room")

    @router.get("/rooms/stats")
    def get_rooms_stats():
        """Получить статистику по комнатам"""
        try:
            game_service = get_game_service()
            stats = game_service.get_rooms_stats()

            logger.debug(f"API: Served rooms stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Failed to get rooms stats: {e}")
            errors_total.labels(type="api").inc()
            raise HTTPException(status_code=500, detail="Failed to get stats")

    @router.get("/health")
    def health_check():
        """Проверка здоровья сервиса"""
        try:
            game_service = get_game_service()
            if game_service is None:
                raise HTTPException(
                    status_code=500, detail="Game service not initialized"
                )

            stats = game_service.get_rooms_stats()

            return {
                "status": "healthy",
                "rooms": stats["total_rooms"],
                "games": stats["active_games"],
                "players": stats["total_players"],
            }

        except Exception as e:
            logger.error(f"Health check failed: {e}")
            errors_total.labels(type="api").inc()
            raise HTTPException(status_code=500, detail="Service unhealthy")

    return router
