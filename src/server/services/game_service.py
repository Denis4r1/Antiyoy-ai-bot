import uuid
import json
import asyncio
from typing import Dict
from ..models.room import Room
from ..models.managers import LobbyConnectionManager, GameConnectionManager
from ..config import GAME_START_COUNTDOWN, ROOM_DELETION_DELAY
from ..monitoring.logging_config import get_logger
from ..monitoring.metrics import games_started_total, game_duration, lobby_wait_time

logger = get_logger("game_service")


class GameService:
    """Сервис для управления игровыми комнатами и играми"""

    def __init__(self, lobby_manager: LobbyConnectionManager, game_manager: GameConnectionManager):
        self.rooms: Dict[str, Room] = {}
        self.lobby_manager = lobby_manager
        self.game_manager = game_manager

    def create_room(self) -> str:
        """Создать новую комнату"""
        while True:
            room_id = uuid.uuid4().hex[:6].upper()
            if room_id not in self.rooms:
                break

        room = Room(room_id)
        self.rooms[room_id] = room

        logger.info(f"Created new room: {room_id}")
        return room_id

    def get_room(self, room_id: str) -> Room:
        """Получить комнату по ID"""
        return self.rooms.get(room_id)

    def remove_room(self, room_id: str):
        """Удалить комнату немедленно"""
        if room_id in self.rooms:
            room = self.rooms[room_id]

            # Отменяем все активные задачи
            if room.game_start_task:
                room.game_start_task.cancel()
            if room.deletion_task:
                room.deletion_task.cancel()

            del self.rooms[room_id]
            logger.info(f"Removed room: {room_id}")

    async def _delayed_room_removal(self, room_id: str):
        """Отложенное удаление комнаты"""
        try:
            await asyncio.sleep(ROOM_DELETION_DELAY)

            # Проверяем что комната все еще должна быть удалена
            room = self.get_room(room_id)
            if not room:
                logger.debug(f"Room {room_id} already removed")
                return

            should_remove = False

            if room.has_game_started():
                # Для игровых комнат проверяем количество подключений к игре
                game_connections = self.game_manager.get_connection_count(room_id)
                if game_connections == 0:
                    should_remove = True
                    logger.info(f"Removing game room {room_id} - no active game connections")
                else:
                    logger.info(f"Room {room_id} has {game_connections} game connections, cancelling deletion")
            else:
                # Для лобби проверяем количество пользователей
                if room.get_user_count() == 0:
                    should_remove = True
                    logger.info(f"Removing lobby room {room_id} - no users")
                else:
                    logger.info(f"Room {room_id} has {room.get_user_count()} users, cancelling deletion")

            if should_remove:
                self.remove_room(room_id)

        except asyncio.CancelledError:
            logger.info(f"Delayed removal of room {room_id} was cancelled")

    def schedule_room_removal(self, room_id: str, force_for_game: bool = False):
        """Запланировать удаление пустой комнаты через 20 секунд

        Args:
            room_id: ID комнаты
            force_for_game: Принудительно планировать удаление даже если игра началась
        """
        room = self.get_room(room_id)
        if not room:
            return

        # Проверяем что комната действительно пуста (только для лобби-комнат)
        if not force_for_game and room.get_user_count() > 0:
            logger.debug(f"Room {room_id} is not empty, not scheduling deletion")
            return

        # Если игра началась, не удаляем комнату (игроки могут переподключиться к лобби)
        # Но если force_for_game=True, то удаляем (все отключились от игры)
        if room.has_game_started() and not force_for_game:
            logger.debug(f"Room {room_id} has active game, not scheduling deletion")
            return

        # Создаем задачу отложенного удаления
        deletion_task = asyncio.create_task(self._delayed_room_removal(room_id))
        room.schedule_deletion(deletion_task)

        game_or_lobby = "game" if room.has_game_started() else "lobby"
        logger.info(f"Scheduled removal of empty {game_or_lobby} room {room_id} in {ROOM_DELETION_DELAY} seconds")

    async def start_game_after_delay(self, room: Room):
        """Запустить игру после обратного отсчета"""
        try:
            # Обратный отсчет
            for i in range(GAME_START_COUNTDOWN, 0, -1):
                await self.lobby_manager.broadcast(room.room_id, f"Игра начнется через {i}")
                await asyncio.sleep(1)

            # Проверяем что все еще готовы
            if room.all_ready():
                player_names = [u["name"] for u in room.users.values()]
                room.start_game(player_names)

                # Разрешаем игрокам подключаться к игре
                tokens = list(room.users.keys())
                self.game_manager.allow_room(room.room_id, tokens)

                # Уведомляем о старте игры
                await self.lobby_manager.broadcast(room.room_id, "Игра начинается!")
                await self.lobby_manager.broadcast(room.room_id, json.dumps({"type": "game_start"}))

                # Метрики
                games_started_total.inc()
                logger.info(f"Game started in room {room.room_id} with {len(player_names)} players")
            else:
                logger.warning(f"Game start cancelled in room {room.room_id} - not all players ready")

        except asyncio.CancelledError:
            await self.lobby_manager.broadcast(room.room_id, "Запуск игры отменён")
            logger.info(f"Game start cancelled in room {room.room_id}")
        except Exception as e:
            logger.error(f"Error starting game in room {room.room_id}: {e}")
            await self.lobby_manager.broadcast(room.room_id, "Ошибка при запуске игры")
        finally:
            room.game_start_task = None

    async def check_game_start(self, room: Room):
        """Проверить условия для запуска игры"""
        if room.all_ready() and len(room.users) >= 2:  # Минимум 2 игрока
            if room.game_start_task is None:
                logger.info(f"Starting countdown for room {room.room_id}")
                room.game_start_task = asyncio.create_task(self.start_game_after_delay(room))
        else:
            if room.game_start_task:
                logger.info(f"Cancelling countdown for room {room.room_id}")
                room.game_start_task.cancel()
                room.game_start_task = None

    async def handle_game_over(self, room_id: str) -> str:
        """Обработать окончание игры и создать новое лобби"""
        room = self.get_room(room_id)
        if not room:
            logger.warning(f"Game over called for non-existent room {room_id}")
            return None

        # Создаем новое лобби
        new_lobby_id = self.create_room()
        new_room = self.get_room(new_lobby_id)

        # Переносим всех игроков из старой комнаты в новую
        for token, user_data in room.users.items():
            new_room.add_user(token, user_data["name"])

        logger.info(f"Game ended in room {room_id}, created new lobby {new_lobby_id}")
        return new_lobby_id

    def get_rooms_stats(self) -> dict:
        """Получить статистику по комнатам"""
        total_rooms = len(self.rooms)
        active_games = sum(1 for room in self.rooms.values() if room.has_game_started())
        total_players = sum(room.get_user_count() for room in self.rooms.values())

        return {
            "total_rooms": total_rooms,
            "active_games": active_games,
            "lobby_rooms": total_rooms - active_games,
            "total_players": total_players,
        }


# Глобальный экземпляр сервиса
game_service = None


def get_game_service() -> GameService:
    """Получить глобальный экземпляр GameService"""
    return game_service


def init_game_service(lobby_manager: LobbyConnectionManager, game_manager: GameConnectionManager):
    """Инициализировать глобальный GameService"""
    global game_service
    game_service = GameService(lobby_manager, game_manager)
    return game_service
