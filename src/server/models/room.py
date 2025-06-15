import asyncio
from typing import Dict, List, Optional
from src.game.gamecontroller import GameController
from ..monitoring.logging_config import get_logger

logger = get_logger("room")


class Room:
    """Класс для управления игровой комнатой"""

    def __init__(self, room_id: str):
        self.room_id = room_id
        self.users: Dict[str, Dict] = {}  # token -> {token, name, ready}
        self.game_start_task: Optional[asyncio.Task] = None
        self.game_controller: Optional[GameController] = None
        self.deletion_task: Optional[asyncio.Task] = None  # Задача отложенного удаления

        logger.info(f"Created room {room_id}")

    def add_user(self, token: str, name: str):
        """Добавить пользователя в комнату"""
        if token not in self.users:
            self.users[token] = {"token": token, "name": name, "ready": False}
            logger.info(
                f"Added new user {name} (token: {token[:8]}...) to room {self.room_id}"
            )
        else:
            old_name = self.users[token]["name"]
            self.users[token]["name"] = name
            logger.info(
                f"Updated user name from {old_name} to {name} in room {self.room_id}"
            )

        # Отменяем удаление комнаты если игрок вернулся
        self.cancel_deletion()

    def remove_user(self, token: str):
        """Удалить пользователя из комнаты"""
        if token in self.users:
            name = self.users[token]["name"]
            del self.users[token]
            logger.info(
                f"Removed user {name} (token: {token[:8]}...) from room {self.room_id}"
            )

    def set_ready(self, token: str, ready: bool):
        """Установить готовность пользователя"""
        if token in self.users:
            old_ready = self.users[token]["ready"]
            self.users[token]["ready"] = ready
            name = self.users[token]["name"]

            if old_ready != ready:
                status = "ready" if ready else "not ready"
                logger.info(f"User {name} is now {status} in room {self.room_id}")

    def get_players_list(self) -> List[Dict]:
        """Получить список игроков для отправки клиентам"""
        return [{"name": u["name"], "ready": u["ready"]} for u in self.users.values()]

    def all_ready(self) -> bool:
        """Проверить, готовы ли все игроки"""
        result = bool(self.users) and all(u["ready"] for u in self.users.values())

        if result and len(self.users) > 1:
            logger.info(
                f"All {len(self.users)} players are ready in room {self.room_id}"
            )

        return result

    def get_user_count(self) -> int:
        """Получить количество пользователей в комнате"""
        return len(self.users)

    def get_ready_count(self) -> int:
        """Получить количество готовых пользователей"""
        return sum(1 for u in self.users.values() if u["ready"])

    def has_game_started(self) -> bool:
        """Проверить, началась ли игра"""
        return self.game_controller is not None

    def start_game(self, players_names: List[str]):
        """Запустить игру"""
        if self.game_controller is None:
            self.game_controller = GameController(players_names)
            logger.info(
                f"Started game in room {self.room_id} with players: {players_names}"
            )
        else:
            logger.warning(
                f"Attempted to start game in room {self.room_id} but game already started"
            )

    def schedule_deletion(self, deletion_task: asyncio.Task):
        """Запланировать удаление комнаты"""
        # Отменяем предыдущую задачу удаления если есть
        self.cancel_deletion()
        self.deletion_task = deletion_task
        logger.info(f"Scheduled deletion for room {self.room_id} in 20 seconds")

    def cancel_deletion(self):
        """Отменить запланированное удаление комнаты"""
        if self.deletion_task and not self.deletion_task.done():
            self.deletion_task.cancel()
            logger.info(f"Cancelled deletion for room {self.room_id}")
            self.deletion_task = None

    def is_deletion_scheduled(self) -> bool:
        """Проверить, запланировано ли удаление"""
        return self.deletion_task is not None and not self.deletion_task.done()
