from typing import Dict, List, Set
from fastapi import WebSocket
from ..monitoring.logging_config import get_logger
from ..monitoring.metrics import (
    websocket_connections_total,
    websocket_disconnections_total,
    errors_total,
)

logger = get_logger("managers")


class LobbyConnectionManager:
    """Менеджер WebSocket соединений для лобби"""

    def __init__(self):
        # room_id -> list[WebSocket]
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, room_id: str, websocket: WebSocket):
        """Подключить WebSocket к лобби комнаты"""
        try:
            await websocket.accept()
            if room_id not in self.active_connections:
                self.active_connections[room_id] = []
            self.active_connections[room_id].append(websocket)

            websocket_connections_total.labels(type="lobby").inc()
            logger.debug(
                f"WebSocket connected to lobby {room_id}. "
                f"Total connections: {len(self.active_connections[room_id])}"
            )
        except Exception as e:
            logger.error(f"Failed to connect WebSocket to lobby {room_id}: {e}")
            errors_total.labels(type="websocket").inc()
            raise

    def disconnect(self, room_id: str, websocket: WebSocket):
        """Отключить WebSocket от лобби комнаты"""
        try:
            if room_id in self.active_connections:
                if websocket in self.active_connections[room_id]:
                    self.active_connections[room_id].remove(websocket)
                    websocket_disconnections_total.labels(type="lobby").inc()

                    remaining = len(self.active_connections[room_id])
                    if remaining == 0:
                        del self.active_connections[room_id]
                        logger.debug(f"Removed empty lobby {room_id}")
                    else:
                        logger.debug(
                            f"WebSocket disconnected from lobby {room_id}. "
                            f"Remaining connections: {remaining}"
                        )
        except Exception as e:
            logger.error(f"Error during lobby disconnect for room {room_id}: {e}")
            errors_total.labels(type="websocket").inc()

    async def broadcast(self, room_id: str, message: str):
        """Отправить сообщение всем подключенным к лобби"""
        if room_id not in self.active_connections:
            logger.debug(f"No connections to broadcast to in lobby {room_id}")
            return

        connections = self.active_connections[room_id].copy()
        failed_connections = []

        for connection in connections:
            try:
                await connection.send_text(message)
            except Exception as e:
                logger.warning(f"Failed to send message to lobby connection: {e}")
                failed_connections.append(connection)
                errors_total.labels(type="websocket").inc()

        # Удаляем неработающие соединения
        for failed_conn in failed_connections:
            self.disconnect(room_id, failed_conn)

        if failed_connections:
            logger.info(
                f"Removed {len(failed_connections)} failed connections from lobby {room_id}"
            )

    def get_connection_count(self, room_id: str) -> int:
        """Получить количество подключений в лобби"""
        return len(self.active_connections.get(room_id, []))

    def get_total_connections(self) -> int:
        """Получить общее количество подключений во всех лобби"""
        return sum(len(connections) for connections in self.active_connections.values())


class GameConnectionManager:
    """Менеджер WebSocket соединений для игр"""

    def __init__(self):
        # room_id -> { token -> WebSocket }
        self.game_connections: Dict[str, Dict[str, WebSocket]] = {}
        # room_id -> set of tokens (кому разрешено подключаться к игре)
        self.allowed_players: Dict[str, Set[str]] = {}

    def allow_room(self, room_id: str, tokens: List[str]):
        """Разрешить этим токенам подключаться к игре в room_id"""
        self.allowed_players[room_id] = set(tokens)
        logger.info(f"Allowed {len(tokens)} players to connect to game {room_id}")

    async def connect(self, room_id: str, token: str, websocket: WebSocket) -> bool:
        """Подключить игрока к игре"""
        try:
            allowed = self.allowed_players.get(room_id)
            if not allowed or token not in allowed:
                logger.warning(
                    f"Unauthorized game connection attempt: "
                    f"room={room_id}, token={token[:8]}..."
                )
                await websocket.close(code=1008)
                return False

            await websocket.accept()
            if room_id not in self.game_connections:
                self.game_connections[room_id] = {}
            self.game_connections[room_id][token] = websocket

            websocket_connections_total.labels(type="game").inc()
            logger.debug(
                f"Player {token[:8]}... connected to game {room_id}. "
                f"Total players: {len(self.game_connections[room_id])}"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to connect player to game {room_id}: {e}")
            errors_total.labels(type="websocket").inc()
            return False

    def disconnect(self, room_id: str, token: str):
        """Отключить игрока от игры"""
        try:
            if room_id in self.game_connections:
                ws_map = self.game_connections[room_id]
                if token in ws_map:
                    del ws_map[token]
                    websocket_disconnections_total.labels(type="game").inc()

                    remaining = len(ws_map)
                    if remaining == 0:
                        del self.game_connections[room_id]
                        # Также очищаем allowed_players для этой комнаты
                        if room_id in self.allowed_players:
                            del self.allowed_players[room_id]
                        logger.debug(f"Removed empty game {room_id}")
                    else:
                        logger.debug(
                            f"Player {token[:8]}... disconnected from game {room_id}. "
                            f"Remaining players: {remaining}"
                        )
        except Exception as e:
            logger.error(f"Error during game disconnect for room {room_id}: {e}")
            errors_total.labels(type="websocket").inc()

    async def broadcast(self, room_id: str, message: str):
        """Отправить сообщение всем игрокам в игре"""
        if room_id not in self.game_connections:
            logger.debug(f"No connections to broadcast to in game {room_id}")
            return

        connections = list(self.game_connections[room_id].items())
        failed_tokens = []

        for token, websocket in connections:
            try:
                await websocket.send_text(message)
            except Exception as e:
                logger.warning(
                    f"Failed to send message to game player {token[:8]}...: {e}"
                )
                failed_tokens.append(token)
                errors_total.labels(type="websocket").inc()

        # Удаляем неработающие соединения
        for failed_token in failed_tokens:
            self.disconnect(room_id, failed_token)

        if failed_tokens:
            logger.info(
                f"Removed {len(failed_tokens)} failed connections from game {room_id}"
            )

    async def send_personal_message(self, room_id: str, token: str, message: str):
        """Отправить персональное сообщение игроку"""
        try:
            if room_id in self.game_connections:
                ws_map = self.game_connections[room_id]
                if token in ws_map:
                    await ws_map[token].send_text(message)
                    return True

            logger.warning(
                f"Could not send personal message: "
                f"room={room_id}, token={token[:8]}... not found"
            )
            return False

        except Exception as e:
            logger.error(f"Failed to send personal message to {token[:8]}...: {e}")
            errors_total.labels(type="websocket").inc()
            self.disconnect(room_id, token)
            return False

    def get_connection_count(self, room_id: str) -> int:
        """Получить количество подключений в игре"""
        return len(self.game_connections.get(room_id, {}))

    def get_total_connections(self) -> int:
        """Получить общее количество подключений во всех играх"""
        return sum(len(connections) for connections in self.game_connections.values())
