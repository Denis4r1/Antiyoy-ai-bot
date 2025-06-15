import json
import time
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from ..services.game_service import get_game_service
from ..models.managers import LobbyConnectionManager, GameConnectionManager
from ..monitoring.logging_config import get_logger
from ..monitoring.metrics import errors_total, lobby_wait_time

logger = get_logger("websockets")


def create_websocket_router(
    lobby_manager: LobbyConnectionManager, game_manager: GameConnectionManager
):
    """Создать роутер для WebSocket endpoints"""
    router = APIRouter()

    @router.websocket("/ws/lobby/{room_id}/{token}")
    async def lobby_ws_endpoint(websocket: WebSocket, room_id: str, token: str):
        """WebSocket endpoint для лобби"""
        game_service = get_game_service()
        room = game_service.get_room(room_id)

        if not room:
            logger.warning(f"WebSocket connection to non-existent lobby: {room_id}")
            await websocket.close(code=1008)
            return

        # Проверяем, что token есть в комнате
        if token not in room.users:
            logger.warning(
                f"WebSocket connection with invalid token {token[:8]}... to lobby {room_id}"
            )
            await websocket.close(code=1008)
            return

        # Записываем время входа в лобби для метрик
        lobby_enter_time = time.time()

        # Подключаем к лобби
        await lobby_manager.connect(room_id, websocket)

        username = room.users[token]["name"]
        logger.info(f"User {username} connected to lobby {room_id}")

        # Отправляем обновление списка игроков
        await lobby_manager.broadcast(
            room_id,
            json.dumps({"type": "players_update", "players": room.get_players_list()}),
        )
        await lobby_manager.broadcast(
            room_id, f"Система: {username} присоединился к комнате"
        )

        try:
            while True:
                data = await websocket.receive_text()

                if data == "READY":
                    room.set_ready(token, True)
                    await lobby_manager.broadcast(
                        room_id,
                        json.dumps(
                            {
                                "type": "players_update",
                                "players": room.get_players_list(),
                            }
                        ),
                    )
                    await game_service.check_game_start(room)

                elif data == "NOT_READY":
                    room.set_ready(token, False)
                    await lobby_manager.broadcast(
                        room_id,
                        json.dumps(
                            {
                                "type": "players_update",
                                "players": room.get_players_list(),
                            }
                        ),
                    )
                    await game_service.check_game_start(room)

                else:
                    # Остальное считаем чат-сообщением
                    message = f"{username}: {data}"
                    await lobby_manager.broadcast(room_id, message)
                    logger.debug(f"Lobby chat in {room_id}: {username}: {data}")

        except WebSocketDisconnect:
            # Записываем время в лобби
            lobby_time = time.time() - lobby_enter_time
            lobby_wait_time.observe(lobby_time)

            # Отключился
            lobby_manager.disconnect(room_id, websocket)

            # Удаляем пользователя из комнаты только если игра не началась
            if not room.has_game_started():
                room.remove_user(token)

                await lobby_manager.broadcast(
                    room_id,
                    json.dumps(
                        {"type": "players_update", "players": room.get_players_list()}
                    ),
                )
                await lobby_manager.broadcast(
                    room_id, f"Система: {username} покинул комнату"
                )
                await game_service.check_game_start(room)

                # Планируем удаление пустой комнаты с задержкой
                if room.get_user_count() == 0 and not room.has_game_started():
                    game_service.schedule_room_removal(room_id)

            logger.info(
                f"User {username} disconnected from lobby {room_id} after {lobby_time:.1f}s"
            )

        except Exception as e:
            logger.error(f"Error in lobby WebSocket for {room_id}: {e}")
            errors_total.labels(type="websocket").inc()

    @router.websocket("/ws/game/{room_id}/{token}/{username}")
    async def game_ws_endpoint(
        websocket: WebSocket, room_id: str, token: str, username: str
    ):
        """WebSocket endpoint для игры"""
        # Проверяем разрешение на подключение
        connected_ok = await game_manager.connect(room_id, token, websocket)

        if not connected_ok:
            logger.warning(
                f"Failed to connect {username} (token: {token[:8]}...) to game {room_id}"
            )
            return

        game_service = get_game_service()
        room = game_service.get_room(room_id)

        if not room or not room.has_game_started():
            logger.warning(f"Game connection to room {room_id} without active game")
            await websocket.close(code=1008)
            return

        logger.info(f"User {username} connected to game {room_id}")

        # Отменяем запланированное удаление комнаты если игрок переподключился
        if room:
            room.cancel_deletion()

        # Отправляем начальное состояние игры
        try:
            initial_state = {
                "type": "game_state_update",
                "players": room.game_controller.players,
                "field": room.game_controller.field.to_dict(),
                "current_player": room.game_controller.players[
                    room.game_controller._current_player_index
                ],
            }
            await game_manager.send_personal_message(
                room_id, token, json.dumps(initial_state)
            )
        except Exception as e:
            logger.error(f"Failed to send initial game state to {username}: {e}")
            errors_total.labels(type="websocket").inc()

        try:
            while True:
                raw_data = await websocket.receive_text()

                try:
                    message = json.loads(raw_data)
                except json.JSONDecodeError:
                    logger.warning(
                        f"Invalid JSON from {username} in game {room_id}: {raw_data}"
                    )
                    continue

                try:
                    # Передаем сообщение на обработку в game_controller
                    result = room.game_controller.process_message(message, username)
                except Exception as e:
                    logger.error(
                        f"Game controller error for {username} in {room_id}: {e}"
                    )
                    await websocket.send_text(
                        json.dumps({"type": "error", "message": str(e)})
                    )
                    errors_total.labels(type="game").inc()
                    continue

                if result:
                    try:
                        if result["type"] == "available_moves":
                            # Отправка только запрашивающему
                            await game_manager.send_personal_message(
                                room_id, token, json.dumps(result)
                            )

                        elif result["type"] == "game_over":
                            # Создаем новое лобби для игроков
                            new_lobby_id = await game_service.handle_game_over(room_id)
                            if new_lobby_id:
                                result["lobby_id"] = new_lobby_id

                            await game_manager.broadcast(room_id, json.dumps(result))
                            logger.info(
                                f"Game over in room {room_id}, new lobby: {new_lobby_id}"
                            )

                        elif result["type"] == "game_state_update":
                            await game_manager.broadcast(room_id, json.dumps(result))

                    except Exception as e:
                        logger.error(f"Failed to handle game result in {room_id}: {e}")
                        errors_total.labels(type="websocket").inc()

        except WebSocketDisconnect:
            game_manager.disconnect(room_id, token)
            logger.info(f"User {username} disconnected from game {room_id}")

            # Если все игроки отключились от игры, планируем удаление комнаты
            if game_manager.get_connection_count(room_id) == 0:
                game_service = get_game_service()
                room = game_service.get_room(room_id)
                if room and room.has_game_started():
                    logger.info(
                        f"All players disconnected from game {room_id}, scheduling room removal"
                    )
                    game_service.schedule_room_removal(room_id, force_for_game=True)

        except Exception as e:
            logger.error(f"Error in game WebSocket for {room_id}: {e}")
            errors_total.labels(type="websocket").inc()
            game_manager.disconnect(room_id, token)

            # Проверяем удаление и здесь
            if game_manager.get_connection_count(room_id) == 0:
                game_service = get_game_service()
                room = game_service.get_room(room_id)
                if room and room.has_game_started():
                    game_service.schedule_room_removal(room_id, force_for_game=True)

    return router
