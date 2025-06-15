import uuid
import json
import asyncio
from typing import Dict, List, Optional, Set

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from src.game.gamecontroller import GameController

app = FastAPI()
app.mount("/static", StaticFiles(directory="web/static"), name="static")


# ======= отдаём лобби-HTML =======
@app.get("/")
async def get_index():
    with open("web/templates/newlobby.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


# ======= отдаём game-HTML =======
@app.get("/game/{room_id}")
async def get_game_page(room_id: str, token: str = Query(None)):
    with open("web/templates/gamepage.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content, status_code=200)


# Разрешаем CORS для отладки
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============== Room (лобби) ==============
class Room:
    def __init__(self, room_id: str):
        self.room_id = room_id
        self.users: Dict[str, Dict] = {}  # token -> {token, name, ready}
        self.game_start_task: Optional[asyncio.Task] = None
        self.game_controller: Optional[GameController] = None

    def add_user(self, token: str, name: str):
        if token not in self.users:
            self.users[token] = {"token": token, "name": name, "ready": False}
        else:
            self.users[token]["name"] = name

    def remove_user(self, token: str):
        if token in self.users:
            del self.users[token]

    def set_ready(self, token: str, ready: bool):
        if token in self.users:
            self.users[token]["ready"] = ready

    def get_players_list(self):
        return [{"name": u["name"], "ready": u["ready"]} for u in self.users.values()]

    def all_ready(self) -> bool:
        return bool(self.users) and all(u["ready"] for u in self.users.values())


rooms: Dict[str, Room] = {}


# ============== LobbyConnectionManager ==============
class LobbyConnectionManager:
    def __init__(self):
        # room_id -> list[WebSocket]
        self.active_connections: Dict[str, List[WebSocket]] = {}

    async def connect(self, room_id: str, websocket: WebSocket):
        await websocket.accept()
        if room_id not in self.active_connections:
            self.active_connections[room_id] = []
        self.active_connections[room_id].append(websocket)

    def disconnect(self, room_id: str, websocket: WebSocket):
        if room_id in self.active_connections:
            self.active_connections[room_id].remove(websocket)
            if not self.active_connections[room_id]:
                del self.active_connections[room_id]

    async def broadcast(self, room_id: str, message: str):
        if room_id in self.active_connections:
            for connection in self.active_connections[room_id]:
                await connection.send_text(message)


lobby_manager = LobbyConnectionManager()


# ============== GameConnectionManager ==============
class GameConnectionManager:
    def __init__(self):
        # room_id -> { token -> WebSocket }
        self.game_connections: Dict[str, Dict[str, WebSocket]] = {}
        # room_id -> set of tokens (кому разрешено подключаться к игре)
        self.allowed_players: Dict[str, Set[str]] = {}

    def allow_room(self, room_id: str, tokens: List[str]):
        """
        Разрешить этим токенам подключаться в room_id.
        в tokens находятся игроки из одного лобби
        """
        self.allowed_players[room_id] = set(tokens)

    async def connect(self, room_id: str, token: str, websocket: WebSocket) -> bool:
        allowed = self.allowed_players.get(room_id)
        if not allowed or token not in allowed:
            # Этот игрок не был в одном лобби
            await websocket.close(code=1008)
            return False

        await websocket.accept()
        if room_id not in self.game_connections:
            self.game_connections[room_id] = {}
        self.game_connections[room_id][token] = websocket
        return True

    def disconnect(self, room_id: str, token: str):
        if room_id in self.game_connections:
            ws_map = self.game_connections[room_id]
            if token in ws_map:
                del ws_map[token]
            if not ws_map:
                del self.game_connections[room_id]

    async def broadcast(self, room_id: str, message: str):
        if room_id in self.game_connections:
            for ws in self.game_connections[room_id].values():
                await ws.send_text(message)

    async def send_personal_message(self, room_id: str, token: str, message: str):
        if room_id in self.game_connections:
            ws_map = self.game_connections[room_id]
            if token in ws_map:
                await ws_map[token].send_text(message)


game_manager = GameConnectionManager()


# ============== Старт игры ==============
async def start_game_after_delay(room: Room):
    try:
        for i in range(5, 0, -1):
            await lobby_manager.broadcast(room.room_id, f"Игра начнется через {i}")
            await asyncio.sleep(1)

        if room.all_ready():
            player_names = [u["name"] for u in room.users.values()]
            room.game_controller = GameController(player_names)

            # Засылаем сообщение о старте игры
            await lobby_manager.broadcast(room.room_id, "Игра начинается!")
            await lobby_manager.broadcast(
                room.room_id, json.dumps({"type": "game_start"})
            )

            # Разрешим этим токенам подключаться к /ws/game/<room_id>:
            tokens = list(room.users.keys())  # Все из комнаты
            game_manager.allow_room(room.room_id, tokens)

    except asyncio.CancelledError:
        await lobby_manager.broadcast(room.room_id, "Запуск игры отменён")
    finally:
        room.game_start_task = None


async def check_game_start(room: Room):
    if room.all_ready():
        if room.game_start_task is None:
            room.game_start_task = asyncio.create_task(start_game_after_delay(room))
    else:
        if room.game_start_task:
            room.game_start_task.cancel()
            room.game_start_task = None


# ============== HTTP API для лобби ==============
@app.post("/create_room")
def create_room():
    while True:
        room_id = uuid.uuid4().hex[:6].upper()
        if room_id not in rooms:
            break
    rooms[room_id] = Room(room_id)
    return {"room_id": room_id}


@app.post("/join_room")
def join_room(
    room_id: str = Query(...), token: str = Query(...), name: str = Query(...)
):
    if room_id not in rooms:
        raise HTTPException(status_code=404, detail="Комната не найдена")
    room = rooms[room_id]

    for user_data in room.users.values():
        if user_data["name"] == name:
            # Если имя не уникально в комнате, не пускаем
            raise HTTPException(status_code=400, detail="Имя занято")

    # Если имя свободно – добавляем
    room.add_user(token, name)
    return {"detail": "Вы успешно подключились к комнате", "room_id": room_id}


# ============== Websocket для lobby  ==============
@app.websocket("/ws/lobby/{room_id}/{token}")
async def lobby_ws_endpoint(websocket: WebSocket, room_id: str, token: str):
    if room_id not in rooms:
        await websocket.close(code=1008)
        return

    room = rooms[room_id]
    # Проверяем, что token есть в комнате
    if token not in room.users:
        await websocket.close(code=1008)
        return

    # Подключаем к лобби
    await lobby_manager.connect(room_id, websocket)
    # Обновим список игроков
    await lobby_manager.broadcast(
        room_id,
        json.dumps({"type": "players_update", "players": room.get_players_list()}),
    )
    await lobby_manager.broadcast(
        room_id, f"Система: {room.users[token]['name']} присоединился к комнате"
    )

    try:
        while True:
            data = await websocket.receive_text()

            if data == "READY":
                room.set_ready(token, True)
                await lobby_manager.broadcast(
                    room_id,
                    json.dumps(
                        {"type": "players_update", "players": room.get_players_list()}
                    ),
                )
                await check_game_start(room)

            elif data == "NOT_READY":
                room.set_ready(token, False)
                await lobby_manager.broadcast(
                    room_id,
                    json.dumps(
                        {"type": "players_update", "players": room.get_players_list()}
                    ),
                )
                await check_game_start(room)

            else:
                # Остальное считаем чат-сообщением
                message = f"{room.users[token]['name']}: {data}"
                await lobby_manager.broadcast(room_id, message)

    except WebSocketDisconnect:
        # Отключился
        lobby_manager.disconnect(room_id, websocket)
        username = room.users[token]["name"]
        room.remove_user(token)

        await lobby_manager.broadcast(
            room_id,
            json.dumps({"type": "players_update", "players": room.get_players_list()}),
        )
        await lobby_manager.broadcast(room_id, f"Система: {username} покинул комнату")
        await check_game_start(room)


# ============== game WebSocket ==============
@app.websocket("/ws/game/{room_id}/{token}/{username}")
async def game_ws_endpoint(
    websocket: WebSocket, room_id: str, token: str, username: str
):
    # Если token не в allowed_players[room_id], не подключаем.
    connected_ok = await game_manager.connect(room_id, token, websocket)

    if not connected_ok:
        return

    # Если всё ок, значит пользователь правильный
    if not token in game_manager.allowed_players[room_id]:
        await websocket.close(code=1008)
    room = rooms.get(room_id)

    initial_state = {
        "type": "game_state_update",
        "players": room.game_controller.players,
        "field": room.game_controller.field.to_dict(),
        "current_player": room.game_controller.players[
            room.game_controller._current_player_index
        ],
    }
    await game_manager.send_personal_message(room_id, token, json.dumps(initial_state))

    try:
        while True:
            raw_data = await websocket.receive_text()

            try:
                message = json.loads(raw_data)
            except:
                continue

            try:
                # Передаем сообщение на обработку в game_controller
                # Он вернет dict с обновлением стейта игры, который нужно разослать игрокам
                result = room.game_controller.process_message(message, username)
            except Exception as e:
                await websocket.send_text(
                    json.dumps({"type": "error", "message": str(e)})
                )
                continue

            if result:
                if result["type"] == "available_moves":
                    # Отправка только запрашивающему
                    await game_manager.send_personal_message(
                        room_id, token, json.dumps(result)
                    )

                elif result["type"] == "game_over":
                    new_lobby_id = uuid.uuid4().hex[:6].upper()
                    new_room = Room(new_lobby_id)

                    # Переносим всех игроков из старой комнаты в новую
                    for tkn, user_data in room.users.items():
                        new_room.add_user(tkn, user_data["name"])
                    rooms[new_lobby_id] = new_room

                    # ID нового лобби
                    result["lobby_id"] = new_lobby_id
                    await game_manager.broadcast(room_id, json.dumps(result))
                elif result["type"] == "game_state_update":
                    await game_manager.broadcast(room_id, json.dumps(result))

    except WebSocketDisconnect:
        game_manager.disconnect(room_id, token)
