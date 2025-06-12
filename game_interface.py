import json
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
import os
import functools
import json
from src.all_turns_ever import all_turns_ever



from src.gamecontroller import GameController
from src.field import EntityType, Field, Cell, TerritoryManager, Territory, COST_WEAK_TOWER, COST_STRONG_TOWER, UNIT_COST

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/state_editor")
async def get_index():
    with open("static/state_editor.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/log_player")
async def get_index():
    with open("static/log_player.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get("/log_player/demo", response_class=HTMLResponse)
async def log_player_demo():
    html = Path("static/log_player.html").read_text(encoding="utf-8")

    # a little hack to inject a demo json file into the page
    injection = """
    <script>
      window.addEventListener('load', async () => {
        // 1) fetch the demo log
        const res  = await fetch('/static/demo_log.jsonl');
        const text = await res.text();

        // 2) wrap it in a File and DataTransfer
        const file = new File([text], 'demo_log.jsonl', { type: 'application/json' });
        const dt   = new DataTransfer();
        dt.items.add(file);

        // 3) shove it into the <input> and fire 'change'
        const input = document.getElementById('fileInput');
        input.files = dt.files;
        input.dispatchEvent(new Event('change'));
      });
    </script>
    """

    html = html.replace("</body>", injection + "\n</body>")
    return html


class CellData(BaseModel):
    """
    Данные одной клетки игрового поля.

    Атрибуты:
        owner (Optional[str]): Игрок, владеющий клеткой, или None, если клетка пустая.
        entity (str): Тип сущности в клетке (например, 'EMPTY', 'UNIT_1', 'FARM')
        has_moved (bool): Если на клетке юнит, то True, если он уже переместился в этом ходу.
    """
    owner: Optional[str]
    entity: str
    has_moved: bool


class FieldData(BaseModel):
    """
    Данные игрового поля.

    Атрибуты:
        height (int): Высота поля (количество строк).
        width (int): Ширина поля (количество столбцов).
        cells (Dict[str, CellData]): Словарь клеток с ключами 'y,x' (строка, столбец) и данными клеток.
        territories (List[Dict[str, List[str]]]): Список территорий в виде [owner: info] где info - какая-то строка
    """
    height: int
    width: int
    cells: Dict[str, CellData]
    territories: List[Dict[str, List[str]]]


class TerritoryData(BaseModel):
    """
    Данные о территории. Территория - связная область клеток, принадлежащая одному игроку.

    Атрибуты:
        owner (str): Игрок, владеющий территорией.
        funds (int): Доступные средства территории.
        tiles (List[List[int]]): Список координат [y, x] клеток территории.
    """
    owner: str
    funds: int
    tiles: List[List[int]]


class GameState(BaseModel):
    """
    Текущее состояние игры.

    Атрибуты:
        players (List[str]): Список имен игроков.
        current_player_index (int): Индекс игрока, чей ход.
        field_data (FieldData): Данные игрового поля.
        territories_data (List[TerritoryData]): Данные о территориях.
    """
    players: List[str]
    current_player_index: int
    field_data: FieldData
    territories_data: List[TerritoryData]


class Action(BaseModel):
    """
    Ход, который может быть сделан в игре.

    Атрибуты:
        action_id (int): Уникальный идентификатор действия.
        action_type (str): Тип действия. Например "move", "build", "spawn_unit".
        params (Dict[str, Any]): Параметры, специфичные для типа действия.
        description (str): Человекочитаемое описание действия.
    """

    action_id: int
    action_type: str
    params: Optional[Dict[str, Any]]
    description: Optional[int]


class ActionRequest(BaseModel):
    """
    Запрос на применение действия к состоянию игры.

    Атрибуты:
        state (GameState): Текущее состояние игры.
        action_type (Optional[str]): Тип действия (если не используется action_id).
        params (Optional[Dict[str, Any]]): Параметры действия (если не используется action_id).
        action_id (Optional[int]): ID действия из /get_actions (альтернатива action_type и params).
        description (Optional[str]): необязательное описание действия.
    """
    state: GameState
    action_type: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    action_id: Optional[int] = None
    description: Optional[str] = None


class ApplyActionResponse(BaseModel):
    """
    Ответ на запрос /apply_action.

    Атрибуты:
        state (GameState): Новое состояние игры после применения действия.
        is_game_over (bool): Завершена ли игра.
        winner (Optional[str]): Победитель, если игра завершена.
    """

    state: GameState
    is_game_over: bool
    winner: Optional[str]


# --------------------------------------------------
# 1) /get_actions — внутренняя кеширующая функция


@functools.lru_cache(maxsize=500)
def _cached_get_actions(state_json: str) -> List[Action]:
    # pydantic v2: модель из JSON
    state = GameState.model_validate_json(state_json)

    gc = reconstruct_game(state)
    current_player = gc.players[gc._current_player_index]

    actions: list[dict] = []
    # всегда доступное
    actions.append({"action_id": 0, "action_type": "end_turn", "params": {}, "description": all_turns_ever["End turn"]})
    action_id = 1

    territories = [t for t in gc.field.territory_manager.territories if t.owner == current_player]
    for territory in territories:
        for y, x in territory.tiles:
            cell = gc.field.cells[(y, x)]
            # пустая клетка — можно строить и спавн
            if cell.entity == EntityType.EMPTY:
                # build farm
                try:
                    if gc.field.has_farm_or_base_neigbour((y, x)):
                        num_farms = sum(1 for p in territory.tiles if gc.field.cells[p].entity == EntityType.FARM)
                        cost = 12 + 2 * num_farms
                        if territory.funds >= cost:
                            actions.append(
                                {
                                    "action_id": action_id,
                                    "action_type": "build_action",
                                    "params": {"x": x, "y": y, "building": "farm"},
                                    "description": all_turns_ever[f"Build farm at ({y},{x})"],
                                }
                            )
                            action_id += 1
                except:
                    pass

                # weak tower
                if territory.funds >= COST_WEAK_TOWER:
                    actions.append(
                        {
                            "action_id": action_id,
                            "action_type": "build_action",
                            "params": {"x": x, "y": y, "building": "weakTower"},
                            "description": all_turns_ever[f"Weak tower at ({y},{x})"],
                        }
                    )
                    action_id += 1

                # strong tower
                if territory.funds >= COST_STRONG_TOWER:
                    actions.append(
                        {
                            "action_id": action_id,
                            "action_type": "build_action",
                            "params": {"x": x, "y": y, "building": "strongTower"},
                            "description": all_turns_ever[f"Strong tower at ({y},{x})"],
                        }
                    )
                    action_id += 1

                # spawn units
                for lvl in range(1, 5):
                    if territory.funds >= UNIT_COST[lvl]:
                        actions.append(
                            {
                                "action_id": action_id,
                                "action_type": "spawn_unit",
                                "params": {"x": x, "y": y, "level": lvl},
                                "description": all_turns_ever[f"Spawn lvl {lvl} unit at ({y},{x})"],
                            }
                        )
                        action_id += 1

            # юнит — можно двигать
            elif "unit" in cell.entity.value and not cell.has_moved:
                try:
                    moves = gc.field.get_moves(x, y, current_player)
                    for m in moves:
                        actions.append(
                            {
                                "action_id": action_id,
                                "action_type": "move_unit",
                                "params": {"from_x": x, "from_y": y, "to_x": m["x"], "to_y": m["y"]},
                                "description": all_turns_ever[f"Move from ({y},{x}) to ({m['y']},{m['x']})"],
                            }
                        )
                        action_id += 1
                except:
                    pass

    # приводим к List[Action]
    return [Action(**a) for a in actions]


# --------------------------------------------------
# 2) /apply_action — внутренняя кеширующая функция


@functools.lru_cache(maxsize=500)
def _cached_apply_action(
    state_json: str, action_id: Optional[int], action_type: Optional[str], params_json: str, description: Optional[str]
) -> ApplyActionResponse:

    # Собираем запрос в словарь и конструируем Pydantic-модель
    req_dict = {
        "state": json.loads(state_json),
        "action_id": action_id,
        "action_type": action_type,
        "params": json.loads(params_json) if params_json else None,
        "description": description,
    }
    req = ActionRequest(**req_dict)

    gc = reconstruct_game(req.state)
    current_player = gc.players[gc._current_player_index]

    # Если передали action_id — находим реальный action_type и params
    if req.action_id is not None:
        act_list = _cached_get_actions(req.state.json())
        act = next((x for x in act_list if x.action_id == req.action_id), None)
        if act is None:
            raise HTTPException(status_code=400, detail=f"Action {req.action_id} not found")
        action_type, params = act.action_type, act.params or {}
    else:
        if not req.action_type:
            raise HTTPException(status_code=400, detail="Missing action_type")
        action_type, params = req.action_type, req.params or {}

    # Применяем ход
    result = gc.process_message({"type": action_type, "payload": params}, current_player)
    if not result:
        raise HTTPException(status_code=400, detail="Invalid action")

    winner = gc.field.is_terminal()
    is_game_over = winner is not None

    # Собираем новый GameState
    fd = gc.field.to_dict()
    td = [{"owner": t.owner, "funds": t.funds, "tiles": list(t.tiles)} for t in gc.field.territory_manager.territories]
    new_state = GameState(
        players=gc.players,
        current_player_index=gc._current_player_index,
        field_data=fd,
        territories_data=td,
    )

    return ApplyActionResponse(
        state=new_state,
        is_game_over=is_game_over,
        winner=winner,
    )


# 1. Генерация нового состояния
@app.post("/generate_state",
           response_model=GameState,
           summary="Генерация нового состояния игры")
def generate_state(num_players: int = 2, random: bool = False):
    """Создает новую игру и возвращает ее состояние"""
    try:
        if (not random):
            with open("static/map_basic_small.json", "r") as f:
                return json.load(f)
            # with open("static/map_basic_10x10.json", "r") as f:
            #     return json.load(f)

        players = [f"player_{i}" for i in range(num_players)]

        # Создаем новую игру
        gc = GameController(players)

        # Преобразуем состояние в dict
        field_data = gc.field.to_dict()

        # Собираем данные о территориях
        territories_data = []
        for territory in gc.field.territory_manager.territories:
            territories_data.append(
                {"owner": territory.owner, "funds": territory.funds, "tiles": list(territory.tiles)}
            )

        # Создаем объект состояния
        state = GameState(
            players=gc.players,
            current_player_index=gc._current_player_index,
            field_data=field_data,
            territories_data=territories_data,
        )

        return state

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate game state: {str(e)}")


# Создание объекта GameController из состояния
def reconstruct_game(state: GameState) -> GameController:
    """Восстанавливает объект GameController из переданного состояния"""
    # Создаем базовый GameController
    gc = GameController(state.players)
    gc._current_player_index = state.current_player_index

    # Получаем данные из состояния
    field_data = state.field_data
    height = field_data.height
    width = field_data.width

    # Создаем поле и полностью его редактируем
    gc.field = Field(height, width, state.players)
    gc.field.cells.clear()

    # Координаты баз
    bases = set()

    # Заполняем клетки
    for cell_key_str, cell_data in field_data.cells.items():

        i, j = map(int, cell_key_str.split(","))

        cell = Cell()
        cell.owner = cell_data.owner
        cell.entity = EntityType(cell_data.entity)
        cell.has_moved = cell_data.has_moved

        if cell.entity == EntityType.BASE:
            bases.add((i, j))

        gc.field.cells[(i, j)] = cell

    territories = []
    for territory_data in state.territories_data:
        territory_tiles = territory_data.tiles

        base_key = None
        for i, j in territory_tiles:
            if (i, j) in bases:
                base_key = (i, j)        

        owner = territory_data.owner
        funds = territory_data.funds

        territory = Territory(owner=owner, field=gc.field, base_key=base_key, funds=funds)
        territory.tiles = set([tuple(i) for i in territory_tiles])
        territories.append(territory)
    gc.field.territory_manager = TerritoryManager(gc.field, territories)
    return gc


@app.post("/get_actions", response_model=List[Action])
def get_actions_endpoint(state: GameState):
    state_json = state.model_dump_json()
    return _cached_get_actions(state_json)


@app.post("/apply_action", response_model=ApplyActionResponse)
def apply_action_endpoint(request: ActionRequest):
    state_json = request.state.model_dump_json()
    params_json = json.dumps(request.params or {}, sort_keys=True)
    return _cached_apply_action(state_json, request.action_id, request.action_type, params_json, request.description)


LOG_DIR = os.path.join(os.path.dirname(__file__), "logs")


@app.get("/logs")
async def list_logs():
    """
    Возвращает список всех JSON-файлов в папке logs.
    """
    try:
        files = [
            fname
            for fname in os.listdir(LOG_DIR)
            if os.path.isfile(os.path.join(LOG_DIR, fname)) and fname.endswith(".json")
        ]
        files.sort()
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Directory logs/ not found")
    return {"logs": files}


@app.get("/logs/{filename}")
async def get_log(filename: str):
    """
    Отдаёт содержимое конкретного лога по имени файла.
    """
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = os.path.join(LOG_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Log not found")
    return FileResponse(path=file_path, media_type="application/json", filename=filename)


if __name__ == "__main__":
    uvicorn.run("game_interface:app", workers=32, host="0.0.0.0", port=8080)
