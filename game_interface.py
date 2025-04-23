import json
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Tuple
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path


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
    description: Optional[str]


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


# 1. Генерация нового состояния
@app.post("/generate_state",
           response_model=GameState,
           summary="Генерация нового состояния игры")
def generate_state(num_players: int = 2, random: bool = False):
    """Создает новую игру и возвращает ее состояние"""
    try:
        if (not random):
            with open("static/map_basic_10x10.json", "r") as f:
                return json.load(f)

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


# 2. Получение доступных действий
@app.post("/get_actions",
           response_model=List[Action],
           summary="Все возможные действия для текущего состояния")
def get_actions(state: GameState):
    """
    Возвращает список возможных действий для текущего игрока на основе состояния игры.
    Аргументы:
        state (GameState): Текущее состояние игры.

    Возвращает:
        List[Action]: Список возможных действий с action_id, action_type, params и description.
    """
    try:
        # Восстанавливаем GameController из состояния
        gc = reconstruct_game(state)

        current_player = gc.players[gc._current_player_index]
        actions = []

        # Завершение хода - всегда доступно
        actions.append({"action_id": 0, "action_type": "end_turn", "params": {}, "description": "End turn"})

        action_id = 1

        # Получаем все территории текущего игрока
        territories = [t for t in gc.field.territory_manager.territories if t.owner == current_player]

        # Перебираем все клетки и собираем возможные действия
        for territory in territories:
            for cell_key in territory.tiles:
                y, x = cell_key
                cell = gc.field.cells[cell_key]

                # Если клетка пустая, можно строить и создавать юнитов
                if cell.entity == EntityType.EMPTY:
                    # Проверяем постройку фермы
                    try:
                        if gc.field.has_farm_or_base_neigbour(cell_key):
                            # Рассчитываем стоимость фермы
                            num_farms = sum(
                                1 for pos in territory.tiles if gc.field.cells[pos].entity == EntityType.FARM
                            )
                            farm_cost = 12 + 2 * num_farms

                            if territory.funds >= farm_cost:
                                actions.append(
                                    {
                                        "action_id": action_id,
                                        "action_type": "build_action",
                                        "params": {"x": x, "y": y, "building": "farm"},
                                        "description": f"Build farm at ({y},{x})",
                                    }
                                )
                                action_id += 1
                    except Exception:
                        pass

                    # Проверяем слабую башню
                    if territory.funds >= COST_WEAK_TOWER:
                        actions.append(
                            {
                                "action_id": action_id,
                                "action_type": "build_action",
                                "params": {"x": x, "y": y, "building": "weakTower"},
                                "description": f"Build weak tower at ({y},{x})",
                            }
                        )
                        action_id += 1

                    # Проверяем сильную башню
                    if territory.funds >= COST_STRONG_TOWER:
                        actions.append(
                            {
                                "action_id": action_id,
                                "action_type": "build_action",
                                "params": {"x": x, "y": y, "building": "strongTower"},
                                "description": f"Build strong tower at ({y},{x})",
                            }
                        )
                        action_id += 1

                    # Проверяем создание юнитов
                    for level in range(1, 5):
                        if territory.funds >= UNIT_COST[level]:
                            actions.append(
                                {
                                    "action_id": action_id,
                                    "action_type": "spawn_unit",
                                    "params": {"x": x, "y": y, "level": level},
                                    "description": f"Spawn level {level} unit at ({y},{x})",
                                }
                            )
                            action_id += 1

                # Если в клетке юнит, проверяем возможные перемещения
                elif "unit" in cell.entity.value and not cell.has_moved:
                    try:
                        moves = gc.field.get_moves(x, y, current_player)
                        for move in moves:
                            to_x = move["x"]
                            to_y = move["y"]
                            actions.append(
                                {
                                    "action_id": action_id,
                                    "action_type": "move_unit",
                                    "params": {"from_x": x, "from_y": y, "to_x": to_x, "to_y": to_y},
                                    "description": f"Move unit from ({y},{x}) to ({to_y},{to_x})",
                                }
                            )
                            action_id += 1
                    except Exception:
                        pass

                # TODO Добавить стакинг юнитов
                # TODO Добавить стакинг башен

        return actions

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get actions: {str(e)}")


# 3. Применение действия к состоянию
@app.post("/apply_action",
           response_model=ApplyActionResponse,
           summary="Применение действия к текущему состоянию")
def apply_action(request: ActionRequest):
    """
    Применяет действие к состоянию игры и возвращает обновленное состояние.

    Действие указывается через `action_id` (из /get_actions), либо через `action_type` и `params`.
    Координаты в `params` указаны как (x, y), где x — столбец, y — строка.

    Аргументы:
        request (ActionRequest): Запрос с текущим состоянием и деталями действия.

    Возвращает:
        ApplyActionResponse: Новое состояние игры, флаг завершения и победитель, если игра окончена.

    Вызывает:
        HTTPException: 400, если действие недопустимо;
    """
    try:
        # Восстанавливаем GameController из состояния
        gc = reconstruct_game(request.state)

        current_player = gc.players[gc._current_player_index]

        # Если передан action_id, найдем соответствующее действие
        if request.action_id is not None:
            # Получаем список доступных действий
            state_copy = request.state.copy()
            actions = get_actions(state_copy)

            # Найдем действие с указанным ID
            action = next((a for a in actions if a["action_id"] == request.action_id), None)

            if not action:
                raise HTTPException(status_code=400, detail=f"Action with ID {request.action_id} not found")

            action_type = action["action_type"]
            params = action["params"]
        else:
            # Используем переданные напрямую параметры
            if not request.action_type:
                raise HTTPException(status_code=400, detail="Missing action_type")

            action_type = request.action_type
            params = request.params or {}

        # Подготовка сообщения для GameController
        message = {"type": action_type, "payload": params}

        # Обработка сообщения и получение результата
        result = gc.process_message(message, current_player)

        if not result:
            raise HTTPException(status_code=400, detail="Invalid action")

        # Создаем новое состояние
        field_data = gc.field.to_dict()
        territories_data = []

        for territory in gc.field.territory_manager.territories:
            territories_data.append(
                {"owner": territory.owner, "funds": territory.funds, "tiles": list(territory.tiles)}
            )

        new_state = GameState(
            players=gc.players,
            current_player_index=gc._current_player_index,
            field_data=field_data,
            territories_data=territories_data,
        )

        # Возвращаем новое состояние и результат операции
        return {
            "state": new_state,
            #"result": result,
            "is_game_over": result.get("type") == "game_over",
            "winner": result.get("winner") if result.get("type") == "game_over" else None,
        }

    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=f"Failed to apply action: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
