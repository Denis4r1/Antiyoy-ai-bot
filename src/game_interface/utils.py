import functools
import json
from .models import GameState, Action, ActionRequest, ApplyActionResponse
from src.game.gamecontroller import GameController
from src.game.core.field import (
    EntityType,
    Field,
    Cell,
    TerritoryManager,
    Territory,
    COST_WEAK_TOWER,
    COST_STRONG_TOWER,
    UNIT_COST,
)
from src.utils.all_turns_ever import all_turns_ever
from fastapi import HTTPException
from typing import Optional, List


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

        territory = Territory(
            owner=owner, field=gc.field, base_key=base_key, funds=funds
        )
        territory.tiles = set([tuple(i) for i in territory_tiles])
        territories.append(territory)
    gc.field.territory_manager = TerritoryManager(gc.field, territories)
    return gc


@functools.lru_cache(maxsize=500)
def _cached_get_actions(state_json: str) -> List[Action]:
    # pydantic v2: модель из JSON
    state = GameState.model_validate_json(state_json)

    gc = reconstruct_game(state)
    current_player = gc.players[gc._current_player_index]

    actions: list[dict] = []
    # всегда доступное
    actions.append(
        {
            "action_id": 0,
            "action_type": "end_turn",
            "params": {},
            "description": all_turns_ever["End turn"],
        }
    )
    action_id = 1

    territories = [
        t for t in gc.field.territory_manager.territories if t.owner == current_player
    ]
    for territory in territories:
        for y, x in territory.tiles:
            cell = gc.field.cells[(y, x)]
            # пустая клетка — можно строить и спавн
            if cell.entity == EntityType.EMPTY:
                # build farm
                try:
                    if gc.field.has_farm_or_base_neigbour((y, x)):
                        num_farms = sum(
                            1
                            for p in territory.tiles
                            if gc.field.cells[p].entity == EntityType.FARM
                        )
                        cost = 12 + 2 * num_farms
                        if territory.funds >= cost:
                            actions.append(
                                {
                                    "action_id": action_id,
                                    "action_type": "build_action",
                                    "params": {"x": x, "y": y, "building": "farm"},
                                    "description": all_turns_ever[
                                        f"Build farm at ({y},{x})"
                                    ],
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
                                "description": all_turns_ever[
                                    f"Spawn lvl {lvl} unit at ({y},{x})"
                                ],
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
                                "params": {
                                    "from_x": x,
                                    "from_y": y,
                                    "to_x": m["x"],
                                    "to_y": m["y"],
                                },
                                "description": all_turns_ever[
                                    f"Move from ({y},{x}) to ({m['y']},{m['x']})"
                                ],
                            }
                        )
                        action_id += 1
                except:
                    pass

    # приводим к List[Action]
    return [Action(**a) for a in actions]


@functools.lru_cache(maxsize=500)
def _cached_apply_action(
    state_json: str,
    action_id: Optional[int],
    action_type: Optional[str],
    params_json: str,
    description: Optional[str],
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
            raise HTTPException(
                status_code=400, detail=f"Action {req.action_id} not found"
            )
        action_type, params = act.action_type, act.params or {}
    else:
        if not req.action_type:
            raise HTTPException(status_code=400, detail="Missing action_type")
        action_type, params = req.action_type, req.params or {}

    # Применяем ход
    result = gc.process_message(
        {"type": action_type, "payload": params}, current_player
    )
    if not result:
        raise HTTPException(status_code=400, detail="Invalid action")

    winner = gc.field.is_terminal()
    is_game_over = winner is not None

    # Собираем новый GameState
    fd = gc.field.to_dict()
    td = [
        {"owner": t.owner, "funds": t.funds, "tiles": list(t.tiles)}
        for t in gc.field.territory_manager.territories
    ]
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
