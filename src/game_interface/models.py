from pydantic import BaseModel
from typing import List, Dict, Any, Optional


class ActionProbability(BaseModel):
    """
    Вероятность конкретного действия в игровом состоянии.

    Атрибуты:
        action_id (int): Уникальный идентификатор действия.
        probability (float): Вероятность выбора данного действия, в диапазоне [0, 1].
    """
    action_id: int
    probability: float


class ProbabilitiesResponse(BaseModel):
    """
    Ответ на запрос вероятностей действий для заданного состояния игры.

    Атрибуты:
        probabilities (List[ActionProbability]): Список объектов, содержащих идентификаторы действий и их вероятности.
        best_action_id (int): Идентификатор действия с наибольшей вероятностью.
    """
    probabilities: List[ActionProbability]
    best_action_id: int


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
        description (int): Глобальный ID хода (всего их ограниченное число для заданного поля)
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
