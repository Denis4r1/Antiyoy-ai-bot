import math
import random
from collections import deque

from enum import Enum

from typing import List, Optional, Dict, Tuple, Set
from pydantic import BaseModel

from .const import *
from .tile import Tile


class FieldGenerator:
    def __init__(self, rows, cols, radius):
        self.rows = rows
        self.cols = cols
        self.radius = radius
        self.grid = [[Tile(i, j, radius) for j in range(cols)] for i in range(rows)]
        self.width = int(cols * math.sqrt(3) * radius * 0.89)
        self.height = int(rows * math.sqrt(3) * radius + 2 * radius)

        self.generateMap()

    def generateMap(self):
        islands = self.genRect()
        self.connectIslands(islands)
        self.addMoreTiles()

    def genRect(self):
        islands = []
        sizes = [((3, 4), (3, 4), 3), ((2, 3), (2, 3), 3), ((1, 2), (1, 2), 5)]

        for size_type in sizes:
            for _ in range(size_type[-1]):
                placed = False
                attempts = 0

                while not placed and attempts < 100:
                    width = random.randint(size_type[0][0], size_type[0][1])
                    height = random.randint(size_type[1][0], size_type[1][1])
                    start_i = random.randint(GAP, max(2, self.rows - height - GAP))
                    start_j = random.randint(GAP, max(2, self.cols - width - GAP))

                    if self.isFree(start_i, start_j, width, height):
                        self.placeRect(start_i, start_j, width, height)
                        islands.append((start_i, start_j, width, height))
                        placed = True

                    attempts += 1

        return islands

    def isFree(self, start_i, start_j, width, height):
        check_start_i = max(0, start_i - 1)
        check_end_i = min(self.rows, start_i + height + 1)
        check_start_j = max(0, start_j - 1)
        check_end_j = min(self.cols, start_j + width + 1)

        for i in range(check_start_i, check_end_i):
            for j in range(check_start_j, check_end_j):
                if self.grid[i][j].tile_type != 0:
                    return False
        return True

    def placeRect(self, start_i, start_j, width, height):
        for i in range(start_i, start_i + height):
            for j in range(start_j, start_j + width):
                self.grid[i][j].tile_type = ISLAND_VAL

    def connectIslands(self, islands):
        centers = set(
            [
                (island[0] + island[3] // 2, island[1] + island[2] // 2)
                for island in islands
            ]
        )
        visited = [[False for _ in range(self.cols)] for _ in range(self.rows)]
        center = centers.pop()
        self.grid[center[0]][center[1]].tile_type = MAP_VAL
        queue = deque([(center[0], center[1], [])])
        visited[center[0]][center[1]] = True

        def addBridge(path):
            for tile in path:
                self.grid[tile[0]][tile[1]].tile_type = MAP_VAL

        while queue:
            i, j, path = queue.popleft()
            for y, x in self.grid[i][j].getNeighbors(self.rows, self.cols):
                if visited[y][x]:
                    continue
                if self.grid[y][x].tile_type == ISLAND_VAL:
                    self.grid[y][x].tile_type = MAP_VAL
                if (y, x) in centers:
                    addBridge(path)
                    path = []
                visited[y][x] = True
                temp = path.copy()
                temp.append((y, x))
                queue.append((y, x, temp))

    def addMoreTiles(self):
        for row in self.grid:
            for tile in row:
                if tile.tile_type == MAP_VAL and tile.getDeg(self.grid, MAP_VAL) <= 2:
                    for i, j in tile.getNeighbors(self.rows, self.cols):
                        if (
                            self.grid[i][j].tile_type != MAP_VAL
                            and 2 <= self.grid[i][j].getDeg(self.grid, MAP_VAL) <= 4
                        ):
                            self.grid[i][j].tile_type = (
                                MAP_VAL
                                if random.randint(0, 10000) % 3 == 0
                                else self.grid[i][j].tile_type
                            )


class UnitType(Enum):
    UNIT1 = "unit1"
    UNIT2 = "unit2"
    UNIT3 = "unit3"
    UNIT4 = "unit4"


class EntityType(Enum):
    EMPTY = "empty"
    UNIT1 = "unit1"
    UNIT2 = "unit2"
    UNIT3 = "unit3"
    UNIT4 = "unit4"
    BASE = "base"
    FARM = "farm"
    WEAK_TOWER = "weakTower"
    STRONG_TOWER = "strongTower"


# Что защищает от юнитов уровней 1-4
PROTECTSFROMUNIT4 = []
PROTECTSFROMUNIT3 = [EntityType.UNIT4, EntityType.UNIT3, EntityType.STRONG_TOWER]
PROTECTSFROMUNIT2 = [*PROTECTSFROMUNIT3, EntityType.UNIT2, EntityType.WEAK_TOWER]
PROTECTSFROMUNIT1 = [*PROTECTSFROMUNIT2, EntityType.UNIT1, EntityType.BASE]

# Клетки в которые может ходить юнит уровня 1-4
ALLOWEDFORUNIT1 = [EntityType.EMPTY, EntityType.FARM]
ALLOWEDFORUNIT2 = [*ALLOWEDFORUNIT1, EntityType.UNIT1, EntityType.BASE]
ALLOWEDFORUNIT3 = [*ALLOWEDFORUNIT2, EntityType.UNIT2, EntityType.WEAK_TOWER]
ALLOWEDFORUNIT4 = [*ALLOWEDFORUNIT3, EntityType.UNIT3, EntityType.UNIT4, EntityType.STRONG_TOWER]


# Максимальная глубина хода юнита (радиус)
MAX_MOVE_RANGE = 4

# Стоимость зданий (ферма: динамически, башни - константа)
COST_WEAK_TOWER = 15
COST_STRONG_TOWER = 35

# Бонус фермы за ход
FARM_INCOME = 4

# Башни: расход за ход (weak = 1, strong = 6)
WEAK_TOWER_UPKEEP = 1
STRONG_TOWER_UPKEEP = 6

# Стоимость и обслуживание юнитов
UNIT_COST = {1: 10, 2: 20, 3: 30, 4: 40}
UNIT_UPKEEP = {1: 2, 2: 4, 3: 12, 4: 18}


def get_unit_level(e: EntityType) -> int:
    """
    Возвращает уровень юнита (1..4) по его типу EntityType.
    Если не юнит, возвращает 0.
    """
    if e == EntityType.UNIT1:
        return 1
    elif e == EntityType.UNIT2:
        return 2
    elif e == EntityType.UNIT3:
        return 3
    elif e == EntityType.UNIT4:
        return 4
    return 0


def get_unit_type_by_level(level: int) -> EntityType:
    """
    Возвращает EntityType соответствующий уровню (1..4).
    Иначе выбрасывает ошибку.
    """
    if level == 1:
        return EntityType.UNIT1
    elif level == 2:
        return EntityType.UNIT2
    elif level == 3:
        return EntityType.UNIT3
    elif level == 4:
        return EntityType.UNIT4
    raise ValueError(f"Invalid unit level {level}")


class Cell():
    def __init__(self):
        self.owner: Optional[str] = None
        self.entity: EntityType = EntityType.EMPTY
        self.has_moved: bool = False
        self.base_cell: Optional[Tuple[int, int]] = None   # ref to Base cell. Base cell references to base cell


class Field:
    """
    Игровое поле (сетка из Cell).
    Содержит логику генерации/инициализации, а также методы для
    постройки, спавна и перемещения юнитов, завершения хода и др.
    """
    def __init__(self, height, width, owners: List[str]):
        """
        Создаёт поле (height x width), затем пытается расставить базы для всех игроков owners.
        Если не удалось после attemts нескольких попыток, выбрасывает исключение.
        Также инициализирует TerritoryManager.
        """
        self.cells: Dict[Tuple[int, int], Cell] = {}

        self.height = height
        self.width = width

        self.__init_field(height, width)

        attemts = 100
        success = False

        for _ in range(attemts):
            if self.create_bases(owners):
                success = True
                break
            else:
                self.__init_field(height, width)

        if not success:
            raise Exception("Failed to create field after 100 attempts")

        territories = []
        for cell_key, cell in self.cells.items():
            if cell.base_cell == cell_key:
                territories.append(Territory(cell.owner, self, cell_key, 10))

        self.territory_manager = TerritoryManager(self, territories)

    def __init_field(self, height, width):
        """
        Создаёт объект FieldGenerator и копирует из него подходящие тайлы (tile_type == 3) в self.cells.
        """
        generator = FieldGenerator(height, width, 20)
        for row in generator.grid:
            for tile in row:
                if tile.tile_type != 3:
                    continue
                i, j = tile.i, tile.j
                self.cells[i, j] = Cell()

    def get_neighbours(self, cell_key: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Возвращает список соседних клеток (координаты i,j) для шестиугольной сетки
        """
        if cell_key not in self.cells:
            raise ValueError(f'cell {cell_key} does not exist')

        i, j = cell_key

        even_offsets = [(0, -1), (0, 1), (-1, 0), (1, -1), (1, 0), (1, 1)]
        odd_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)]

        directions = odd_offsets if j % 2 == 0 else even_offsets

        return [(i + di, j + dj) for di, dj in directions if (i + di, j + dj) in self.cells]

    def get_distance_bfs(self, start: Tuple[int, int], end: Tuple[int, int]) -> int:
        """
        Находит кратчайшее расстояние (в шагах) между клетками start и end с помощью BFS.
        Возвращает -1, если путь не найден.
        """
        if start not in self.cells or end not in self.cells:
            raise ValueError(f"One or both cells {start}, {end} do not exist")

        if start == end:
            return 0

        queue = deque([(start, 0)])  # Очередь BFS (кортеж: (клетка, шаги))
        visited = set([start])       # Посещенные клетки

        while queue:
            current, steps = queue.popleft()

            for neighbor in self.get_neighbours(current):
                if neighbor == end:
                    return steps + 1  # Найден кратчайший путь

                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, steps + 1))

        return -1  # Если путь не найден

    def create_bases(self, owners: List[str]) -> bool:
        """
        Пытается создать базы (по 5 клеток) для всех игроков из списка owners.
        Если для кого-то не получилось, сбрасывает все клетки и возвращает False.
        Иначе True.
        """
        for owner in owners:
            if not self.__create_base(owner):

                # Откатываем изменения
                for cell in self.cells.values():
                    cell.owner = None
                    cell.entity = EntityType.EMPTY
                    cell.base_cell = None

                return False

        return True

    def __create_base(self, owner: str) -> bool:
        """
        Пытается создать базу (5 клеток) для конкретного игрока owner.
        Возвращает True при успехе, иначе False.
        """
        # Фильтруем клетки, которые еще не заняты (owner is None)
        free_cells = [cell_key for cell_key in self.cells if self.cells[cell_key].owner is None]
        if not free_cells:
            return False  # Нет свободных клеток для создания базы

        # Выбираем случайную стартовую клетку из свободных
        start_cell_key = random.choice(free_cells)
        base_cells: Set[Tuple[int, int]] = {start_cell_key}

        self.cells[start_cell_key].owner = owner
        self.cells[start_cell_key].entity = EntityType.BASE

        # Инициализируем frontier только незанятыми соседями стартовой клетки
        frontier: Set[Tuple[int, int]] = set(
            neighbor for neighbor in self.get_neighbours(start_cell_key)
            if self.cells[neighbor].owner is None
        )

        while len(base_cells) < 5:
            if frontier:
                # Выбираем случайного соседа из текущего frontier
                next_cell = random.choice(list(frontier))
                base_cells.add(next_cell)
                self.cells[next_cell].owner = owner
                self.cells[next_cell].entity = EntityType.EMPTY
                frontier.remove(next_cell)
            else:
                # Если frontier пуст, вычисляем новый frontier
                # как объединение свободных соседей всех клеток базы
                new_frontier: Set[Tuple[int, int]] = set()
                for cell in base_cells:
                    for neighbor in self.get_neighbours(cell):
                        if neighbor not in base_cells and self.cells[neighbor].owner is None:
                            new_frontier.add(neighbor)
                if not new_frontier:
                    return False  # Расширить базу не удалось
                frontier = new_frontier

        for cell_key in base_cells:
            self.cells[cell_key].base_cell = start_cell_key

        return True

    def check_unit_range(self, cell_key: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Возвращает список координат клеток, куда может походить юнит из cell_key (не более 4 шагов).
        При этом промежуточные шаги разрешены только по клеткам того же владельца,
        а конечная клетка может быть вражеской/ничьей, но должна удовлетворять правилам.
        """
        # Получаем исходную клетку и параметры перемещения
        start = cell_key
        start_cell = self.cells[start]
        moving_owner = start_cell.owner
        unit_type = start_cell.entity  # Должен быть UNIT1, UNIT2, UNIT3 или UNIT4

        # Проверяем перемещался ли юнит
        if start_cell.has_moved:
            return []

        max_range = 4
        queue = deque([(start, 0)])
        visited = {start}
        candidates = set()

        while queue:
            current, dist = queue.popleft()
            # Не продолжаем дальше, если достигли максимального радиуса
            if dist >= max_range:
                continue
            for neighbor in self.get_neighbours(current):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                nd = dist + 1
                # Если сосед принадлежит той же территории – используем для продолжения пути
                if self.cells[neighbor].owner == moving_owner:
                    candidates.add(neighbor)
                    queue.append((neighbor, nd))
                else:
                    # Если клетка не принадлежит движущемуся игроку, считаем её кандидатом и не продолжаем путь
                    if nd <= max_range:
                        candidates.add(neighbor)

        # Фильтруем кандидатов согласно ограничениям для конкретного типа юнита
        valid = []

        for cand in candidates:
            if self._is_valid_destination(unit_type, cand, moving_owner):
                valid.append(cand)

        return valid

    def _is_valid_destination(
        self, unit_type: EntityType, cand: Tuple[int, int], moving_owner: str
    ) -> bool:
        """
        Проверяет, можно ли походить юнитом unit_type на клетку cand, учитывая владельца moving_owner и правила.
        Есть логика merge, если там дружественный юнит (сумма <= 4).
        """
        cell = self.cells[cand]

        # Можем переместиться в клетку своей территории если...
        if cell.owner == moving_owner:
            # Если она пустая
            if cell.entity == EntityType.EMPTY:
                return True

            # Если там уже стоит дружественный юнит, пробуем merge
            if cell.entity in (EntityType.UNIT1, EntityType.UNIT2, EntityType.UNIT3, EntityType.UNIT4):
                moving_unit_level = get_unit_level(unit_type)
                target_unit_level = get_unit_level(cell.entity)
                if moving_unit_level + target_unit_level <= 4:
                    return True
                else:
                    return False

            # Иначе там здание - нельз
            return False

        # Для ниченых территорий ограничений нет
        if cell.owner is None:
            return True

        # Остается проверить только вражеские клетки

        # Получаем список соседних клеток кандидата для проверки ограничений
        neighbors = self.get_neighbours(cand)

        if unit_type == EntityType.UNIT1:
            # Клетка должна быть посильна для UNIT1
            if cell.entity not in ALLOWEDFORUNIT1:
                return False

            # Клетка не должна быть защищена от UNIT1
            for nb in neighbors:
                if (self.cells[nb].owner != moving_owner and (self.cells[nb].entity in PROTECTSFROMUNIT1)):
                    return False
            return True

        elif unit_type == EntityType.UNIT2:
            if cell.entity not in ALLOWEDFORUNIT2:
                return False
            # Клетка не должна быть защищена от UNIT2
            for nb in neighbors:
                nb_cell = self.cells[nb]
                if nb_cell.owner != moving_owner and nb_cell.entity in PROTECTSFROMUNIT2:
                    return False
            return True

        elif unit_type == EntityType.UNIT3:
            if cell.entity not in ALLOWEDFORUNIT3:
                return False
            # Клетка не должна быть защищена от UNIT3
            for nb in neighbors:
                nb_cell = self.cells[nb]
                if nb_cell.owner != moving_owner and nb_cell.entity in PROTECTSFROMUNIT3:
                    return False
            return True

        elif unit_type == EntityType.UNIT4:
            # UNIT4 может перемещаться на любую enemy клетку
            return True
        return False

    def has_farm_or_base_neigbour(self, cell_key: Tuple[int, int]):
        """
        Проверяет, есть ли рядом (в соседях) ферма или база того же владельца.
        Это нужно для строительства фермы (farm).
        """
        cell = self.cells[cell_key]
        neighbors = self.get_neighbours(cell_key)
        for nb in neighbors:
            if (self.cells[nb].owner == cell.owner and self.cells[nb].entity in [EntityType.BASE, EntityType.FARM]):
                return True
        return False

    def to_dict(self) -> dict:
        """
        Возвращает словарь, описывающий текущее состояние поля:
         - размеры (height, width)
         - cells (dict с ключами "i,j" и значениями {owner, entity})
         - territories (список словарей с информацией о каждой территории, включая funds и расчётный income).
        """
        # Формируем информацию о клетках
        cells_dict = {
            f"{i},{j}": {"owner": cell.owner, "entity": cell.entity.value, "has_moved": cell.has_moved} 
            for (i, j), cell in self.cells.items()
        }

        # Группируем территории по владельцам
        territory_info = {}
        for territory in self.territory_manager.territories:
            territory_income = self.territory_manager.calculate_income_for_territory(territory)
            info_str = (
                f"territory {len(territory.tiles)} tiles, "
                f"{territory.funds} funds, "
                f"{'+' if territory_income >= 0 else ''}{territory_income} income"
            )
            territory_info.setdefault(territory.owner, []).append(info_str)

        # Преобразуем в список словарей, где каждый словарь имеет вид {player_name: [info_str, ...]}
        territories_list = [{owner: info_list} for owner, info_list in territory_info.items()]

        return {
            "height": self.height,
            "width": self.width,
            "cells": cells_dict,
            "territories": territories_list
        }

    def get_territory_for_cell(self, cell_key: Tuple[int, int], player_name: str):
        """
        Возвращает объект Territory, к которому принадлежит клетка cell_key, если владелец — player_name.
        Если такой территории нет, выбрасывает ошибку.
        """
        for territory in self.territory_manager.territories:
            if territory.owner == player_name and cell_key in territory.tiles:
                return territory
        raise Exception("No territory found for the given cell")

    def get_moves(self, x: int, y: int, player_name: str) -> List[Dict]:
        """
        Возвращает список доступных ходов для юнита, стоящего в (y,x),
        в формате [{x, y}, ...] - список словарей.
        """
        key = (y, x)
        if key not in self.cells:
            raise Exception("Invalid cell coordinates")
        cell = self.cells[key]
        if cell.owner != player_name or "unit" not in cell.entity.value:
            raise Exception("No unit in the specified cell")
        moves = self.check_unit_range(key)
        return [{"x": move[1], "y": move[0]} for move in moves]

    def build(self, x: int, y: int, player_name: str, building: str):
        """
        Строит здание (farm/weakTower/strongTower) в клетке (y,x) игрока player_name.
        Если это ферма — её стоимость рассчитывается динамически.
        Если средств не хватает или клетка занята, выбрасывается ошибка.
        """
        key = (y, x)
        if key not in self.cells:
            raise Exception("Invalid cell coordinates")
        cell = self.cells[key]
        if cell.owner != player_name:
            raise Exception("You can only build on your own territory")
        if cell.entity != EntityType.EMPTY:
            raise Exception("Cell is not empty")

        territory = self.get_territory_for_cell(key, player_name)
        # Определяем стоимость в зависимости от типа здания
        if building.lower() == "farm":
            # Стоимость фермы = 12 + (колво ферм)
            num_farms = sum(
                1 for pos in territory.tiles
                if self.cells[pos].entity == EntityType.FARM
            )
            cost = 12 + 2 * num_farms

            if not self.has_farm_or_base_neigbour(key):
                raise Exception("You can only build farms near farms or base")
            # Если средств территории недостаточно – ошибка
            territory = self.get_territory_for_cell(key, player_name)
            if territory.funds < cost:
                raise Exception("Not enough funds in your territory to build a farm")
            territory.funds -= cost
            cell.entity = EntityType.FARM

        elif building.lower() == "weaktower":
            cost = COST_WEAK_TOWER
            territory = self.get_territory_for_cell(key, player_name)
            if territory.funds < cost:
                raise Exception("Not enough funds in your territory to build a weak tower")
            territory.funds -= cost
            cell.entity = EntityType.WEAK_TOWER

        elif building.lower() == "strongtower":
            cost = COST_STRONG_TOWER
            territory = self.get_territory_for_cell(key, player_name)
            if territory.funds < cost:
                raise Exception("Not enough funds in your territory to build a strong tower")
            territory.funds -= cost
            cell.entity = EntityType.STRONG_TOWER

        else:
            raise Exception("Invalid building type")

    def spawn_unit(self, x: int, y: int, player_name: str, level: int):
        """
        Порождает юнита нужного уровня (1..4) в клетке (y,x).
        Если там уже есть дружественный юнит, суммируем уровни (merge),
        если сумма превышает 4 — выбрасываем ошибку.
        """
        key = (y, x)
        if key not in self.cells:
            raise Exception("Invalid cell coordinates")
        cell = self.cells[key]

        ### cell может быть nonempty (при merge)
        # if cell.entity != EntityType.EMPTY:
        #    raise Exception("Cell is not empty")

        if cell.owner != player_name:
            raise Exception("You can only spawn units on your own territory")
        # Определяем стоимость спауна юнита
        if level not in UNIT_COST:
            raise Exception("Invalid unit level")
        cost = UNIT_COST[level]

        territory = self.get_territory_for_cell(key, player_name)
        if territory.funds < cost:
            raise Exception("Not enough funds in your territory to spawn this unit")

        if cell.entity.value.startswith("unit"):
            # --- MERGE при спавне ---
            # Уже есть дружественный юнит (иначе был бы owner != player_name)
            existing_level = get_unit_level(cell.entity)
            new_level = existing_level + level
            if new_level > 4:
                raise Exception("Cannot merge units: sum of levels would exceed 4")
            # Иначе объединяем
            cell.entity = get_unit_type_by_level(new_level)
        else:
            # Если нет юнита, должна быть пустая клетка или здание
            if cell.entity != EntityType.EMPTY:
                raise Exception("Cell is not empty (no merging logic for non-unit)")

            # Создаем юнит обычным способом
            cell.entity = get_unit_type_by_level(level)
            
        cell.has_moved = True
        territory.funds -= cost

    def move_unit(self, from_x: int, from_y: int, to_x: int, to_y: int, player_name: str):
        """
        Перемещает юнита из клетки (from_y, from_x) в клетку (to_y, to_x).
        Проверяет:
         - Что в исходной клетке есть юнит игрока, который ещё не двигался
         - Что цель (to_x, to_y) входит в список допустимых ходов (check_unit_range)
         - Если там дружественный юнит, пытаемся слить (merge), если уровень > 4, ошибка
         - Иначе обычное перемещение
        Затем вызывает update_territories.
        """
        from_key = (from_y, from_x)
        to_key = (to_y, to_x)
        if from_key not in self.cells or to_key not in self.cells:
            raise Exception("Invalid cell coordinates")

        from_cell = self.cells[from_key]
        to_cell = self.cells[to_key]
        if from_cell.owner != player_name or "unit" not in from_cell.entity.value:
            raise Exception("No unit to move on the source cell")
        if from_cell.has_moved:
            raise Exception("This unit has already moved this turn")

        valid_moves = self.check_unit_range(from_key)
        if to_key not in valid_moves:
            raise Exception("Target cell is out of range or invalid")

        if to_cell.owner == player_name and to_cell.entity.value.startswith("unit"):
            # --- MERGE  при перемещении ---
            level_from = get_unit_level(from_cell.entity)
            level_to = get_unit_level(to_cell.entity)
            merged_level = level_from + level_to
            if merged_level > 4:
                raise Exception("Cannot merge units: sum of levels would exceed 4")

            # Сливаем два юнита в один
            to_cell.entity = get_unit_type_by_level(merged_level)
            to_cell.has_moved = True  # Новый юнит считается переместившимся
            from_cell.entity = EntityType.EMPTY
        else:
            # Обычное перемещение
            to_cell.owner = player_name
            to_cell.entity = from_cell.entity
            to_cell.has_moved = True
            from_cell.entity = EntityType.EMPTY

        # После перемещения обновляем территории
        self.territory_manager.update_territories()

    def end_turn(self, current_turn: str, players: List) -> Optional[str]:
        """
        Завершает ход для игрока current_turn:
         - Сбрасывает has_moved у всех клеток
         - Обновляет фонды (update_funds)
         - Проверяет, остался ли один владелец (если да, он победил)
         Возвращает имя победителя или None.
        """
        # Сбрасываем флаг перемещения для всех клеток
        for cell in self.cells.values():
            cell.has_moved = False

        # Обновляем фонды каждой территории (внутри TerritoryManager)
        self.territory_manager.update_funds(current_turn)
        # Пересчитываем и объединяем/разделяем территории, если необходимо
        # self.territory_manager.update_territories()

        # Проверяем условие победы: если на поле присутствует только один владелец клеток, он выигрывает.
        owners_present = {t.owner for t in self.territory_manager.territories if t.owner is not None}
        if len(owners_present) == 1:
            return owners_present.pop()

        return None


class Territory:
    """
    Территория (набор связанных клеток одного владельца).
    Имеет базовую клетку (base_key), funds, владелец owner,
    а также набор клеток tiles.
    """
    def __init__(self, owner: str, field: Field, base_key: Tuple[int, int], funds=0):
        """
        Инициализирует территорию, вычисляет её компоненту связности (tiles)
        исходя из base.
        """
        self.owner: str = owner
        self.funds: int = funds
        self.field: Field = field
        self.base_key: Tuple[int, int]= base_key
        self.tiles: Set[Tuple[int, int]] = set()
        self.tiles = self.get_territory_component(self.base_key)

    def get_territory_component(self, base_key: Tuple[int, int]) -> Set[Tuple[int, int]]:
        """
        Находит все клетки, принадлежащие тому же владельцу, что и base_key,
        и достижимые через соседей. Возвращает их в виде множества.
        """
        if base_key not in self.field.cells:
            raise ValueError(f"Клетка {base_key} не существует")

        base_owner = self.field.cells[base_key].owner
        if base_owner is None:
            return set()

        component = set()
        queue = deque([base_key])
        component.add(base_key)

        while queue:
            current = queue.popleft()
            for nb in self.field.get_neighbours(current):
                if nb not in component and self.field.cells[nb].owner == base_owner:
                    component.add(nb)
                    queue.append(nb)

        return component

    def update_component(self):
        """
        Пересчитывает set клеток (tiles) на случай, если территория расширилась или уменьшилась.
        Если базовая клетка изменила владельца, выбирается новая база случайно из списка tiles.
        """
        if self.base_key not in self.field.cells or self.field.cells[self.base_key].owner != self.owner:
            if self.tiles:
                self.base_key = random.choice(list(self.tiles))
            else:
                return  # территории нет
        comp = set()
        queue = deque([self.base_key])
        comp.add(self.base_key)
        while queue:
            current = queue.popleft()
            for neighbor in self.field.get_neighbours(current):
                if (
                    neighbor in self.field.cells
                    and self.field.cells[neighbor].owner == self.owner
                    and neighbor not in comp
                ):
                    comp.add(neighbor)
                    queue.append(neighbor)
        self.tiles = comp


class TerritoryManager:
    """
    Управляет всеми территориями (список Territory).
    Позволяет обновлять территории (слияние, разделение) и пересчитывать фонды.
    """
    def __init__(self, field: Field, territories: List[Territory]):
        self.field = field
        self.territories = territories


    def has_territories(self, owner: str) -> bool:
        """
        Проверяет, имеет ли игрок owner хотя бы одну территорию.
        """
        return any(territory.owner == owner for territory in self.territories)

    def update_territories(self):
        """
        Обновляет территории для всех владельцев:
         1) Находит компоненты связности клеток одного владельца
         2) Определяет, куда переходят funds при слиянии/разделении
         3) Создаёт новые объекты Territory
        """
        new_territories = []

        # Группируем старые территории по владельцам
        territories_by_owner = {}
        for t in self.territories:
            territories_by_owner.setdefault(t.owner, []).append(t)

        for owner, old_territories in territories_by_owner.items():
            # Собираем все клетки, принадлежащие владельцу (актуальное состояние поля)
            owner_cells = {pos for pos, cell in self.field.cells.items() if cell.owner == owner}
            # Вычисляем компоненты связности для этих клеток
            new_components = self._compute_connected_components(owner_cells)
            # Для удобства работаем с ключами-компонентами в виде frozenset
            comp_contributions = {}  # key: frozenset(component), value: dict с суммой фондов и кандидатами для базы
            comp_map = {}  # отображает ключ (frozenset) на исходное множество клеток
            for comp in new_components:
                comp_key = frozenset(comp)
                comp_contributions[comp_key] = {"funds": 0, "bases": []}
                comp_map[comp_key] = comp

            # Для каждого старого объекта Territory определяем, в какую новую компоненту попало
            # наибольшее число его клеток. Именно эта компонента получит все его фонды.
            territory_best_component = {}  # key: объект Territory, value: ключ (frozenset) новой компоненты
            for t in old_territories:
                best_comp_key = None
                best_intersection = 0
                for comp in new_components:
                    comp_key = frozenset(comp)
                    # Пересечение с исходной территорией (старый набор клеток)
                    inter_size = len(comp & t.tiles)
                    if inter_size > best_intersection:
                        best_intersection = inter_size
                        best_comp_key = comp_key
                territory_best_component[t] = best_comp_key

            # Каждая старая территория, если она оказалась в новой компоненте, «передаёт» свои фонды
            # именно в ту компоненту, где пересечение было максимальным.
            for t, comp_key in territory_best_component.items():
                if comp_key is not None:
                    comp_contributions[comp_key]["funds"] += t.funds
                    # Если базовая клетка старой территории попала в эту компоненту – используем её как кандидат
                    if t.base_key in comp_map[comp_key]:
                        comp_contributions[comp_key]["bases"].append(t.base_key)

            # Создаём новые объекты Territory для каждой найденной компоненты
            for comp_key, info in comp_contributions.items():

                comp = comp_map[comp_key]
                # Компонента из одной клетки не считается территорией
                if len(comp) == 1:
                    cell_key = list(comp)[0]
                    self.field.cells[cell_key].entity = EntityType.EMPTY
                    continue

                    # TODO (убрать владение территорией размера 1?)

                funds = info["funds"]
                # Если кандидат для базы есть – выбираем случайно, иначе выбираем случайную клетку из компоненты
                if info["bases"]:
                    base_key = random.choice(info["bases"])

                    # Удаляем остальные базы
                    for k in info["bases"]:
                        if k != base_key:
                            self.field.cells[k].entity = EntityType.EMPTY
                else:
                    empty_cells = [cell_key for cell_key in list(comp) if self.field.cells[cell_key].entity == EntityType.EMPTY]

                    if not empty_cells:
                        # Если все клетки заняты, на базу заменяем случайную клетку
                        base_key = random.choice(list(comp))
                    else:
                        # Иначе выбираем только из пустых
                        base_key = random.choice(empty_cells)

                    # Создаем базу
                    self.field.cells[base_key].entity = EntityType.BASE
                new_territory = Territory(owner, self.field, base_key, funds)
                new_territory.tiles = comp
                new_territories.append(new_territory)

        self.territories = new_territories

    def calculate_income_for_territory(self, territory: Territory) -> int:
        """
        Считает, какой доход (income) територия получила бы на текущий ход.
        Учитывает кол-во клеток, ферм, башен, юнитов.
        """
        unit_levels = {
            EntityType.UNIT1: 1,
            EntityType.UNIT2: 2,
            EntityType.UNIT3: 3,
            EntityType.UNIT4: 4,
        }
        captured = len(territory.tiles)
        num_farms = sum(1 for pos in territory.tiles if self.field.cells[pos].entity == EntityType.FARM)
        weak_towers = sum(1 for pos in territory.tiles if self.field.cells[pos].entity == EntityType.WEAK_TOWER)
        strong_towers = sum(1 for pos in territory.tiles if self.field.cells[pos].entity == EntityType.STRONG_TOWER)
        unit_upkeep_cost = sum(
            UNIT_UPKEEP[unit_levels[self.field.cells[pos].entity]]
            for pos in territory.tiles
            if self.field.cells[pos].entity in unit_levels
        )
        income = (
            captured
            + FARM_INCOME * num_farms
            - (WEAK_TOWER_UPKEEP * weak_towers + STRONG_TOWER_UPKEEP * strong_towers)
            - unit_upkeep_cost
        )
        return income

    def _compute_connected_components(self, cells: Set[Tuple[int, int]]) -> List[Set[Tuple[int, int]]]:
        """
        Находит компоненты связности клеток cells (принадлежащих одному владельцу).
        Возвращает список множеств клеток.
        """
        components = []
        visited = set()
        for cell in cells:
            if cell in visited:
                continue
            comp = set()
            queue = deque([cell])
            visited.add(cell)
            comp.add(cell)
            while queue:
                current = queue.popleft()
                for neighbor in self.field.get_neighbours(current):
                    if neighbor in cells and neighbor not in visited:
                        visited.add(neighbor)
                        comp.add(neighbor)
                        queue.append(neighbor)
            components.append(comp)
        return components

    def update_funds(self, owner):
        """
        Начисляет доход (income) всем территориям игрока owner 
        (по формуле calculate_income_for_territory) и прибавляет его к funds.
        Если funds уходит в минус, убиваем все юниты (entity=EMPTY), funds=0.
        """
        unit_levels = {
            EntityType.UNIT1: 1,
            EntityType.UNIT2: 2,
            EntityType.UNIT3: 3,
            EntityType.UNIT4: 4,
        }
        for territory in self.territories:
            if territory.owner != owner:
                continue
            # Определяем доход

            income = self.calculate_income_for_territory(territory)
            territory.funds += income

            # Если доход отрицательный – убиваем юнитов и обнуляем funds
            if territory.funds < 0:
                for pos in territory.tiles:
                    if self.field.cells[pos].entity in unit_levels:
                        self.field.cells[pos].entity = EntityType.EMPTY
                territory.funds = 0
