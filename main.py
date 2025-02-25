from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
import random

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")  # Подключаем статические файлы (HTML)

# Модель клетки: хранит владельца, здание и юнит
class Cell(BaseModel):
    owner: Optional[str] = None
    building: Optional[str] = None
    unit: Optional[int] = None

# Модель игрока: имя, монеты, количество ферм
class Player(BaseModel):
    name: str
    coins: int = 100
    farms_bought: int = 0

# Модель состояния игры: карта, игроки, текущий ход
class GameState(BaseModel):
    rows: int
    cols: int
    grid: List[List[Cell]]
    players: List[Player]
    current_turn: str

# Модели запросов для действий игрока
class BuildRequest(BaseModel):
    building: str
    x: int
    y: int
    player_name: str

class SpawnUnitRequest(BaseModel):
    level: int
    x: int
    y: int
    player_name: str

class MoveUnitRequest(BaseModel):
    from_x: int
    from_y: int
    to_x: int
    to_y: int
    player_name: str

class GetMovesRequest(BaseModel):
    x: int
    y: int
    player_name: str

# Инициализация игры: создает карту и базы игроков
def initialize_game_state(rows: int = 10, cols: int = 10):
    grid = [[Cell() for _ in range(cols)] for _ in range(rows)]
    players = [Player(name="player1"), Player(name="player2")]

    # Генерация баз 3x3 для двух игроков
    base_x1 = random.randint(0, cols - 3)
    base_y1 = random.randint(0, rows - 3)
    base_x2 = random.randint(0, cols - 3)
    base_y2 = random.randint(0, rows - 3)
    while abs(base_x1 - base_x2) < 3 and abs(base_y1 - base_y2) < 3:
        base_x2 = random.randint(0, cols - 3)
        base_y2 = random.randint(0, rows - 3)

    for dx in range(3):
        for dy in range(3):
            grid[base_y1 + dy][base_x1 + dx].owner = "player1"
            grid[base_y2 + dy][base_x2 + dx].owner = "player2"
    grid[base_y1 + 1][base_x1 + 1].building = "townhall"
    grid[base_y2 + 1][base_x2 + 1].building = "townhall"

    return GameState(rows=rows, cols=cols, grid=grid, players=players, current_turn="player1")

game_state = initialize_game_state()

# Расчет дохода: клетки + фермы - расходы башен
def calculate_income(player: Player) -> int:
    captured = sum(1 for row in game_state.grid for cell in row if cell.owner == player.name)
    farm_bonus = 4 * sum(1 for row in game_state.grid for cell in row if cell.building == "farm" and cell.owner == player.name)
    tower_cost = sum(1 for row in game_state.grid for cell in row if cell.building == "weak_tower" and cell.owner == player.name) + \
                 6 * sum(1 for row in game_state.grid for cell in row if cell.building == "strong_tower" and cell.owner == player.name)
    return captured + farm_bonus - tower_cost

# Проверка возможности постройки фермы: нужна близость к ратуше или другой ферме
def can_build_farm(x: int, y: int, player_name: str) -> bool:
    if game_state.grid[y][x].owner != player_name or game_state.grid[y][x].building:
        return False
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < game_state.rows and 0 <= nx < game_state.cols:
                if game_state.grid[ny][nx].building in ["townhall", "farm"]:
                    return True
    return False

# Проверка блокировки башни: блокирует только врагов
def is_blocked_by_tower(x: int, y: int, unit_level: int, player_name: str) -> bool:
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            ny, nx = y + dy, x + dx
            if 0 <= ny < game_state.rows and 0 <= nx < game_state.cols:
                building = game_state.grid[ny][nx].building
                owner = game_state.grid[ny][nx].owner
                if owner != player_name:  # Башни противника
                    if building == "weak_tower" and unit_level < 3:
                        return True
                    if building == "strong_tower" and unit_level < 4:
                        return True
    return False

# Получение доступных ходов: DFS с глубиной 4, учитывает соседство для нейтральных клеток
def get_available_moves(x: int, y: int, player_name: str) -> List[Dict[str, int]]:
    unit_level = game_state.grid[y][x].unit
    if not unit_level:
        return []
    
    available = set()
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1 if x % 2 else 1), (1, -1 if x % 2 else 1)]

    def dfs(cx: int, cy: int, depth: int):
        if depth > 4 or (cx, cy) in available:
            return
        if not (0 <= cx < game_state.cols and 0 <= cy < game_state.rows):
            return
        if is_blocked_by_tower(cx, cy, unit_level, player_name):
            return
        if depth > 0 and game_state.grid[cy][cx].unit and game_state.grid[cy][cx].owner != player_name:
            return

        # Проверка соседей для нейтральных/вражеских клеток
        has_player_neighbor = False
        for dy, dx in directions:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < game_state.rows and 0 <= nx < game_state.cols and game_state.grid[ny][nx].owner == player_name:
                has_player_neighbor = True
                break
        
        if game_state.grid[cy][cx].owner != player_name and not has_player_neighbor:
            return
        
        available.add((cx, cy))
        if game_state.grid[cy][cx].owner != player_name:
            return
        
        for dx, dy in directions:
            nx, ny = cx + dx, cy + dy
            dfs(nx, ny, depth + 1)

    dfs(x, y, 0)
    return [{"x": cx, "y": cy} for cx, cy in available]

# Эндпоинты
@app.get("/")
async def redirect_to_game():
    return RedirectResponse(url="/static/index.html")  # Перенаправление на игру

@app.get("/state")
def get_state():
    return game_state  # Возвращает текущее состояние игры

@app.post("/end_turn")
def end_turn():
    player = next(p for p in game_state.players if p.name == game_state.current_turn)
    player.coins += calculate_income(player)  # Начисляем доход перед сменой хода
    game_state.current_turn = "player2" if game_state.current_turn == "player1" else "player1"
    return {"status": "success"}

@app.post("/build")
def build(request: BuildRequest):
    if game_state.current_turn != request.player_name:
        raise HTTPException(status_code=400, detail="Не ваш ход")
    if not (0 <= request.x < game_state.cols and 0 <= request.y < game_state.rows):
        raise HTTPException(status_code=400, detail="Недопустимые координаты")
    if game_state.grid[request.y][request.x].owner != request.player_name or game_state.grid[request.y][request.x].building:
        raise HTTPException(status_code=400, detail="Нельзя строить здесь")
    
    player = next(p for p in game_state.players if p.name == request.player_name)
    cost = {"farm": 12 + 2 * player.farms_bought, "weak_tower": 15, "strong_tower": 35}.get(request.building, 0)
    if cost == 0:
        raise HTTPException(status_code=400, detail="Недопустимый тип здания")
    if player.coins < cost:
        raise HTTPException(status_code=400, detail="Недостаточно монет")
    if request.building == "farm" and not can_build_farm(request.x, request.y, request.player_name):
        raise HTTPException(status_code=400, detail="Ферму можно строить только рядом с ратушей или другой фермой")
    
    player.coins -= cost
    if request.building == "farm":
        player.farms_bought += 1
    game_state.grid[request.y][request.x].building = request.building
    return {"status": "success"}

@app.post("/spawn_unit")
def spawn_unit(request: SpawnUnitRequest):
    if game_state.current_turn != request.player_name:
        raise HTTPException(status_code=400, detail="Не ваш ход")
    if not (1 <= request.level <= 4):
        raise HTTPException(status_code=400, detail="Недопустимый уровень юнита")
    if not (0 <= request.x < game_state.cols and 0 <= request.y < game_state.rows):
        raise HTTPException(status_code=400, detail="Недопустимые координаты")
    if game_state.grid[request.y][request.x].owner != request.player_name or game_state.grid[request.y][request.x].unit:
        raise HTTPException(status_code=400, detail="Нельзя разместить юнита здесь")
    if is_blocked_by_tower(request.x, request.y, request.level, request.player_name):
        raise HTTPException(status_code=400, detail="Клетка заблокирована башней")
    
    player = next(p for p in game_state.players if p.name == request.player_name)
    cost = request.level * 10
    if player.coins < cost:
        raise HTTPException(status_code=400, detail="Недостаточно монет")
    
    player.coins -= cost
    game_state.grid[request.y][request.x].unit = request.level
    return {"status": "success"}

@app.post("/move_unit")
def move_unit(request: MoveUnitRequest):
    if game_state.current_turn != request.player_name:
        raise HTTPException(status_code=400, detail="Не ваш ход")
    if not (0 <= request.from_x < game_state.cols and 0 <= request.from_y < game_state.rows and
            0 <= request.to_x < game_state.cols and 0 <= request.to_y < game_state.rows):
        raise HTTPException(status_code=400, detail="Недопустимые координаты")
    if game_state.grid[request.from_y][request.from_x].unit is None or game_state.grid[request.from_y][request.from_x].owner != request.player_name:
        raise HTTPException(status_code=400, detail="На этой клетке нет вашего юнита")
    
    moves = get_available_moves(request.from_x, request.from_y, request.player_name)
    if not any(m["x"] == request.to_x and m["y"] == request.to_y for m in moves):
        raise HTTPException(status_code=400, detail="Юнит не может туда переместить")
    
    unit = game_state.grid[request.from_y][request.from_x].unit
    target_unit = game_state.grid[request.to_y][request.to_x].unit
    
    if target_unit and game_state.grid[request.to_y][request.to_x].owner == request.player_name:
        # Слияние юнитов
        new_level = min(4, unit + target_unit)
        game_state.grid[request.from_y][request.from_x].unit = None
        game_state.grid[request.to_y][request.to_x].unit = new_level
    else:
        # Перемещение
        game_state.grid[request.from_y][request.from_x].unit = None
        game_state.grid[request.to_y][request.to_x].unit = unit
        game_state.grid[request.to_y][request.to_x].owner = request.player_name
    
    return {"status": "success"}

@app.post("/get_moves")
def get_moves(request: GetMovesRequest):
    if game_state.current_turn != request.player_name:
        raise HTTPException(status_code=400, detail="Не ваш ход")
    if not (0 <= request.x < game_state.cols and 0 <= request.y < game_state.rows):
        raise HTTPException(status_code=400, detail="Недопустимые координаты")
    if game_state.grid[request.y][request.x].unit is None or game_state.grid[request.y][request.x].owner != request.player_name:
        raise HTTPException(status_code=400, detail="На этой клетке нет вашего юнита")
    
    moves = get_available_moves(request.x, request.y, request.player_name)
    return {"status": "success", "moves": moves}
