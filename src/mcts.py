import asyncio
import aiohttp
import math
import random
import logging
import time
import os
import json
import uuid
from typing import Any, Dict, List, Optional, Tuple
from tqdm import tqdm
from pydantic import BaseModel, Field
from aiohttp import ClientOSError, ServerDisconnectedError, ClientResponseError, ClientConnectionError
from dataclasses import dataclass


# Pydantic models remain unchanged
class CellData(BaseModel):
    owner: Optional[str]
    entity: str
    has_moved: bool


class FieldData(BaseModel):
    height: int
    width: int
    cells: Dict[str, CellData]
    territories: List[Dict[str, List[str]]]


class TerritoryData(BaseModel):
    owner: str
    funds: int
    tiles: List[List[int]]


class GameState(BaseModel):
    players: List[str]
    current_player_index: int
    field_data: FieldData
    territories_data: List[TerritoryData]
    is_game_over: bool = False
    winner: Optional[str] = None
    last_action: Optional[str] = None


class Action(BaseModel):
    action_id: int
    action_type: str
    params: Optional[Dict[str, Any]] = None
    description: Optional[str] = None


class ActionRequest(BaseModel):
    state: GameState
    action_id: Optional[int] = None


class ApplyActionResponse(BaseModel):
    state: GameState
    is_game_over: bool
    winner: Optional[str]


# Asynchronous Game API client


class AsyncGameApi:
    def __init__(
        self,
        base_url: str,
        timeout: int = 300,
        max_concurrent: int = 100,
        max_retries: int = 10,
    ):
        self.base = base_url.rstrip("/")
        self.timeout = timeout
        self.query_count = 0
        self._sem = asyncio.Semaphore(max_concurrent)
        self._max_retries = max_retries
        self._session: Optional[aiohttp.ClientSession] = None

        logging.basicConfig(
            filename="mcst_api_async.log",
            format="%(asctime)s %(levelname)s %(message)s",
            level=logging.INFO,
        )

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            timeout_cfg = aiohttp.ClientTimeout(
                total=self.timeout,
                connect=self.timeout,
                sock_read=self.timeout,
            )
            self._session = aiohttp.ClientSession(timeout=timeout_cfg)

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()

    async def _inc(self):
        self.query_count += 1

    async def _request_with_retry(
        self,
        method: str,
        endpoint: str,
        params: Dict[str, Any] = None,
        json_data: Any = None,
    ) -> Any:
        await self._ensure_session()
        url = f"{self.base}/{endpoint.lstrip('/')}"
        backoff = 0.5

        for attempt in range(1, self._max_retries + 1):
            await self._inc()
            try:
                async with self._sem:
                    async with getattr(self._session, method)(url, params=params, json=json_data) as resp:
                        resp.raise_for_status()
                        return await resp.json()

            except ClientResponseError as e:
                # Ретрайим на 5xx
                if 500 <= e.status < 600:
                    err = e
                else:
                    raise

            except (ClientOSError, ServerDisconnectedError, ClientConnectionError, asyncio.TimeoutError) as e:
                # Сетевая ошибка или таймаут
                err = e

            else:
                # Если не было исключений — выходим
                break

            # Если мы здесь, значит err установлено и нужно ретраить
            logging.warning(f"[Attempt {attempt}] {method.upper()} {endpoint} failed: {err}")
            if attempt == self._max_retries:
                logging.error(f"All {attempt} retries failed for {method.upper()} {endpoint}")
                raise err
            await asyncio.sleep(backoff)
            backoff *= 2

    async def generate_state(self, num_players: int = 2, randomize: bool = False) -> GameState:
        data = await self._request_with_retry(
            method="post",
            endpoint="/generate_state",
            params={"num_players": num_players, "random": str(randomize).lower()},
        )
        return GameState.model_validate(data)

    async def get_actions(self, state: GameState) -> List[Action]:
        data = await self._request_with_retry(
            method="post",
            endpoint="/get_actions",
            json_data=state.model_dump(),
        )
        return [Action.model_validate(o) for o in data]

    async def apply_action(self, state: GameState, action: Action) -> ApplyActionResponse:
        req = ActionRequest(state=state, action_id=action.action_id)
        data = await self._request_with_retry(
            method="post",
            endpoint="/apply_action",
            json_data=req.model_dump(),
        )
        return ApplyActionResponse.model_validate(data)

    @staticmethod
    def is_terminal(state: GameState) -> bool:
        return state.is_game_over

    @staticmethod
    def current_player(state: GameState) -> str:
        return state.players[state.current_player_index]

    @staticmethod
    def evaluate_state(state: GameState, player_id: str) -> float:
        cells = state.field_data.cells
        total = len(cells)
        cnt_cur = sum(1 for c in cells.values() if c.owner == player_id)
        return cnt_cur / max(1, total)


# MCTS Node (no locks needed with asyncio)
class MCTSNode:
    def __init__(self, state: GameState, parent: Optional["MCTSNode"] = None, action: Optional[Action] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions: Optional[List[Action]] = None

    def is_fully_expanded(self) -> bool:
        return self.untried_actions is not None and not self.untried_actions

    def best_uct_child(self, c: float) -> "MCTSNode":
        if not self.children:
            return None

        def uct_score(child):
            if child.visits == 0:
                return float("inf")
            return (child.value / child.visits) + c * math.sqrt(math.log(self.visits) / child.visits)

        return max(self.children, key=uct_score)

    def find_child_by_action(self, action_id: int) -> Optional["MCTSNode"]:
        for child in self.children:
            if child.action and child.action.action_id == action_id:
                return child
        return None


# Asynchronous MCTS Player
class MCTSPlayer:
    def __init__(
        self,
        player_id: str,
        game_api: AsyncGameApi,
        c: float = math.sqrt(2),
        max_depth: int = 200,
        iterations: int = 100,
    ):
        self.player_id = player_id
        self.game = game_api
        self.c = c
        self.max_depth = max_depth
        self.root = None
        self.iterations = iterations

    async def initialize(self, state: GameState):
        self.root = MCTSNode(state)
        self.root.untried_actions = await self.game.get_actions(state)

    async def _select(self, node: MCTSNode) -> MCTSNode:
        while not self.game.is_terminal(node.state) and node.is_fully_expanded():
            child = node.best_uct_child(self.c)
            if not child:
                break
            node = child
        return node

    async def _expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        if node.untried_actions is None:
            node.untried_actions = await self.game.get_actions(node.state)
        if not node.untried_actions:
            return None

        actions = node.untried_actions
        idx = random.randrange(len(actions))
        actions[idx], actions[-1] = actions[-1], actions[idx]
        action = actions.pop()

        res = await self.game.apply_action(node.state, action)
        child = MCTSNode(res.state, parent=node, action=action)
        child.untried_actions = await self.game.get_actions(res.state)
        node.children.append(child)
        return child

    async def _simulate(self, node: MCTSNode) -> float:
        state = node.state
        depth = 0
        total_reward = 0.0

        while depth < self.max_depth and not self.game.is_terminal(state):
            actions = await self.game.get_actions(state)
            if not actions:
                break

            action = random.choice(actions)
            res = await self.game.apply_action(state, action)
            if res.is_game_over:
                if res.winner == self.player_id:
                    return 3
                else:
                    return -3
            new_state = res.state
            prev_owned = {key for key, cell in state.field_data.cells.items() if cell.owner == self.player_id}
            new_owned = {key for key, cell in new_state.field_data.cells.items() if cell.owner == self.player_id}

            opp_prev_owned = {
                key for key, cell in state.field_data.cells.items() if cell.owner and cell.owner != self.player_id
            }
            opp_new_owned = {
                key for key, cell in new_state.field_data.cells.items() if cell.owner and cell.owner != self.player_id
            }

            opponent_captured = len(opp_new_owned - opp_prev_owned)
            opponent_lost = len(opp_prev_owned - opp_new_owned)

            captured = len(new_owned - prev_owned)
            lost = len(prev_owned - new_owned)

            gamma = 0.999
            step_reward = (0.1 * captured + 0.6 * opponent_lost - 0.2 * opponent_captured - 0.6 * lost) * (
                gamma**depth
            )
            total_reward += step_reward

            depth += 1
            state = new_state

        return total_reward

    def _backpropagate(self, node: MCTSNode, reward: float):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    async def _run_iteration(self):
        node = await self._select(self.root)
        if not self.game.is_terminal(node.state):
            new_node = await self._expand(node)
            if new_node:
                node = new_node
        reward = await self._simulate(node)
        self._backpropagate(node, reward)

    async def search(self, iterations: int, show_progress: bool = False, desc: str = "MCTS") -> Action:
        if not self.root:
            raise ValueError("MCTS дерево не инициализировано")

        if self.root.untried_actions is None:
            self.root.untried_actions = await self.game.get_actions(self.root.state)

        if show_progress:
            pbar = tqdm(total=iterations, desc=desc, unit="iter")
            start_time = time.time()
            start_queries = self.game.query_count

        # tasks = [self._run_iteration() for _ in range(iterations)]
        # for i, task in enumerate(asyncio.as_completed(tasks)):
        #     await task
        #     if show_progress:
        #         elapsed = time.time() - start_time
        #         queries_done = self.game.query_count - start_queries
        #         qps = queries_done / elapsed if elapsed > 0 else 0
        #         pbar.set_postfix({"iter": i + 1, "queries": queries_done, "QPS": f"{qps:.1f}"})
        #         pbar.update(1)
        tasks = [self._run_iteration() for _ in range(iterations)]
        for i, task in enumerate(tasks):
            await task
            if show_progress:
                elapsed = time.time() - start_time
                queries_done = self.game.query_count - start_queries
                qps = queries_done / elapsed if elapsed > 0 else 0
                pbar.set_postfix({"iter": i + 1, "queries": queries_done, "QPS": f"{qps:.1f}"})
                pbar.update(1)

        if show_progress:
            pbar.close()

        best_child = None
        best_visits = -1

        for child in self.root.children:
            if child.visits > best_visits:
                best_visits = child.visits
                best_child = child

        if not best_child:
            if not self.root.untried_actions:
                self.root.untried_actions = await self.game.get_actions(self.root.state)
            if self.root.untried_actions:
                return random.choice(self.root.untried_actions)
            else:
                raise ValueError("No valid actions found!")
            
            # Собираем статистику
        stats = {
            "visit_distribution": {},
            "action_values": {},
            "total_visits": self.root.visits
        }
        
        total_visits = sum(child.visits for child in self.root.children)
        
        for child in self.root.children:
            action_id = child.action.action_id
            # Нормализованные визиты
            stats["visit_distribution"][action_id] = child.visits / total_visits
            # Средние значения
            stats["action_values"][action_id] = (
                child.value / child.visits if child.visits > 0 else 0.0
            )
        
        return best_child.action, stats

        return best_child.action

    async def update_root(self, action_id: int, new_state: GameState) -> bool:
        if not self.root or not self.root.children:
            await self.initialize(new_state)
            return False

        child = self.root.find_child_by_action(action_id)
        if child:
            child.parent = None
            self.root = child
            if self.root.untried_actions is None:
                self.root.untried_actions = await self.game.get_actions(self.root.state)
            return True
        else:
            await self.initialize(new_state)
            return False


# Asynchronous game simulation
async def simulate_game(
    api: AsyncGameApi,
    max_moves: int = 50,
    mcts_iters: int = 500,
    c: float = 1.4,
    max_depth: int = 200,
) -> None:
    os.makedirs("logs", exist_ok=True)
    session_id = uuid.uuid4().hex
    date_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_path = os.path.join("logs", f"{session_id}_{date_str}.json")

    print(f"Log file: {log_path}")

    state = await api.generate_state(num_players=2, randomize=False)

    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(state.model_dump(), ensure_ascii=False) + "\n")

    player1 = MCTSPlayer(player_id=state.players[0], game_api=api, c=c, max_depth=max_depth, iterations=mcts_iters)
    player2 = MCTSPlayer(player_id=state.players[1], game_api=api, c=c, max_depth=max_depth, iterations=mcts_iters)

    await player1.initialize(state)
    await player2.initialize(state)

    history = []

    for move_no in tqdm(range(1, max_moves + 1), desc="Game moves"):
        current_player_idx = state.current_player_index
        current_player_id = state.players[current_player_idx]

        active_player = player1 if current_player_id == player1.player_id else player2

        print(f"\nMove {move_no} - Player {current_player_id}")

        action = await active_player.search(
            iterations=mcts_iters, show_progress=True, desc=f"Move {move_no} (Player {current_player_id})"
        )

        if active_player.root and active_player.root.children:
            sorted_children = sorted(active_player.root.children, key=lambda child: child.visits, reverse=True)
            print("Root children stats (action_id: visits, value):")
            for child in sorted_children:
                aid = child.action.description if child.action else None
                print(f"  {aid}: {child.visits}, {child.value:.2f}")
        else:
            print("No children at root yet.")

        print(f"Selected action: {action.action_id} ({action.action_type})")

        response = await api.apply_action(state, action)
        new_state = response.state

        history.append((move_no, current_player_id, action.action_id))

        print(f"Player1 размер дерева: {len(player1.root.children) if player1.root else 0}")
        print(f"Player2 размер дерева: {len(player2.root.children) if player2.root else 0}")

        player1_updated = await player1.update_root(action.action_id, new_state)
        player2_updated = await player2.update_root(action.action_id, new_state)

        state = new_state

        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(state.model_dump(), ensure_ascii=False) + "\n")

        if response.is_game_over:
            break

    winner = state.winner
    print(f"{move_no} ходов, победил {winner}")


@dataclass
class MCTSExperience:
    # Состояние
    state: Dict

    # Распределение визитов MCTS (нормализованное)
    visit_distribution: Dict[int, float]  # {action_id: probability}

    # Q-значения для действий
    action_values: Dict[int, float]  # {action_id: average_value}

    # Метаданные
    total_visits: int  # общее количество симуляций MCTS
    selected_action_id: int  # какое действие было выбрано
    game_progress: float  # 0.0 - начало, 1.0 - конец игры

    # Результат игры (заполняется в конце)
    game_outcome: Optional[float] = None  # 1.0 если выиграл, -1.0 если проиграл
    moves_to_end: Optional[int] = None  # сколько ходов до конца игры


class BootstrapDataCollector:
    def __init__(self, api: AsyncGameApi, mcts_params: Dict):
        self.api = api
        self.mcts_params = mcts_params
        self.experiences = []

    def get_temperature(self, move_count: int) -> float:
        """Температура в зависимости от стадии игры"""
        schedule = self.mcts_params["temperature_schedule"]
        for threshold in sorted(schedule.keys(), reverse=True):
            if move_count >= threshold:
                return schedule[threshold]
        return 1.0

    async def collect_game(self, game_id: int) -> Dict:
        """Собирает данные одной игры"""

        # Инициализация
        state = await self.api.generate_state()
        players = [
            MCTSPlayer(
                game_api=self.api, max_depth=self.mcts_params["max_depth"], player_id=0, c=self.mcts_params["c"]
            ),
            MCTSPlayer(
                game_api=self.api, max_depth=self.mcts_params["max_depth"], player_id=1, c=self.mcts_params["c"]
            ),
        ]
        await players[0].initialize(state)
        await players[1].initialize(state)

        game_experiences = []
        move_count = 0

        # Играем игру
        while move_count < 1000:  # Защита от бесконечных игр
            current_player_idx = state.current_player_index
            current_player = players[current_player_idx]

            # Температура для текущего хода
            temperature = self.get_temperature(move_count)

            # MCTS поиск
            action_id, stats = await current_player.search(
                iterations=self.mcts_params["iterations"],
                show_progress=True,
            )

            # Создаем experience
            experience = MCTSExperience(
                state=state,
                visit_distribution=stats["visit_distribution"],
                action_values=stats["action_values"],
                total_visits=stats["total_visits"],
                selected_action_id=-999,
                game_progress=move_count / 100.0,  # Нормализуем к ожидаемой длине
            )

            game_experiences.append(experience)

            # Применяем действие
            result = await self.api.apply_action(state, action_id)
            state = result.state
            move_count += 1

            # Проверяем конец игры
            if result.is_game_over:
                # Заполняем результаты
                winner = result.winner
                for i, exp in enumerate(game_experiences):
                    player_idx = i % 2
                    if winner == f"player_{player_idx}":
                        exp.game_outcome = 1.0
                    elif winner == f"player_{1-player_idx}":
                        exp.game_outcome = -1.0
                    else:
                        exp.game_outcome = 0.0

                    exp.moves_to_end = len(game_experiences) - i

                break

            player1_updated = await players[0].update_root(action_id, state)
            player2_updated = await players[1].update_root(action_id, state)

        return {
            "game_id": game_id,
            "experiences": game_experiences,
            "total_moves": move_count,
            "winner": result.winner,
        }


if __name__ == "__main__":
    api = AsyncGameApi("http://localhost:8080")
    try:
        asyncio.run(simulate_game(api, max_moves=1500, mcts_iters=100, c=1.4, max_depth=20))
    finally:
        # Закрываем сессию (вариант: внутри simulate_game)
        asyncio.run(api.close())
