import math
import random
import logging
import time
import requests
from typing import Any, Dict, List, Optional, Tuple
import os
import json
import uuid
import concurrent.futures
from tqdm import tqdm
from pydantic import BaseModel, Field
import threading

import numpy as np
from typing import List, Tuple


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


class GameApi:
    """
    Синхронный клиент для взаимодействия с API игры
    """

    def __init__(self, base_url: str, timeout: int = 30):
        self.base = base_url.rstrip("/")
        self.session = requests.Session()
        self.session.timeout = timeout
        self.query_count = 0
        logging.basicConfig(
            filename="mcst_api_sync.log",
            format="%(asctime)s %(levelname)s %(message)s",
            level=logging.INFO,
        )

    def _inc(self):
        self.query_count += 1

    def generate_state(self, num_players: int = 2, randomize: bool = False) -> GameState:
        self._inc()
        url = f"{self.base}/generate_state"
        r = self.session.post(
            url,
            params={"num_players": num_players, "random": str(randomize).lower()},
        )
        r.raise_for_status()
        return GameState.model_validate(r.json())

    def get_actions(self, state: GameState) -> List[Action]:
        self._inc()
        url = f"{self.base}/get_actions"
        r = self.session.post(url, json=state.model_dump())
        r.raise_for_status()
        return [Action.model_validate(o) for o in r.json()]

    def apply_action(self, state: GameState, action: Action) -> ApplyActionResponse:
        try:
            self._inc()
            url = f"{self.base}/apply_action"
            req = ActionRequest(state=state, action_id=action.action_id)
            r = self.session.post(url, json=req.model_dump())
            r.raise_for_status()
            return ApplyActionResponse.model_validate(r.json())
        except Exception as e:
            print(req.model_dump_json())
            raise e

    @staticmethod
    def is_terminal(state: GameState) -> bool:
        return state.is_game_over

    @staticmethod
    def current_player(state: GameState) -> str:
        return state.players[state.current_player_index]

    @staticmethod
    def evaluate_state(state: GameState, player_id: str) -> float:
        """
        Эвристическая оценка состояния

        Пока для игрока player_id

        """
        cells = state.field_data.cells
        total = len(cells)
        # cur = state.players[state.current_player_index]
        # opp = state.players[1 - state.current_player_index]

        # Подсчет количества захваченных клеток для каждого игрока
        cnt_cur = sum(1 for c in cells.values() if c.owner == player_id)
        # cnt_opp = sum(1 for c in cells.values() if c.owner == opp)
        diff = cnt_cur / total
        return (cnt_cur) / max(1, total)


class MCTSNode:
    def __init__(self, state: GameState, parent: Optional["MCTSNode"] = None, action: Optional[Action] = None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions: Optional[List[Action]] = None
        self.lock = threading.Lock()

    def is_fully_expanded(self) -> bool:
        with self.lock:
            return self.untried_actions is not None and not self.untried_actions


    def best_uct_child(self, c: float) -> "MCTSNode":
        with self.lock:
            if not self.children:
                return None

            def uct_score(child):
                if child.visits == 0:
                    return float("inf")  # Приоритет для непосещённых узлов
                return (child.value / child.visits) + c * math.sqrt(math.log(self.visits) / child.visits)

            return max(self.children, key=uct_score)

    def find_child_by_action(self, action_id: int) -> Optional["MCTSNode"]:
        """Поиск дочернего узла по ID действия"""
        with self.lock:
            for child in self.children:
                if child.action and child.action.action_id == action_id:
                    return child
            return None


class MCTSPlayer:
    def __init__(
        self, player_id: str, game_api: GameApi, c: float = math.sqrt(2), max_depth: int = 200, workers: int = 1, iterations: int = 100
    ):
        self.player_id = player_id
        self.game = game_api
        self.c = c
        self.max_depth = max_depth
        self.root = None
        self.iterations = iterations
        self.workers = workers

    def initialize(self, state: GameState):
        """Инициализирует корневой узел дерева"""
        self.root = MCTSNode(state)
        self.root.untried_actions = self.game.get_actions(state)

    def _select(self, node: MCTSNode) -> MCTSNode:
        """Выбор узла для расширения"""
        while not self.game.is_terminal(node.state) and node.is_fully_expanded():
            child = node.best_uct_child(self.c)
            if not child:
                break
            node = child
        return node

    def _expand(self, node: MCTSNode) -> Optional[MCTSNode]:
        """Расширение узла новым дочерним узлом"""
        with node.lock:
            if node.untried_actions is None:
                node.untried_actions = self.game.get_actions(node.state)
            if not node.untried_actions:
                return None

            actions = node.untried_actions
            # 1) случайно выбираем индекс
            idx = random.randrange(len(actions))
            # 2) меняем выбранный элемент с последним
            actions[idx], actions[-1] = actions[-1], actions[idx]
            # 3) удаляем и возвращаем последний — pop()
            action = actions.pop()

            if action.action_type == "end_turn" and node.state.current_player_index == 1:
                pass
                # print('debug: end_turn')

            res = self.game.apply_action(node.state, action)
            child = MCTSNode(res.state, parent=node, action=action)
            child.untried_actions = self.game.get_actions(res.state)  # Сразу загружаем действия
            node.children.append(child)
            return child

    def _simulate(self, node: MCTSNode) -> float:
        """Симуляция из текущего узла"""
        state = node.state
        depth = 0

        while depth < self.max_depth and not self.game.is_terminal(state):
            actions = self.game.get_actions(state)
            if not actions:
                break

            action = random.choice(actions)
            res = self.game.apply_action(state, action)
            state = res.state
            depth += 1

        if self.game.is_terminal(state):
            return 1 if state.winner == self.player_id else -1

        # Пока что посчитаем тут
        cells = state.field_data.cells
        total = len(cells)
        my_cells = sum(1 for c in cells.values() if c.owner == self.player_id)
        opp_cells = sum(1 for c in cells.values() if c.owner and c.owner != self.player_id)

        # Увеличиваем вес в 10 раз, чтобы охотнее захватывало клетки
        return (my_cells - opp_cells) / max(1, total)

        # Эвристическая оценка для состояний без победы/поражения
        return GameApi.evaluate_state(state, self.player_id)

    def _backpropagate(self, node: MCTSNode, reward: float):
        """Обратное распространение результата симуляции"""
        while node:
            with node.lock:
                node.visits += 1
                node_player = self.game.current_player(node.state)

                # Более четкое определение, какой игрок получает положительную награду
                # if node_player == self.player_id:
                node.value += reward

            node = node.parent

    def _run_iteration(self):
        """Выполняет одну итерацию MCTS"""
        # 1. Выбор
        node = self._select(self.root)

        # 2. Расширение
        if not self.game.is_terminal(node.state):
            new_node = self._expand(node)
            if new_node:
                node = new_node

        # 3. Симуляция
        reward = self._simulate(node)

        # 4. Распространение
        self._backpropagate(node, reward)

    def search(self, iterations: int, show_progress: bool = False, desc: str = "MCTS") -> Action:
        """Выполняет поиск лучшего хода"""
        if not self.root:
            raise ValueError("MCTS дерево не инициализировано")
        
        iterations = self.iterations

        print(f"Searching for state {self.root.state.model_dump_json()}")

        # Проверяем, что у корня есть untried_actions
        if self.root.untried_actions is None:
            self.root.untried_actions = self.game.get_actions(self.root.state)

        if show_progress:
            pbar = tqdm(total=iterations, desc=desc, unit="iter")
            start_time = time.time()
            start_queries = self.game.query_count

        if self.workers > 1:
            iterations_per_worker = iterations

            # Так как боттлнек в игровом движке (который дергается через API), то можно запускать несколько потоков
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
                futures = []

                for i in range(iterations_per_worker):
                    futures.append(executor.submit(self._run_iteration))

                for i, future in enumerate(concurrent.futures.as_completed(futures)):
                    if show_progress:
                        elapsed = time.time() - start_time
                        queries_done = self.game.query_count - start_queries
                        qps = queries_done / elapsed if elapsed > 0 else 0
                        pbar.set_postfix({"iter": i + 1, "queries": queries_done, "QPS": f"{qps:.1f}"})
                        pbar.update(1)
        else:
            for i in range(iterations):
                self._run_iteration()

                if show_progress:
                    elapsed = time.time() - start_time
                    queries_done = self.game.query_count - start_queries
                    qps = queries_done / elapsed if elapsed > 0 else 0
                    pbar.set_postfix({"iter": i + 1, "queries": queries_done, "QPS": f"{qps:.1f}"})
                    pbar.update(1)

        if show_progress:
            pbar.close()

        print(f"Root has {len(self.root.children)} children after search")

        best_child = None
        best_visits = -1

        sort_by_value = sorted(self.root.children, key=lambda x: x.value)
        best_value_child = max(sort_by_value, key=lambda x: x.value)

        #self.root = best_value_child
        #self.root.parent = None 

        return best_value_child.action

        for child in self.root.children:
            if child.visits > best_visits:
                best_visits = child.visits
                best_child = child

        if not best_child:
            if not self.root.untried_actions:
                self.root.untried_actions = self.game.get_actions(self.root.state)
            if self.root.untried_actions:
                return random.choice(self.root.untried_actions)
            else:
                raise ValueError("No valid actions found!")

        return best_child.action

    def update_root(self, action_id: int, new_state: GameState) -> bool:
        """
        Обновляет корень дерева после выполнения хода.
        Возвращает True если успешно обновили корень, иначе False.
        """
        if not self.root or not self.root.children:
            self.initialize(new_state)
            return False

        child = self.root.find_child_by_action(action_id)

        if child:
            # Если нашли подходящий узел, делаем его новым корнем
            child.parent = None  # Отсоединяем от родителя
            self.root = child
            # Проверяем, есть ли у корня доступные действия
            if self.root.untried_actions is None:
                self.root.untried_actions = self.game.get_actions(self.root.state)
            return True
        else:
            self.initialize(new_state)
            return False


def simulate_game(
    api: GameApi,
    max_moves: int = 50,
    mcts_iters: int = 500,
    c: float = 1.4,
    max_depth: int = 200,
    workers: int = 1,
) -> None:
    """
    Симуляция игры двух MCTS-игроков
    """
    os.makedirs("logs", exist_ok=True)
    session_id = uuid.uuid4().hex
    date_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_path = os.path.join("logs", f"{session_id}_{date_str}.json")

    print(f"Log file: {log_path}")

    state = api.generate_state(num_players=2, randomize=False)
    #override_json_str = '{"players":["player_0","player_1"],"current_player_index":0,"field_data":{"height":12,"width":12,"cells":{"2,2":{"owner":"player_0","entity":"base","has_moved":false},"2,3":{"owner":"player_1","entity":"weakTower","has_moved":false},"2,4":{"owner":"player_0","entity":"weakTower","has_moved":false},"2,5":{"owner":"player_0","entity":"empty","has_moved":false},"2,7":{"owner":"player_0","entity":"empty","has_moved":false},"2,8":{"owner":"player_0","entity":"empty","has_moved":false},"2,9":{"owner":"player_0","entity":"empty","has_moved":false},"3,2":{"owner":"player_0","entity":"farm","has_moved":false},"3,3":{"owner":"player_1","entity":"base","has_moved":false},"3,4":{"owner":"player_0","entity":"empty","has_moved":false},"3,5":{"owner":"player_0","entity":"weakTower","has_moved":false},"3,7":{"owner":"player_0","entity":"empty","has_moved":false},"3,8":{"owner":"player_0","entity":"empty","has_moved":false},"3,9":{"owner":"player_0","entity":"farm","has_moved":false},"4,2":{"owner":"player_0","entity":"unit2","has_moved":true},"4,3":{"owner":"player_0","entity":"empty","has_moved":false},"4,4":{"owner":"player_0","entity":"empty","has_moved":false},"4,5":{"owner":"player_0","entity":"weakTower","has_moved":false},"4,6":{"owner":"player_0","entity":"empty","has_moved":false},"4,7":{"owner":"player_0","entity":"empty","has_moved":false},"4,8":{"owner":"player_0","entity":"farm","has_moved":false},"4,9":{"owner":"player_0","entity":"empty","has_moved":false},"4,10":{"owner":"player_0","entity":"farm","has_moved":false},"5,4":{"owner":"player_1","entity":"base","has_moved":false},"5,6":{"owner":"player_0","entity":"unit1","has_moved":false},"6,3":{"owner":"player_1","entity":"empty","has_moved":false},"6,4":{"owner":"player_1","entity":"weakTower","has_moved":false},"6,5":{"owner":"player_1","entity":"empty","has_moved":false},"6,6":{"owner":"player_1","entity":"unit1","has_moved":false},"7,3":{"owner":"player_1","entity":"empty","has_moved":false},"7,4":{"owner":"player_1","entity":"farm","has_moved":false},"7,5":{"owner":"player_1","entity":"weakTower","has_moved":false},"7,6":{"owner":"player_1","entity":"empty","has_moved":false},"7,7":{"owner":"player_1","entity":"empty","has_moved":false},"7,8":{"owner":"player_1","entity":"empty","has_moved":false},"7,9":{"owner":"player_1","entity":"unit3","has_moved":false},"8,3":{"owner":"player_1","entity":"farm","has_moved":false},"8,4":{"owner":"player_1","entity":"empty","has_moved":false},"8,5":{"owner":"player_1","entity":"unit1","has_moved":false},"8,6":{"owner":"player_1","entity":"empty","has_moved":false},"8,8":{"owner":"player_1","entity":"empty","has_moved":false},"8,9":{"owner":"player_1","entity":"unit1","has_moved":false},"9,8":{"owner":"player_0","entity":"empty","has_moved":false},"9,9":{"owner":"player_0","entity":"unit1","has_moved":false},"9,10":{"owner":"player_0","entity":"base","has_moved":false}},"territories":[{"player_0":["territory 22 tiles, 5 funds, +29 income","territory 3 tiles, 3 funds, +1 income"]},{"player_1":["territory 18 tiles, 13 funds, +6 income","territory 2 tiles, 12 funds, +1 income"]}]},"territories_data":[{"owner":"player_0","funds":5,"tiles":[[3,4],[4,3],[4,9],[3,7],[4,6],[2,2],[2,5],[2,8],[4,2],[4,5],[3,9],[5,6],[4,8],[2,4],[2,7],[3,2],[4,7],[3,5],[4,4],[4,10],[3,8],[2,9]]},{"owner":"player_0","funds":3,"tiles":[[9,10],[9,8],[9,9]]},{"owner":"player_1","funds":13,"tiles":[[8,8],[7,4],[8,4],[7,7],[6,5],[5,4],[6,4],[7,3],[7,9],[8,3],[7,6],[8,9],[8,6],[6,6],[7,5],[6,3],[8,5],[7,8]]},{"owner":"player_1","funds":12,"tiles":[[2,3],[3,3]]}],"is_game_over":false,"winner":null,"last_action":null}'
    #state = GameState.parse_raw(override_json_str)

    # Записываем начальное состояние в лог
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(state.model_dump(), ensure_ascii=False) + "\n")

    player1 = MCTSPlayer(player_id=state.players[0], game_api=api, c=c, max_depth=max_depth, workers=workers, iterations=mcts_iters)
    player2 = MCTSPlayer(player_id=state.players[1], game_api=api, c=c, max_depth=max_depth, workers=workers, iterations=mcts_iters)

    player1.initialize(state)
    player2.initialize(state)

    # История ходов
    history = []

    # Основной цикл игры
    for move_no in tqdm(range(1, max_moves + 1), desc="Game moves"):
        # Определяем активного игрока
        current_player_idx = state.current_player_index
        current_player_id = state.players[current_player_idx]

        # Выбираем активного игрока
        active_player = player1 if current_player_id == player1.player_id else player2

        print(f"\nMove {move_no} - Player {current_player_id}")

        # Поиск лучшего хода
        action = active_player.search(
            iterations=mcts_iters, show_progress=True, desc=f"Move {move_no} (Player {current_player_id})"
        )

        print(f"Selected action: {action.action_id} ({action.action_type})")

        response = api.apply_action(state, action)
        new_state = response.state

        history.append((move_no, current_player_id, action.action_id))

        print(f"Player1 размер дерева: {len(player1.root.children) if player1.root else 0}")
        print(f"Player2 размер дерева: {len(player2.root.children) if player2.root else 0}")

        player1_updated = player1.update_root(action.action_id, new_state)
        player2_updated = player2.update_root(action.action_id, new_state)

        # print(f"player1_updated: {player1_updated}, player2_updated: {player2_updated}")

        state = new_state

        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(state.model_dump(), ensure_ascii=False) + "\n")

        if response.is_game_over:
            break

    winner = state.winner
    print(f"{move_no} ходов, победил {winner}")


api = GameApi("http://localhost:8080")

simulate_game(
    api,
    max_moves=1500,  # Максимальное количество ходов в игре
    mcts_iters=200,  # Количество итераций MCTS при расчете хода
    c=1.4,
    max_depth=40,  # Отсечка глубины симуляции
    workers=32,
)
