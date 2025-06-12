# collect_bootstrap_data.py
import asyncio
import pickle
import json
from pathlib import Path
from datetime import datetime
from tqdm.asyncio import tqdm_asyncio
import logging
from mcts import BootstrapDataCollector, AsyncGameApi, MCTSExperience
from tqdm import tqdm
import numpy as np
from typing import Dict, List, Any
import time


class JSONGameData:
    """Класс для сериализации игровых данных в JSON"""

    @staticmethod
    def experience_to_dict(exp: MCTSExperience) -> Dict:
        """Конвертирует MCTSExperience в словарь"""
        # Если state - это Pydantic модель, конвертируем
        if hasattr(exp.state, "model_dump"):
            state_dict = exp.state.model_dump()
        else:
            state_dict = exp.state

        return {
            "state": state_dict,
            "visit_distribution": exp.visit_distribution,
            "action_values": exp.action_values,
            "total_visits": exp.total_visits,
            "selected_action_id": exp.selected_action_id,
            "game_progress": exp.game_progress,
            "game_outcome": exp.game_outcome,
            "moves_to_end": exp.moves_to_end,
        }

    @staticmethod
    def save_game(game_data: Dict, filepath: Path):
        """Сохраняет данные игры в JSON"""
        json_data = {
            "game_id": game_data["game_id"],
            "total_moves": game_data["total_moves"],
            "winner": game_data["winner"],
            "experiences": [JSONGameData.experience_to_dict(exp) for exp in game_data["experiences"]],
        }

        # Сохраняем с ensure_ascii=False для читаемости
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)


async def collect_single_game(collector: BootstrapDataCollector, game_id: int, progress_bar: tqdm_asyncio) -> Dict:
    """Собирает данные одной игры"""
    try:
        start_time = time.time()

        # Собираем данные игры
        game_data = await collector.collect_game(game_id)

        # Сохраняем в JSON
        output_path = Path(f"bootstrap_data/games/game_{game_id:04d}.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)

        JSONGameData.save_game(game_data, output_path)

        duration = time.time() - start_time

        # Обновляем прогресс
        progress_bar.update(1)
        progress_bar.set_postfix(
            {"moves": game_data["total_moves"], "winner": game_data["winner"], "time": f"{duration:.1f}s"}
        )

        return {
            "game_id": game_id,
            "duration": duration,
            "moves": game_data["total_moves"],
            "winner": game_data["winner"],
            "experiences": len(game_data["experiences"]),
        }

    except Exception as e:
        logging.error(f"Game {game_id} failed: {e}", exc_info=True)
        print(e)
        return {"game_id": game_id, "error": str(e)}


async def main():
    # Настройки
    NUM_GAMES = 100
    CONCURRENT_GAMES = 1  # Параллельные игры
    MCTS_ITERATIONS = 100  # Итераций MCTS на ход

    # Создаем директории
    Path("bootstrap_data/games").mkdir(parents=True, exist_ok=True)
    Path("bootstrap_data/logs").mkdir(parents=True, exist_ok=True)

    # Настройка логирования
    logging.basicConfig(
        filename=f"bootstrap_data/logs/collection_{datetime.now():%Y%m%d_%H%M%S}.log",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Инициализация
    print("🚀 Инициализация сборщика данных...")

    # API с пулом соединений
    api = AsyncGameApi(
        base_url="http://localhost:8080", max_concurrent=CONCURRENT_GAMES * 2  # Запас для параллельных запросов
    )

    # Параметры MCTS
    mcts_params = {
        "iterations": MCTS_ITERATIONS,
        "c": 1.4,  # exploration constant
        "max_depth": 25,
        "temperature_schedule": {
            0: 1.0,  # Начало игры - больше exploration
            20: 0.5,  # Середина
            60: 0.1,  # Конец - exploitation
        },
    }

    collector = BootstrapDataCollector(api, mcts_params)

    print(f"Планируется собрать {NUM_GAMES} игр")
    print(f"Параллельно: {CONCURRENT_GAMES} игр")
    print(f"MCTS итераций на ход: {MCTS_ITERATIONS}")

    # Прогресс бар
    progress_bar = tqdm_asyncio(total=NUM_GAMES, desc="Сбор игр", unit="games")

    # Запускаем сбор батчами
    results = []
    for batch_start in range(0, NUM_GAMES, CONCURRENT_GAMES):
        batch_end = min(batch_start + CONCURRENT_GAMES, NUM_GAMES)
        batch_tasks = [
            collect_single_game(collector, game_id, progress_bar) for game_id in range(batch_start, batch_end)
        ]

        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)

    progress_bar.close()

    # Сохраняем метаданные
    await save_metadata(results, mcts_params)

    # Закрываем API
    await api.close()

    # Выводим статистику
    print_statistics(results)


async def save_metadata(results: List[Dict], mcts_params: Dict):
    """Сохраняет метаданные о собранных данных"""

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    metadata = {
        "collection_date": datetime.now().isoformat(),
        "total_games": len(results),
        "successful_games": len(successful),
        "failed_games": len(failed),
        "mcts_params": mcts_params,
        "statistics": {
            "total_experiences": sum(r.get("experiences", 0) for r in successful),
            "avg_game_length": sum(r.get("moves", 0) for r in successful) / len(successful) if successful else 0,
            "avg_collection_time": (
                sum(r.get("duration", 0) for r in successful) / len(successful) if successful else 0
            ),
            "win_distribution": {},
        },
        "failed_games": [{"game_id": r["game_id"], "error": r["error"]} for r in failed],
    }

    # Подсчет побед
    for r in successful:
        winner = r.get("winner", "unknown")
        metadata["statistics"]["win_distribution"][winner] = (
            metadata["statistics"]["win_distribution"].get(winner, 0) + 1
        )

    with open("bootstrap_data/metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def print_statistics(results: List[Dict]):
    """Выводит финальную статистику"""
    successful = [r for r in results if "error" not in r]

    print("\n" + "=" * 50)
    print("СТАТИСТИКА СБОРА ДАННЫХ")
    print("=" * 50)

    print(f"Успешно собрано игр: {len(successful)}/{len(results)}")

    if successful:
        total_moves = sum(r["moves"] for r in successful)
        total_time = sum(r["duration"] for r in successful)
        total_exp = sum(r["experiences"] for r in successful)

        print(f"Всего ходов: {total_moves}")
        print(f"Всего experiences: {total_exp}")
        print(f"Общее время: {total_time/60:.1f} минут")
        print(f"Среднее время на игру: {total_time/len(successful):.1f} сек")
        print(f"Средняя длина игры: {total_moves/len(successful):.1f} ходов")

        # Распределение побед
        winners = {}
        for r in successful:
            winner = r.get("winner", "unknown")
            winners[winner] = winners.get(winner, 0) + 1

        print("\n🏆 Распределение побед:")
        for winner, count in winners.items():
            print(f"   {winner}: {count} ({count/len(successful)*100:.1f}%)")


if __name__ == "__main__":
    asyncio.run(main())
