# collect_bootstrap_data.py
import asyncio
import pickle
import json
import uuid
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
            "experiences": [
                JSONGameData.experience_to_dict(exp) for exp in game_data["experiences"]
            ],
        }

        # Сохраняем с ensure_ascii=False для читаемости
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)


async def collect_single_game(
    collector: BootstrapDataCollector,
    game_id: int,
    progress_bar: tqdm_asyncio,
    session_uuid: str,
) -> Dict:
    """Собирает данные одной игры"""
    try:
        start_time = time.time()

        # Собираем данные игры
        game_data = await collector.collect_game(game_id)

        # Сохраняем в JSON в папку с UUID сессии
        output_path = Path(
            f"bootstrap_data/games_{session_uuid}/game_{game_id:04d}.json"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        JSONGameData.save_game(game_data, output_path)

        duration = time.time() - start_time

        # Обновляем прогресс
        progress_bar.update(1)
        progress_bar.set_postfix(
            {
                "moves": game_data["total_moves"],
                "winner": game_data["winner"],
                "time": f"{duration:.1f}s",
            }
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
    NUM_GAMES = 8
    CONCURRENT_GAMES = 8  # Параллельные игры
    MCTS_ITERATIONS = 10  # Итераций MCTS на ход

    # Генерируем UUID для этой сессии сбора данных
    session_uuid = str(uuid.uuid4())[:8]  # Используем короткий UUID для читаемости
    session_folder = f"bootstrap_data/games_{session_uuid}"

    print(f"📁 Сессия сбора данных: {session_uuid}")
    print(f"📂 Папка для сохранения: {session_folder}")

    # Создаем директории
    Path(session_folder).mkdir(parents=True, exist_ok=True)
    Path("bootstrap_data/logs").mkdir(parents=True, exist_ok=True)

    # Настройка логирования
    log_filename = f"bootstrap_data/logs/collection_{session_uuid}_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    # Инициализация
    print("🚀 Инициализация сборщика данных...")

    # API с пулом соединений
    api = AsyncGameApi(
        base_url="http://localhost:8080",
        max_concurrent=CONCURRENT_GAMES * 2,  # Запас для параллельных запросов
    )

    # Параметры MCTS
    mcts_params = {
        "iterations": MCTS_ITERATIONS,
        "c": 1.4,  # exploration constant
        "max_depth": 30,
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
            collect_single_game(collector, game_id, progress_bar, session_uuid)
            for game_id in range(batch_start, batch_end)
        ]

        batch_results = await asyncio.gather(*batch_tasks)
        results.extend(batch_results)

    progress_bar.close()

    # Сохраняем метаданные в папку сессии
    await save_metadata(results, mcts_params, session_uuid, session_folder)

    # Закрываем API
    await api.close()

    # Выводим статистику
    print_statistics(results, session_uuid)


async def save_metadata(
    results: List[Dict], mcts_params: Dict, session_uuid: str, session_folder: str
):
    """Сохраняет метаданные о собранных данных"""

    successful = [r for r in results if "error" not in r]
    failed = [r for r in results if "error" in r]

    metadata = {
        "session_uuid": session_uuid,
        "collection_date": datetime.now().isoformat(),
        "total_games": len(results),
        "successful_games": len(successful),
        "failed_games": len(failed),
        "mcts_params": mcts_params,
        "statistics": {
            "total_experiences": sum(r.get("experiences", 0) for r in successful),
            "avg_game_length": (
                sum(r.get("moves", 0) for r in successful) / len(successful)
                if successful
                else 0
            ),
            "avg_collection_time": (
                sum(r.get("duration", 0) for r in successful) / len(successful)
                if successful
                else 0
            ),
            "win_distribution": {},
        },
        "failed_games": [
            {"game_id": r["game_id"], "error": r["error"]} for r in failed
        ],
    }

    # Подсчет побед
    for r in successful:
        winner = r.get("winner", "unknown")
        metadata["statistics"]["win_distribution"][winner] = (
            metadata["statistics"]["win_distribution"].get(winner, 0) + 1
        )

    # Сохраняем метаданные в папку сессии
    metadata_path = Path(session_folder) / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # Также сохраняем общий индекс всех сессий
    await update_sessions_index(session_uuid, metadata)


async def update_sessions_index(session_uuid: str, session_metadata: Dict):
    """Обновляет индекс всех сессий сбора данных"""
    index_path = Path("bootstrap_data/sessions_index.json")

    # Загружаем существующий индекс или создаем новый
    if index_path.exists():
        with open(index_path, "r", encoding="utf-8") as f:
            index = json.load(f)
    else:
        index = {"sessions": []}

    # Добавляем информацию о текущей сессии
    session_info = {
        "uuid": session_uuid,
        "date": session_metadata["collection_date"],
        "total_games": session_metadata["total_games"],
        "successful_games": session_metadata["successful_games"],
        "total_experiences": session_metadata["statistics"]["total_experiences"],
        "folder": f"games_{session_uuid}",
    }

    index["sessions"].append(session_info)
    index["last_updated"] = datetime.now().isoformat()

    # Сохраняем обновленный индекс
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, indent=2, ensure_ascii=False)


def print_statistics(results: List[Dict], session_uuid: str):
    """Выводит финальную статистику"""
    successful = [r for r in results if "error" not in r]

    print("\n" + "=" * 50)
    print("СТАТИСТИКА СБОРА ДАННЫХ")
    print("=" * 50)
    print(f"Сессия: {session_uuid}")

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

        print(f"\n📁 Данные сохранены в: bootstrap_data/games_{session_uuid}/")


if __name__ == "__main__":
    asyncio.run(main())
