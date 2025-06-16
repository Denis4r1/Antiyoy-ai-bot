import json
import os
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pathlib import Path
from .models import GameState, Action, ActionRequest, ApplyActionResponse
from .utils import reconstruct_game, _cached_get_actions, _cached_apply_action
from src.game.gamecontroller import GameController
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="web/static"), name="static")

LOG_DIR = "logs"


@app.get(
    "/state_editor",
    response_class=HTMLResponse,
    tags=["UI"],
    summary="Редактор состояний игры",
    description="HTML страница редактора состояний игры",
)
async def get_state_editor():
    with open("web/templates/state_editor.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get(
    "/log_player",
    response_class=HTMLResponse,
    tags=["UI"],
    summary="Проигрыватель логов",
    description="HTML страница проигрывателя логов",
)
async def get_log_player():
    with open("web/templates/log_player.html", "r", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.get(
    "/log_player/demo",
    response_class=HTMLResponse,
    tags=["UI"],
    summary="Демо проигрывателя логов",
    description="HTML страница проигрывателя логов с предзагруженным демо логом",
)
async def log_player_demo():
    html = Path("web/templates/log_player.html").read_text(encoding="utf-8")

    # a little hack to inject a demo json file into the page
    injection = """
    <script>
      window.addEventListener('load', async () => {
        // 1) fetch the demo log
        const res  = await fetch('/static/misc/demo_log.jsonl');
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


@app.post(
    "/generate_state",
    response_model=GameState,
    tags=["Game State"],
    summary="Создание нового состояния игры",
    description="Генерирует state с указанным количеством игроков. Может создать случайное состояние или загрузить базовую карту.",
)
def generate_state(
    num_players: int = Query(
        2, ge=2, le=4, description="Количество игроков (от 1 до 4), проверено для 2"
    ),
    random: bool = Query(
        False,
        description="Создать случайную (True) или загрузить базовую карту (False)",
    ),
):
    """Создает новую игру и возвращает ее состояние"""
    try:
        if not random:
            with open("web/static/maps/map_basic_small.json", "r") as f:
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
                {
                    "owner": territory.owner,
                    "funds": territory.funds,
                    "tiles": list(territory.tiles),
                }
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
        raise HTTPException(
            status_code=500, detail=f"Failed to generate game state: {str(e)}"
        )


@app.post(
    "/get_actions",
    response_model=List[Action],
    tags=["Game State"],
    summary="Получить доступные действия",
    description="Возвращает список всех разрешенных действий для заданного state",
)
def get_actions_endpoint(state: GameState):
    state_json = state.model_dump_json()
    return _cached_get_actions(state_json)


@app.post(
    "/apply_action",
    response_model=ApplyActionResponse,
    tags=["Game State"],
    summary="Применить действие",
    description="Применяет указанное действие к state и возвращает новый state",
)
def apply_action_endpoint(request: ActionRequest):
    state_json = request.state.model_dump_json()
    params_json = json.dumps(request.params or {}, sort_keys=True)
    return _cached_apply_action(
        state_json,
        request.action_id,
        request.action_type,
        params_json,
        request.description,
    )


@app.get(
    "/logs",
    tags=["Logs"],
    summary="Список логов",
    description="Возвращает список всех доступных файлов логов в директории logs/ , полезно для отслеживания игр",
)
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


@app.get(
    "/logs/{filename}",
    tags=["Logs"],
    summary="Скачать лог",
    description="Возвращает содержимое указанного файла лога, полезно для отслеживания игры в проигрывателе логов в реальном времени",
)
async def get_log(filename: str):
    """
    Отдаёт содержимое конкретного лога по имени файла.
    """
    if ".." in filename or filename.startswith("/"):
        raise HTTPException(status_code=400, detail="Invalid filename")
    file_path = os.path.join(LOG_DIR, filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail="Log not found")
    return FileResponse(
        path=file_path, media_type="application/json", filename=filename
    )
