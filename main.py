# main.py

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from pydantic import BaseModel
from typing import List
import random

from src.field import Field, EntityType  # Импорт игровых классов


# Импорт моделей запросов
class Player(BaseModel):
    name: str
    coins: int = 100
    farms_bought: int = 0


class GameState(BaseModel):
    rows: int
    cols: int
    field: Field
    players: List[Player]
    current_turn: str

    class Config:
        arbitrary_types_allowed = True
        json_encoders = {Field: lambda f: f.to_dict(), EntityType: lambda e: e.value}


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


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


def initialize_game_state(width: int = 15, height: int = 15) -> GameState:
    players_names = ["player1", "player2"]
    players = [Player(name=name) for name in players_names]
    field = Field(height=height, width=width, owners=players_names)
    return GameState(
        rows=height, cols=width, field=field, players=players, current_turn="player1"
    )


game_state = initialize_game_state()


@app.get("/")
async def redirect_to_game():
    return RedirectResponse(url="/static/index.html")


@app.get("/state")
def get_state():
    return game_state


@app.post("/build")
def build(req: BuildRequest):
    try:
        game_state.field.build(req.x, req.y, req.player_name, req.building)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"detail": "Building placed successfully"}


@app.post("/spawn_unit")
def spawn_unit(req: SpawnUnitRequest):
    try:
        game_state.field.spawn_unit(req.x, req.y, req.player_name, req.level)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"detail": "Unit spawned successfully"}


@app.post("/move_unit")
def move_unit(req: MoveUnitRequest):
    try:
        game_state.field.move_unit(
            req.from_x, req.from_y, req.to_x, req.to_y, req.player_name
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"detail": "Unit moved successfully"}


@app.post("/get_moves")
def get_moves(req: GetMovesRequest):
    try:
        moves = game_state.field.get_moves(req.x, req.y, req.player_name)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"moves": moves}


@app.post("/end_turn")
def end_turn():
    current = game_state.current_turn
    winner = game_state.field.end_turn(current, game_state.players)
    if winner is not None:
        return {"detail": f"Turn ended. {winner} wins!", "current_turn": winner}
    next_turn = next(
        player.name for player in game_state.players if player.name != current
    )
    game_state.current_turn = next_turn
    return {
        "detail": "Turn ended successfully",
        "current_turn": game_state.current_turn,
    }
