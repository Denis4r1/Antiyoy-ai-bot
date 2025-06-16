from pydantic import BaseModel, Field
from typing import List, Optional


class CreateRoomResponse(BaseModel):
    room_id: str = Field(
        ..., description="Уникальный ID созданной комнаты", example="room_abc123"
    )


class JoinRoomResponse(BaseModel):
    detail: str = Field(..., description="Сообщение об успешном подключении")
    room_id: str = Field(..., description="ID комнаты", example="room_abc123")
    players_count: int = Field(
        ..., description="Количество игроков в комнате", ge=1, le=8
    )


class RoomStats(BaseModel):
    total_rooms: int = Field(..., description="Общее количество комнат", ge=0)
    active_games: int = Field(..., description="Количество активных игр", ge=0)
    total_players: int = Field(..., description="Общее количество игроков", ge=0)
    lobby_connections: int = Field(
        ..., description="WebSocket подключений к лобби", ge=0
    )
    game_connections: int = Field(
        ..., description="WebSocket подключений к играм", ge=0
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="Статус сервиса", example="healthy")
    rooms: int = Field(..., description="Количество комнат", ge=0)
    games: int = Field(..., description="Количество активных игр", ge=0)
    players: int = Field(..., description="Количество игроков", ge=0)


class ErrorResponse(BaseModel):
    detail: str = Field(..., description="Описание ошибки")
