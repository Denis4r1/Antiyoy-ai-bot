import random
from typing import Dict, Any, Optional, List

from src.game.core.field import Field, EntityType


class GameController:
    def __init__(self, players: List[str]):
        self.field = Field(8, 8, players)
        self.players = players
        self._current_player_index = random.randint(0, len(players) - 1)

    def ensure_player_turn(self, player_name: str):
        current_player = self.players[self._current_player_index]
        if current_player != player_name:
            raise Exception("Not your turn")

    def build_action(self, x: int, y: int, player_name: str, building: str):
        self.ensure_player_turn(player_name)
        self.field.build(x, y, player_name, building)

    def spawn_unit_action(self, x: int, y: int, player_name: str, level: int):
        self.ensure_player_turn(player_name)
        self.field.spawn_unit(x, y, player_name, level)

    def move_unit_action(
        self, from_x: int, from_y: int, to_x: int, to_y: int, player_name: str
    ):
        self.ensure_player_turn(player_name)
        self.field.move_unit(from_x, from_y, to_x, to_y, player_name)

    def get_moves_action(self, x: int, y: int, player_name: str):
        self.ensure_player_turn(player_name)
        return self.field.get_moves(x, y, player_name)

    def end_turn(self, player_name: str):
        self.ensure_player_turn(player_name)
        current = self.players[self._current_player_index]

        winner = self.field.end_turn(current, self.players)
        if winner is not None:
            return winner

        num_players = len(self.players)
        for _ in range(num_players):
            self._current_player_index += 1
            self._current_player_index %= len(self.players)

            current_player = self.players[self._current_player_index]
            if self.field.territory_manager.has_territories(current_player):
                break

    def process_message(self, message: dict, player_name: str) -> dict:
        """
        Принимает dict, описывающий действие (например, {"type": "move_unit", ...}).
        Возвращает dict, который надо отослать клиентам (или None)
        возвращает None если не удалось распознать action
        """

        msg_type = message.get("type")

        if msg_type == "build_action":
            x = message["payload"]["x"]
            y = message["payload"]["y"]
            building = message["payload"]["building"]

            self.build_action(x, y, player_name, building)

            return {
                "type": "game_state_update",
                "field": self.field.to_dict(),
                "current_player": self.players[self._current_player_index],
                "players": self.players,
            }

        elif msg_type == "move_unit":
            from_x = message["payload"]["from_x"]
            from_y = message["payload"]["from_y"]
            to_x = message["payload"]["to_x"]
            to_y = message["payload"]["to_y"]

            self.move_unit_action(from_x, from_y, to_x, to_y, player_name)

            return {
                "type": "game_state_update",
                "field": self.field.to_dict(),
                "current_player": self.players[self._current_player_index],
                "players": self.players,
            }

        elif msg_type == "spawn_unit":
            x = message["payload"]["x"]
            y = message["payload"]["y"]
            level = message["payload"]["level"]

            self.spawn_unit_action(x, y, player_name, level)

            return {
                "type": "game_state_update",
                "field": self.field.to_dict(),
                "current_player": self.players[self._current_player_index],
                "players": self.players,
            }

        elif msg_type == "get_moves":
            x = message["payload"]["x"]
            y = message["payload"]["y"]

            moves = self.get_moves_action(x, y, player_name)

            return {"type": "available_moves", "moves": moves}

        elif msg_type == "end_turn":
            winner = self.end_turn(player_name)
            if winner:
                # Игра окончена, возвращаем сообщение о победителе
                return {"type": "game_over", "winner": winner}
            else:
                return {
                    "type": "game_state_update",
                    "field": self.field.to_dict(),
                    "current_player": self.players[self._current_player_index],
                    "players": self.players,
                }

        elif msg_type == "get_update":
            return {
                "type": "game_state_update",
                "field": self.field.to_dict(),
                "current_player": self.players[self._current_player_index],
                "players": self.players,
            }

        return None
