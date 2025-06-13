from __future__ import annotations
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch import nn


ENTITY_TO_CH: Dict[str, int] = {
    "empty": 0, "base": 1,
    "unit1": 2, "unit2": 3, "unit3": 4, "unit4": 5,
    "farm": 6, "weakTower": 7, "strongTower": 8,
}
N_ENT   = len(ENTITY_TO_CH)          # 9
IDX_OWN = N_ENT                      # 9
IDX_ENE = N_ENT + 1                  # 10
IDX_MOV = N_ENT + 2                  # 11
N_FEATS = N_ENT + 3                  # 12 (модель обучена именно на 12-канальных входах)

# ---------- STATE → FEATURE VECTOR -----------------------------------
def encode_cell(cell: dict, cur_player: dict) -> np.ndarray:
    """Вернёт 12-мерный вектор признаков одной клетки."""
    v = np.zeros(N_FEATS, dtype=np.float32)
    v[ENTITY_TO_CH[cell["entity"]]] = 1.0
    if cell["owner"] == cur_player:
        v[IDX_OWN] = 1.0
    elif cell["owner"] is not None:
        v[IDX_ENE] = 1.0
    if cell["has_moved"]:
        v[IDX_MOV] = 1.0
    return v

def to_tensor_pred(state: dict) -> torch.Tensor:
    """State → tensor[C,H,W] для подачи в сеть."""
    field = state["field_data"]
    H, W  = field["height"], field["width"]
    cur   = state["players"][state["current_player_index"]]

    X = np.zeros((N_FEATS, H, W), dtype=np.float32)
    for key, cell in field["cells"].items():
        i, j = map(int, key.split(","))
        X[:, i, j] = encode_cell(cell, cur)
    return torch.from_numpy(X)       # [12,H,W]

# ---------- AUTO-BUILD NETWORK ---------------------------------------
def build_from_sd(sd: dict, device="cpu") -> nn.Module:
    """Создаёт nn.Module с размерами, полностью совпадающими с весами."""
    # Conv-слои
    c1o, c1i = sd["conv1.weight"].shape[:2]
    c2o, c2i = sd["conv2.weight"].shape[:2]
    c3o, c3i = sd["conv3.weight"].shape[:2]
    # FC-слои
    fc1o, fc1i = sd["fc1.weight"].shape
    act_dim    = sd["fc2.weight"].shape[0]        # 291

    class Net(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(c1i, c1o, 3, padding=1)
            self.conv2 = nn.Conv2d(c2i, c2o, 3, padding=1)
            self.conv3 = nn.Conv2d(c3i, c3o, 3, padding=1)
            self.relu  = nn.ReLU(inplace=True)
            self.fc1   = nn.Linear(fc1i, fc1o)
            self.fc2   = nn.Linear(fc1o, act_dim)
        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = x.flatten(1)
            x = self.relu(self.fc1(x))
            return self.fc2(x)

    net = Net().to(device)
    net.load_state_dict(sd, strict=True)
    net.eval()
    return net

# ---------- PUBLIC API -----------------------------------------------
def predict_q(state: dict, weights: str | Path, device: str | torch.device = "cpu") -> np.ndarray:
    """Вернёт np.array (291,) с Q-значениями."""
    sd = torch.load(weights, map_location=device)      # OrderedDict
    model = build_from_sd(sd, device)
    with torch.no_grad():
        x = to_tensor_pred(state).to(device).unsqueeze(0)   # [1,12,H,W]
        q = model(x).squeeze(0).cpu().numpy().astype(np.float32)
    return q


#predict_q(state, "hex_dqn_masked.pth", device="cuda")