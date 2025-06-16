from __future__ import annotations
import argparse, asyncio, json, math, random, time, uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from .mcts import GameState, Action, AsyncGameApi

C, A_GLOBAL = 12, 291
BOARD_SIZE: Tuple[int, int] = (8, 8)
ENTITY_CHANNELS = {
    "base": 0,
    "farm": 1,
    "weakTower": 2,
    "strongTower": 3,
    "unit1": 4,
    "unit2": 5,
    "unit3": 6,
    "unit4": 7,
}
OWNER_CHANNELS = {"player_0": 8, "player_1": 9}
HAS_MOVED_CHANNEL, FUNDS_CHANNEL = 10, 11


def state_to_tensor(state: dict) -> torch.Tensor:
    t = torch.zeros((C, *BOARD_SIZE), dtype=torch.float32)
    cells = state["field_data"]["cells"]
    funds = {tr["owner"]: tr["funds"] for tr in state.get("territories_data", [])}
    for key, cell in cells.items():
        y, x = map(int, key.split(","))
        ent, own = cell["entity"], cell["owner"]
        if ent in ENTITY_CHANNELS:
            t[ENTITY_CHANNELS[ent], y, x] = 1.0
        if own in OWNER_CHANNELS:
            t[OWNER_CHANNELS[own], y, x] = 1.0
        if cell.get("has_moved"):
            t[HAS_MOVED_CHANNEL, y, x] = 1.0
        if own in funds:
            t[FUNDS_CHANNEL, y, x] = funds[own] / 100
    return t


class StateEncoder:
    def encode(self, s: GameState) -> torch.Tensor:
        return state_to_tensor(s.model_dump())


class Residual(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, stride=1, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, stride=1, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)

    def forward(self, x):
        y = torch.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return torch.relu(x + y)


class PolicyValueNet(nn.Module):
    def __init__(self):
        super().__init__()
        h, w = BOARD_SIZE

        self.head = nn.Sequential(
            nn.Conv2d(C, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            *(Residual(64) for _ in range(6)),
        )

        self.p_conv = nn.Sequential(
            nn.Conv2d(64, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
        )
        self.p_fc = nn.Linear(2 * h * w, A_GLOBAL)

        self.v_conv = nn.Sequential(
            nn.Conv2d(64, 1, 1, bias=False),
            nn.BatchNorm2d(1),
            nn.ReLU(),
        )
        self.v_fc1 = nn.Linear(h * w, 256)
        self.v_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        x = self.head(x)
        pi = self.p_fc(self.p_conv(x).flatten(1))
        v = torch.tanh(self.v_fc2(torch.relu(self.v_fc1(self.v_conv(x).flatten(1)))))
        return torch.log_softmax(pi, 1), v.squeeze(1)


class AZNode:
    def __init__(
        self,
        st: Optional[GameState],
        prior: float,
        par: Optional["AZNode"],
        act: Optional[Action],
    ):
        self.state, self.prior, self.par, self.act = st, prior, par, act
        self.children: Dict[int, "AZNode"] = {}
        self.v_sum = 0.0
        self.n = 0

    def q(self):
        return 0 if self.n == 0 else self.v_sum / self.n

    def u(self, c):
        return c * self.prior * math.sqrt(self.par.n) / (1 + self.n)

    def best(self, c):
        return max(self.children.values(), key=lambda n: n.q() + n.u(c))


class AlphaZeroMCTS:
    def __init__(
        self,
        api: AsyncGameApi,
        pid: str,
        net: PolicyValueNet,
        enc: StateEncoder,
        sims=400,
        cpuct=1.4,
        dev="cpu",
    ):
        self.api, self.pid, self.net, self.enc = api, pid, net.to(dev), enc
        self.S, self.c, self.dev = sims, cpuct, dev
        self.root: Optional[AZNode] = None

    async def initialize(self, s: GameState):
        self.root = await self._new_root(s)

    async def select(self):

        for _ in range(self.S):
            await self._sim(self.root)

        visits = {aid: n.n for aid, n in self.root.children.items()}
        tot = sum(visits.values())

        # --- τ (температура) — используем softmax(visits/τ) ---------------
        τ = 1.0 if self.S < 100 else 0.3  # при больших sim делаем острее
        probs = np.array([visits[k] ** (1 / τ) for k in visits])
        probs /= probs.sum()
        aid = np.random.choice(list(visits.keys()), p=probs)

        return self.root.children[aid].act, {
            "visit_distribution": {k: v / tot for k, v in visits.items()}
        }

    async def update(self, aid: int, s: GameState):
        self.root = self.root.children.get(aid) or await self._new_root(s)
        self.root.par = None
        self.root.state = s

    async def _new_root(self, s: GameState):
        pri, v = await self._eval(s)
        # --- Dirichlet noise ------------------------------------------------
        alpha, eps = 0.3, 0.25  # α=0.3 как в A0, eps=25 %
        noise = np.random.dirichlet([alpha] * len(pri))
        for i, k in enumerate(pri):
            pri[k] = (1 - eps) * pri[k] + eps * noise[i]
        root = AZNode(s, 1.0, None, None)
        for a in await self.api.get_actions(s):
            root.children[a.action_id] = AZNode(
                None, pri.get(a.action_id, 1e-3), root, a
            )
        root.n, root.v_sum = 1, v
        return root

    async def _sim(self, node):
        path = [node]
        while path[-1].children:
            path.append(path[-1].best(self.c))
        leaf = path[-1]
        if leaf.state is None and leaf.par:
            r = await self.api.apply_action(leaf.par.state, leaf.act)
            leaf.state, term, win = r.state, r.is_game_over, r.winner
        else:
            term = self.api.is_terminal(leaf.state)
            win = leaf.state.winner if term else None
        if term:
            val = 1.0 if win == self.pid else -1.0 if win else 0.0
        else:
            pri, val = await self._eval(leaf.state)
            if leaf.n == 0:
                for a in await self.api.get_actions(leaf.state):
                    leaf.children[a.action_id] = AZNode(
                        None, pri.get(a.action_id, 1e-3), leaf, a
                    )
        for n in reversed(path):
            n.n += 1
            n.v_sum += val
            val = -val

    async def _eval(self, s):
        x = self.enc.encode(s).unsqueeze(0).to(self.dev)
        with torch.no_grad():
            log_pi, v = self.net(x)
        p = log_pi.exp().squeeze(0).cpu().tolist()
        acts = await self.api.get_actions(s)
        pri = {a.action_id: p[i] if i < len(p) else 1e-3 for i, a in enumerate(acts)}
        z = sum(pri.values())
        pri = {k: v / z for k, v in pri.items()}
        return pri, float(v.item())


async def play_game(
    api_url: str,
    net: PolicyValueNet,
    states_dir: Path,
    train_dir: Path,
    sims: int,
    cpuct: float,
    max_mv: int,
    game_number: int,
):

    api = AsyncGameApi(api_url, timeout=300)
    enc = StateEncoder()
    st = await api.generate_state(num_players=2)

    pl = {pid: AlphaZeroMCTS(api, pid, net, enc, sims, cpuct) for pid in st.players}
    for p in pl.values():
        await p.initialize(st)

    sid, ts = uuid.uuid4().hex, time.strftime("%Y%m%d_%H%M%S")
    f_states = (states_dir / f"{game_number}_{sid}_{ts}.jsonl").open("w")
    f_train = (train_dir / f"{game_number}_{sid}_{ts}.json").open("w")

    f_states.write(json.dumps(st.model_dump(), ensure_ascii=False) + "\n")
    winner = None
    for mv in range(max_mv):
        pid = st.players[st.current_player_index]
        mcts = pl[pid]
        act, stats = await mcts.select()
        f_train.write(
            json.dumps(
                {
                    "move": mv,
                    "player": pid,
                    "state": st.model_dump(),
                    "mcts_stats": stats,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        r = await api.apply_action(st, act)
        if r.state.current_player_index == st.current_player_index:
            et = next(
                (
                    a
                    for a in await api.get_actions(r.state)
                    if a.action_type == "endTurn"
                ),
                None,
            )
            if et:
                r = await api.apply_action(r.state, et)
        for p in pl.values():
            await p.update(act.action_id, r.state)
        st = r.state
        f_states.write(json.dumps(st.model_dump(), ensure_ascii=False) + "\n")
        if r.is_game_over:
            winner = r.winner
            break

    f_train.write(json.dumps(st.model_dump(), ensure_ascii=False) + "\n")
    f_states.close()
    f_train.close()
    await api.close()
    return winner


class TrainDataset(Dataset):
    def __init__(self, d: Path):
        self.samples = []
        for fp in d.glob("*.json"):
            lines = [json.loads(l) for l in fp.read_text().splitlines() if l.strip()]
            if not lines:
                continue
            winner = lines[-1].get("winner")
            for e in lines[:-1]:
                s = e["state"]
                pi_t = torch.zeros(A_GLOBAL)
                for aid, prob in e["mcts_stats"]["visit_distribution"].items():
                    aid = int(aid)
                    0 <= aid < A_GLOBAL and pi_t.__setitem__(aid, prob)
                z = torch.tensor(
                    1.0 if winner == e["player"] else -1.0 if winner else 0.0
                )
                self.samples.append((state_to_tensor(s), pi_t, z))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        return self.samples[i]


def make_loaders(train_dir: Path, batch: int):
    ds = TrainDataset(train_dir)
    if not ds:
        raise RuntimeError("train-dataset empty")
    n_val = max(1, int(0.1 * len(ds)))
    tr, val = random_split(ds, [len(ds) - n_val, n_val])
    col = lambda b: tuple(torch.stack(x) for x in zip(*b))
    return DataLoader(tr, batch, True, collate_fn=col), DataLoader(
        val, batch, False, collate_fn=col
    )


def train(net, tr, val, ep, lr, dev, tb=None):
    net.to(dev)
    opt = torch.optim.Adam(net.parameters(), lr, weight_decay=1e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=2, factor=0.5)
    for e in range(1, ep + 1):
        net.train()
        tl = vl = n = 0
        for x, pi, z in tqdm(tr, f"train {e}/{ep}", leave=False):
            x, pi, z = x.to(dev), pi.to(dev), z.to(dev)
            lp, v = net(x)
            loss_pi = -(pi * lp).sum(1).mean()
            loss_v = F.mse_loss(v, z)
            loss = loss_pi + loss_v
            opt.zero_grad()
            loss.backward()
            opt.step()
            tl += loss.item() * x.size(0)
            n += x.size(0)
        net.eval()
        with torch.no_grad():
            for x, pi, z in val:
                lp, v = net(x.to(dev))
                vl += F.mse_loss(v.to(dev), z.to(dev)).item() * x.size(0)
        sch.step(vl / n)
        print(f"[{e}/{ep}] train {tl/n:.4f} | val {vl/n:.4f}")
        tb and tb.add_scalar("val/loss", vl / n, e)


def get_args():
    p = argparse.ArgumentParser("AlphaZero trainer")
    p.add_argument("--url", default="http://localhost:8080")
    p.add_argument("--states", default="logs/states")
    p.add_argument("--train", default="logs/train")
    p.add_argument("--out", default="models")
    p.add_argument("--games", type=int, default=10)
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--sim", type=int, default=30)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument(
        "--max-moves",
        type=int,
        default=200,
        help="сколько полуходов максимум играет одна партия",
    )
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--tb", action="store_true")
    return p.parse_args()


async def worker(sem: asyncio.Semaphore, url, net, sd, td, sims, cpuct, maxm):
    async with sem:
        return await play_game(url, net, sd, td, sims, cpuct, maxm)


async def main():
    args = get_args()
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    states_dir = Path(args.states)
    states_dir.mkdir(parents=True, exist_ok=True)
    train_dir = Path(args.train)
    train_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    net = PolicyValueNet().to(device)
    # encoder = StateEncoder()
    for p in net.parameters():
        p.requires_grad_(False)
    net.eval()

    # главный цикл: одна игра → один шаг обучения ------------------------------
    for g in range(1, args.games + 1):
        print(f"\n=== Game {g}/{args.games} ===")
        winner = await play_game(
            api_url=args.url,
            net=net,
            states_dir=states_dir,
            train_dir=train_dir,
            sims=args.sim,
            cpuct=1.4,
            max_mv=args.max_moves,
            game_number=g,
        )
        print("winner:", winner)

        # разморозка
        for p in net.parameters():
            p.requires_grad_(True)
        net.train()

        tr_loader, val_loader = make_loaders(train_dir, args.batch)

        train(
            net,  # модель
            tr_loader,  # тренировочный DataLoader
            val_loader,  # валидационный
            args.epochs,  # сколько эпох
            args.lr,  # learning-rate
            device,  # cpu / cuda
            SummaryWriter() if args.tb else None,
        )

        # заморозка
        net.eval()
        for p in net.parameters():
            p.requires_grad_(False)

    torch.save(net.state_dict(), out_dir / "model.pt")
    torch.jit.script(net.cpu()).save(out_dir / "model.ts")
    print("\nFINISHED.  Models saved to:", out_dir.resolve())


if __name__ == "__main__":
    asyncio.run(main())
