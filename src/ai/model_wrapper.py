import torch
from .rl import state_to_tensor, PolicyValueNet


class StateEvaluator:
    def __init__(self, model_path="src/ai/models/model.pt"):
        self.net = PolicyValueNet()
        self.net.load_state_dict(torch.load(model_path, map_location="cpu"))

    def evaluate(self, state_dict, top_k=10, return_all=False):
        """
        Оценка заданного состояния и возврат предсказанного значения и действий.

        Args:
            state_dict (dict): Игровое состояние в виде словаря.
            top_k (int): Количество топовых действий для возврата (по умолчанию 10).
            return_all (bool): Если True, возвращает вероятности всех действий, иначе топ-k действий.

        Returns:
            tuple: (value, actions), где:
                - value (float): Предсказанное значение состояния.
                - actions: Список вероятностей всех действий, если return_all=True,
                           иначе список кортежей (action_id, probability) для топ-k действий.
        """
        self.net.eval()
        x = state_to_tensor(state_dict).unsqueeze(0)
        with torch.no_grad():
            log_p, v = self.net(x)
        probs = log_p.exp().squeeze(0).tolist()
        if return_all:
            return v.item(), probs
        else:
            top = sorted(enumerate(probs), key=lambda kv: -kv[1])[:top_k]
            return v.item(), [(aid, p) for aid, p in top]
