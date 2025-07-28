from typing import Optional

import torch

from tensordict.tensordict import TensorDict
from torchrl.data import Bounded, Composite, Unbounded

from rl4co.envs.common.base import RL4COEnvBase
from rl4co.utils.ops import gather_by_index
from rl4co.utils.pylogger import get_pylogger

from .generator import KnapsackGenerator

log = get_pylogger(__name__)


class KnapsackEnv(RL4COEnvBase):
    """0-1 Knapsack Problem environment.

    At each step, the agent chooses an item to put into the knapsack. The episode
    ends when the agent selects action 0 ("finish"). The reward is the sum of the
    values of the selected items.

    Observations:
        - weight and value of each item
        - current used capacity of the knapsack
        - visited items

    Constraints:
        - each item can be selected at most once
        - the total weight of selected items cannot exceed the capacity

    Reward:
        - sum of the values of selected items
    """

    name = "knapsack"

    def __init__(
        self,
        generator: KnapsackGenerator = None,
        generator_params: dict = {},
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        if generator is None:
            generator = KnapsackGenerator(**generator_params)
        self.generator = generator
        self._make_spec(self.generator)

    def _step(self, td: TensorDict) -> TensorDict:
        current_item = td["action"][:, None]
        n_items = td["demand"].size(-1)

        selected_weight = gather_by_index(
            td["demand"], torch.clamp(current_item - 1, 0, n_items - 1), squeeze=False
        )
        selected_value = gather_by_index(
            td["values"], torch.clamp(current_item - 1, 0, n_items - 1), squeeze=False
        )

        used_capacity = td["used_capacity"] + selected_weight * (current_item != 0).float()
        total_value = td["total_value"] + selected_value * (current_item != 0).float()

        assert current_item.min() >= 0, f"current_item.min()={current_item.min()} < 0"
        assert current_item.max() < td["visited"].size(-1), f"current_item.max()={current_item.max()} >= visited.size(-1)"
        visited = td["visited"].scatter(-1, current_item, 1)
        done = (current_item.squeeze(-1) == 0) & (td["i"] > 0)

        reward = torch.zeros_like(done)

        td.update(
            {
                "current_node": current_item,
                "used_capacity": used_capacity,
                "total_value": total_value,
                "visited": visited,
                "i": td["i"] + 1,
                "reward": reward,
                "done": done,
            }
        )
        td.set("action_mask", self.get_action_mask(td))
        return td

    def _reset(
        self, td: Optional[TensorDict] = None, batch_size: Optional[list] = None
    ) -> TensorDict:
        device = td.device

        td_reset = TensorDict(
            {
                "weights": td["weights"],
                "demand": td["demand"],
                "values": td["values"],
                "locs": td["locs"],
                "capacity": td["capacity"],
                "current_node": torch.zeros(*batch_size, 1, dtype=torch.long, device=device),
                "used_capacity": torch.zeros((*batch_size, 1), device=device),
                "total_value": torch.zeros((*batch_size, 1), device=device),
                "visited": torch.zeros(
                    (*batch_size, td["demand"].shape[-1] + 1), dtype=torch.bool, device=device
                ),
                "i": torch.zeros((*batch_size, 1), dtype=torch.int64, device=device),
            },
            batch_size=batch_size,
        )
        td_reset.set("action_mask", self.get_action_mask(td_reset))
        return td_reset

    @staticmethod
    def get_action_mask(td: TensorDict) -> torch.Tensor:
        exceeds_cap = td["demand"] + td["used_capacity"] > td["capacity"] + 1e-5
        mask = td["visited"][..., 1:].to(exceeds_cap.dtype) | exceeds_cap
        action_mask = ~mask
        action_mask = torch.cat((torch.ones_like(action_mask[..., :1]), action_mask), -1)
        return action_mask

    def _get_reward(self, td: TensorDict, actions: torch.Tensor) -> torch.Tensor:
        values = torch.cat(
            (torch.zeros_like(td["values"][..., :1]), td["values"]), dim=-1
        )
        collected = values.gather(1, actions)
        return collected.sum(-1)

    @staticmethod
    def check_solution_validity(td: TensorDict, actions: torch.Tensor) -> None:
        sorted_actions = actions.data.sort(1)[0]
        assert (
            (sorted_actions[:, 1:] == 0)
            | (sorted_actions[:, 1:] > sorted_actions[:, :-1])
        ).all(), "Duplicates"

        weights = torch.cat((torch.zeros_like(td["demand"][..., :1]), td["demand"]), dim=-1)
        total_weight = weights.gather(1, actions).sum(-1)
        assert (total_weight <= td["capacity"].squeeze(-1) + 1e-5).all(), "Capacity exceeded"

    def _make_spec(self, generator: KnapsackGenerator):
        self.observation_spec = Composite(
            locs=Bounded(
                low=0.0,
                high=1.0,
                shape=(generator.num_items + 1, 2),
                dtype=torch.float32,
            ),
            weights=Bounded(
                low=generator.min_weight,
                high=generator.max_weight,
                shape=(generator.num_items,),
                dtype=torch.float32,
            ),
            demand=Bounded(
                low=generator.min_weight,
                high=generator.max_weight,
                shape=(generator.num_items,),
                dtype=torch.float32,
            ),
            values=Bounded(
                low=generator.min_value,
                high=generator.max_value,
                shape=(generator.num_items,),
                dtype=torch.float32,
            ),
            capacity=Unbounded(shape=(1,), dtype=torch.float32),
            current_item=Unbounded(shape=(1,), dtype=torch.int64),
            used_capacity=Unbounded(shape=(1,), dtype=torch.float32),
            total_value=Unbounded(shape=(1,), dtype=torch.float32),
            visited=Unbounded(shape=(generator.num_items + 1,), dtype=torch.bool),
            i=Unbounded(shape=(1,), dtype=torch.int64),
            action_mask=Unbounded(shape=(generator.num_items + 1,), dtype=torch.bool),
            shape=(),
        )
        self.action_spec = Bounded(
            shape=(1,),
            dtype=torch.int64,
            low=0,
            high=generator.num_items + 1,
        )
        self.reward_spec = Unbounded(shape=(1,))
        self.done_spec = Unbounded(shape=(1,), dtype=torch.bool)