from typing import Callable

import torch

from tensordict.tensordict import TensorDict
from torch.distributions import Uniform

from rl4co.envs.common.utils import Generator, get_sampler
from rl4co.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

# Default capacities for common instance sizes
CAPACITIES = {50: 12.5, 100: 25.0}


class KnapsackGenerator(Generator):
    """Data generator for the 0-1 Knapsack problem.

    Args:
        num_items: number of items in the knapsack instance
        min_weight: minimum value for item weights
        max_weight: maximum value for item weights
        min_value: minimum value for item values
        max_value: maximum value for item values
        weight_distribution: distribution for the item weights
        value_distribution: distribution for the item values
        capacity: knapsack capacity
    """

    def __init__(
        self,
        num_items: int = 50,
        min_weight: float = 0.0,
        max_weight: float = 1.0,
        min_value: float = 0.0,
        max_value: float = 1.0,
        weight_distribution: int | float | str | type | Callable = Uniform,
        value_distribution: int | float | str | type | Callable = Uniform,
        capacity: float = None,
        **kwargs,
    ) -> None:
        self.num_items = num_items
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.min_value = min_value
        self.max_value = max_value

        if kwargs.get("weight_sampler", None) is not None:
            self.weight_sampler = kwargs["weight_sampler"]
        else:
            self.weight_sampler = get_sampler(
                "weight", weight_distribution, min_weight, max_weight, **kwargs
            )

        if kwargs.get("value_sampler", None) is not None:
            self.value_sampler = kwargs["value_sampler"]
        else:
            self.value_sampler = get_sampler(
                "value", value_distribution, min_value, max_value, **kwargs
            )

        if capacity is None:
            capacity = CAPACITIES.get(num_items, None)
        if capacity is None:
            capacity = num_items / 4.0
            log.warning(
                f"The capacity for {num_items} items is not defined. Using {capacity}."
            )
        self.capacity = capacity

    def _generate(self, batch_size) -> TensorDict:
        weights = self.weight_sampler.sample((*batch_size, self.num_items))
        values = self.value_sampler.sample((*batch_size, self.num_items))

        items = torch.stack((weights, values), dim=-1)
        depot = torch.zeros(*batch_size, 1, 2, device=items.device, dtype=items.dtype)
        locs = torch.cat((depot, items), dim=-2)

        capacity = torch.full((*batch_size, 1), self.capacity)

        return TensorDict(
            {
                "weights": weights,
                "demand": weights,
                "values": values,
                "locs": locs,
                "vehicle_capacity": capacity,
            },
            batch_size=batch_size,
        )