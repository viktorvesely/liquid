from typing import TYPE_CHECKING, Callable
from abc import ABC, abstractmethod
from flax import linen as nn
import jax

if TYPE_CHECKING:
    from structs import TrainParams

class Learner[
    TrainReturn
    ](ABC):

    @staticmethod
    @abstractmethod
    def get_model(
        train_params: "TrainParams",
        param_budget: int | None = None,
        dummy_input: jax.Array | None = None
    ) -> nn.Module:
        ...

    @staticmethod
    @abstractmethod
    def forward(
        key: jax.Array,
        x: jax.Array,
        model: nn.Module,
        params: dict
    ) -> tuple[jax.Array, TrainReturn]:
        ...

    @staticmethod
    @abstractmethod
    def auxillary_losses(
        key: jax.Array,
        model: nn.Module,
        params: dict,
        train_return: TrainReturn
    ) -> dict[str, jax.Array]:
        ...


def to_param(scaled_param: float) -> int:
    return max(int(round(scaled_param)), 1)

def find_best_matching_architecture_scalar(
    param_budget: int, 
    dummy_input: jax.Array,
    model_builder: Callable[[float], nn.Module],
    max_iter: int = 50
) -> tuple[float, int]:
    """
    Finds the alpha that matches the param_budget using binary search.
    """
    def get_param_count(alpha: float) -> int:
        model = model_builder(alpha)
        # eval_shape gets shapes without allocating memory, making this very fast
        variables = jax.eval_shape(lambda: model.init(jax.random.PRNGKey(0), dummy_input))
        return sum(x.size for x in jax.tree.leaves(variables.get('params', {})))

    # 1. Exponentially search for an upper bound
    low, high = 0.001, 1.0
    while get_param_count(high) < param_budget:
        low = high
        high *= 2.0

    # 2. Binary search for the closest match
    best_alpha = high
    best_diff = float('inf')
    actual_params = 0

    for _ in range(max_iter):
        mid = (low + high) / 2.0
        current_params = get_param_count(mid)
        diff = abs(current_params - param_budget)

        if diff < best_diff:
            best_diff = diff
            best_alpha = mid
            actual_params = current_params

        if current_params == param_budget:
            break
        elif current_params < param_budget:
            low = mid
        else:
            high = mid

    return best_alpha, actual_params