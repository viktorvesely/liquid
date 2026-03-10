from abc import ABC, abstractmethod
from flax import linen as nn
import jax 


class Learner[
    TrainReturn
    ](ABC):

    @staticmethod
    @abstractmethod
    def get_model() -> nn.Module:
        ...

    @staticmethod
    @abstractmethod
    def forward(
        key: jax.Array,
        x: jax.Array,
        model: nn.Module,
        params: dict,
    ) -> tuple[jax.Array, TrainReturn]:
        ...

    @staticmethod
    @abstractmethod
    def auxillary_losses(
        key: jax.Array,
        train_return: TrainReturn,
        where: jax.Array,
    ) -> dict[str, jax.Array]:
        ...