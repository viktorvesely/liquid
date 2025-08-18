from dataclasses import dataclass


@dataclass
class GlobalConfig:
    make_delegation_uniform: bool = False

    display_loss_landscape: bool = False
    ll_every: int = 3
    ll_start: int = 3
    ll_distance: float = 0.3
    ll_resolution: int = 15


config = GlobalConfig()