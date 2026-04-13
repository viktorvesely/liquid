from dataclasses import dataclass
from functools import partial
import math
from typing import Literal

from flax import struct
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pathlib import Path

@struct.dataclass
class LEInfo:
    delegation: jax.Array 
    power: jax.Array

@dataclass(frozen=True)
class LEsolver:

    load_distribution_lambda: float = 0.0
    load_distribution_temperature: float = 0.5
    specialization_lambda: float = 0.0
    epsilon: float = 1e-3
    long_delegations_penalty: float = 0.95
    solver: Literal["sink_one", "sink_many"] = "sink_many"

    def load_distribution_loss(
        self,
        le_info: LEInfo        
    ):
        n_models = le_info.power.shape[-1]

        # Chair is the model how has the most power for a given region
        # Calculated as power.argmax(dim=1) 
        soft_chair = jax.nn.softmax(le_info.power / self.load_distribution_temperature, axis=1)
        
        # How much were they active across batch
        soft_chair_dist = soft_chair.mean(axis=0)
        
        # Make it a valid distribution
        soft_chair_dist = soft_chair_dist / soft_chair_dist.sum()

        # Make it like uniform (more stable than maximizing entropy)
        non_uniformity = n_models * jnp.sum(soft_chair_dist ** 2) - 1
        
        # Range before: [0, n_models - 1], now [0, 1]
        non_uniformity = non_uniformity / (n_models - 1)

        # Maximimize the uniformity to use all the modes across the batch
        return self.load_distribution_lambda * non_uniformity 
    

    def specialization_loss(
        self,
        le_info: LEInfo
    ):
        n_models = le_info.power.shape[-1]

        # Make the power into a distribution
        power_dist = le_info.power / le_info.power.sum(axis=-1, keepdims=True)

        # Calculate non-uniformity for each batch element
        non_uniformities = n_models * jnp.sum(power_dist ** 2, axis=-1) - 1

        # Range before: [0, n_models - 1], now [0, 1]
        non_uniformities = non_uniformities / (n_models - 1)

        # Also known as gini impurity
        uniformities = 1 - non_uniformities

        # Minimize the uniformity to push for specialization
        return self.specialization_lambda * jnp.mean(uniformities)

    def solve_power(
        self,
        delegation: jax.Array # (batch, from_voter, to_voter)
    ) -> LEInfo:
        
        if self.solver == "sink_one":
            power = self._solve_one_sink(delegation, self.epsilon)
        elif self.solver == "sink_many":
            power = self._solve_many_sinks(delegation, self.epsilon, self.long_delegations_penalty)
        else:
            raise ValueError(f"Invalid solver = {self.solver}")

        return LEInfo(
            delegation=delegation,
            power=power
        )
        
   
    @staticmethod
    @partial(jax.vmap, in_axes=(0, None, None))
    def _solve_many_sinks(
        delegation: jax.Array, 
        epsilon: float,
        penalty: float
    ) -> jax.Array:
        
        delegation = delegation.T
        n_models = delegation.shape[-1]
        n_ext = n_models * 2
        I = jnp.eye(n_ext) 

        D_ext = jnp.zeros((n_ext, n_ext))
        
        # 1. Real models keep their off-diagonal flow, SCALED BY PENALTY
        off_diag_mask = 1 - jnp.eye(n_models)
        D_no_diag = delegation * off_diag_mask * penalty  # <-- APPLY HERE
        D_ext = D_ext.at[:n_models, :n_models].set(D_no_diag)
        
        # 2. Sinks absorb the self-delegation plus epsilon
        self_connections = jnp.diagonal(delegation) + epsilon
        D_ext = D_ext.at[n_models:, :n_models].set(jnp.diag(self_connections))
        
        # 3. The "shadow models" delegate only to themselves
        D_ext = D_ext.at[n_models:, n_models:].set(jnp.eye(n_models))
    
        # 4. ONLY the real models begin with one vote 
        power_start = jnp.zeros((n_ext,))
        power_start = power_start.at[:n_models].set(1.0)
        
        D_no_diag_ext = D_ext * (1 - I)

        power_ext = jnp.linalg.solve(I - D_no_diag_ext, power_start)

        power = power_ext[n_models:]
        power = (power / power.sum()) * n_models
        
        return power
    
    @staticmethod
    @partial(jax.vmap, in_axes=(0, None))
    def _solve_one_sink(
        delegation: jax.Array, # (from_voter, to_voter)
        epsilon: float
    ) -> jax.Array:
        
        # Now (to_voter, from_voter)
        delegation = delegation.T
        n_models = delegation.shape[-1]
        I = jnp.eye(n_models + 1) 

        # Extend the delegation matrix to include the additional "sink model"
        D_ext = jnp.zeros((n_models + 1, n_models + 1))
        
        # All models will sink epsilon power to "sink model"
        D_ext = D_ext.at[n_models, :n_models].set(epsilon)
        
        # The remaining voting vector needs to be readjusted to sum to 1
        D_ext = D_ext.at[:n_models, :n_models].set((1 - epsilon) * delegation)

        # The "sink model" delegates only to itself (closing the loop)
        D_ext = D_ext.at[n_models, n_models].set(1.0)
    
        # Everyone begins with one vote
        power_start = jnp.ones((n_models + 1,))
        
        # Solve flow of power, for which we need to remove the retention of power (the diagonal)
        D_no_diag = D_ext * (1 - I)
        
        # Keep the self delegation (diagonal)
        self_delegation = jnp.diagonal(D_ext)

        # Solve the flow
        power_ext = jnp.linalg.solve(I - D_no_diag, power_start)

        # Consider self delegation
        power_ext = power_ext * self_delegation

        # Remove the "sink model"
        power = power_ext[:-1]

        # Return the lost power from the sink uniformly back
        power = power + (power_ext[-1] / n_models)

        # Remove the artificial power brought by the "sink model"
        power = power - (1 / n_models)
        
        return power



if __name__ == "__main__":

    import time

    
    solver = LEsolver(
        solver="sink_many"
    )

    @jax.jit
    def test_solve(delegations: jax.Array):
        info = solver.solve_power(delegations)
        return info.power
    

    delegations = jnp.stack([
        jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ]),
        jnp.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0]
        ]),
        jnp.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0]
        ]),
        jnp.array([
            [0.5, 0.5, 0.0],
            [0.5, 0.0, 0.5],
            [0.0, 0.0, 1.0]
        ])
    ], axis= 0)

    assert jnp.allclose(delegations.sum(axis=-1), 1.0)

    power = test_solve(delegations)

    assert jnp.allclose(power.sum(axis=-1), 3.0, atol=1e-3), jnp.round(power.sum(axis=-1), 3)

    print(jnp.round(power, 2))


    key = jax.random.key(42)
    rand_delegations = jax.random.uniform(key, (2048, 100, 100))
    rand_delegations = rand_delegations / rand_delegations.sum(axis=-1, keepdims=True) 

    print("compiling")
    res = test_solve(rand_delegations)
    jax.block_until_ready(res)

    print("running")

    s = time.perf_counter()
    res = test_solve(rand_delegations)
    jax.block_until_ready(res)
    e = time.perf_counter()
    print(e - s)

    print((~jnp.isfinite(res)).sum())