from typing import Literal

from flax import linen as nn
import jax 
import jax.numpy as jnp

from learner_base import Learner, find_best_matching_architecture_scalar, to_param
from atomic_networks import get_cnn_layers, get_layers, forward
from task_base import Task
from liquid_solver import LEsolver, LEInfo
from math_utils import mix_weighted_logits, mix_weighted_mean
from structs import TrainParams

class ExpertBodyMlp(nn.Module):
    body: tuple[int, ...]
    
    def setup(self):
        self.body_layers = get_layers(self.body)

    def __call__(self, x: jax.Array):
        h_body = forward(x, last_linear=False, layers=self.body_layers)
        return h_body
    
class ExpertBodyCNN(nn.Module):
    body: tuple[int, ...]
    
    def setup(self):
        self.body_layers = get_cnn_layers(self.body, kernel_size=3, stride=2)

    def __call__(self, x: jax.Array):
        h_body = forward(x, last_linear=False, layers=self.body_layers)
        batch_shape = h_body.shape[:-3]
        h_body = h_body.reshape(batch_shape + (-1,))
        return h_body

class ExpertHead(nn.Module):
    out: tuple[int, ...]
    delegation: tuple[int, ...]
    
    def setup(self):
        self.out_layers = get_layers(self.out)
        self.delegation_layers = get_layers(self.delegation)

    def __call__(self, h_body: jax.Array):
        y = forward(h_body, last_linear=True, layers=self.out_layers)
        d = forward(h_body, last_linear=True, layers=self.delegation_layers)
        return y, d

class Le(nn.Module):
    n_models: int
    body: tuple[int, ...]
    out: tuple[int, ...]
    delegation: tuple[int, ...]
    solver: LEsolver
    output_mixing: Literal["classification", "regression"]
    cnn_body: bool

    def setup(self):
        assert self.delegation[-1] == self.n_models, "Last dim of delegation needs to equal to n_models"

        BodyClass = ExpertBodyCNN if self.cnn_body else ExpertBodyMlp 

        # Vmap over the bodies to get intermediate representations
        self.vmapped_body = nn.vmap(
            BodyClass,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=None, # Do not vmap over batch elements
            axis_size=self.n_models,
            out_axes=1 # Stack the model outputs to axis=1
        )(body=self.body)

        # Vmap over the heads, which will take the orthogonalized bodies
        self.vmapped_head = nn.vmap(
            ExpertHead,
            variable_axes={'params': 0},
            split_rngs={'params': True},
            in_axes=1, # Take h_bodies where axis=1 is the expert dimension
            axis_size=self.n_models,
            out_axes=1 
        )(out=self.out, delegation=self.delegation)

    def __call__(self, x: jax.Array):

        if not self.cnn_body:
            x = x if x.ndim == 2 else x.reshape((x.shape[0], -1))

        # 1. Get body representations: Shape (bs, n_models, body_dim)
        h_bodies = self.vmapped_body(x)
        body_dim = h_bodies.shape[-1]
        
        assert body_dim >= self.n_models, "Body dimension must be >= n_models for full orthogonalization."

        # 2. Enforce Orthogonality via Gram-Schmidt (QR Decomposition)
        # Swap axes to (bs, body_dim, n_models) so columns are the expert vectors
        A = jnp.swapaxes(h_bodies, 1, 2)
        
        # --- FIX: Inject a tiny full-rank matrix to prevent NaN gradients ---
        eps = 1e-4
        rank_stabilizer = jnp.eye(A.shape[1], A.shape[2])
        A = A + (eps * rank_stabilizer)
        # --------------------------------------------------------------------
        
        Q, R = jnp.linalg.qr(A)
        
        # Swap back to (bs, n_models, body_dim)
        h_bodies_ortho = jnp.swapaxes(Q, 1, 2)
        
        # Scale by sqrt(body_dim) so the variance of the features remains ~1 (since Q has L2 norm of 1)
        # This prevents the orthogonalized features from vanishing and choking the linear layers
        h_bodies_ortho = h_bodies_ortho * jnp.sqrt(body_dim)

        # 3. Pass orthogonalized features to the heads
        ys, delegation = self.vmapped_head(h_bodies_ortho)
        
        delegation = nn.softmax(delegation, axis=-1)
        leinfo = self.solver.solve_power(delegation)
        
        # 4. Calculate Cosine Similarity (should be close to 0 due to QR forcing orthogonality)
        # Note: we use h_bodies_ortho here, which mathematically bounds off-diagonals near 0.
        h_bodies_norm = h_bodies_ortho / (jnp.linalg.norm(h_bodies_ortho, axis=-1, keepdims=True) + 1e-8)
        cosine_similarity_matrix = jnp.einsum("bxd,byd->bxy", h_bodies_norm, h_bodies_norm)
        off_diag_mask = 1 - jnp.eye(self.n_models)
        n = self.n_models
        num_off_diag = n * (n - 1)
        cosine_similarity = jnp.sum(cosine_similarity_matrix * off_diag_mask) / (h_bodies_norm.shape[0] * num_off_diag)

        if self.output_mixing == "classification":
            y = mix_weighted_logits(ys, leinfo.power)
        elif self.output_mixing == "regression":
            y = mix_weighted_mean(ys, leinfo.power)

        return y, leinfo, cosine_similarity
    
            

class LeMnistLearner(Learner[LEInfo]):

    @staticmethod
    def get_model(
        train_params: TrainParams,
        param_budget: int | None = None,
        dummy_input: jax.Array | None = None
    ) -> nn.Module:
        

        builder = lambda alpha: Le(
            n_models=train_params.n_models_in_ensemble,
            body=(to_param(9 * alpha), to_param(16 * alpha), to_param(32 * alpha)),
            out=(train_params.task.out_dim(),),
            delegation=(train_params.n_models_in_ensemble,),
            solver=LEsolver(
                load_distribution_lambda=0.1,
                specialization_lambda=0.0,
            ),
            output_mixing=train_params.task.task_type(),
            cnn_body=True
        )

        if param_budget is None:
            model = builder(alpha=1.0)
        else:
            alpha, actual_params = find_best_matching_architecture_scalar(
                param_budget=param_budget, 
                dummy_input=dummy_input, 
                model_builder=builder
            )

            diff = abs(actual_params - param_budget)
            print(f"Difference between requested and actual #params = {diff}")
            assert abs((diff / actual_params) - 1) > 0.2, r"Difference is more than 20% of the budget" 
            model = builder(alpha=alpha)
            
        return model


    @staticmethod
    def forward(key: jax.Array, x: jax.Array, model: Le, params: dict) -> tuple[jax.Array, LEInfo]:
        return model.apply({"params": params}, x)
    
    @staticmethod
    def auxillary_losses(
        key: jax.Array,
        model: nn.Module,
        params: dict,
        train_return: LEInfo
    ) -> dict[str, jax.Array]:
        
        return {
            "load_distribution_loss": model.solver.load_distribution_loss(train_return),
            "specialization_loss": model.solver.specialization_loss(train_return)
        }