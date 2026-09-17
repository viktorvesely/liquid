from dataclasses import dataclass
from functools import partial
import math
import time
from typing import Callable, Literal

from jax.flatten_util import ravel_pytree
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn, struct
import tqdm

from structs import TrainParams, InOutData, Predictors, Delegators
from math_utils import (
    eval_loss,
    eval_predictor_delegator_decomposition,
    aggregate_delegators,
    predictor_error_ambiguity_decomposition,
    delegator_error_ambiguity_decomposition,
)
from utils import train_loader
from atomic_networks import three_layer_mlp, Mlp, CnnMlp
from architectures import Ensemble, get_modules

USE_THREE_LAYER_DELEGATOR = False
THREE_LAYER_DELEGATOR_BASE = 8

ORACLE_SYSTEM: Literal["original", "continuation", "continuation_linear"] = "continuation_linear"
# Diagnostic mode: validation labels also participate in oracle training.
CONTINUATION_ORACLE_USE_VALIDATION = True
# Penalize KL(original aggregate routing || current aggregate routing).
CONTINUATION_ORACLE_TETHER = 0.1

# Last-layer-only fitting. Validation is always included; training is optional.
CONTINUATION_LINEAR_INCLUDE_TRAIN = False
CONTINUATION_LINEAR_MAX_STEPS = 100
CONTINUATION_LINEAR_GRAD_TOL = 1e-6

@struct.dataclass
class InOutDataOracle:
    x: jax.Array
    y: jax.Array
    predictions: jax.Array

@struct.dataclass
class InOutDataContinuation:
    x: jax.Array
    y: jax.Array
    predictions: jax.Array
    original_log_weights: jax.Array

@dataclass(frozen=True)
class JittedFunctions:
    train_oracle_batch: Callable
    validate_and_update_best: Callable
    apply_delegators_agg: Callable
    apply_oracle_delegators_agg: Callable
    apply_predictors: Callable
    oracle_optimizer: optax.TransformUpdateExtraArgsFn
    oracle_delegators: nn.Module

def jit_functions(
    predictors: nn.Module,
    delegators: nn.Module,
    train_params: TrainParams,
) -> JittedFunctions:


    oracle_optimizer = optax.adamw(learning_rate=train_params.lr * 3)
    _apply_delegators_agg = jax.jit(partial(
        apply_delegators_agg,
        delegators=delegators,
        train_params=train_params
    ))

    if USE_THREE_LAYER_DELEGATOR and ORACLE_SYSTEM == "original":

        assert train_params.architecture.cnn == 0, "Implement some powerful oracle for CNNs"

        architecture = three_layer_mlp.determine_size(
            predictor_base=1,
            delegator_base=THREE_LAYER_DELEGATOR_BASE,
            out_dim=train_params.task.out_dim(),
            n_predictors=train_params.n_predictors
        )
        ensemble = Ensemble(
            n_predictors=train_params.n_predictors,
            n_delegators=train_params.n_delegators,
            predictor=architecture.predictor,
            delegator=architecture.delegator,
            n_cnn_layers=architecture.cnn   
        ) 
        _, oracle_delegators = get_modules(ensemble)
        
        _apply_oracle_delegators_agg = jax.jit(partial(
            apply_delegators_agg,
            delegators=oracle_delegators,
            train_params=train_params
        ))

    else:
        oracle_delegators = delegators
        _apply_oracle_delegators_agg = _apply_delegators_agg

    return JittedFunctions(
        train_oracle_batch=jax.jit(partial(
            train_oracle_batch,
            optimizer=oracle_optimizer,
            delegators=oracle_delegators,
            train_params=train_params
        )),
        validate_and_update_best=jax.jit(partial(
            validate_and_update_best,
            delegators=oracle_delegators,
            train_params=train_params
        )),
        apply_delegators_agg=_apply_delegators_agg,
        apply_oracle_delegators_agg=_apply_oracle_delegators_agg,
        apply_predictors=jax.jit(partial(
            apply_predictors,
            predictors=predictors
        )),
        oracle_optimizer=oracle_optimizer,
        oracle_delegators=oracle_delegators
    )

def oracle_loss_fn(
    one_seed_params,
    inout_batch: InOutDataOracle,
    delegators: Delegators,
    train_params: TrainParams,
):
    delegations_logits = delegators.apply(
        {"params": one_seed_params},
        inout_batch.x,
    )
    predictions = jax.lax.stop_gradient(inout_batch.predictions)
    y = inout_batch.y
    agg_weights = aggregate_delegators(train_params, delegations_logits)

    if train_params.ambiguity_gradient_delegators:
        performance, ambiguity = predictor_error_ambiguity_decomposition(
            predictions=predictions,
            y=y,
            task_type=train_params.task.task_type(),
            agg_delegation=agg_weights,
        )

        weighted_performance = jnp.sum(
            agg_weights * performance, axis=-1
        )
        weighted_ambiguity = jnp.sum(
            agg_weights * ambiguity, axis=-1
        )

        if train_params.ambiguity_gradient_predictors == "none":
            weighted_ambiguity = jax.lax.stop_gradient(weighted_ambiguity)

        training_loss = jnp.mean(
            weighted_performance - weighted_ambiguity
        )

    else:
        performance, ambiguity = delegator_error_ambiguity_decomposition(
            delegations=delegations_logits,
            predictions=predictions,
            y=y,
            train_params=train_params,
            stop_delegations_ambiguity_gradient=True,
        )
        training_loss = jnp.mean(performance - ambiguity)

    actual_loss = jnp.mean(
        eval_loss(agg_weights, predictions, y, train_params)
    )

    # Actual task-loss value, with the intended training gradients.
    return training_loss + jax.lax.stop_gradient(
        actual_loss - training_loss
    )

def train_continuation_oracle(
    key: jax.Array,
    delegators: Delegators,
    delegators_params: dict,
    predictors_params: dict,
    inout_train: InOutData,
    inout_valid: InOutData,
    train_params: TrainParams,
    jit_funcs: JittedFunctions,
):
    """Continue every original seed together, with cached, fixed predictions.

    Each seed has its own optimizer state and best validation checkpoint.
    Epoch -1 denotes the original parameters; no random restarts are used.
    Optional validation training is diagnostic, not held-out evaluation.
    A fixed original routing distribution softly tethers every update.
    """
    gpu = jax.devices("gpu")[0]
    gpu_batch_size = train_params.batch_size * train_params.preload_batches_to_gpu
    apply_all_predictors = jax.jit(jax.vmap(
        jit_funcs.apply_predictors, in_axes=(0, None), out_axes=1
    ))

    def routing_log_probs(params, x):
        logits = delegators.apply({"params": params}, x)
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        if train_params.delegators_mixing == "sum":
            return jax.nn.log_softmax(
                jax.scipy.special.logsumexp(log_probs, axis=1), axis=-1
            )
        return jax.nn.log_softmax(jnp.mean(log_probs, axis=1), axis=-1)

    apply_original_routing = jax.jit(jax.vmap(
        routing_log_probs, in_axes=(0, None), out_axes=1
    ))
    fitting_data = inout_train
    if CONTINUATION_ORACLE_USE_VALIDATION:
        fitting_data = jax.tree.map(
            lambda train, valid: np.concatenate(
                (np.asarray(jax.device_get(train)), np.asarray(jax.device_get(valid))),
                axis=0,
            ),
            inout_train, inout_valid,
        )

    # Example axis first, so the loader shuffles x/y/all seeds' predictions together.
    # Cache on the host rather than retaining the entire training set on the GPU.
    prediction_batches = []
    original_routing_batches = []
    for batch, key in train_loader(
        key, fitting_data, batch_size=gpu_batch_size, serve_as_is=True
    ):
        batch_x = jax.device_put(batch.x, device=gpu)
        predictions = apply_all_predictors(
            predictors_params, batch_x
        )  # (examples, seeds, predictors, out)
        prediction_batches.append(np.asarray(jax.device_get(predictions)))
        original_routing_batches.append(np.asarray(jax.device_get(
            apply_original_routing(delegators_params, batch_x)
        )))  # (examples, seeds, predictors)
    train_data = InOutDataContinuation(
        x=fitting_data.x,
        y=fitting_data.y,
        predictions=np.concatenate(prediction_batches, axis=0),
        original_log_weights=np.concatenate(original_routing_batches, axis=0),
    )
    del prediction_batches, original_routing_batches
    valid_data = InOutDataOracle(
        x=jax.device_put(inout_valid.x, device=gpu),
        y=jax.device_put(inout_valid.y, device=gpu),
        predictions=apply_all_predictors(predictors_params, inout_valid.x),
    )

    # Match the original oracle's optimizer and epoch budget.
    optimizer = jit_funcs.oracle_optimizer
    params = delegators_params
    states = jax.vmap(optimizer.init)(params)

    def one_seed_loss(params, x, y, predictions, original_log_weights):
        task_loss = oracle_loss_fn(
            params, InOutDataOracle(x=x, y=y, predictions=predictions),
            delegators, train_params,
        )
        if CONTINUATION_ORACLE_TETHER == 0:
            return task_loss
        original_log_weights = jax.lax.stop_gradient(original_log_weights)
        current_log_weights = routing_log_probs(params, x)
        tether = jnp.mean(jnp.sum(
            jnp.exp(original_log_weights) * (original_log_weights - current_log_weights),
            axis=-1,
        ))
        return task_loss + CONTINUATION_ORACLE_TETHER * tether

    def update_one(params, state, x, y, predictions, original_log_weights):
        gradient = jax.grad(one_seed_loss)(params, x, y, predictions, original_log_weights)
        updates, state = optimizer.update(gradient, state, params)
        return optax.apply_updates(params, updates), state

    @jax.jit
    def train_batch(params, states, data):
        bs = train_params.batch_size
        assert data.x.shape[0] % bs == 0
        batches = jax.tree.map(
            lambda value: value.reshape((-1, bs) + value.shape[1:]), data
        )

        def step(carry, batch):
            params, states = jax.vmap(
                update_one, in_axes=(0, 0, None, None, 1, 1)
            )(*carry, batch.x, batch.y, batch.predictions, batch.original_log_weights)
            return (params, states), None

        return jax.lax.scan(step, (params, states), batches)[0]

    def one_seed_validation(params, predictions, x, y):
        logits = delegators.apply({"params": params}, x)
        weights = aggregate_delegators(train_params, logits)
        return jnp.mean(eval_loss(weights, predictions, y, train_params))

    validate = jax.jit(jax.vmap(
        one_seed_validation, in_axes=(0, 1, None, None)
    ))
    best_params = params
    best_losses = validate(params, valid_data.predictions, valid_data.x, valid_data.y)
    best_epochs = jnp.full(best_losses.shape, -1, dtype=jnp.int32)

    @jax.jit
    def checkpoint(params, best_params, best_losses, best_epochs, epoch, data):
        losses = validate(params, data.predictions, data.x, data.y)
        improved = jnp.isfinite(losses) & (losses < best_losses)
        best_params = jax.tree.map(
            lambda new, old: jnp.where(
                improved.reshape((improved.shape[0],) + (1,) * (new.ndim - 1)),
                new, old,
            ), params, best_params,
        )
        return (
            best_params,
            jnp.where(improved, losses, best_losses),
            jnp.where(improved, epoch, best_epochs),
        )

    n_batches = math.ceil(train_data.x.shape[0] / gpu_batch_size)
    epochs = max(1, round(train_params.epochs / 5))
    for epoch in tqdm.tqdm(range(epochs), desc="Continuing oracle seeds", disable=True):
        for batch, key in train_loader(
            key, train_data, batch_size=gpu_batch_size, desired_batches=n_batches
        ):
            batch = jax.tree.map(lambda value: jax.device_put(value, device=gpu), batch)
            params, states = train_batch(params, states, batch)
        best_params, best_losses, best_epochs = checkpoint(
            params, best_params, best_losses, best_epochs, jnp.array(epoch), valid_data
        )

    apply_all_delegators = jax.jit(jax.vmap(
        jit_funcs.apply_delegators_agg, in_axes=(0, None)
    ))
    best_weights, _ = apply_all_delegators(best_params, valid_data.x)
    return jnp.swapaxes(valid_data.predictions, 0, 1), best_weights, best_epochs


class _MlpHiddenFeatures(Mlp):
    def __call__(self, x):
        for layer in self.body_layers[:-1]:
            x = nn.relu(layer(x))
        return x


class _CnnMlpHiddenFeatures(CnnMlp):
    def __call__(self, x):
        for layer in self.cnn_layers:
            x = nn.relu(layer(x))
        x = x.reshape(x.shape[:-3] + (-1,))
        for layer in self.mlp_layers[:-1]:
            x = nn.relu(layer(x))
        return x


class _DelegatorHiddenFeatures(nn.Module):
    """Same Flax parameter paths as Delegators, stopping before the last Dense."""
    architecture: tuple[int, ...]
    n_delegators: int
    n_cnn_layers: int
    kernel_size: int

    @nn.compact
    def __call__(self, x):
        mlp = self.architecture[self.n_cnn_layers:]
        if self.n_cnn_layers:
            base = _CnnMlpHiddenFeatures
            kwargs = dict(cnn=self.architecture[:self.n_cnn_layers], mlp=mlp,
                          kernel_size=self.kernel_size, stride=2)
        else:
            base = _MlpHiddenFeatures
            kwargs = dict(body=mlp)
            x = x.reshape((x.shape[0], -1))
        mapped = nn.vmap(
            base, variable_axes={"params": 0}, split_rngs={"params": True},
            in_axes=None, out_axes=1, axis_size=self.n_delegators,
        )
        return mapped(name="delegators", **kwargs)(x)


def train_continuation_linear_oracle(
    key: jax.Array,
    delegators: Delegators,
    delegators_params: dict,
    predictors_params: dict,
    inout_train: InOutData,
    inout_valid: InOutData,
    train_params: TrainParams,
    jit_funcs: JittedFunctions,
):
    """Fit only existing final Dense kernels/biases with full-batch L-BFGS.

    Hidden features and predictions are cached once and never updated. Optimize
    actual aggregate CE/MSE without tethering, regularization or gradient
    ablations. Validation-only by default; optionally include training examples.
    Each seed retains its best validation iterate, including initialization (-1).
    The routing softmax means last-layer fitting is not generally convex.
    """
    assert CONTINUATION_LINEAR_MAX_STEPS >= 0
    assert CONTINUATION_LINEAR_GRAD_TOL >= 0
    if delegators.n_delegators < 1:
        raise ValueError("The last-layer oracle requires at least one delegator.")
    mlp = delegators.delegator[delegators.n_cnn_layers:]
    if not mlp:
        raise ValueError("The last-layer oracle requires a final Dense layer.")

    gpu = jax.devices("gpu")[0]
    gpu_batch_size = train_params.batch_size * train_params.preload_batches_to_gpu
    feature_model = _DelegatorHiddenFeatures(
        architecture=delegators.delegator,
        n_delegators=delegators.n_delegators,
        n_cnn_layers=delegators.n_cnn_layers,
        kernel_size=delegators.kernel_size,
    )
    layer_prefix = "mlp_layers" if delegators.n_cnn_layers else "body_layers"
    head_params = delegators_params["delegators"][f"{layer_prefix}_{len(mlp) - 1}"]
    # These are the only parameters passed to the optimizer.
    heads = {"kernel": head_params["kernel"], "bias": head_params["bias"]}

    def cache_one(predictor_params, delegator_params, x):
        predictions = jit_funcs.apply_predictors(predictor_params, x)
        features = feature_model.apply({"params": delegator_params}, x)
        return predictions, features

    cache_all = jax.jit(jax.vmap(cache_one, in_axes=(0, 0, None), out_axes=1))

    def cache_split(data, loader_key):
        chunks = []
        for batch, loader_key in train_loader(
            loader_key, data, batch_size=gpu_batch_size, serve_as_is=True
        ):
            values = cache_all(
                predictors_params, delegators_params,
                jax.device_put(batch.x, device=gpu),
            )
            chunks.append(jax.tree.map(lambda value: np.asarray(jax.device_get(value)), values))
        return jax.tree.map(
            lambda *values: jax.device_put(np.concatenate(values, axis=0), device=gpu),
            *chunks,
        )

    valid_key, train_key = jax.random.split(key)
    valid_predictions, valid_features = cache_split(inout_valid, valid_key)
    valid_y = jax.device_put(inout_valid.y, device=gpu)
    fit_predictions, fit_features, fit_y = valid_predictions, valid_features, valid_y
    if CONTINUATION_LINEAR_INCLUDE_TRAIN:
        train_predictions, train_features = cache_split(inout_train, train_key)
        fit_predictions = jnp.concatenate((train_predictions, valid_predictions), axis=0)
        fit_features = jnp.concatenate((train_features, valid_features), axis=0)
        fit_y = jnp.concatenate((jax.device_put(inout_train.y, device=gpu), valid_y), axis=0)

    # Preserve the original validation routing exactly if no iterate improves it.
    apply_all_delegators = jax.jit(jax.vmap(
        jit_funcs.apply_delegators_agg, in_axes=(0, None), out_axes=1
    ))
    original_weights, _ = apply_all_delegators(
        delegators_params, jax.device_put(inout_valid.x, device=gpu)
    )
    classification = train_params.task.task_type() == "classification"

    def fit_one(head, fit_features, fit_predictions, valid_features,
                valid_predictions, original_weights, fit_y, valid_y):
        fit_outputs = (
            jax.nn.log_softmax(fit_predictions, axis=-1) if classification else fit_predictions
        )

        def routing(head, features):
            logits = jnp.einsum("ndh,dhp->ndp", features, head["kernel"]) + head["bias"]
            return aggregate_delegators(train_params, logits)

        def objective(head):
            weights = routing(head, fit_features)
            prediction = jnp.sum(weights[..., None] * fit_outputs, axis=1)
            if classification:
                return jnp.mean(optax.softmax_cross_entropy_with_integer_labels(prediction, fit_y))
            return jnp.mean((prediction - fit_y) ** 2)

        optimizer = optax.lbfgs(memory_size=10)
        state = optimizer.init(head)
        value_and_grad = optax.value_and_grad_from_state(objective)
        baseline_loss = jnp.mean(eval_loss(original_weights, valid_predictions, valid_y, train_params))

        def step(carry):
            iteration, head, state, best_weights, best_loss, best_iteration, _ = carry
            value, gradient = value_and_grad(head, state=state)
            gradient_norm = optax.global_norm(gradient)
            active = jnp.isfinite(value) & jnp.isfinite(gradient_norm) & (
                gradient_norm > CONTINUATION_LINEAR_GRAD_TOL
            )

            def update(_):
                updates, new_state = optimizer.update(
                    gradient, state, head, value=value, grad=gradient, value_fn=objective
                )
                return optax.apply_updates(head, updates), new_state

            head, state = jax.lax.cond(active, update, lambda _: (head, state), operand=None)
            weights = routing(head, valid_features)
            score = jnp.mean(eval_loss(weights, valid_predictions, valid_y, train_params))
            improved = jnp.isfinite(score) & (score < best_loss)
            return (
                iteration + 1, head, state,
                jnp.where(improved, weights, best_weights),
                jnp.where(improved, score, best_loss),
                jnp.where(improved, iteration, best_iteration),
                active & jnp.isfinite(score),
            )

        initial = (
            jnp.array(0, dtype=jnp.int32), head, state, original_weights, baseline_loss,
            jnp.array(-1, dtype=jnp.int32), jnp.array(True),
        )
        _, _, _, best_weights, _, best_iteration, _ = jax.lax.while_loop(
            lambda carry: (carry[0] < CONTINUATION_LINEAR_MAX_STEPS) & carry[-1],
            step, initial,
        )
        return best_weights, best_iteration

    # Independent optimizer histories/checkpoints, batched across original seeds.
    fit_all = jax.jit(jax.vmap(fit_one, in_axes=(0, 1, 1, 1, 1, 1, None, None)))
    best_weights, best_iterations = fit_all(
        heads, fit_features, fit_predictions, valid_features, valid_predictions,
        original_weights, fit_y, valid_y,
    )
    return jnp.swapaxes(valid_predictions, 0, 1), best_weights, best_iterations


def train_oracle_batch(
    inout_data: InOutDataOracle,
    delegator_params: dict,
    opt_states: dict,
    optimizer: optax.GradientTransformationExtraArgs,
    delegators: nn.Module,
    train_params: TrainParams
):
    # print(f" Compiling {train_oracle_batch.__name__}")
    
    batch_size = train_params.batch_size
    assert inout_data.x.shape[0] % train_params.batch_size == 0, (
        "GPU batch needs to be divisible by batch_size"
    )

    n_batches = inout_data.x.shape[0] // batch_size

    inout_data = jax.tree.map(
        lambda x: x.reshape(
            (n_batches, batch_size) + x.shape[1:]
        ),
        inout_data,
    )


    loss_and_grad_fn = jax.value_and_grad(oracle_loss_fn)

    def update_one_seed(
        one_seed_params,
        one_seed_opt_state,
        inout_batch: InOutData,
    ):
        _, grads = loss_and_grad_fn(
            one_seed_params,
            inout_batch,
            delegators,
            train_params,
        )

        updates, one_seed_opt_state = optimizer.update(
            grads,
            one_seed_opt_state,
            one_seed_params,
        )

        one_seed_params = optax.apply_updates(
            one_seed_params,
            updates,
        )

        return one_seed_params, one_seed_opt_state

    def train_step(carry, inout_batch: InOutData):
        params, states = carry

        params, states = jax.vmap(
            update_one_seed,
            in_axes=(0, 0, None),
        )(
            params,
            states,
            inout_batch,
        )

        return (params, states), None

    (delegator_params, opt_states), _ = jax.lax.scan(
        train_step,
        (delegator_params, opt_states),
        inout_data,
    )

    return delegator_params, opt_states


def validate_and_update_best(
    params,
    current_best_params,
    current_best_losses,
    current_best_epoch,
    valid_data: InOutDataOracle,
    epoch: jax.Array,
    delegators: nn.Module, 
    train_params: TrainParams
):
    # print(f" Compiling {validate_and_update_best.__name__}")
    valid_losses = jax.vmap(
        oracle_loss_fn,
        in_axes=(0, None, None, None),
    )(
        params,
        valid_data,
        delegators,
        train_params
    )

    epochs = jnp.broadcast_to(epoch, current_best_epoch.shape)

    improved = valid_losses < current_best_losses

    current_best_params = jax.tree.map(
        lambda new, old: jnp.where(
            improved.reshape(
                (improved.shape[0],)
                + (1,) * (new.ndim - 1)
            ),
            new,
            old,
        ),
        params,
        current_best_params,
    )

    current_best_epoch = jax.tree.map(
        lambda new, old: jnp.where(improved, new, old),
        epochs,
        current_best_epoch
    )

    current_best_losses = jax.tree.map(
        lambda new, old: jnp.where(improved, new, old),
        valid_losses,
        current_best_losses
    )

    return (
        current_best_params,
        current_best_losses,
        current_best_epoch,
        valid_losses,
    )

def train_oracle(
    key: jax.Array,
    train_predictions: jax.Array,
    valid_predictions: jax.Array,
    agg_delegations: jax.Array,
    inout_train_predictions: InOutData,
    inout_valid_predictions: InOutData,
    train_params: TrainParams,
    jit_funcs: JittedFunctions,
    selected_delegator_params: dict,
    epochs_p: float = (1/3),
    n_seeds: int = 5,
):
    gpu = jax.devices("gpu")[0]

    k_init, k_perturb, k_loader = jax.random.split(key, 3)


    # print("Finding unrestricted optimal weights")
    # all_y = jnp.concatenate(
    #     (
    #         inout_train_predictions.y,
    #         inout_valid_predictions.y,
    #     ),
    #     axis=0,
    # )
    # optimal_weights = optimal_convex_weights(
    #     y=all_y,
    #     predictions=predictions,
    #     weights0=agg_delegations,
    #     train_params=train_params,
    # )

    # verify_weights_improvement(
    #     y=all_y,
    #     predictions=predictions,
    #     oracle_weights=optimal_weights,
    #     weights=agg_delegations,
    #     train_params=train_params,
    #     verbal=True
    # )
    # n_train = inout_train_predictions.y.shape[0]

    inout_train_delegations = InOutDataOracle(
        x=inout_train_predictions.x,
        y=inout_train_predictions.y,
        predictions=train_predictions
    )

    inout_valid_delegations = InOutDataOracle(
        x=inout_valid_predictions.x,
        y=inout_valid_predictions.y,
        predictions=valid_predictions
    )

    optimizer = jit_funcs.oracle_optimizer
    delegators = jit_funcs.oracle_delegators


    def perturb_params(params, key, sigma=0.05):
        flatten, reverse_fn = ravel_pytree(params)
        perturbed_params = flatten + jax.random.normal(key, shape=flatten.shape) * sigma
        perturbed_params = reverse_fn(perturbed_params)
        opt_state = optimizer.init(perturbed_params)
        return perturbed_params, opt_state
    
    def init_one_seed(init_key: jax.Array):
        params = delegators.init(
            init_key,
            inout_train_delegations.x[[0], ...],
        )["params"]

        opt_state = optimizer.init(params)

        return params, opt_state


    init_keys = jax.random.split(k_init, n_seeds)
    delegator_params, opt_states = jax.vmap(
        init_one_seed,
    )(init_keys)

    # perturb_keys = jax.random.split(k_perturb, n_seeds)
    # delegator_params, opt_states = jax.vmap(
    #     perturb_params, in_axes=(None, 0)
    # )(selected_delegator_params, perturb_keys)

    best_delegator_params = delegator_params
    best_valid_losses = jnp.full((n_seeds,), jnp.inf,)
    best_delegator_epoch = jnp.full((n_seeds,), -1)

   

    batch_size = train_params.batch_size

    # Number of ordinary minibatches loaded onto GPU together.
    gpu_batch_size = (
        batch_size
        * train_params.preload_batches_to_gpu
    )

    n_train_examples = inout_train_delegations.x.shape[0]


    n_gpu_batches = math.ceil(
        n_train_examples / gpu_batch_size
    )

    epochs = max(1, round(train_params.epochs * epochs_p))

    for i_epoch in tqdm.tqdm(
        range(epochs),
        desc="Training oracle",
        position=1,
        disable=True
    ):
        for inout_batch, k_loader in train_loader(
            k_loader,
            inout_train_delegations,
            batch_size=gpu_batch_size,
            desired_batches=n_gpu_batches,
        ):
    
            inout_batch = jax.tree.map(
                lambda x: jax.device_put(x, device=gpu),
                inout_batch,
            )

            delegator_params, opt_states = jit_funcs.train_oracle_batch(
                inout_data=inout_batch,
                delegator_params=delegator_params,
                opt_states=opt_states,
            )

        (
            best_delegator_params,
            best_valid_losses,
            best_delegator_epoch,
            valid_losses,
        ) = jit_funcs.validate_and_update_best(
            params=delegator_params,
            current_best_params=best_delegator_params,
            current_best_losses=best_valid_losses,
            current_best_epoch=best_delegator_epoch,
            valid_data=inout_valid_delegations,
            epoch=jnp.array(i_epoch)
        )

    best_seed = jnp.argmin(best_valid_losses)

    return jax.tree.map(
        lambda x: x[best_seed],
        best_delegator_params,
    ), best_delegator_epoch[best_seed]


def apply_delegators_agg(
    delegator_params: dict,
    x: jax.Array,
    delegators: nn.Module,
    train_params: TrainParams
):
    # print(f" Compiling {apply_delegators_agg.__name__}")
    delegations = delegators.apply({"params": delegator_params}, x)
    agg_delegations = aggregate_delegators(train_params, delegations)
    return agg_delegations, delegations

def apply_predictors(
    predictor_params: dict,
    x: jax.Array,
    predictors: nn.Module,
):
    # print(f" Compiling {apply_predictors.__name__}")
    predictions = predictors.apply({"params": predictor_params}, x)
    return predictions

def get_evaluation_metrics(
    key: jax.Array,
    delegators: Delegators,
    delegators_params: dict,
    predictors: Predictors,
    predictors_params: dict,
    inout_train_predictions: InOutData,
    inout_valid_predictions: InOutData,
    train_params: TrainParams,
):
    if ORACLE_SYSTEM not in {"original", "continuation", "continuation_linear"}:
        raise ValueError(f"Unknown oracle system: {ORACLE_SYSTEM}")

    n_seeds = jax.tree.leaves(delegators_params)[0].shape[0] 
    jit_funcs = jit_functions(
        predictors=predictors,
        delegators=delegators,
        train_params=train_params
    )

    all_metrics = []


    start = time.perf_counter()
    fitted_oracle_results = None
    if ORACLE_SYSTEM in {"continuation", "continuation_linear"}:
        fit_oracle = train_continuation_oracle if ORACLE_SYSTEM == "continuation" else train_continuation_linear_oracle
        fitted_oracle_results = fit_oracle(
            key=key,
            delegators=delegators,
            delegators_params=delegators_params,
            predictors_params=predictors_params,
            inout_train=inout_train_predictions,
            inout_valid=inout_valid_predictions,
            train_params=train_params,
            jit_funcs=jit_funcs,
        )

    for use_seed in tqdm.tqdm(range(n_seeds), position=0, desc="Seeds", disable=True):
        metrics = one_get_evaluation_metrics(
            key=key,
            delegators=delegators,
            delegators_params=delegators_params,
            predictors=predictors,
            predictors_params=predictors_params,
            inout_train_predictions=inout_train_predictions,
            inout_valid_predictions=inout_valid_predictions,
            train_params=train_params,
            jit_funcs=jit_funcs,
            use_seed=use_seed,
            fitted_oracle_result=(
                tuple(value[use_seed] for value in fitted_oracle_results)
                if fitted_oracle_results is not None else None
            ),
        )
        all_metrics.append(metrics)
    all_metrics = jax.tree.map(lambda *values: jnp.stack(values), *all_metrics)
    end = time.perf_counter()

    print(f"Eval took {(end - start):.3f} seconds")
    
    # for k, v in all_metrics.items():
        #     print(k, v.shape)

    return all_metrics

def one_get_evaluation_metrics(
    key: jax.Array,
    delegators: Delegators,
    delegators_params: dict,
    predictors: Predictors,
    predictors_params: dict,
    inout_train_predictions: InOutData,
    inout_valid_predictions: InOutData,
    train_params: TrainParams,
    jit_funcs: JittedFunctions,
    use_seed: int,
    fitted_oracle_result: tuple | None = None,
):
    gpu = jax.devices("gpu")[0]

    assert (inout_train_predictions.y.ndim == inout_valid_predictions.y.ndim) and inout_train_predictions.y.ndim <= 2, f"{inout_train_predictions.y.shape}, {inout_valid_predictions.y.shape}" 
    key, k_loader, k_train_oracle = jax.random.split(key, 3)
    
    # Select seeds
    selected_delegator_params = jax.tree.map(lambda x: x[use_seed, ...], delegators_params)
    selected_predictor_params = jax.tree.map(lambda x: x[use_seed, ...], predictors_params)

    if fitted_oracle_result is not None:
        valid_predictions, valid_oracle_agg_delegations, from_epoch = fitted_oracle_result
        valid_agg_delegations, valid_delegations = jit_funcs.apply_delegators_agg(
            selected_delegator_params, inout_valid_predictions.x
        )
    else:
        # Get predictions and current delegations for all data
        gpu_batch_size = train_params.batch_size * train_params.preload_batches_to_gpu

        train_predictions = []
        agg_delegations = []

        for inout_batch, k_loader in train_loader(
                k_loader,
                inout_train_predictions,
                batch_size=gpu_batch_size,
                serve_as_is=True
            ):
                inout_batch: InOutData = jax.tree.map(
                    lambda x: jax.device_put(x, device=gpu),
                    inout_batch,
                )

                batch_predictions = jit_funcs.apply_predictors(selected_predictor_params, inout_batch.x)
                train_predictions.append(batch_predictions)
            
                batch_agg_delegations, _ = jit_funcs.apply_delegators_agg(selected_delegator_params, inout_batch.x)
                agg_delegations.append(batch_agg_delegations)
    

        # Validation
        valid_predictions = jit_funcs.apply_predictors(selected_predictor_params, inout_valid_predictions.x)
        valid_agg_delegations, valid_delegations = jit_funcs.apply_delegators_agg(selected_delegator_params, inout_valid_predictions.x)
        agg_delegations.append(valid_agg_delegations)

        train_predictions = jnp.concatenate(train_predictions, axis=0)
        agg_delegations = jnp.concatenate(agg_delegations, axis=0)

        # Train original oracle
        oracle_delegator_params, from_epoch = train_oracle(
            key=k_train_oracle,
            train_predictions=train_predictions,
            valid_predictions=valid_predictions,
            agg_delegations=agg_delegations,
            inout_train_predictions=inout_train_predictions,
            inout_valid_predictions=inout_valid_predictions,
            train_params=train_params,
            jit_funcs=jit_funcs,
            selected_delegator_params=selected_delegator_params
        )
        valid_oracle_agg_delegations, valid_oracle_delegations = jit_funcs.apply_oracle_delegators_agg(oracle_delegator_params, inout_valid_predictions.x)
 


    (predictor_loss, delegator_regret_loss), (loss, loss_under_oracle) = eval_predictor_delegator_decomposition(
        predictions=valid_predictions,
        agg_delegations=valid_agg_delegations,
        agg_oracle_delegations=valid_oracle_agg_delegations,
        y=inout_valid_predictions.y,
        train_params=train_params,
        use="loss"
    )
    
    metrics = dict(
        predictor_loss=predictor_loss,
        delegator_regret_loss=delegator_regret_loss,
        loss=loss,
        loss_under_oracle=loss_under_oracle,
        from_epoch=from_epoch
    )

    (predictor_metric, delegator_regret_metric), (metric, metric_under_oracle) = eval_predictor_delegator_decomposition(
        predictions=valid_predictions,
        agg_delegations=valid_agg_delegations,
        agg_oracle_delegations=valid_oracle_agg_delegations,
        y=inout_valid_predictions.y,
        train_params=train_params,
        use="metric"
    )

    metrics |= dict(
        predictor_metric=predictor_metric,
        delegator_regret_metric=delegator_regret_metric,
        metric=metric,
        metric_under_oracle=metric_under_oracle
    )


    print(f"From epoch = {from_epoch}")
    print(f"Loss = {loss:.3f} Loss oracle {loss_under_oracle:.3f}")

    # Error ambiguity decompositions

    # For predictors use the truly optimal weights 
    oracle_better = loss_under_oracle <= loss
    valid_superior_agg_delegations = valid_oracle_agg_delegations if oracle_better else valid_agg_delegations
    predictors_perfomance_per_model, predictors_ambiguity_per_model = predictor_error_ambiguity_decomposition(
        predictions=valid_predictions,
        y=inout_valid_predictions.y,
        task_type=train_params.task.task_type(),
        agg_delegation=valid_superior_agg_delegations
    )

    metrics |= dict(
        predictors_perfomance_per_model=predictors_perfomance_per_model,
        predictors_ambiguity_per_model=predictors_ambiguity_per_model,
        predictors_weight_per_model=valid_superior_agg_delegations
    )

    # For delegators use the oracle weights
    delegators_perfomance_per_model, delegators_ambiguity_per_model = (
        delegator_error_ambiguity_decomposition(
            delegations=valid_delegations,
            predictions=valid_predictions,
            y=inout_valid_predictions.y,
            train_params=train_params,
        )
    )

    metrics |= dict(
        delegators_perfomance_per_model=delegators_perfomance_per_model,
        delegators_ambiguity_per_model=delegators_ambiguity_per_model
    )


    metrics = {k: jnp.array(v) for k, v in metrics.items()}

    return metrics
    
