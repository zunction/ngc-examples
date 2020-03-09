# Copyright 2019 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""NovoGrad for TensorFlow."""

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike

from typing import Union, Callable
from typeguard import typechecked

# TODO: Find public API alternatives to these
from tensorflow.python.training import training_ops

class NovoGrad(tf.keras.optimizers.Optimizer):
    """The NovoGrad Optimizer was first proposed in [Stochastic Gradient
    Methods with Layerwise Adaptvie Moments for training of Deep
    Networks](https://arxiv.org/pdf/1905.11286.pdf)

    NovoGrad is a first-order SGD-based algorithm, which computes second
    moments per layer instead of per weight as in Adam. Compared to Adam,
    NovoGrad takes less memory, and has been found to be more numerically
    stable. More specifically we compute (for more information on the
    computation please refer to this
    [link](https://nvidia.github.io/OpenSeq2Seq/html/optimizers.html):

    Second order moment = exponential moving average of Layer-wise square
    of grads:
        v_t <-- beta_2 * v_{t-1} + (1-beta_2) * (g_t)^2
    First order moment in one of four modes:
        1. moment of grads normalized by v_t:
            m_t <- beta_1 * m_{t-1} + [ g_t / (sqrt(v_t)+epsilon)]
        2. moment similar to Adam: exponential moving average of grads
        normalized by v_t (set grad_averaging = True to use this):
            m_t <- beta_1 * m_{t-1} +
                   [(1 - beta_1) * (g_t / (sqrt(v_t) + epsilon))]
        3. weight decay adds a w_d term after grads are rescaled by
        1/sqrt(v_t) (set weight_decay > 0 to use this0:
            m_t <- beta_1 * m_{t-1} +
                   [(g_t / (sqrt(v_t) + epsilon)) + (w_d * w_{t-1})]
        4. weight decay + exponential moving average from Adam:
            m_t <- beta_1 * m_{t-1} +
                   [(1 - beta_1) * ((g_t / (sqrt(v_t + epsilon)) +
                   (w_d * w_{t-1}))]
    Weight update:
        w_t <- w_{t-1} - lr_t * m_t

    Example of usage:
    ```python
    opt = tfa.optimizers.NovoGrad(
        lr=1e-3,
        beta_1=0.9,
        beta_2=0.999,
        weight_decay=0.001,
        grad_averaging=False
    )
    ```
    """

    @typechecked
    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable] = 0.001,
        beta_1: FloatTensorLike = 0.9,
        beta_2: FloatTensorLike = 0.999,
        epsilon: FloatTensorLike = 1e-7,
        weight_decay: FloatTensorLike = 0.0,
        grad_averaging: bool = False,
        amsgrad: bool = False,
        name: str = "NovoGrad",
        **kwargs
    ):
        r"""Construct a new NovoGrad optimizer.

        Args:
            learning_rate: A `Tensor` or a floating point value. or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`
                The learning rate.
            beta_1: A float value or a constant float tensor.
                The exponential decay rate for the 1st moment estimates.
            beta_2: A float value or a constant float tensor.
                The exponential decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability.
            weight_decay: A floating point value. Weight decay for each param.
            grad_averaging: determines whether to use Adam style exponential
                moving averaging for the first order moments.
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients
                by norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(name, **kwargs)
        if weight_decay < 0.0:
            raise ValueError("Weight decay rate cannot be negative")
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("weight_decay", weight_decay)
        self._set_hyper("grad_averaging", grad_averaging)
        self.amsgrad = amsgrad
        self.epsilon = epsilon or tf.keras.backend.epsilon()

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var=var, slot_name="m", initializer="zeros")
        for var in var_list:
            self.add_slot(
                var=var, slot_name="v", initializer=tf.zeros(shape=[], dtype=var.dtype)
            )
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, "vhat")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)
        beta_1_t = tf.identity(self._get_hyper("beta_1", var_dtype))
        beta_2_t = tf.identity(self._get_hyper("beta_2", var_dtype))
        apply_state[(var_device, var_dtype)].update(
            dict(
                epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_2_t=beta_2_t,
                one_minus_beta_2_t=1 - beta_2_t,
                one_minus_beta_1_t=1 - beta_1_t,
            )
        )

    def set_weights(self, weights):
        params = self.weights
        # If the weights are generated by Keras V1 optimizer, it includes vhats
        # even without amsgrad, i.e, V1 optimizer has 3x + 1 variables, while V2
        # optimizer has 2x + 1 variables. Filter vhats out for compatibility.
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super().set_weights(weights)

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
        weight_decay = self._get_hyper("weight_decay")
        grad_averaging = self._get_hyper("grad_averaging")

        v = self.get_slot(var, "v")
        g_2 = tf.reduce_sum(tf.square(tf.cast(grad, tf.float32)))
        v_t = tf.cond(
            tf.equal(self.iterations, 0),
            lambda: g_2,
            lambda: v * coefficients["beta_2_t"]
            + g_2 * coefficients["one_minus_beta_2_t"],
        )
        v_t = v.assign(v_t, use_locking=self._use_locking)

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            grad = grad / (tf.sqrt(vhat_t) + self.epsilon)
        else:
            grad = grad / (tf.sqrt(v_t) + self.epsilon)
        grad = tf.cond(
            tf.greater(weight_decay, 0), lambda: grad + weight_decay * var, lambda: grad
        )
        grad = tf.cond(
            tf.logical_and(grad_averaging, tf.not_equal(self.iterations, 0)),
            lambda: grad * coefficients["one_minus_beta_1_t"],
            lambda: grad,
        )
        m = self.get_slot(var, "m")
        return training_ops.resource_apply_keras_momentum(
            var.handle,
            m.handle,
            coefficients["lr_t"],
            grad,
            coefficients["beta_1_t"],
            use_locking=self._use_locking,
            use_nesterov=False,
        )

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)
        weight_decay = self._get_hyper("weight_decay")
        grad_averaging = self._get_hyper("grad_averaging")

        v = self.get_slot(var, "v")
        g_2 = tf.reduce_sum(tf.square(tf.cast(grad, tf.float32)))
        # v is just a scalar and does not need to involve sparse tensors.
        v_t = tf.cond(
            tf.equal(self.iterations, 0),
            lambda: g_2,
            lambda: v * coefficients["beta_2_t"]
            + g_2 * coefficients["one_minus_beta_2_t"],
        )
        v_t = v.assign(v_t, use_locking=self._use_locking)

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            grad = grad / (tf.sqrt(vhat_t) + self.epsilon)
        else:
            grad = grad / (tf.sqrt(v_t) + self.epsilon)
        grad = tf.cond(
            tf.greater(weight_decay, 0), lambda: grad + weight_decay * var, lambda: grad
        )
        grad = tf.cond(
            tf.logical_and(grad_averaging, tf.not_equal(self.iterations, 0)),
            lambda: grad * coefficients["one_minus_beta_1_t"],
            lambda: grad,
        )
        m = self.get_slot(var, "m")
        return training_ops.resource_sparse_apply_keras_momentum(
            var.handle,
            m.handle,
            coefficients["lr_t"],
            tf.gather(grad, indices),
            indices,
            coefficients["beta_1_t"],
            use_locking=self._use_locking,
            use_nesterov=False,
        )

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "epsilon": self.epsilon,
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
                "grad_averaging": self._serialize_hyperparameter("grad_averaging"),
            }
        )
        return config
    
    
import re
from typing import Optional, Union, Callable
from typeguard import typechecked

class LAMB(tf.keras.optimizers.Optimizer):
    """Optimizer that implements the Layer-wise Adaptive Moments (LAMB).
    See paper [Large Batch Optimization for Deep Learning: Training BERT
    in 76 minutes](https://arxiv.org/abs/1904.00962).
    """

    @typechecked
    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable] = 0.001,
        beta_1: FloatTensorLike = 0.9,
        beta_2: FloatTensorLike = 0.999,
        epsilon: FloatTensorLike = 1e-6,
        weight_decay_rate: FloatTensorLike = 0.0,
        exclude_from_weight_decay: Optional[str] = None,
        exclude_from_layer_adaptation: Optional[str] = None,
        name: str = "LAMB",
        **kwargs
    ):
        """Construct a new LAMB optimizer.
        Args:
            learning_rate: A `Tensor` or a floating point value. or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`
                The learning rate.
            beta_1: A `float` value or a constant `float` tensor.
              The exponential decay rate for the 1st moment estimates.
            beta_2: A `float` value or a constant `float` tensor.
              The exponential decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability.
            weight_decay_rate: weight decay rate.
            exclude_from_weight_decay: comma separated name patterns of
              variables excluded from weight decay. Variables whose name
              contain a substring matching the pattern will be excluded.
            exclude_from_layer_adaptation: comma separated name patterns of
              variables excluded from layer adaptation. Variables whose name
              contain a substring matching the pattern will be excluded.
            name: Optional name for the operations created when applying
              gradients. Defaults to "LAMB".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
              `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients by
              norm; `clipvalue` is clip gradients by value, `decay` is
              included for backward compatibility to allow time inverse
              decay of learning rate. `lr` is included for backward
              compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(name, **kwargs)

        # Just adding the square of the weights to the loss function is *not*
        # the correct way of using L2 regularization/weight decay with Adam,
        # since that will interact with the m and v parameters in strange ways.
        #
        # Instead we want to decay the weights in a manner that doesn't interact
        # with the m/v parameters.
        self._set_hyper("weight_decay_rate", weight_decay_rate)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))

        # This is learning rate decay for using keras learning rate schedule.
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self.epsilon = epsilon or tf.backend_config.epsilon()
        self.exclude_from_weight_decay = exclude_from_weight_decay
        # exclude_from_layer_adaptation is set to exclude_from_weight_decay if
        # the arg is None.
        if exclude_from_layer_adaptation:
            self.exclude_from_layer_adaptation = exclude_from_layer_adaptation
        else:
            self.exclude_from_layer_adaptation = exclude_from_weight_decay

    def _create_slots(self, var_list):
        # Create slots for the first and second moments.
        # Separate for-loops to respect the ordering of slot variables from v1.
        for var in var_list:
            self.add_slot(var, "m")
        for var in var_list:
            self.add_slot(var, "v")

    def _prepare_local(self, var_device, var_dtype, apply_state):
        super()._prepare_local(var_device, var_dtype, apply_state)

        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_t = tf.identity(self._get_hyper("beta_1", var_dtype))
        beta_2_t = tf.identity(self._get_hyper("beta_2", var_dtype))
        weight_decay_rate = tf.identity(self._get_hyper("weight_decay_rate", var_dtype))
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)
        apply_state[(var_device, var_dtype)].update(
            dict(
                weight_decay_rate=weight_decay_rate,
                epsilon=tf.convert_to_tensor(self.epsilon, var_dtype),
                beta_1_t=beta_1_t,
                beta_1_power=beta_1_power,
                one_minus_beta_1_t=1 - beta_1_t,
                beta_2_t=beta_2_t,
                beta_2_power=beta_2_power,
                one_minus_beta_2_t=1 - beta_2_t,
            )
        )

    def _resource_apply_dense(self, grad, var, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * coefficients["one_minus_beta_1_t"]
        m_t = m * coefficients["beta_1_t"] + m_scaled_g_values
        m_t = m.assign(m_t, use_locking=self._use_locking)
        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * coefficients["one_minus_beta_2_t"]
        v_t = v * coefficients["beta_2_t"] + v_scaled_g_values
        v_t = v.assign(v_t, use_locking=self._use_locking)

        m_t_hat = m_t / (1.0 - coefficients["beta_1_power"])
        v_t_hat = v_t / (1.0 - coefficients["beta_2_power"])

        v_sqrt = tf.sqrt(v_t_hat)
        update = m_t_hat / (v_sqrt + coefficients["epsilon"])

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            update += coefficients["weight_decay_rate"] * var

        ratio = 1.0
        if self._do_layer_adaptation(var_name):
            w_norm = tf.norm(var, ord=2)
            g_norm = tf.norm(update, ord=2)
            ratio = tf.where(
                tf.greater(w_norm, 0),
                tf.where(tf.greater(g_norm, 0), (w_norm / g_norm), 1.0),
                1.0,
            )

        var_update = var - ratio * coefficients["lr_t"] * update
        return var.assign(var_update, use_locking=self._use_locking).op

    def _resource_apply_sparse(self, grad, var, indices, apply_state=None):
        var_device, var_dtype = var.device, var.dtype.base_dtype
        coefficients = (apply_state or {}).get(
            (var_device, var_dtype)
        ) or self._fallback_apply_state(var_device, var_dtype)

        # m_t = beta1 * m + (1 - beta1) * g_t
        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * coefficients["one_minus_beta_1_t"]
        m_t = m.assign(m * coefficients["beta_1_t"], use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)

        # v_t = beta2 * v + (1 - beta2) * (g_t * g_t)
        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * coefficients["one_minus_beta_2_t"]
        v_t = v.assign(v * coefficients["beta_2_t"], use_locking=self._use_locking)
        with tf.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        m_t_hat = m_t / (1.0 - coefficients["beta_1_power"])
        v_t_hat = v_t / (1.0 - coefficients["beta_2_power"])

        v_sqrt = tf.sqrt(v_t_hat)
        update = m_t_hat / (v_sqrt + coefficients["epsilon"])

        var_name = self._get_variable_name(var.name)
        if self._do_use_weight_decay(var_name):
            update += coefficients["weight_decay_rate"] * var

        ratio = 1.0
        if self._do_layer_adaptation(var_name):
            w_norm = tf.norm(var, ord=2)
            g_norm = tf.norm(update, ord=2)
            ratio = tf.where(
                tf.greater(w_norm, 0),
                tf.where(tf.greater(g_norm, 0), (w_norm / g_norm), 1.0),
                1.0,
            )

        var_update = var.assign_sub(
            ratio * coefficients["lr_t"] * update, use_locking=self._use_locking
        )
        return tf.group(*[var_update, m_t, v_t])

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "weight_decay_rate": self._serialize_hyperparameter(
                    "weight_decay_rate"
                ),
                "decay": self._serialize_hyperparameter("decay"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "epsilon": self.epsilon,
            }
        )
        return config

    def _do_use_weight_decay(self, param_name):
        """Whether to use L2 weight decay for `param_name`."""
        if self.exclude_from_weight_decay:
            for r in self.exclude_from_weight_decay:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _do_layer_adaptation(self, param_name):
        """Whether to do layer-wise learning rate adaptation for
        `param_name`."""
        if self.exclude_from_layer_adaptation:
            for r in self.exclude_from_layer_adaptation:
                if re.search(r, param_name) is not None:
                    return False
        return True

    def _get_variable_name(self, param_name):
        """Get the variable name from the tensor name."""
        m = re.match("^(.*):\\d+$", param_name)
        if m is not None:
            param_name = m.group(1)
        return param_name
    

class RectifiedAdam(tf.keras.optimizers.Optimizer):
    """Variant of the Adam optimizer whose adaptive learning rate is rectified
    so as to have a consistent variance.
    It implements the Rectified Adam (a.k.a. RAdam) proposed by
    Liyuan Liu et al. in [On The Variance Of The Adaptive Learning Rate
    And Beyond](https://arxiv.org/pdf/1908.03265v1.pdf).
    Example of usage:
    ```python
    opt = tfa.optimizers.RectifiedAdam(lr=1e-3)
    ```
    Note: `amsgrad` is not described in the original paper. Use it with
          caution.
    RAdam is not a placement of the heuristic warmup, the settings should be
    kept if warmup has already been employed and tuned in the baseline method.
    You can enable warmup by setting `total_steps` and `warmup_proportion`:
    ```python
    opt = tfa.optimizers.RectifiedAdam(
        lr=1e-3,
        total_steps=10000,
        warmup_proportion=0.1,
        min_lr=1e-5,
    )
    ```
    In the above example, the learning rate will increase linearly
    from 0 to `lr` in 1000 steps, then decrease linearly from `lr` to `min_lr`
    in 9000 steps.
    Lookahead, proposed by Michael R. Zhang et.al in the paper
    [Lookahead Optimizer: k steps forward, 1 step back]
    (https://arxiv.org/abs/1907.08610v1), can be integrated with RAdam,
    which is announced by Less Wright and the new combined optimizer can also
    be called "Ranger". The mechanism can be enabled by using the lookahead
    wrapper. For example:
    ```python
    radam = tfa.optimizers.RectifiedAdam()
    ranger = tfa.optimizers.Lookahead(radam, sync_period=6, slow_step_size=0.5)
    ```
    """

    @typechecked
    def __init__(
        self,
        learning_rate: Union[FloatTensorLike, Callable] = 0.001,
        beta_1: FloatTensorLike = 0.9,
        beta_2: FloatTensorLike = 0.999,
        epsilon: FloatTensorLike = 1e-7,
        weight_decay: FloatTensorLike = 0.0,
        amsgrad: bool = False,
        sma_threshold: FloatTensorLike = 5.0,
        total_steps: int = 0,
        warmup_proportion: FloatTensorLike = 0.1,
        min_lr: FloatTensorLike = 0.0,
        name: str = "RectifiedAdam",
        **kwargs
    ):
        r"""Construct a new RAdam optimizer.
        Args:
            learning_rate: A `Tensor` or a floating point value. or a schedule
                that is a `tf.keras.optimizers.schedules.LearningRateSchedule`
                The learning rate.
            beta_1: A float value or a constant float tensor.
                The exponential decay rate for the 1st moment estimates.
            beta_2: A float value or a constant float tensor.
                The exponential decay rate for the 2nd moment estimates.
            epsilon: A small constant for numerical stability.
            weight_decay: A floating point value. Weight decay for each param.
            amsgrad: boolean. Whether to apply AMSGrad variant of this
                algorithm from the paper "On the Convergence of Adam and
                beyond".
            sma_threshold. A float value.
                The threshold for simple mean average.
            total_steps: An integer. Total number of training steps.
                Enable warmup by setting a positive value.
            warmup_proportion: A floating point value.
                The proportion of increasing steps.
            min_lr: A floating point value. Minimum learning rate after warmup.
            name: Optional name for the operations created when applying
                gradients. Defaults to "RectifiedAdam".
            **kwargs: keyword arguments. Allowed to be {`clipnorm`,
                `clipvalue`, `lr`, `decay`}. `clipnorm` is clip gradients
                by norm; `clipvalue` is clip gradients by value, `decay` is
                included for backward compatibility to allow time inverse
                decay of learning rate. `lr` is included for backward
                compatibility, recommended to use `learning_rate` instead.
        """
        super().__init__(name, **kwargs)
        self._set_hyper("learning_rate", kwargs.get("lr", learning_rate))
        self._set_hyper("beta_1", beta_1)
        self._set_hyper("beta_2", beta_2)
        self._set_hyper("decay", self._initial_decay)
        self._set_hyper("weight_decay", weight_decay)
        self._set_hyper("sma_threshold", sma_threshold)
        self._set_hyper("total_steps", float(total_steps))
        self._set_hyper("warmup_proportion", warmup_proportion)
        self._set_hyper("min_lr", min_lr)
        self.epsilon = epsilon or tf.keras.backend.epsilon()
        self.amsgrad = amsgrad
        self._initial_weight_decay = weight_decay
        self._initial_total_steps = total_steps

    def _create_slots(self, var_list):
        for var in var_list:
            self.add_slot(var, "m")
        for var in var_list:
            self.add_slot(var, "v")
        if self.amsgrad:
            for var in var_list:
                self.add_slot(var, "vhat")

    def set_weights(self, weights):
        params = self.weights
        num_vars = int((len(params) - 1) / 2)
        if len(weights) == 3 * num_vars + 1:
            weights = weights[: len(params)]
        super().set_weights(weights)

    def _resource_apply_dense(self, grad, var):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        m = self.get_slot(var, "m")
        v = self.get_slot(var, "v")
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)

        if self._initial_total_steps > 0:
            total_steps = self._get_hyper("total_steps", var_dtype)
            warmup_steps = total_steps * self._get_hyper("warmup_proportion", var_dtype)
            min_lr = self._get_hyper("min_lr", var_dtype)
            decay_steps = tf.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr_t) / decay_steps
            lr_t = tf.where(
                local_step <= warmup_steps,
                lr_t * (local_step / warmup_steps),
                lr_t + decay_rate * tf.minimum(local_step - warmup_steps, decay_steps),
            )

        sma_inf = 2.0 / (1.0 - beta_2_t) - 1.0
        sma_t = sma_inf - 2.0 * local_step * beta_2_power / (1.0 - beta_2_power)

        m_t = m.assign(
            beta_1_t * m + (1.0 - beta_1_t) * grad, use_locking=self._use_locking
        )
        m_corr_t = m_t / (1.0 - beta_1_power)

        v_t = v.assign(
            beta_2_t * v + (1.0 - beta_2_t) * tf.square(grad),
            use_locking=self._use_locking,
        )
        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            v_corr_t = tf.sqrt(vhat_t / (1.0 - beta_2_power))
        else:
            vhat_t = None
            v_corr_t = tf.sqrt(v_t / (1.0 - beta_2_power))

        r_t = tf.sqrt(
            (sma_t - 4.0)
            / (sma_inf - 4.0)
            * (sma_t - 2.0)
            / (sma_inf - 2.0)
            * sma_inf
            / sma_t
        )

        sma_threshold = self._get_hyper("sma_threshold", var_dtype)
        var_t = tf.where(
            sma_t >= sma_threshold, r_t * m_corr_t / (v_corr_t + epsilon_t), m_corr_t
        )

        if self._initial_weight_decay > 0.0:
            var_t += self._get_hyper("weight_decay", var_dtype) * var

        var_update = var.assign_sub(lr_t * var_t, use_locking=self._use_locking)

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def _resource_apply_sparse(self, grad, var, indices):
        var_dtype = var.dtype.base_dtype
        lr_t = self._decayed_lr(var_dtype)
        beta_1_t = self._get_hyper("beta_1", var_dtype)
        beta_2_t = self._get_hyper("beta_2", var_dtype)
        epsilon_t = tf.convert_to_tensor(self.epsilon, var_dtype)
        local_step = tf.cast(self.iterations + 1, var_dtype)
        beta_1_power = tf.pow(beta_1_t, local_step)
        beta_2_power = tf.pow(beta_2_t, local_step)

        if self._initial_total_steps > 0:
            total_steps = self._get_hyper("total_steps", var_dtype)
            warmup_steps = total_steps * self._get_hyper("warmup_proportion", var_dtype)
            min_lr = self._get_hyper("min_lr", var_dtype)
            decay_steps = tf.maximum(total_steps - warmup_steps, 1)
            decay_rate = (min_lr - lr_t) / decay_steps
            lr_t = tf.where(
                local_step <= warmup_steps,
                lr_t * (local_step / warmup_steps),
                lr_t + decay_rate * tf.minimum(local_step - warmup_steps, decay_steps),
            )

        sma_inf = 2.0 / (1.0 - beta_2_t) - 1.0
        sma_t = sma_inf - 2.0 * local_step * beta_2_power / (1.0 - beta_2_power)

        m = self.get_slot(var, "m")
        m_scaled_g_values = grad * (1 - beta_1_t)
        m_t = m.assign(m * beta_1_t, use_locking=self._use_locking)
        with tf.control_dependencies([m_t]):
            m_t = self._resource_scatter_add(m, indices, m_scaled_g_values)
        m_corr_t = m_t / (1.0 - beta_1_power)

        v = self.get_slot(var, "v")
        v_scaled_g_values = (grad * grad) * (1 - beta_2_t)
        v_t = v.assign(v * beta_2_t, use_locking=self._use_locking)
        with tf.control_dependencies([v_t]):
            v_t = self._resource_scatter_add(v, indices, v_scaled_g_values)

        if self.amsgrad:
            vhat = self.get_slot(var, "vhat")
            vhat_t = vhat.assign(tf.maximum(vhat, v_t), use_locking=self._use_locking)
            v_corr_t = tf.sqrt(vhat_t / (1.0 - beta_2_power))
        else:
            vhat_t = None
            v_corr_t = tf.sqrt(v_t / (1.0 - beta_2_power))

        r_t = tf.sqrt(
            (sma_t - 4.0)
            / (sma_inf - 4.0)
            * (sma_t - 2.0)
            / (sma_inf - 2.0)
            * sma_inf
            / sma_t
        )

        sma_threshold = self._get_hyper("sma_threshold", var_dtype)
        var_t = tf.where(
            sma_t >= sma_threshold, r_t * m_corr_t / (v_corr_t + epsilon_t), m_corr_t
        )

        if self._initial_weight_decay > 0.0:
            var_t += self._get_hyper("weight_decay", var_dtype) * var

        with tf.control_dependencies([var_t]):
            var_update = self._resource_scatter_add(
                var, indices, tf.gather(-lr_t * var_t, indices)
            )

        updates = [var_update, m_t, v_t]
        if self.amsgrad:
            updates.append(vhat_t)
        return tf.group(*updates)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "learning_rate": self._serialize_hyperparameter("learning_rate"),
                "beta_1": self._serialize_hyperparameter("beta_1"),
                "beta_2": self._serialize_hyperparameter("beta_2"),
                "decay": self._serialize_hyperparameter("decay"),
                "weight_decay": self._serialize_hyperparameter("weight_decay"),
                "sma_threshold": self._serialize_hyperparameter("sma_threshold"),
                "epsilon": self.epsilon,
                "amsgrad": self.amsgrad,
                "total_steps": self._serialize_hyperparameter("total_steps"),
                "warmup_proportion": self._serialize_hyperparameter(
                    "warmup_proportion"
                ),
                "min_lr": self._serialize_hyperparameter("min_lr"),
            }
        )
        return config
    
    