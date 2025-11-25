import copy
import pickle
import time

import numpy as np
import ray
import torch as T
from ray.actor import ActorProxy
from torch import optim
from torch.nn import functional as F
from typing_extensions import Any, Self

from mu_zero_smt.models import (
    dict_to_cpu,
    one_hot_encode,
    scalar_to_support,
    support_to_scalar,
)
from mu_zero_smt.replay_buffer import ReplayBuffer
from mu_zero_smt.shared_storage import SharedStorage
from mu_zero_smt.utils.config import MuZeroConfig


class Trainer:
    """
    Class which run in a dedicated thread to train a neural network and save it
    in the shared storage.
    """

    def __init__(
        self: Self, initial_checkpoint: dict[str, Any], config: MuZeroConfig
    ) -> None:
        self.config = config

        # Fix random generator seed
        np.random.seed(self.config.seed)
        T.manual_seed(self.config.seed)

        # Initialize the network
        self.model = self.config.network.from_config(self.config)
        self.model.load_state_dict(copy.deepcopy(initial_checkpoint["weights"]))
        self.model.to(T.device("cpu"))
        self.model.train()

        self.training_step = initial_checkpoint["training_step"]

        # Initialize the optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.lr_init,
            weight_decay=self.config.weight_decay,
        )

        if initial_checkpoint["optimizer_state"] is not None:
            print("Loading optimizer...\n")
            self.optimizer.load_state_dict(
                copy.deepcopy(initial_checkpoint["optimizer_state"])
            )

    @ray.method
    def continuous_update_weights(
        self: Self,
        replay_buffer: ActorProxy[ReplayBuffer],
        shared_storage: ActorProxy[SharedStorage],
    ) -> None:
        # Wait for the replay buffer to be filled
        while ray.get(shared_storage.get_info.remote("num_played_games")) < 1:
            time.sleep(0.1)

        # Training loop
        while self.training_step < self.config.training_steps and not ray.get(
            shared_storage.get_info.remote("terminate")
        ):
            index_batch, batch = ray.get(replay_buffer.get_batch.remote())

            self.update_lr()
            (
                priorities,
                total_loss,
                value_loss,
                reward_loss,
                policy_loss,
            ) = self.update_weights(batch)

            # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
            replay_buffer.update_priorities.remote(priorities, index_batch)

            # Save to the shared storage
            if self.training_step % self.config.checkpoint_interval == 0:
                shared_storage.set_info_batch.remote(
                    {
                        "weights": copy.deepcopy(dict_to_cpu(self.model.state_dict())),
                        "optimizer_state": copy.deepcopy(
                            dict_to_cpu(self.optimizer.state_dict())
                        ),
                    }
                )

                # Save model weights
                shared_storage.save_checkpoint.remote(None)

                # Save replay buffer
                replay_buffer_path = self.config.results_path / "replay_buffer.pkl"
                pickle.dump(
                    {
                        "buffer": ray.get(replay_buffer.get_buffer.remote()),
                        "num_played_games": ray.get(
                            shared_storage.get_info.remote("num_played_games")
                        ),
                        "num_played_steps": ray.get(
                            shared_storage.get_info.remote("num_played_steps")
                        ),
                        "num_reanalysed_games": ray.get(
                            shared_storage.get_info.remote("num_reanalysed_games")
                        ),
                    },
                    open(replay_buffer_path, "wb"),
                )

            shared_storage.set_info_batch.remote(
                {
                    "training_step": self.training_step,
                    "lr": self.optimizer.param_groups[0]["lr"],
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "reward_loss": reward_loss,
                    "policy_loss": policy_loss,
                }
            )

            # Managing the self-play / training ratio
            if self.config.training_delay:
                time.sleep(self.config.training_delay)
            if self.config.ratio:
                while (
                    self.training_step
                    / max(
                        1, ray.get(shared_storage.get_info.remote("num_played_steps"))
                    )
                    > self.config.ratio
                    and self.training_step < self.config.training_steps
                    and not ray.get(shared_storage.get_info.remote("terminate"))
                ):
                    time.sleep(0.5)

    def update_weights(self, batch):
        """
        Perform one training step.
        """

        (
            observation_batch,
            action_batch,
            param_batch,
            target_value,
            target_reward,
            target_policy,
            weight_batch,
            gradient_scale_batch,
        ) = batch

        # Keep values as scalars for calculating the priorities for the prioritized replay
        target_value_scalar = np.array(target_value, dtype="float32")
        priorities = np.zeros_like(target_value_scalar)

        device = next(self.model.parameters()).device

        weight_batch = T.tensor(weight_batch.copy()).float().to(device)
        observation_batch = T.tensor(np.array(observation_batch)).float().to(device)
        action_batch = T.tensor(action_batch).long().to(device).unsqueeze(-1)
        param_batch = T.tensor(np.array(param_batch), dtype=T.float32, device=device)
        target_value = T.tensor(target_value).float().to(device)
        target_reward = T.tensor(target_reward).float().to(device)
        target_policy = T.tensor(target_policy).float().to(device)
        gradient_scale_batch = T.tensor(gradient_scale_batch).float().to(device)
        # observation_batch: batch, channels, height, width
        # action_batch: batch, num_unroll_steps+1, 1 (unsqueeze)
        # target_value: batch, num_unroll_steps+1
        # target_reward: batch, num_unroll_steps+1
        # target_policy: batch, num_unroll_steps+1, len(action_space)
        # gradient_scale_batch: batch, num_unroll_steps+1

        target_value = scalar_to_support(target_value, self.config.support_size)
        target_reward = scalar_to_support(target_reward, self.config.support_size)
        # target_value: batch, num_unroll_steps+1, 2*support_size+1
        # target_reward: batch, num_unroll_steps+1, 2*support_size+1

        ## Generate predictions
        value, reward, policy_logits, hidden_state = self.model.initial_inference(
            observation_batch
        )

        policy = policy_logits[:, : self.config.discrete_action_space]
        continuous_params = policy_logits[:, self.config.discrete_action_space :]

        predictions = [(value, reward, policy, continuous_params)]
        for i in range(1, action_batch.shape[1]):
            value, reward, policy_logits, hidden_state = self.model.recurrent_inference(
                hidden_state,
                T.concat(
                    (
                        one_hot_encode(
                            action_batch[:, i], self.config.discrete_action_space
                        ),
                        param_batch[:, i],
                    ),
                    dim=1,
                ),
            )

            policy = policy_logits[:, : self.config.discrete_action_space]
            continuous_params = policy_logits[:, self.config.discrete_action_space :]

            # Scale the gradient at the start of the dynamics function (See paper appendix Training)
            hidden_state.register_hook(lambda grad: grad * 0.5)
            predictions.append((value, reward, policy, continuous_params))
        # predictions: num_unroll_steps+1, 3, batch, 2*support_size+1 | 2*support_size+1 | 9 (according to the 2nd dim)

        ## Compute losses
        value_loss, reward_loss, policy_loss = (0, 0, 0)
        value, reward, policy_logits, params = predictions[0]
        # Ignore reward loss for the first batch step
        current_value_loss, _, current_policy_loss = self.loss_function(
            value.squeeze(-1),
            reward.squeeze(-1),
            policy_logits,
            params,
            target_value[:, 0],
            target_reward[:, 0],
            target_policy[:, 0],
            param_batch[:, 0],
        )
        value_loss += current_value_loss
        policy_loss += current_policy_loss
        # Compute priorities for the prioritized replay (See paper appendix Training)
        pred_value_scalar = (
            support_to_scalar(value, self.config.support_size)
            .detach()
            .cpu()
            .numpy()
            .squeeze()
        )
        priorities[:, 0] = (
            np.abs(pred_value_scalar - target_value_scalar[:, 0])
            ** self.config.priority_alpha
        )

        for i in range(1, len(predictions)):
            value, reward, policy_logits, params = predictions[i]
            (
                current_value_loss,
                current_reward_loss,
                current_policy_loss,
            ) = self.loss_function(
                value.squeeze(-1),
                reward.squeeze(-1),
                policy_logits,
                params,
                target_value[:, i],
                target_reward[:, i],
                target_policy[:, i],
                param_batch[:, i],
            )

            # Scale gradient by the number of unroll steps (See paper appendix Training)
            current_value_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            current_reward_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )
            current_policy_loss.register_hook(
                lambda grad: grad / gradient_scale_batch[:, i]
            )

            value_loss += current_value_loss
            reward_loss += current_reward_loss
            policy_loss += current_policy_loss

            # Compute priorities for the prioritized replay (See paper appendix Training)
            pred_value_scalar = (
                support_to_scalar(value, self.config.support_size)
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            priorities[:, i] = (
                np.abs(pred_value_scalar - target_value_scalar[:, i])
                ** self.config.priority_alpha
            )

        # Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
        loss = value_loss * self.config.value_loss_weight + reward_loss + policy_loss

        # Correct PER bias by using importance-sampling (IS) weights
        loss *= weight_batch
        # Mean over batch dimension (pseudocode do a sum)
        loss = loss.mean()

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.training_step += 1

        return (
            priorities,
            # For log purpose
            loss.item(),
            value_loss.mean().item(),
            reward_loss.mean().item(),
            policy_loss.mean().item(),
        )

    def update_lr(self):
        """
        Update learning rate
        """
        lr = self.config.lr_init * self.config.lr_decay_rate ** (
            self.training_step / self.config.lr_decay_steps
        )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    @staticmethod
    def loss_function(
        value: T.Tensor,
        reward: T.Tensor,
        policy_logits: T.Tensor,
        params: T.Tensor,
        target_value: T.Tensor,
        target_reward: T.Tensor,
        target_policy: T.Tensor,
        target_params: T.Tensor,
    ) -> tuple[T.Tensor, T.Tensor, T.Tensor]:
        # Cross-entropy seems to have a better convergence than MSE
        value_loss = (-target_value * F.log_softmax(value, dim=1)).sum(1)
        reward_loss = (-target_reward * F.log_softmax(reward, dim=1)).sum(1)
        policy_loss = (-target_policy * F.log_softmax(policy_logits, dim=1)).sum(1)

        # If we have continuous parameters add it to the policy loss
        if params.numel() != 0:
            policy_loss += F.gaussian_nll_loss(params[:, 0], target_params, 1.0)

        return value_loss, reward_loss, policy_loss
