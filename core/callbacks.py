import threading
import time
from collections import deque

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class LiveMetricsCallback(BaseCallback):
    """Captures training metrics for live UI updates and supports pause/stop."""

    def __init__(
        self,
        metrics: dict,
        stop_event: threading.Event,
        pause_event: threading.Event,
        moving_avg_window: int = 20,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.metrics = metrics
        self.stop_event = stop_event
        self.pause_event = pause_event
        self.moving_avg_window = moving_avg_window
        self._reward_window = deque(maxlen=moving_avg_window)

    def _on_step(self) -> bool:
        if self.stop_event.is_set():
            return False

        while self.pause_event.is_set() and not self.stop_event.is_set():
            time.sleep(0.05)

        if self.stop_event.is_set():
            return False

        if hasattr(self.model, "exploration_rate"):
            self.metrics["epsilon"].append(float(self.model.exploration_rate))

        loss_keys = ("train/loss", "train/policy_loss", "train/value_loss")
        for key in loss_keys:
            if key in self.logger.name_to_value:
                self.metrics["loss"].append(float(self.logger.name_to_value[key]))
                break

        self.metrics["timesteps"].append(int(self.num_timesteps))
        return True

    def _on_rollout_end(self) -> None:
        if not self.model.ep_info_buffer:
            return

        episode_rewards = [float(info["r"]) for info in self.model.ep_info_buffer]
        for reward in episode_rewards:
            self.metrics["episode_rewards"].append(reward)
            self._reward_window.append(reward)
            self.metrics["episodes"].append(len(self.metrics["episode_rewards"]))
            if self._reward_window:
                self.metrics["moving_avg_rewards"].append(
                    float(np.mean(self._reward_window))
                )

        self.metrics["status"] = "training"
