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
        lock: threading.Lock,
        moving_avg_window: int = 20,
        log_every: int = 128,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.metrics = metrics
        self.stop_event = stop_event
        self.pause_event = pause_event
        self._lock = lock
        self.moving_avg_window = moving_avg_window
        self.log_every = log_every
        self._reward_window = deque(maxlen=moving_avg_window)
        self._prev_buffer_len = 0

    def _on_step(self) -> bool:
        if self.stop_event.is_set():
            return False

        while self.pause_event.is_set() and not self.stop_event.is_set():
            time.sleep(0.05)

        if self.stop_event.is_set():
            return False

        step = int(self.num_timesteps)
        with self._lock:
            self.metrics["current_timestep"] = step
            if step % self.log_every == 0:
                self.metrics["timesteps"].append(step)
                if hasattr(self.model, "exploration_rate"):
                    self.metrics["epsilon"].append(float(self.model.exploration_rate))
                loss_keys = ("train/loss", "train/policy_loss", "train/value_loss")
                for key in loss_keys:
                    if key in self.logger.name_to_value:
                        self.metrics["loss"].append(float(self.logger.name_to_value[key]))
                        break

        return True

    def _on_rollout_end(self) -> None:
        if not self.model.ep_info_buffer:
            return

        buffer = list(self.model.ep_info_buffer)
        if len(buffer) > self._prev_buffer_len:
            new_infos = buffer[self._prev_buffer_len :]
        elif buffer:
            new_infos = [buffer[-1]]
        else:
            new_infos = []
        self._prev_buffer_len = len(buffer)

        with self._lock:
            for info in new_infos:
                reward = float(info["r"])
                self.metrics["episode_rewards"].append(reward)
                self._reward_window.append(reward)
                self.metrics["episodes"].append(len(self.metrics["episode_rewards"]))
                if self._reward_window:
                    self.metrics["moving_avg_rewards"].append(float(np.mean(self._reward_window)))
                self.metrics["recent_rewards"] = list(self.metrics["episode_rewards"][-5:])
            self.metrics["status"] = "training"
