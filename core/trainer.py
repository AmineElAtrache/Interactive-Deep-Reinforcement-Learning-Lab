import threading
import time
from typing import Callable, Optional

from core.callbacks import LiveMetricsCallback
from core.models import create_model, make_env, save_model


def empty_metrics() -> dict:
    return {
        "episode_rewards": [],
        "moving_avg_rewards": [],
        "episodes": [],
        "loss": [],
        "epsilon": [],
        "timesteps": [],
        "status": "idle",
        "error": None,
        "saved_model": None,
        "elapsed_seconds": 0.0,
        "algorithm": None,
        "environment": None,
    }


class TrainingSession:
    """Runs SB3 training in a background thread with pause/stop control."""

    def __init__(self):
        self.metrics = empty_metrics()
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._model = None
        self._env = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def model(self):
        return self._model

    def reset_metrics(self, algorithm: str, environment: str):
        with self._lock:
            self.metrics = empty_metrics()
            self.metrics["algorithm"] = algorithm
            self.metrics["environment"] = environment

    def start(
        self,
        algorithm: str,
        env_id: str,
        total_timesteps: int,
        learning_rate: float,
        gamma: float,
        chunk_size: int = 2048,
        on_complete: Optional[Callable] = None,
    ):
        if self.is_running:
            raise RuntimeError("Training is already running.")

        self._stop_event.clear()
        self._pause_event.clear()
        self.reset_metrics(algorithm, env_id)

        def _train():
            start_time = time.time()
            try:
                self._env = make_env(env_id)
                run_name = f"{algorithm}_{env_id}"
                self._model = create_model(
                    algorithm, self._env, learning_rate, gamma, run_name
                )
                callback = LiveMetricsCallback(
                    self.metrics, self._stop_event, self._pause_event
                )

                remaining = total_timesteps
                self.metrics["status"] = "training"

                while remaining > 0 and not self._stop_event.is_set():
                    step = min(chunk_size, remaining)
                    self._model.learn(
                        total_timesteps=step,
                        callback=callback,
                        reset_num_timesteps=False,
                        tb_log_name=run_name,
                    )
                    remaining -= step

                self.metrics["elapsed_seconds"] = time.time() - start_time
                if self._stop_event.is_set():
                    self.metrics["status"] = "stopped"
                else:
                    self.metrics["status"] = "completed"
            except Exception as exc:
                self.metrics["error"] = str(exc)
                self.metrics["status"] = "error"
            finally:
                if self._env is not None:
                    self._env.close()
                    self._env = None
                if on_complete:
                    on_complete()

        self._thread = threading.Thread(target=_train, daemon=True)
        self._thread.start()

    def pause(self):
        self._pause_event.set()
        if self.metrics["status"] == "training":
            self.metrics["status"] = "paused"

    def resume(self):
        self._pause_event.clear()
        if self.metrics["status"] == "paused":
            self.metrics["status"] = "training"

    def stop(self):
        self._stop_event.set()
        self._pause_event.clear()

    def save_current_model(self, label: Optional[str] = None) -> Optional[str]:
        if self._model is None or self.metrics["algorithm"] is None:
            return None
        saved = save_model(self._model, self.metrics["algorithm"], label)
        self.metrics["saved_model"] = saved
        return saved

    def get_snapshot(self) -> dict:
        with self._lock:
            return {key: (list(val) if isinstance(val, list) else val) for key, val in self.metrics.items()}
