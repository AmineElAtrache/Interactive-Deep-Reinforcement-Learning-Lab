import copy
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from core.callbacks import LiveMetricsCallback
from core.env_utils import make_env
from core.models import create_model, save_model


@dataclass
class ComparisonResult:
    environment: str
    total_timesteps: int
    algorithms: List[str] = field(default_factory=list)
    metrics: Dict[str, dict] = field(default_factory=dict)
    elapsed_seconds: Dict[str, float] = field(default_factory=dict)
    saved_models: Dict[str, str] = field(default_factory=dict)
    status: str = "idle"
    error: Optional[str] = None
    current_algorithm: Optional[str] = None

    def summary_table(self) -> List[dict]:
        rows = []
        for algo in self.algorithms:
            m = self.metrics.get(algo, {})
            rewards = m.get("episode_rewards", [])
            if not rewards:
                rows.append(
                    {
                        "Algorithm": algo,
                        "Avg Reward": "-",
                        "Best Reward": "-",
                        "Stability (std)": "-",
                        "Episodes": 0,
                        "Time (s)": round(self.elapsed_seconds.get(algo, 0), 1),
                    }
                )
                continue
            rows.append(
                {
                    "Algorithm": algo,
                    "Avg Reward": round(float(np.mean(rewards)), 2),
                    "Best Reward": round(float(max(rewards)), 2),
                    "Stability (std)": round(float(np.std(rewards)), 2),
                    "Episodes": len(rewards),
                    "Time (s)": round(self.elapsed_seconds.get(algo, 0), 1),
                }
            )
        return rows

    def fastest_algorithm(self) -> Optional[str]:
        if not self.elapsed_seconds:
            return None
        return min(self.elapsed_seconds, key=self.elapsed_seconds.get)

    def best_average_algorithm(self) -> Optional[str]:
        best_algo, best_avg = None, float("-inf")
        for algo in self.algorithms:
            rewards = self.metrics.get(algo, {}).get("episode_rewards", [])
            if rewards:
                avg = float(np.mean(rewards))
                if avg > best_avg:
                    best_avg = avg
                    best_algo = algo
        return best_algo


def empty_algo_metrics() -> dict:
    return {
        "episode_rewards": [],
        "moving_avg_rewards": [],
        "episodes": [],
        "recent_rewards": [],
        "loss": [],
        "epsilon": [],
        "timesteps": [],
        "current_timestep": 0,
    }


class ComparisonSession:
    def __init__(self):
        self.result = ComparisonResult(environment="", total_timesteps=0)
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(
        self,
        env_id: str,
        algorithms: List[str],
        total_timesteps: int,
        learning_rate: float,
        gamma: float,
        chunk_size: int = 512,
        save_models: bool = True,
    ):
        if self.is_running:
            raise RuntimeError("Comparison is already running.")

        self._stop_event.clear()
        self._pause_event.clear()
        self.result = ComparisonResult(
            environment=env_id,
            total_timesteps=total_timesteps,
            algorithms=algorithms,
            status="running",
        )

        def _run():
            try:
                for algo in algorithms:
                    if self._stop_event.is_set():
                        break

                    self.result.current_algorithm = algo
                    self.result.metrics[algo] = empty_algo_metrics()
                    env = make_env(env_id)
                    model = create_model(algo, env, learning_rate, gamma, f"compare_{algo}")
                    callback = LiveMetricsCallback(
                        self.result.metrics[algo],
                        self._stop_event,
                        self._pause_event,
                        self._lock,
                    )

                    start = time.time()
                    remaining = total_timesteps
                    while remaining > 0 and not self._stop_event.is_set():
                        step = min(chunk_size, remaining)
                        model.learn(
                            total_timesteps=step,
                            callback=callback,
                            reset_num_timesteps=True,
                            tb_log_name=f"compare_{algo}_{env_id}",
                        )
                        remaining -= step

                    self.result.elapsed_seconds[algo] = time.time() - start
                    if save_models and not self._stop_event.is_set():
                        self.result.saved_models[algo] = save_model(
                            model, algo, f"compare_{env_id}_{algo}", env_id=env_id
                        )
                    env.close()

                self.result.status = "stopped" if self._stop_event.is_set() else "completed"
            except Exception as exc:
                self.result.error = str(exc)
                self.result.status = "error"
            finally:
                self.result.current_algorithm = None

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def pause(self):
        self._pause_event.set()

    def resume(self):
        self._pause_event.clear()

    def stop(self):
        self._stop_event.set()
        self._pause_event.clear()

    def get_result(self) -> ComparisonResult:
        return copy.deepcopy(self.result)
