# 🧠 Interactive Deep Reinforcement Learning Research Lab

An interactive, visual, multi-algorithm reinforcement learning lab built on **Stable Baselines 3**. Train, compare, evaluate, and watch DQN, PPO, and A2C agents on classic control environments — all from a demo-ready web UI.

## Features

| Feature | Description |
|---------|-------------|
| 🎮 **Interactive Training** | Select environment, algorithm, and hyperparameters; train with one click |
| ▶️ **Live Progress** | Real-time reward curves, pause/stop controls |
| 🧠 **Multi-Algorithm** | DQN, PPO, A2C |
| 📊 **Live Visualization** | Reward, moving average, loss, epsilon decay |
| ⚖️ **Comparison Mode** | Run all algorithms on the same env; compare reward, stability, speed |
| 🎬 **Agent Visualization** | Watch trained agents play; export GIF/MP4 |
| 💾 **Model Save/Load** | Persist and reload trained models locally |
| 📈 **Analytics Dashboard** | Best reward, convergence, success rate |
| 🧪 **Evaluation Mode** | Test agents over multiple episodes without training |

## Prerequisites

- Python 3.8+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) package manager

## Setup

```bash
uv sync
```

## Launch the Lab

```bash
uv run lab
```

Open **http://localhost:8501** in your browser.

## Environments

- **CartPole-v1** — balance a pole on a cart (default, fast to train)
- **LunarLander-v2** — land a spacecraft (requires `box2d`, slower)

## CLI (Legacy)

Original command-line scripts are still available:

```bash
# Batch-train all three algorithms
uv run train 25

# View TensorBoard metrics
uv run metrics

# Load and render a saved model
uv run load PPO 50000.zip
```

## Project Structure

```
├── app.py              # Streamlit interactive lab UI
├── config.py           # Paths, environments, defaults
├── pyproject.toml      # Project metadata & dependencies (uv)
├── core/
│   ├── callbacks.py    # Live training metrics callback
│   ├── trainer.py      # Training session (pause/stop)
│   ├── comparison.py   # Multi-algorithm comparison
│   ├── evaluator.py    # Evaluation mode
│   ├── visualization.py# Agent rendering & video export
│   └── models.py       # Model create/load/save
├── train_models.py     # Legacy batch training
├── load_model.py       # Legacy model viewer
├── models/             # Saved model checkpoints
├── logs/               # TensorBoard logs
└── exports/            # Exported GIF/MP4 episodes
```

## Resources

1. [Stable Baselines 3 Documentation](https://stable-baselines3.readthedocs.io/)
2. [OpenAI Gym Environments](https://gym.openai.com/envs/)
3. [Streamlit Documentation](https://docs.streamlit.io/)
4. [uv Documentation](https://docs.astral.sh/uv/)
