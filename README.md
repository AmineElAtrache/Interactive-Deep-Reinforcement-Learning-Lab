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

## CI/CD

[![CI](https://github.com/OWNER/REPO/actions/workflows/ci.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/ci.yml)
[![CD](https://github.com/OWNER/REPO/actions/workflows/cd.yml/badge.svg)](https://github.com/OWNER/REPO/actions/workflows/cd.yml)

> Replace `OWNER/REPO` in the badge URLs with your GitHub username and repository name after pushing.

### Continuous Integration (CI)

Runs on every push and pull request to `main` / `master`:

- **Ruff** lint and format check
- **Compile** all Python modules
- **Pytest** smoke tests (env reset, model creation — no full training)

Run locally:

```bash
uv sync --group dev
uv run ruff check .
uv run ruff format --check .
uv run pytest tests/ -v
```

### Continuous Deployment (CD)

Runs on push to `main` / `master` and version tags (`v*`):

1. **Docker** — builds and pushes the app image to [GitHub Container Registry](https://docs.github.com/en/packages/working-with-a-github-packages-registry/working-with-the-container-registry) (`ghcr.io/<owner>/<repo>`)
2. **Verify** — confirms the Streamlit app loads after deploy

Pull and run the published image:

```bash
docker pull ghcr.io/OWNER/REPO:latest
docker run -p 8501:8501 ghcr.io/OWNER/REPO:latest
```

### Streamlit Community Cloud (optional)

You can also deploy for free without Docker:

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the repo — main file: `app.py`
4. Python version: 3.12 — install command: `uv sync` or use the Docker image above

## Setup

```bash
uv sync
```

## Launch the Lab

```bash
uv run streamlit run app.py
```

Open **http://localhost:8501** in your browser.

## Environments

- **CartPole-v1** — balance a pole on a cart (default, fast to train)
- **LunarLander-v2** — land a spacecraft (requires `box2d`, slower)

## CLI (Legacy)

Original command-line scripts are still available:

```bash
# Batch-train all three algorithms
uv run python train_models.py 25

# View TensorBoard metrics
uv run tensorboard --logdir=logs

# Load and render a saved model
uv run python load_model.py PPO 50000.zip
```

## Project Structure

```
├── app.py              # Streamlit interactive lab UI
├── config.py           # Paths, environments, defaults
├── pyproject.toml      # Project metadata & dependencies (uv)
├── Dockerfile          # Container image for CD
├── .github/workflows/  # CI & CD pipelines
├── tests/              # Smoke tests for CI
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
