"""Interactive Deep Reinforcement Learning Research Lab — Streamlit UI."""

import time
from datetime import timedelta
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import ALGORITHMS, CHART_COLORS, DEFAULT_HYPERPARAMS, ENVIRONMENTS, ensure_dirs
from core.comparison import ComparisonSession
from core.env_utils import make_env
from core.evaluator import evaluate_saved_model
from core.models import list_saved_models, load_model
from core.trainer import TrainingSession
from core.visualization import export_episode, run_agent_episode

ensure_dirs()

st.set_page_config(
    page_title="DRL Research Lab",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header { font-size: 2rem; font-weight: 700; margin-bottom: 0.2rem; }
    .sub-header { color: #666; margin-bottom: 1.5rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "training_session" not in st.session_state:
    st.session_state.training_session = TrainingSession()
if "comparison_session" not in st.session_state:
    st.session_state.comparison_session = ComparisonSession()
if "last_eval_result" not in st.session_state:
    st.session_state.last_eval_result = None


def plot_reward_curve(metrics, title="Reward vs Episodes", color="#3498db"):
    fig = go.Figure()
    if metrics.get("episodes") and metrics.get("episode_rewards"):
        fig.add_trace(
            go.Scatter(
                x=metrics["episodes"],
                y=metrics["episode_rewards"],
                mode="lines",
                name="Episode Reward",
                line=dict(color=color, width=1),
                opacity=0.5,
            )
        )
    if metrics.get("episodes") and metrics.get("moving_avg_rewards"):
        fig.add_trace(
            go.Scatter(
                x=metrics["episodes"][: len(metrics["moving_avg_rewards"])],
                y=metrics["moving_avg_rewards"],
                mode="lines",
                name="Moving Average",
                line=dict(color=color, width=2.5),
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title="Episode",
        yaxis_title="Reward",
        height=320,
        margin=dict(l=40, r=20, t=40, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_loss_curve(metrics):
    fig = go.Figure()
    if metrics.get("loss"):
        fig.add_trace(
            go.Scatter(
                y=metrics["loss"],
                mode="lines",
                name="Loss",
                line=dict(color="#9b59b6", width=2),
            )
        )
    fig.update_layout(
        title="Training Loss",
        xaxis_title="Update Step",
        yaxis_title="Loss",
        height=280,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def plot_epsilon_curve(metrics):
    fig = go.Figure()
    if metrics.get("epsilon"):
        fig.add_trace(
            go.Scatter(
                y=metrics["epsilon"],
                mode="lines",
                name="Epsilon",
                line=dict(color="#e67e22", width=2),
            )
        )
    fig.update_layout(
        title="Exploration Decay (ε)",
        xaxis_title="Step",
        yaxis_title="Epsilon",
        height=280,
        margin=dict(l=40, r=20, t=40, b=40),
    )
    return fig


def plot_comparison_rewards(result):
    fig = go.Figure()
    for algo in result.algorithms:
        m = result.metrics.get(algo, {})
        if m.get("episodes") and m.get("moving_avg_rewards"):
            fig.add_trace(
                go.Scatter(
                    x=m["episodes"][: len(m["moving_avg_rewards"])],
                    y=m["moving_avg_rewards"],
                    mode="lines",
                    name=algo,
                    line=dict(color=CHART_COLORS.get(algo, "#333"), width=2),
                )
            )
    fig.update_layout(
        title="Algorithm Comparison — Moving Average Reward",
        xaxis_title="Episode",
        yaxis_title="Moving Avg Reward",
        height=380,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def render_training_progress(metrics, is_running: bool):
    """Live training panel — updates every 0.5s while training."""
    status = metrics.get("status", "idle")
    current = metrics.get("current_timestep", 0)
    total = metrics.get("total_timesteps", 1) or 1
    elapsed = metrics.get("elapsed_seconds", 0.0)
    rewards = metrics.get("episode_rewards", [])

    status_colors = {
        "idle": "⚪",
        "training": "🟢",
        "paused": "🟡",
        "completed": "✅",
        "stopped": "🟠",
        "error": "🔴",
    }
    st.info(f"{status_colors.get(status, '⚪')} Status: **{status.upper()}**")

    if metrics.get("error"):
        st.error(metrics["error"])

    pct = min(current / total, 1.0)
    st.progress(
        pct,
        text=(
            f"{'Training' if is_running else 'Progress'}: "
            f"{current:,} / {total:,} timesteps ({pct * 100:.1f}%)"
        ),
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Episodes", len(rewards))
    c2.metric("Latest Reward", f"{rewards[-1]:.1f}" if rewards else "—")
    c3.metric("Best Reward", f"{max(rewards):.1f}" if rewards else "—")
    c4.metric("Elapsed (s)", f"{elapsed:.0f}")

    if is_running and not rewards:
        st.caption(
            "Collecting first rollout — watch the timestep counter and progress bar update live."
        )
    recent = metrics.get("recent_rewards", [])
    if recent:
        st.caption(f"Recent episode rewards: {', '.join(f'{r:.1f}' for r in recent)}")

    algorithm = metrics.get("algorithm", "PPO")
    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        st.plotly_chart(
            plot_reward_curve(metrics, color=CHART_COLORS.get(algorithm, "#3498db")),
            width="stretch",
        )
    with chart_col2:
        if algorithm == "DQN" and metrics.get("epsilon"):
            st.plotly_chart(plot_epsilon_curve(metrics), width="stretch")
        elif metrics.get("loss"):
            st.plotly_chart(plot_loss_curve(metrics), width="stretch")
        elif is_running:
            st.caption("Loss / epsilon charts appear after the first gradient updates.")
        else:
            st.caption("Loss / epsilon data not recorded for this run.")


def analytics_from_metrics(metrics, success_threshold):
    rewards = metrics.get("episode_rewards", [])
    if not rewards:
        return {}
    import numpy as np

    best = float(max(rewards))
    avg = float(np.mean(rewards))
    conv_episode = next(
        (
            i + 1
            for i, r in enumerate(metrics.get("moving_avg_rewards", []))
            if r >= success_threshold * 0.9
        ),
        None,
    )
    success_rate = sum(1 for r in rewards if r >= success_threshold) / len(rewards)
    return {
        "best_reward": best,
        "avg_reward": avg,
        "total_episodes": len(rewards),
        "convergence_episode": conv_episode,
        "success_rate": success_rate,
        "elapsed_seconds": metrics.get("elapsed_seconds", 0),
    }


st.markdown(
    '<p class="main-header">🧠 Interactive Deep Reinforcement Learning Research Lab</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">Interactive · Visual · Multi-algorithm · Comparison-based · Demo-ready</p>',
    unsafe_allow_html=True,
)

tab_train, tab_compare, tab_eval, tab_watch, tab_dashboard = st.tabs(
    ["🎮 Train", "⚖️ Compare", "🧪 Evaluate", "🎬 Watch Agent", "📈 Dashboard"]
)

with st.sidebar:
    st.header("Configuration")
    env_name = st.selectbox("Environment", list(ENVIRONMENTS.keys()))
    env_id = ENVIRONMENTS[env_name]["id"]
    success_threshold = ENVIRONMENTS[env_name]["success_threshold"]

    st.divider()
    st.subheader("Hyperparameters")
    total_timesteps = st.number_input(
        "Total Timesteps",
        min_value=5_000,
        max_value=500_000,
        value=DEFAULT_HYPERPARAMS["total_timesteps"],
        step=5_000,
    )
    learning_rate = st.number_input(
        "Learning Rate",
        min_value=1e-5,
        max_value=1e-2,
        value=DEFAULT_HYPERPARAMS["learning_rate"],
        format="%.5f",
    )
    gamma = st.slider(
        "Gamma (γ)", min_value=0.90, max_value=0.999, value=DEFAULT_HYPERPARAMS["gamma"], step=0.001
    )

with tab_train:
    st.subheader("🎮 Interactive Training System")
    session = st.session_state.training_session

    col_cfg, col_ctrl = st.columns([2, 1])
    with col_cfg:
        algorithm = st.selectbox("Algorithm", ALGORITHMS, key="train_algo")
    with col_ctrl:
        st.write("")
        btn1, btn2, btn3 = st.columns(3)
        with btn1:
            start_train = st.button("▶️ Train", type="primary", width="stretch")
        with btn2:
            pause_train = st.button("⏸️ Pause", width="stretch")
        with btn3:
            stop_train = st.button("⏹️ Stop", width="stretch")

    if start_train and not session.is_running:
        session.start(
            algorithm=algorithm,
            env_id=env_id,
            total_timesteps=int(total_timesteps),
            learning_rate=float(learning_rate),
            gamma=float(gamma),
        )
    if pause_train and session.is_running:
        if session.metrics["status"] == "paused":
            session.resume()
        else:
            session.pause()
    if stop_train:
        session.stop()

    @st.fragment(run_every=timedelta(milliseconds=500))
    def live_training_panel():
        metrics = session.get_snapshot()
        show = (
            session.is_running
            or metrics.get("status") in ("training", "paused")
            or metrics.get("episode_rewards")
            or metrics.get("current_timestep", 0) > 0
        )
        if show:
            render_training_progress(metrics, session.is_running)

    live_training_panel()

    st.divider()
    st.subheader("💾 Save Model")
    save_col1, save_col2 = st.columns([2, 1])
    with save_col1:
        save_label = st.text_input("Model name (optional)", key="save_label")
    with save_col2:
        st.write("")
        if st.button("💾 Save Trained Model", width="stretch"):
            saved = session.save_current_model(save_label or None)
            if saved:
                st.success(f"Model saved: `models/{algorithm}/{saved}`")
            else:
                st.warning("No trained model available to save.")

    saved_models = list_saved_models(algorithm)
    if saved_models.get(algorithm):
        st.caption(f"Existing {algorithm} models: {', '.join(saved_models[algorithm][-5:])}")

with tab_compare:
    st.subheader("⚖️ Algorithm Comparison Mode")
    st.caption("Run DQN, PPO, and A2C on the same environment and compare performance.")

    cmp_session = st.session_state.comparison_session
    cmp_algos = st.multiselect("Algorithms to compare", ALGORITHMS, default=ALGORITHMS)

    c1, c2, c3 = st.columns(3)
    with c1:
        cmp_start = st.button("▶️ Run Comparison", type="primary")
    with c2:
        cmp_pause = st.button("⏸️ Pause Comparison")
    with c3:
        cmp_stop = st.button("⏹️ Stop Comparison")

    if cmp_start and cmp_algos and not cmp_session.is_running:
        cmp_session.start(
            env_id=env_id,
            algorithms=cmp_algos,
            total_timesteps=int(total_timesteps),
            learning_rate=float(learning_rate),
            gamma=float(gamma),
        )
    if cmp_pause and cmp_session.is_running:
        if cmp_session._pause_event.is_set():
            cmp_session.resume()
        else:
            cmp_session.pause()
    if cmp_stop:
        cmp_session.stop()

    @st.fragment(run_every=timedelta(milliseconds=500))
    def live_comparison_panel():
        result = cmp_session.get_result()
        if result.current_algorithm:
            st.info(f"Currently training: **{result.current_algorithm}**")

        if result.metrics:
            for algo, m in result.metrics.items():
                if cmp_session.is_running and algo == result.current_algorithm:
                    current = m.get("current_timestep", 0)
                    total = result.total_timesteps or 1
                    st.progress(
                        min(current / total, 1.0),
                        text=f"{algo}: {current:,} / {total:,} timesteps",
                    )

            st.plotly_chart(plot_comparison_rewards(result), width="stretch")

            table_data = result.summary_table()
            if table_data:
                st.subheader("Comparison Results")
                st.dataframe(pd.DataFrame(table_data), width="stretch", hide_index=True)

                best_avg = result.best_average_algorithm()
                fastest = result.fastest_algorithm()
                if best_avg or fastest:
                    r1, r2 = st.columns(2)
                    r1.metric("Best Average Reward", best_avg or "—")
                    r2.metric("Fastest Training", fastest or "—")

            if result.saved_models:
                st.caption(
                    "Saved comparison models: "
                    + ", ".join(f"{k}: {v}" for k, v in result.saved_models.items())
                )

    if cmp_session.is_running or cmp_session.get_result().metrics:
        live_comparison_panel()

with tab_eval:
    st.subheader("🧪 Evaluation Mode")
    st.caption("Test a trained agent without further training.")

    eval_algo = st.selectbox("Algorithm", ALGORITHMS, key="eval_algo")
    available = list_saved_models(eval_algo).get(eval_algo, [])

    if not available:
        st.warning(f"No saved {eval_algo} models found. Train and save a model first.")
    else:
        eval_model_file = st.selectbox("Model file", available)
        n_eval_episodes = st.slider("Evaluation Episodes", 1, 50, 10)

        if st.button("🧪 Run Evaluation", type="primary"):
            with st.spinner("Evaluating agent..."):
                eval_result = evaluate_saved_model(
                    eval_algo, eval_model_file, env_id, n_eval_episodes
                )
                st.session_state.last_eval_result = eval_result

        if st.session_state.last_eval_result:
            er = st.session_state.last_eval_result
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Mean Reward", f"{er.mean_reward:.2f}")
            e2.metric("Std Dev", f"{er.std_reward:.2f}")
            e3.metric("Best Episode", f"{er.best_reward:.2f}")
            e4.metric("Success Rate", f"{er.success_rate * 100:.1f}%")

            st.line_chart(pd.DataFrame({"Reward": er.episode_rewards}))

with tab_watch:
    st.subheader("🎬 Agent Visualization")
    st.caption("Watch a trained agent play and export episodes as GIF or MP4.")

    watch_algo = st.selectbox("Algorithm", ALGORITHMS, key="watch_algo")
    watch_models = list_saved_models(watch_algo).get(watch_algo, [])

    if not watch_models:
        st.warning("No saved models available. Train and save a model first.")
    else:
        watch_file = st.selectbox("Model", watch_models, key="watch_file")
        export_fmt = st.radio("Export format", ["gif", "mp4"], horizontal=True)

        w1, w2 = st.columns(2)
        with w1:
            run_watch = st.button("▶️ Play Episode", type="primary")
        with w2:
            export_watch = st.button("💾 Save Episode Video")

        if run_watch or export_watch:
            env = make_env(env_id)
            model = load_model(watch_algo, watch_file, env=env)
            frames, reward, steps = run_agent_episode(model, env_id)
            env.close()

            st.success(f"Episode finished — Reward: **{reward:.1f}**, Steps: **{steps}**")

            if frames:
                frame_slot = st.empty()
                for i, frame in enumerate(frames):
                    frame_slot.image(frame, caption=f"Step {i + 1}/{len(frames)}", width="stretch")
                    time.sleep(0.03)

            if export_watch and frames:
                path, _, _, _ = export_episode(
                    model, env_id, export_fmt, label=f"{watch_algo}_{env_id}"
                )
                st.download_button(
                    label=f"⬇️ Download {export_fmt.upper()}",
                    data=open(path, "rb").read(),
                    file_name=Path(path).name,
                    mime="video/mp4" if export_fmt == "mp4" else "image/gif",
                )
                st.caption(f"Saved to `{path}`")

with tab_dashboard:
    st.subheader("📈 Performance Analytics Dashboard")

    dash_source = st.radio(
        "Data source",
        ["Current Training Session", "Comparison Results"],
        horizontal=True,
    )

    if dash_source == "Current Training Session":
        dash_metrics = st.session_state.training_session.get_snapshot()
        stats = analytics_from_metrics(dash_metrics, success_threshold)
        if stats:
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("Best Episode Reward", f"{stats['best_reward']:.1f}")
            d2.metric("Average Reward", f"{stats['avg_reward']:.1f}")
            d3.metric("Total Episodes", stats["total_episodes"])
            d4.metric("Convergence Episode", stats["convergence_episode"] or "—")
            d5.metric("Success Rate", f"{stats['success_rate'] * 100:.1f}%")

            st.plotly_chart(
                plot_reward_curve(
                    dash_metrics,
                    title="Training Progress",
                    color=CHART_COLORS.get(dash_metrics.get("algorithm", "PPO"), "#3498db"),
                ),
                width="stretch",
            )
        else:
            st.info("Start training to see analytics here.")
    else:
        cmp_result = st.session_state.comparison_session.get_result()
        if cmp_result.metrics:
            rows = cmp_result.summary_table()
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)
            st.plotly_chart(plot_comparison_rewards(cmp_result), width="stretch")

            speeds = cmp_result.elapsed_seconds
            if speeds:
                speed_fig = go.Figure(
                    data=[
                        go.Bar(
                            x=list(speeds.keys()),
                            y=list(speeds.values()),
                            marker_color=[CHART_COLORS.get(a, "#333") for a in speeds],
                        )
                    ]
                )
                speed_fig.update_layout(title="Training Speed (seconds)", height=300)
                st.plotly_chart(speed_fig, width="stretch")
        else:
            st.info("Run a comparison to see analytics here.")
