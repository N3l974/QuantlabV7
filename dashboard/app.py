"""
Streamlit Dashboard for Quantlab V7.
Visualize meta-optimization results, profiles, and portfolios.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.meta_optimizer import load_profiles, MetaProfile
from strategies.registry import list_strategies


st.set_page_config(
    page_title="Quantlab V7 — Meta-Optimizer",
    page_icon="📈",
    layout="wide",
)


def load_results_files() -> list[str]:
    """Find all result JSON files."""
    results_dir = Path("results")
    if not results_dir.exists():
        return []
    return sorted(results_dir.glob("meta_profiles_*.json"), reverse=True)


def plot_profiles_scatter(profiles: list[MetaProfile]):
    """Scatter plot: Sharpe vs Max Drawdown, colored by strategy."""
    data = []
    for p in profiles:
        data.append({
            "Strategy": p.strategy_name,
            "Timeframe": p.timeframe,
            "Sharpe": p.metrics.get("sharpe", 0),
            "Max Drawdown": abs(p.metrics.get("max_drawdown", 0)) * 100,
            "Total Return": p.metrics.get("total_return", 0) * 100,
            "Score": p.score,
            "Reoptim": p.reoptim_frequency,
            "Window": p.training_window,
        })
    df = pd.DataFrame(data)

    fig = px.scatter(
        df, x="Max Drawdown", y="Sharpe",
        color="Strategy", size="Score",
        hover_data=["Timeframe", "Reoptim", "Window", "Total Return"],
        title="Meta-Profiles: Sharpe vs Max Drawdown",
        labels={"Max Drawdown": "Max Drawdown (%)", "Sharpe": "Sharpe Ratio"},
    )
    fig.update_layout(height=500)
    return fig


def plot_profiles_bar(profiles: list[MetaProfile], top_n: int = 15):
    """Bar chart of top profiles by composite score."""
    top = profiles[:top_n]
    labels = [f"{p.strategy_name[:15]}/{p.timeframe}" for p in top]
    scores = [p.score for p in top]

    fig = go.Figure(go.Bar(
        x=scores, y=labels, orientation="h",
        marker_color=px.colors.qualitative.Set2[:len(top)],
        text=[f"{s:.2f}" for s in scores],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Top {top_n} Profiles by Composite Score",
        xaxis_title="Composite Score",
        yaxis=dict(autorange="reversed"),
        height=max(400, top_n * 35),
    )
    return fig


def plot_strategy_distribution(profiles: list[MetaProfile]):
    """Pie chart of strategy distribution in valid profiles."""
    strats = [p.strategy_name for p in profiles]
    df = pd.DataFrame({"Strategy": strats})
    counts = df["Strategy"].value_counts().reset_index()
    counts.columns = ["Strategy", "Count"]

    fig = px.pie(counts, values="Count", names="Strategy",
                 title="Strategy Distribution in Valid Profiles")
    fig.update_layout(height=400)
    return fig


def plot_meta_param_analysis(profiles: list[MetaProfile]):
    """Analyze which meta-parameters lead to best scores."""
    data = []
    for p in profiles:
        data.append({
            "Reoptim Frequency": p.reoptim_frequency,
            "Training Window": p.training_window,
            "Bounds Scale": p.param_bounds_scale,
            "Optim Metric": p.optim_metric,
            "Score": p.score,
        })
    df = pd.DataFrame(data)

    fig = px.box(
        df, x="Reoptim Frequency", y="Score", color="Training Window",
        title="Score Distribution by Reoptim Frequency & Training Window",
    )
    fig.update_layout(height=450)
    return fig


def main():
    st.title("📈 Quantlab V7 — Meta-Optimizer Dashboard")
    st.markdown("**Meta-optimization results for BTC trading strategies**")

    # Sidebar
    st.sidebar.header("Configuration")
    result_files = load_results_files()

    if not result_files:
        st.warning("No results found. Run `python main.py optimize` first.")
        st.info("Expected results in `results/meta_profiles_*.json`")

        # Show demo mode
        st.sidebar.info("No data available — showing empty dashboard")
        return

    selected_file = st.sidebar.selectbox(
        "Select results file",
        result_files,
        format_func=lambda x: x.name,
    )

    profiles = load_profiles(str(selected_file))
    st.sidebar.metric("Total Profiles", len(profiles))

    # Filter
    min_score = st.sidebar.slider("Min Score", 0.0, 5.0, 0.0, 0.1)
    filtered = [p for p in profiles if p.score >= min_score]
    st.sidebar.metric("Filtered Profiles", len(filtered))

    strategy_filter = st.sidebar.multiselect(
        "Filter Strategies",
        list_strategies(),
        default=list_strategies(),
    )
    filtered = [p for p in filtered if p.strategy_name in strategy_filter]

    if not filtered:
        st.warning("No profiles match the current filters.")
        return

    # Main content
    tab1, tab2, tab3, tab4 = st.tabs([
        "Overview", "Profile Details", "Meta-Parameter Analysis", "Comparison"
    ])

    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        best = filtered[0]
        col1.metric("Best Score", f"{best.score:.3f}")
        col2.metric("Best Sharpe", f"{best.metrics.get('sharpe', 0):.2f}")
        col3.metric("Best Return", f"{best.metrics.get('total_return', 0):.1%}")
        col4.metric("Best MaxDD", f"{best.metrics.get('max_drawdown', 0):.1%}")

        st.plotly_chart(plot_profiles_bar(filtered), use_container_width=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.plotly_chart(plot_profiles_scatter(filtered), use_container_width=True)
        with col_b:
            st.plotly_chart(plot_strategy_distribution(filtered), use_container_width=True)

    with tab2:
        st.subheader("Top Profiles Detail")
        top_n = st.slider("Show top N", 5, 50, 10)
        for i, p in enumerate(filtered[:top_n]):
            with st.expander(f"#{i+1} — {p.strategy_name} / {p.timeframe} (Score: {p.score:.3f})"):
                col1, col2, col3 = st.columns(3)
                col1.markdown(f"""
                **Strategy:** {p.strategy_name}
                **Timeframe:** {p.timeframe}
                **Type:** {p.optim_metric}
                """)
                col2.markdown(f"""
                **Reoptim:** {p.reoptim_frequency}
                **Window:** {p.training_window}
                **Bounds Scale:** {p.param_bounds_scale:.1f}
                **Inner Trials:** {p.n_optim_trials}
                """)
                col3.markdown(f"""
                **Sharpe:** {p.metrics.get('sharpe', 0):.2f}
                **Sortino:** {p.metrics.get('sortino', 0):.2f}
                **Calmar:** {p.metrics.get('calmar', 0):.2f}
                **Max DD:** {p.metrics.get('max_drawdown', 0):.2%}
                **Return:** {p.metrics.get('total_return', 0):.2%}
                **Stability:** {p.metrics.get('stability', 0):.2f}
                **Trades:** {p.metrics.get('n_trades', 0)}
                **Win Rate:** {p.metrics.get('win_rate', 0):.1%}
                """)

    with tab3:
        st.subheader("Meta-Parameter Analysis")
        st.plotly_chart(plot_meta_param_analysis(filtered), use_container_width=True)

        # Heatmap: strategy x timeframe
        data_matrix = {}
        for p in filtered:
            key = (p.strategy_name, p.timeframe)
            if key not in data_matrix or p.score > data_matrix[key]:
                data_matrix[key] = p.score

        strats = sorted(set(p.strategy_name for p in filtered))
        tfs = sorted(set(p.timeframe for p in filtered))
        matrix = np.zeros((len(strats), len(tfs)))
        for i, s in enumerate(strats):
            for j, t in enumerate(tfs):
                matrix[i, j] = data_matrix.get((s, t), 0)

        fig = go.Figure(go.Heatmap(
            z=matrix, x=tfs, y=strats,
            colorscale="RdYlGn",
            text=np.round(matrix, 2),
            texttemplate="%{text}",
        ))
        fig.update_layout(
            title="Best Score: Strategy × Timeframe",
            height=max(300, len(strats) * 40),
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Compare Profiles")
        compare_indices = st.multiselect(
            "Select profiles to compare",
            range(len(filtered[:20])),
            default=[0, 1] if len(filtered) >= 2 else [0],
            format_func=lambda i: f"#{i+1} {filtered[i].strategy_name}/{filtered[i].timeframe}",
        )

        if compare_indices:
            compare_data = []
            for idx in compare_indices:
                p = filtered[idx]
                row = {
                    "Profile": f"#{idx+1} {p.strategy_name}/{p.timeframe}",
                    "Score": p.score,
                    **{k: v for k, v in p.metrics.items() if isinstance(v, (int, float))},
                    "Reoptim": p.reoptim_frequency,
                    "Window": p.training_window,
                }
                compare_data.append(row)
            st.dataframe(pd.DataFrame(compare_data), use_container_width=True)


if __name__ == "__main__":
    main()
