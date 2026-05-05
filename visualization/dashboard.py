from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml

ROOT = Path(__file__).resolve().parents[1]
LOGS_DIR = ROOT / "logs"
PARAMS_PATH = ROOT / "configs" / "params.yaml"


def _to_datetime(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, errors="coerce")


def list_run_dirs(logs_dir: Path) -> list[Path]:
    if not logs_dir.exists():
        return []
    runs = [p for p in logs_dir.iterdir() if p.is_dir() and p.name.startswith("run_")]
    return sorted(runs, key=lambda p: p.name, reverse=True)


def load_daily_performance(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "2_daily_performance.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = _to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_trade_history(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "3_trade_history.csv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = _to_datetime(df["date"])
    return df.sort_values(["date", "step"], ascending=[False, False]).reset_index(drop=True)


def load_optional_csv(run_dir: Path, filename: str, date_col: str | None = None) -> pd.DataFrame:
    path = run_dir / filename
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if date_col and date_col in df.columns:
        df[date_col] = _to_datetime(df[date_col])
    return df


def load_execution_rows(run_dir: Path) -> list[dict]:
    path = run_dir / "execution_logs.jsonl"
    if not path.exists():
        return []

    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def parse_allocations(execution_rows: list[dict]) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for row in execution_rows:
        alloc = row.get("allocations")
        if not isinstance(alloc, dict):
            continue
        date = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(date):
            continue
        step = row.get("step")
        for ticker, weight in alloc.items():
            try:
                w = float(weight)
            except (TypeError, ValueError):
                continue
            if w <= 0.0:
                continue
            records.append({
                "date": date,
                "step": step,
                "ticker": str(ticker),
                "weight": w,
            })

    if not records:
        return pd.DataFrame(columns=["date", "step", "ticker", "weight"])

    df = pd.DataFrame(records)
    df = df.sort_values(["date", "step", "ticker"]).drop_duplicates(
        subset=["date", "ticker"], keep="last"
    )
    return df.reset_index(drop=True)


def extract_latest_allocation(execution_rows: list[dict]) -> tuple[pd.Timestamp | None, int | None, dict[str, float]]:
    latest_date = None
    latest_step = None
    latest_alloc: dict[str, float] = {}

    for row in execution_rows:
        alloc = row.get("allocations")
        if not isinstance(alloc, dict):
            continue

        step = row.get("step")
        date = pd.to_datetime(row.get("date"), errors="coerce")
        if pd.isna(date):
            continue

        if latest_date is None or date > latest_date or (date == latest_date and (step or -1) > (latest_step or -1)):
            latest_date = date
            latest_step = step
            latest_alloc = {str(k): float(v) for k, v in alloc.items() if float(v) > 0.0}

    return latest_date, latest_step, latest_alloc


def classify_trade_action(row: pd.Series) -> str:
    action = str(row.get("action", "")).strip().upper()
    if action in {"BUY", "SELL"}:
        return action
    shares = float(row.get("shares", 0.0))
    return "BUY" if shares >= 0 else "SELL"


def aggregate_trade_flow(trades: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if trades.empty:
        return pd.DataFrame(), pd.DataFrame()

    tdf = trades.copy()
    tdf["action_std"] = tdf.apply(classify_trade_action, axis=1)
    tdf["value_abs"] = tdf["value"].abs() if "value" in tdf.columns else 0.0

    daily = (
        tdf.groupby(["date", "action_std"], dropna=False)
        .agg(
            gross_value=("value_abs", "sum"),
            trade_count=("ticker", "count"),
            shares=("shares", lambda s: s.abs().sum()),
            realized_pnl=("realized_pnl", "sum") if "realized_pnl" in tdf.columns else ("ticker", "count"),
        )
        .reset_index()
    )

    pivot = (
        daily.pivot(index="date", columns="action_std", values="gross_value")
        .fillna(0.0)
        .rename(columns={"BUY": "buy_value", "SELL": "sell_value"})
    )
    if "buy_value" not in pivot.columns:
        pivot["buy_value"] = 0.0
    if "sell_value" not in pivot.columns:
        pivot["sell_value"] = 0.0
    pivot["net_flow"] = pivot["buy_value"] - pivot["sell_value"]

    ticker = (
        tdf.groupby(["ticker", "action_std"], dropna=False)["value_abs"]
        .sum()
        .unstack(fill_value=0.0)
        .rename(columns={"BUY": "buy_value", "SELL": "sell_value"})
    )
    if "buy_value" not in ticker.columns:
        ticker["buy_value"] = 0.0
    if "sell_value" not in ticker.columns:
        ticker["sell_value"] = 0.0
    ticker["net_flow"] = ticker["buy_value"] - ticker["sell_value"]
    ticker = ticker.sort_values("net_flow", ascending=False).reset_index()

    return pivot.sort_index(), ticker


def allocation_change_table(alloc_df: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp) -> pd.DataFrame:
    if alloc_df.empty:
        return pd.DataFrame()

    window_df = alloc_df[(alloc_df["date"] >= window_start) & (alloc_df["date"] <= window_end)].copy()
    if window_df.empty:
        return pd.DataFrame()

    first_day = window_df["date"].min()
    last_day = window_df["date"].max()

    start_alloc = (
        window_df[window_df["date"] == first_day][["ticker", "weight"]]
        .groupby("ticker", as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "start_weight"})
    )
    end_alloc = (
        window_df[window_df["date"] == last_day][["ticker", "weight"]]
        .groupby("ticker", as_index=False)["weight"]
        .sum()
        .rename(columns={"weight": "end_weight"})
    )

    merged = start_alloc.merge(end_alloc, on="ticker", how="outer").fillna(0.0)
    merged["delta_weight"] = merged["end_weight"] - merged["start_weight"]
    merged["start_%"] = merged["start_weight"] * 100.0
    merged["end_%"] = merged["end_weight"] * 100.0
    merged["delta_%"] = merged["delta_weight"] * 100.0
    merged["abs_delta_%"] = merged["delta_%"].abs()

    merged = merged.sort_values("abs_delta_%", ascending=False).reset_index(drop=True)
    return merged


def estimate_totals(cash: float | None, cash_pct: float | None, nav: float | None) -> tuple[float | None, float | None]:
    if cash is None or cash_pct is None:
        return None, None
    if cash_pct <= 0.0 or cash_pct >= 1.0:
        return None, None

    total_value = cash / cash_pct
    invested_value = total_value - cash
    return total_value, invested_value


def make_holdings_df(allocation: dict[str, float], total_value: float | None) -> pd.DataFrame:
    rows = []
    for ticker, w in sorted(allocation.items(), key=lambda x: x[1], reverse=True):
        est_value = (w * total_value) if total_value is not None else None
        rows.append({
            "ticker": ticker,
            "weight": float(w),
            "weight_pct": float(w) * 100.0,
            "est_value": est_value,
        })
    return pd.DataFrame(rows)


def show_allocation_pie(holdings: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        holdings["weight"],
        labels=holdings["ticker"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax.axis("equal")
    st.pyplot(fig, clear_figure=True)


@st.cache_data(show_spinner=False)
def load_training_config(params_path: str) -> dict[str, Any]:
    path = Path(params_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    return cfg


@st.cache_data(show_spinner=True)
def load_price_panel_from_cfg(cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str], int]:
    tickers = list(cfg.get("tickers", []))
    if not tickers:
        return pd.DataFrame(), [], int(cfg.get("window_length", 126))

    parquet_paths: list[str] = []
    for t in tickers:
        p = ROOT / "nifty200" / f"{t}_merged.parquet"
        if p.exists():
            parquet_paths.append(str(p))
            continue
        matches = glob.glob(str(ROOT / "nifty200" / f"{t}*.parquet"))
        if matches:
            parquet_paths.append(matches[0])

    if not parquet_paths:
        return pd.DataFrame(), [], int(cfg.get("window_length", 126))

    close_frames: list[pd.DataFrame] = []
    used_tickers: list[str] = []
    for p in parquet_paths:
        try:
            df = pd.read_parquet(p, columns=["Date", "Close"])
        except Exception:
            continue

        if "Date" not in df.columns or "Close" not in df.columns:
            continue

        name = Path(p).stem
        ticker = name.replace("_merged", "")
        tmp = df[["Date", "Close"]].copy()
        tmp["Date"] = pd.to_datetime(tmp["Date"], errors="coerce")
        tmp = tmp.dropna(subset=["Date"]).drop_duplicates(subset=["Date"], keep="last")
        tmp = tmp.sort_values("Date")
        tmp = tmp.rename(columns={"Close": ticker})
        close_frames.append(tmp.set_index("Date"))
        used_tickers.append(ticker)

    if not close_frames:
        return pd.DataFrame(), [], int(cfg.get("window_length", 126))

    panel = pd.concat(close_frames, axis=1).sort_index()
    return panel, used_tickers, int(cfg.get("window_length", 126))


def build_corr_and_adj(
    close_window: pd.DataFrame,
    method: str,
    k: int,
    corr_thr: float,
    absolute: bool,
) -> tuple[np.ndarray, np.ndarray]:
    prices = close_window.to_numpy(dtype=np.float64).T
    n = prices.shape[0]
    if prices.shape[1] < 3 or n == 0:
        eye = np.eye(n, dtype=np.float64)
        return eye, eye

    rets = np.diff(prices, axis=1) / (prices[:, :-1] + 1e-8)
    rets = np.nan_to_num(rets, nan=0.0, posinf=0.0, neginf=0.0)

    std = np.std(rets, axis=1)
    active_idx = np.flatnonzero(std > 1e-12)

    corr_full = np.eye(n, dtype=np.float64)
    adj_full = np.eye(n, dtype=np.float64)
    if active_idx.size <= 1:
        return corr_full, adj_full

    active_rets = rets[active_idx]
    corr = np.corrcoef(active_rets)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr_for_graph = np.abs(corr) if absolute else corr

    sub_adj = np.zeros_like(corr_for_graph, dtype=np.float64)
    if method == "corr_threshold":
        sub_adj[corr_for_graph >= corr_thr] = 1.0
    else:
        nn_k = max(1, min(k, active_idx.size - 1))
        for i in range(active_idx.size):
            order = np.argsort(-corr_for_graph[i])
            neighbors = [j for j in order if j != i][:nn_k]
            sub_adj[i, neighbors] = 1.0

    np.fill_diagonal(sub_adj, 1.0)
    row_sum = sub_adj.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0.0] = 1.0
    sub_adj = sub_adj / row_sum

    for local_i, global_i in enumerate(active_idx):
        corr_full[global_i, active_idx] = corr_for_graph[local_i]
        adj_full[global_i, active_idx] = sub_adj[local_i]

    return corr_full, adj_full


def build_message_passing_features(adj: np.ndarray, returns_vec: np.ndarray) -> pd.DataFrame:
    x0 = np.nan_to_num(returns_vec.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    x1 = adj @ x0
    delta = x1 - x0
    return pd.DataFrame({
        "raw_return": x0,
        "gnn_agg_return": x1,
        "message_delta": delta,
        "abs_message_delta": np.abs(delta),
    })


def draw_stock_network(
    tickers: list[str],
    corr: np.ndarray,
    message_strength: np.ndarray,
    edge_threshold: float,
    max_nodes: int,
    top_edges: int,
) -> plt.Figure:
    n_all = len(tickers)
    if n_all == 0:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_title("No stocks available")
        ax.axis("off")
        return fig

    node_scores = np.nan_to_num(message_strength, nan=0.0, posinf=0.0, neginf=0.0)
    node_order = np.argsort(-node_scores)
    keep = node_order[: max(2, min(max_nodes, n_all))]
    keep_set = set(int(i) for i in keep)

    # Circle layout keeps deterministic ordering across time for easier comparison.
    theta = np.linspace(0, 2 * np.pi, len(keep), endpoint=False)
    coords = {int(idx): (float(np.cos(t)), float(np.sin(t))) for idx, t in zip(keep, theta)}

    edge_items: list[tuple[float, int, int]] = []
    for i in keep:
        for j in keep:
            if j <= i:
                continue
            w = float(corr[int(i), int(j)])
            if w >= edge_threshold:
                edge_items.append((w, int(i), int(j)))

    edge_items.sort(key=lambda x: x[0], reverse=True)
    edge_items = edge_items[:top_edges]

    fig, ax = plt.subplots(figsize=(9, 9))
    ax.set_facecolor("#f8f9fb")

    for w, i, j in edge_items:
        xi, yi = coords[i]
        xj, yj = coords[j]
        alpha = min(0.9, 0.2 + 0.8 * w)
        ax.plot([xi, xj], [yi, yj], color="#1f77b4", linewidth=1.0 + 2.5 * w, alpha=alpha)

    xs = [coords[i][0] for i in keep]
    ys = [coords[i][1] for i in keep]
    sizes = [120 + 900 * float(node_scores[i]) for i in keep]
    ax.scatter(xs, ys, s=sizes, c="#ef6c00", alpha=0.9, edgecolors="black", linewidths=0.5)

    for i in keep:
        x, y = coords[i]
        ax.text(x, y, tickers[i], fontsize=8, ha="center", va="center", color="black")

    ax.set_title("Correlation Knowledge Graph (Node Size = GNN Message Impact)")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    return fig


def render_graph_evolution_view() -> None:
    st.subheader("Stock Correlation Graph Over Time")
    st.caption(
        "Inspect the dynamic correlation graph used as GNN input. "
        "Node size reflects how much message passing changes each stock's short-term signal."
    )

    cfg = load_training_config(str(PARAMS_PATH))
    if not cfg:
        st.info("Could not load configs/params.yaml, so graph view is unavailable.")
        return

    graph_cfg = cfg.get("graph", {})
    default_mode = graph_cfg.get("mode", "knn")
    default_k = int(graph_cfg.get("k", 8))
    default_thr = float(graph_cfg.get("corr_threshold", 0.6))
    default_abs = bool(graph_cfg.get("absolute_corr", True))

    panel, tickers, window_len = load_price_panel_from_cfg(cfg)
    if panel.empty or len(tickers) < 2:
        st.info("No usable parquet panel found under nifty200/ for configured tickers.")
        return

    with st.expander("Graph Controls", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            method = st.selectbox("Graph mode", ["knn", "corr_threshold"], index=0 if default_mode == "knn" else 1)
        with c2:
            k = st.slider("K neighbors", min_value=1, max_value=20, value=max(1, min(default_k, 20)), step=1)
        with c3:
            corr_thr = st.slider("Correlation threshold", min_value=0.1, max_value=0.95, value=float(np.clip(default_thr, 0.1, 0.95)), step=0.05)
        with c4:
            absolute = st.checkbox("Use absolute correlation", value=default_abs)

        valid_dates = panel.dropna(axis=0, how="all").index
        if len(valid_dates) <= window_len:
            st.warning("Not enough dates to form rolling windows for graph view.")
            return

        end_date_candidates = valid_dates[window_len - 1 :]
        end_idx = st.slider("Window end date index", min_value=0, max_value=len(end_date_candidates) - 1, value=len(end_date_candidates) - 1, step=1)
        end_date = end_date_candidates[end_idx]

        d1, d2, d3 = st.columns(3)
        with d1:
            edge_threshold = st.slider("Edge display threshold", min_value=0.1, max_value=0.95, value=0.5, step=0.05)
        with d2:
            max_nodes = st.slider("Max nodes shown", min_value=10, max_value=min(80, len(tickers)), value=min(35, len(tickers)), step=5)
        with d3:
            top_edges = st.slider("Max edges shown", min_value=20, max_value=300, value=120, step=10)

    window_start = end_date - pd.Timedelta(days=window_len * 3)
    candidate = panel[(panel.index <= end_date) & (panel.index >= window_start)].tail(window_len)
    if len(candidate) < window_len:
        candidate = panel[panel.index <= end_date].tail(window_len)
    close_window = candidate.dropna(axis=1, thresh=max(5, int(window_len * 0.7)))

    if close_window.shape[1] < 2:
        st.warning("Too few stocks with enough data in selected window.")
        return

    selected_tickers = list(close_window.columns)
    corr, adj = build_corr_and_adj(close_window, method=method, k=k, corr_thr=corr_thr, absolute=absolute)

    returns_last = close_window.pct_change().iloc[-1].to_numpy(dtype=np.float64)
    msg_df = build_message_passing_features(adj, returns_last)
    msg_df.insert(0, "ticker", selected_tickers)

    m1, m2, m3, m4 = st.columns(4)
    off_diag = corr.copy()
    np.fill_diagonal(off_diag, np.nan)
    avg_corr = float(np.nanmean(off_diag)) if np.isfinite(np.nanmean(off_diag)) else 0.0
    edge_density = float(np.mean(off_diag >= edge_threshold)) if off_diag.size else 0.0
    mean_shift = float(msg_df["abs_message_delta"].mean()) if not msg_df.empty else 0.0
    max_shift = float(msg_df["abs_message_delta"].max()) if not msg_df.empty else 0.0
    m1.metric("Window End", str(pd.to_datetime(end_date).date()))
    m2.metric("Avg Pair Corr", f"{avg_corr:.3f}")
    m3.metric("Edge Density", f"{edge_density * 100:.1f}%")
    m4.metric("Avg Message Shift", f"{mean_shift:.4f}")

    fig = draw_stock_network(
        tickers=selected_tickers,
        corr=corr,
        message_strength=msg_df["abs_message_delta"].to_numpy(dtype=float),
        edge_threshold=edge_threshold,
        max_nodes=max_nodes,
        top_edges=top_edges,
    )
    st.pyplot(fig, clear_figure=True)

    st.write("Top Correlation Relationships in Selected Window")
    pairs: list[dict[str, Any]] = []
    n = len(selected_tickers)
    for i in range(n):
        for j in range(i + 1, n):
            w = float(corr[i, j])
            if w >= edge_threshold:
                pairs.append({"stock_a": selected_tickers[i], "stock_b": selected_tickers[j], "corr": w})
    pair_df = pd.DataFrame(pairs).sort_values("corr", ascending=False).head(50) if pairs else pd.DataFrame(columns=["stock_a", "stock_b", "corr"])
    st.dataframe(pair_df, use_container_width=True)

    st.write("GNN Message Passing View (Before vs After Neighbor Aggregation)")
    st.caption("Using one-step GraphSAGE-style aggregation: h^(1) = A h^(0).")
    show_msg = msg_df.sort_values("abs_message_delta", ascending=False).head(25)
    st.dataframe(show_msg, use_container_width=True)
    st.metric("Largest Message Shift", f"{max_shift:.4f}")

    st.write("Graph Evolution (Recent Windows)")
    max_hist = min(120, len(end_date_candidates))
    hist_end_dates = end_date_candidates[max(0, end_idx - max_hist + 1) : end_idx + 1]
    history_rows: list[dict[str, Any]] = []
    prev_binary: np.ndarray | None = None
    for d in hist_end_dates:
        tmp = panel[panel.index <= d].tail(window_len)
        tmp = tmp.dropna(axis=1, thresh=max(5, int(window_len * 0.7)))
        if tmp.shape[1] < 2:
            continue
        cmat, _ = build_corr_and_adj(tmp, method=method, k=k, corr_thr=corr_thr, absolute=absolute)
        off = cmat.copy()
        np.fill_diagonal(off, np.nan)
        avg_c = float(np.nanmean(off)) if np.isfinite(np.nanmean(off)) else 0.0
        binary = (cmat >= edge_threshold).astype(np.int8)
        np.fill_diagonal(binary, 0)
        density = float(binary.sum() / max(1, binary.size - len(binary)))
        stability = np.nan
        if prev_binary is not None and prev_binary.shape == binary.shape:
            union = np.logical_or(prev_binary, binary).sum()
            inter = np.logical_and(prev_binary, binary).sum()
            stability = float(inter / union) if union > 0 else 1.0
        prev_binary = binary
        history_rows.append({"date": d, "avg_corr": avg_c, "edge_density": density, "edge_stability": stability})

    if history_rows:
        hist_df = pd.DataFrame(history_rows).set_index("date")
        st.line_chart(hist_df[["avg_corr", "edge_density", "edge_stability"]], height=260)
    else:
        st.info("Not enough historical windows for evolution chart.")


def main() -> None:
    st.set_page_config(page_title="Portfolio Dashboard", layout="wide")
    st.title("Portfolio Dashboard")
    st.caption("Simple view of current investment state from training run logs.")

    run_dirs = list_run_dirs(LOGS_DIR)
    if not run_dirs:
        st.error("No run directories found under logs/. Run training first.")
        return

    run_names = [p.name for p in run_dirs]
    selected_name = st.sidebar.selectbox("Run", run_names, index=0)
    run_dir = LOGS_DIR / selected_name
    st.sidebar.caption(f"Using: {run_dir}")

    daily_df = load_daily_performance(run_dir)
    trade_df = load_trade_history(run_dir)
    execution_rows = load_execution_rows(run_dir)
    alloc_df = parse_allocations(execution_rows)
    meta_df = load_optional_csv(run_dir, "1_meta_strategy.csv", date_col="week_start_date")
    ticker_stats_df = load_optional_csv(run_dir, "6_ticker_trade_stats.csv")
    trade_summary_df = load_optional_csv(run_dir, "7_trade_summary.csv")
    selection_freq_df = load_optional_csv(run_dir, "5_stock_selection_frequency.csv")

    if daily_df.empty:
        st.error(f"Missing or empty 2_daily_performance.csv in {run_dir}")
        return

    latest = daily_df.iloc[-1]
    latest_date = latest.get("date")
    latest_nav = float(latest.get("nav")) if pd.notna(latest.get("nav")) else None
    latest_cash = float(latest.get("cash")) if pd.notna(latest.get("cash")) else None
    latest_cash_pct = float(latest.get("cash_pct")) if pd.notna(latest.get("cash_pct")) else None
    latest_daily_ret = float(latest.get("daily_return")) if pd.notna(latest.get("daily_return")) else None

    total_value, invested_value = estimate_totals(latest_cash, latest_cash_pct, latest_nav)

    alloc_date, alloc_step, alloc = extract_latest_allocation(execution_rows)
    holdings = make_holdings_df(alloc, total_value)

    st.subheader("Current Snapshot")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Date", str(latest_date.date()) if pd.notna(latest_date) else "N/A")
    m2.metric("NAV", f"{latest_nav:,.2f}" if latest_nav is not None else "N/A")
    m3.metric("Cash", f"{latest_cash:,.2f}" if latest_cash is not None else "N/A")
    m4.metric("Daily Return", f"{latest_daily_ret * 100:.2f}%" if latest_daily_ret is not None else "N/A")

    n1, n2, n3 = st.columns(3)
    n1.metric("Cash %", f"{latest_cash_pct * 100:.2f}%" if latest_cash_pct is not None else "N/A")
    n2.metric("Invested (Estimated)", f"{invested_value:,.2f}" if invested_value is not None else "N/A")
    n3.metric("Total Value (Estimated)", f"{total_value:,.2f}" if total_value is not None else "N/A")

    st.subheader("Allocation")
    if holdings.empty:
        st.warning("No allocation data found in execution logs.")
    else:
        st.caption(
            f"Latest allocation from execution logs: {alloc_date.date() if alloc_date is not None else 'N/A'}"
            f" (step {alloc_step if alloc_step is not None else 'N/A'})"
        )

        c1, c2 = st.columns([1, 1])
        with c1:
            st.write("Top Holdings")
            st.bar_chart(holdings.set_index("ticker")["weight_pct"])
        with c2:
            st.write("Allocation Mix")
            show_allocation_pie(holdings)

        display_df = holdings[["ticker", "weight_pct", "est_value"]].copy()
        display_df = display_df.rename(columns={
            "weight_pct": "weight_%",
            "est_value": "estimated_value",
        })
        st.dataframe(display_df, use_container_width=True)

    st.subheader("Last 1 Month: Allocation + Buy/Sell Changes")
    if daily_df["date"].notna().any():
        end_date = daily_df["date"].max()
        start_date = end_date - pd.DateOffset(months=1)
        st.caption(f"Window: {start_date.date()} to {end_date.date()} (based on latest run date)")

        daily_last = daily_df[(daily_df["date"] >= start_date) & (daily_df["date"] <= end_date)].copy()
        trade_last = trade_df[(trade_df["date"] >= start_date) & (trade_df["date"] <= end_date)].copy() if not trade_df.empty else pd.DataFrame()
        alloc_last = alloc_df[(alloc_df["date"] >= start_date) & (alloc_df["date"] <= end_date)].copy() if not alloc_df.empty else pd.DataFrame()

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Trading Days", f"{daily_last.shape[0]}")
        a2.metric("Window NAV Change", f"{(daily_last['nav'].iloc[-1] - daily_last['nav'].iloc[0]):,.2f}" if len(daily_last) > 1 else "N/A")
        a3.metric("Window Return", f"{((daily_last['nav'].iloc[-1] / daily_last['nav'].iloc[0] - 1) * 100):.2f}%" if len(daily_last) > 1 else "N/A")
        a4.metric("Trades in Window", f"{trade_last.shape[0]}")

        if not alloc_last.empty:
            alloc_change = allocation_change_table(alloc_df, start_date, end_date)

            left, right = st.columns([1.1, 1])
            with left:
                st.write("Top Allocation Changes (Start vs End)")
                show_cols = ["ticker", "start_%", "end_%", "delta_%"]
                st.dataframe(alloc_change[show_cols].head(20), use_container_width=True)

            with right:
                st.write("Largest Absolute Allocation Moves")
                top_moves = alloc_change.head(15).set_index("ticker")["delta_%"]
                st.bar_chart(top_moves)

            st.write("Allocation Trend (Top Movers in Last Month)")
            movers = alloc_change.head(8)["ticker"].tolist()
            alloc_ts = (
                alloc_last[alloc_last["ticker"].isin(movers)]
                .pivot_table(index="date", columns="ticker", values="weight", aggfunc="sum")
                .fillna(0.0)
            )
            if not alloc_ts.empty:
                st.area_chart(alloc_ts * 100.0, height=280)
        else:
            st.warning("No allocation entries found in execution logs for the last 1 month window.")

        if not trade_last.empty:
            flow_daily, flow_ticker = aggregate_trade_flow(trade_last)

            t1, t2, t3 = st.columns(3)
            t1.metric("Gross Buy Value", f"{flow_daily['buy_value'].sum():,.2f}" if not flow_daily.empty else "0")
            t2.metric("Gross Sell Value", f"{flow_daily['sell_value'].sum():,.2f}" if not flow_daily.empty else "0")
            t3.metric("Net Flow (Buy-Sell)", f"{flow_daily['net_flow'].sum():,.2f}" if not flow_daily.empty else "0")

            st.write("Daily Buy/Sell Points")
            st.line_chart(flow_daily[["buy_value", "sell_value", "net_flow"]], height=280)

            c1, c2 = st.columns([1.1, 1])
            with c1:
                st.write("Ticker Net Flow in Last Month")
                st.dataframe(flow_ticker.head(25), use_container_width=True)
            with c2:
                st.write("Top Net Buys")
                top_buys = flow_ticker.sort_values("net_flow", ascending=False).head(12)
                st.bar_chart(top_buys.set_index("ticker")["net_flow"])

            st.write("All Buy/Sell Trade Points (Last Month)")
            trade_points_cols = [c for c in ["date", "step", "ticker", "action", "shares", "price", "value", "commission", "realized_pnl"] if c in trade_last.columns]
            st.dataframe(
                trade_last.sort_values(["date", "step"], ascending=[False, False])[trade_points_cols],
                use_container_width=True,
                height=320,
            )
        else:
            st.warning("No trades found in the last 1 month window.")
    else:
        st.warning("Could not infer time window because date column is empty.")

    st.subheader("Performance Over Time")
    perf_cols = [c for c in ["date", "nav", "cash", "cash_pct", "daily_return", "reward"] if c in daily_df.columns]
    st.line_chart(daily_df[perf_cols].set_index("date"), height=320)

    if not trade_df.empty:
        st.subheader("Recent Trades")
        show_cols = [
            c for c in [
                "date", "step", "ticker", "action", "shares", "price", "value", "commission", "realized_pnl"
            ] if c in trade_df.columns
        ]
        st.dataframe(trade_df[show_cols].head(30), use_container_width=True)

    with st.expander("Run Summary Tables", expanded=False):
        s1, s2 = st.columns(2)
        with s1:
            st.write("Trade Summary")
            if not trade_summary_df.empty:
                st.dataframe(trade_summary_df, use_container_width=True)
            else:
                st.info("7_trade_summary.csv not found")

            st.write("Meta Strategy (latest 12 weeks)")
            if not meta_df.empty:
                st.dataframe(meta_df.tail(12), use_container_width=True)
            else:
                st.info("1_meta_strategy.csv not found")

        with s2:
            st.write("Ticker Trade Stats (Top by realized PnL)")
            if not ticker_stats_df.empty and "realized_pnl" in ticker_stats_df.columns:
                st.dataframe(
                    ticker_stats_df.sort_values("realized_pnl", ascending=False).head(20),
                    use_container_width=True,
                )
            elif not ticker_stats_df.empty:
                st.dataframe(ticker_stats_df.head(20), use_container_width=True)
            else:
                st.info("6_ticker_trade_stats.csv not found")

            st.write("Selection Frequency (Top 20)")
            if not selection_freq_df.empty:
                st.dataframe(selection_freq_df.head(20), use_container_width=True)
            else:
                st.info("5_stock_selection_frequency.csv not found")

    render_graph_evolution_view()


if __name__ == "__main__":
    main()
