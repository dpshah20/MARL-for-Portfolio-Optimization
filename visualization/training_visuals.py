import argparse
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import pandas as pd


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def _shade_test_region(ax, df, date_col="date"):
    if "phase" not in df.columns:
        return
    test_df = df[df["phase"].astype(str).str.lower() == "test"]
    if test_df.empty:
        return
    start = test_df[date_col].min()
    end = test_df[date_col].max()
    ax.axvspan(start, end, alpha=0.12, color="orange", label="test-phase")


def plot_nav_and_cash(perf, out_dir):
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.plot(perf["date"], perf["nav"], color="tab:blue", label="NAV")
    ax1.set_title("NAV and Cash %")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("NAV", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(perf["date"], perf["cash_pct"], color="tab:red", alpha=0.65, label="cash_pct")
    ax2.set_ylabel("Cash %", color="tab:red")
    ax2.tick_params(axis="y", labelcolor="tab:red")

    _shade_test_region(ax1, perf)
    ax1.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "nav_cash.png"), dpi=140)
    plt.close(fig)


def plot_reward_decomposition(perf, out_dir):
    required = ["term_ret", "term_vol", "term_cvar", "term_mdd", "reward_raw", "reward"]
    if any(c not in perf.columns for c in required):
        return

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.plot(perf["date"], perf["term_ret"], label="term_ret", lw=1.1)
    ax.plot(perf["date"], -perf["term_vol"], label="-term_vol", lw=1.0)
    ax.plot(perf["date"], -perf["term_cvar"], label="-term_cvar", lw=1.0)
    ax.plot(perf["date"], -perf["term_mdd"], label="-term_mdd", lw=1.0)
    ax.plot(perf["date"], perf["reward_raw"], label="reward_raw", lw=1.5, color="black")
    ax.plot(perf["date"], perf["reward"], label="reward_clipped", lw=1.3, color="tab:purple")
    _shade_test_region(ax, perf)
    ax.set_title("Reward Decomposition Through Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Reward Components")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "reward_decomposition.png"), dpi=140)
    plt.close(fig)


def plot_risk_metrics(perf, out_dir):
    cols = [c for c in ["vol_30d", "cvar_30d", "mdd_30d"] if c in perf.columns]
    if not cols:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    for c in cols:
        ax.plot(perf["date"], perf[c], label=c)
    _shade_test_region(ax, perf)
    ax.set_title("Risk Metrics")
    ax.set_xlabel("Date")
    ax.set_ylabel("Metric Value")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "risk_metrics.png"), dpi=140)
    plt.close(fig)


def plot_meta_policy(meta, out_dir):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    ax1.plot(meta["week_start_date"], meta["rho_cash"], label="rho_cash", color="tab:red")
    ax1.set_title("Meta Policy: Cash Exposure")
    ax1.set_ylabel("rho")
    ax1.grid(alpha=0.25)
    ax1.legend()

    for c in ["w_ret", "w_vol", "w_cvar", "w_mdd"]:
        if c in meta.columns:
            ax2.plot(meta["week_start_date"], meta[c], label=c)
    ax2.set_title("Meta Policy: Reward Weights")
    ax2.set_xlabel("Week")
    ax2.set_ylabel("Weight")
    ax2.grid(alpha=0.25)
    ax2.legend(ncol=4)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "meta_policy.png"), dpi=140)
    plt.close(fig)


def plot_training_losses(internal, out_dir):
    if not {"actor_loss", "critic_loss"}.issubset(internal.columns):
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(internal["step"], internal["actor_loss"], label="actor_loss")
    ax.plot(internal["step"], internal["critic_loss"], label="critic_loss")
    ax.set_title("Training Losses")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "losses.png"), dpi=140)
    plt.close(fig)


def plot_trade_activity(trades, out_dir):
    if trades.empty:
        return

    daily = trades.groupby("date").agg(
        trades=("ticker", "count"),
        gross_value=("value", lambda x: x.abs().sum()),
        realized_pnl=("realized_pnl", "sum") if "realized_pnl" in trades.columns else ("value", lambda x: 0.0),
    ).reset_index()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax1.bar(daily["date"], daily["trades"], color="tab:blue", alpha=0.7)
    ax1.set_title("Daily Number of Trades")
    ax1.set_ylabel("Trade Count")
    ax1.grid(alpha=0.25)

    ax2.plot(daily["date"], daily["gross_value"], label="gross traded value", color="tab:orange")
    if "realized_pnl" in daily.columns:
        ax2.plot(daily["date"], daily["realized_pnl"], label="realized pnl", color="tab:green", alpha=0.8)
    ax2.set_title("Trade Value and Realized PnL")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Amount")
    ax2.grid(alpha=0.25)
    ax2.legend()

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trade_activity.png"), dpi=140)
    plt.close(fig)


def plot_selection_frequency(run_dir, out_dir):
    freq_path = os.path.join(run_dir, "5_stock_selection_frequency.csv")
    if os.path.exists(freq_path):
        freq = pd.read_csv(freq_path)
    else:
        logs = _read_jsonl(os.path.join(run_dir, "execution_logs.jsonl"))
        counts = Counter()
        for row in logs:
            selected = row.get("selected_tickers", [])
            if not selected:
                alloc = row.get("allocations", {})
                selected = [t for t, w in alloc.items() if float(w) > 0.0]
            for t in selected:
                counts[t] += 1
        freq = pd.DataFrame({"ticker": list(counts.keys()), "selected_days": list(counts.values())})

    if freq.empty or "selected_days" not in freq.columns:
        return

    freq = freq.sort_values("selected_days", ascending=False).head(25)

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.barh(freq["ticker"][::-1], freq["selected_days"][::-1], color="tab:cyan")
    ax.set_title("Top 25 Most Frequently Selected Stocks")
    ax.set_xlabel("Selected Days")
    ax.grid(alpha=0.25, axis="x")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "selection_frequency_top25.png"), dpi=140)
    plt.close(fig)


def build_training_visuals(run_dir):
    out_dir = os.path.join(run_dir, "plots")
    _ensure_dir(out_dir)

    perf = pd.read_csv(os.path.join(run_dir, "2_daily_performance.csv"))
    perf["date"] = pd.to_datetime(perf["date"])

    meta = pd.read_csv(os.path.join(run_dir, "1_meta_strategy.csv"))
    meta["week_start_date"] = pd.to_datetime(meta["week_start_date"])

    internal_path = os.path.join(run_dir, "4_training_internals.csv")
    internal = pd.read_csv(internal_path) if os.path.exists(internal_path) else pd.DataFrame()

    trades_path = os.path.join(run_dir, "3_trade_history.csv")
    trades = pd.read_csv(trades_path) if os.path.exists(trades_path) else pd.DataFrame()
    if not trades.empty and "date" in trades.columns:
        trades["date"] = pd.to_datetime(trades["date"])

    plot_nav_and_cash(perf, out_dir)
    plot_reward_decomposition(perf, out_dir)
    plot_risk_metrics(perf, out_dir)
    plot_meta_policy(meta, out_dir)
    if not internal.empty:
        plot_training_losses(internal, out_dir)
    if not trades.empty:
        plot_trade_activity(trades, out_dir)
    plot_selection_frequency(run_dir, out_dir)

    print(f"[TrainingVisuals] Generated plots in: {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Generate training explainability plots for a run directory")
    parser.add_argument("--run_dir", required=True, help="Path to logs/run_YYYYMMDD_HHMMSS")
    args = parser.parse_args()

    if not os.path.isdir(args.run_dir):
        raise FileNotFoundError(f"Run directory not found: {args.run_dir}")

    build_training_visuals(args.run_dir)


if __name__ == "__main__":
    main()
