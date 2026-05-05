"""
training/workforward.py
=======================
Walk-Forward Validation Orchestrator
-------------------------------------
Splits the full dataset (2015-01-01 → 2026-04-30) into rolling windows:
  - Train: 4 years  (always starting from MIN_DATE so the model sees the full history)
  - Test : 1 year   (out-of-sample)
  - Step : 1 year   (slide forward)

Windows generated:
  W1  Train 2015→2018  Test 2019
  W2  Train 2015→2019  Test 2020
  W3  Train 2015→2020  Test 2021
  W4  Train 2015→2021  Test 2022
  W5  Train 2015→2022  Test 2023
  W6  Train 2015→2023  Test 2024
  W7  Train 2015→2024  Test 2025
  W8  Train 2015→2025  Test Jan-Apr 2026  (partial year)

Usage:
  python -m training.workforward                   # run all windows
  python -m training.workforward --windows W1 W3   # run specific windows
  python -m training.workforward --summarize        # just print results from existing runs
  python -m training.workforward --smoke            # one window, smoke mode (quick test)
"""

import argparse
import csv
import os
import subprocess
import sys
import json
import math
from datetime import datetime

# ---------------------------------------------------------------------------
# Window schedule
# ---------------------------------------------------------------------------
MIN_DATE = "2015-01-01"

WINDOWS = [
    {"label": "W1_2019", "train_end": "2018-12-31", "test_start": "2019-01-01", "test_end": "2019-12-31"},
    {"label": "W2_2020", "train_end": "2019-12-31", "test_start": "2020-01-01", "test_end": "2020-12-31"},
    {"label": "W3_2021", "train_end": "2020-12-31", "test_start": "2021-01-01", "test_end": "2021-12-31"},
    {"label": "W4_2022", "train_end": "2021-12-31", "test_start": "2022-01-01", "test_end": "2022-12-31"},
    {"label": "W5_2023", "train_end": "2022-12-31", "test_start": "2023-01-01", "test_end": "2023-12-31"},
    {"label": "W6_2024", "train_end": "2023-12-31", "test_start": "2024-01-01", "test_end": "2024-12-31"},
    {"label": "W7_2025", "train_end": "2024-12-31", "test_start": "2025-01-01", "test_end": "2025-12-31"},
    {"label": "W8_2026", "train_end": "2025-12-31", "test_start": "2026-01-01", "test_end": "2026-04-30"},
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _existing_run_dirs(base="logs"):
    return set(
        d for d in os.listdir(base)
        if os.path.isdir(os.path.join(base, d)) and d.startswith("run_")
    )


def _find_new_run_dir(before: set, base="logs") -> str | None:
    after = _existing_run_dirs(base)
    new = after - before
    if not new:
        return None
    return os.path.join(base, sorted(new)[-1])


def _compute_metrics(rows: list[dict]) -> dict:
    """Compute test-phase performance metrics from daily_performance rows."""
    navs = [float(r["nav"]) for r in rows]
    rets = [float(r["daily_return"]) for r in rows]

    if not navs:
        return {}

    start_nav = navs[0]
    end_nav   = navs[-1]
    cum_ret   = (end_nav - start_nav) / start_nav * 100

    # Annualised return (CAGR-style)
    n_days = len(navs)
    n_years = n_days / 252.0
    cagr = ((end_nav / start_nav) ** (1.0 / max(n_years, 1e-6)) - 1) * 100 if n_years > 0 else 0.0

    # Sharpe (daily, annualised, rf=0)
    if len(rets) > 1:
        mean_r = sum(rets) / len(rets)
        std_r  = (sum((r - mean_r) ** 2 for r in rets) / (len(rets) - 1)) ** 0.5
        sharpe = (mean_r / std_r * (252 ** 0.5)) if std_r > 0 else 0.0
    else:
        sharpe = 0.0

    # Max drawdown
    peak = navs[0]
    mdd  = 0.0
    for n in navs:
        if n > peak:
            peak = n
        dd = (n - peak) / peak
        if dd < mdd:
            mdd = dd

    return {
        "n_days":   n_days,
        "start_nav": start_nav,
        "end_nav":   end_nav,
        "cum_ret":   cum_ret,
        "cagr":      cagr,
        "sharpe":    sharpe,
        "mdd":       mdd * 100,
    }


def _load_test_metrics(run_dir: str) -> dict | None:
    perf_path = os.path.join(run_dir, "2_daily_performance.csv")
    if not os.path.exists(perf_path):
        return None
    with open(perf_path) as f:
        rows = [r for r in csv.DictReader(f) if r.get("phase") == "test"]
    if not rows:
        return None
    return _compute_metrics(rows)


def _window_label_from_syslog(run_dir: str) -> str | None:
    slog = os.path.join(run_dir, "system.log")
    if not os.path.exists(slog):
        return None
    with open(slog) as f:
        for line in f:
            if "Walk-forward window:" in line:
                return line.split("Walk-forward window:")[-1].strip()
    return None


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_window(win: dict, mode: str = "train", seed: int = 42) -> str | None:
    """Spawn run_trainer for one window. Returns the run directory path."""
    before = _existing_run_dirs()
    cmd = [
        sys.executable, "-m", "training.run_trainer",
        "--mode", mode,
        "--min_date",        MIN_DATE,
        "--train_end_date",  win["train_end"],
        "--test_start_date", win["test_start"],
        "--test_end_date",   win["test_end"],
        "--wf_label",        win["label"],
        "--seed",            str(seed),
    ]
    print(f"\n{'='*60}")
    print(f"  Window {win['label']}  |  Train: {MIN_DATE} → {win['train_end']}  |  Test: {win['test_start']} → {win['test_end']}")
    print(f"{'='*60}")

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    if result.returncode != 0:
        print(f"  [ERROR] Window {win['label']} exited with code {result.returncode}")

    run_dir = _find_new_run_dir(before)
    if run_dir:
        print(f"  Logs saved to: {run_dir}")
    return run_dir


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------
def print_summary(results: list[dict]) -> None:
    print(f"\n{'='*90}")
    print("  WALK-FORWARD VALIDATION SUMMARY")
    print(f"{'='*90}")
    hdr = f"{'Window':<12} {'Train end':<12} {'Test period':<25} {'CumRet':>8} {'CAGR':>7} {'Sharpe':>7} {'MaxDD':>8} {'Days':>6}"
    print(hdr)
    print("-" * 90)

    cum_rets, cagrs, sharpes, mdds = [], [], [], []

    for r in results:
        if not r.get("metrics"):
            print(f"  {r['label']:<10}  (no test data)")
            continue
        m = r["metrics"]
        cum_rets.append(m["cum_ret"])
        cagrs.append(m["cagr"])
        sharpes.append(m["sharpe"])
        mdds.append(m["mdd"])
        test_period = f"{r['win']['test_start']} → {r['win']['test_end']}"
        print(
            f"  {r['label']:<10}  {r['win']['train_end']:<12}  {test_period:<25}"
            f"  {m['cum_ret']:>+7.1f}%  {m['cagr']:>+6.1f}%  {m['sharpe']:>6.2f}  {m['mdd']:>+7.1f}%  {m['n_days']:>5}"
        )

    if cum_rets:
        n = len(cum_rets)
        print("-" * 90)
        avg_cr = sum(cum_rets) / n
        avg_cagr = sum(cagrs) / n
        avg_sh   = sum(sharpes) / n
        avg_mdd  = sum(mdds) / n
        std_cr   = (sum((x - avg_cr) ** 2 for x in cum_rets) / max(n - 1, 1)) ** 0.5
        std_cagr = (sum((x - avg_cagr) ** 2 for x in cagrs) / max(n - 1, 1)) ** 0.5
        print(
            f"  {'AVG':<10}  {'':12}  {'':25}"
            f"  {avg_cr:>+7.1f}%  {avg_cagr:>+6.1f}%  {avg_sh:>6.2f}  {avg_mdd:>+7.1f}%"
        )
        print(
            f"  {'STD':<10}  {'':12}  {'':25}"
            f"  {std_cr:>7.1f}%  {std_cagr:>6.1f}%  {'':>6}  {'':>7}"
        )
    print(f"{'='*90}\n")

    # Save to CSV next to logs
    out_path = os.path.join("logs", f"wf_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "label", "train_end", "test_start", "test_end",
            "cum_ret_pct", "cagr_pct", "sharpe", "mdd_pct", "n_days", "run_dir"
        ])
        writer.writeheader()
        for r in results:
            m = r.get("metrics") or {}
            writer.writerow({
                "label":       r["label"],
                "train_end":   r["win"]["train_end"],
                "test_start":  r["win"]["test_start"],
                "test_end":    r["win"]["test_end"],
                "cum_ret_pct": round(m.get("cum_ret", float("nan")), 4),
                "cagr_pct":    round(m.get("cagr", float("nan")), 4),
                "sharpe":      round(m.get("sharpe", float("nan")), 4),
                "mdd_pct":     round(m.get("mdd", float("nan")), 4),
                "n_days":      m.get("n_days", ""),
                "run_dir":     r.get("run_dir", ""),
            })
    print(f"  Summary CSV saved to: {out_path}")


# ---------------------------------------------------------------------------
# Summarise-only mode: scan existing runs for wf_label
# ---------------------------------------------------------------------------
def summarize_existing(base="logs") -> None:
    all_dirs = sorted(_existing_run_dirs(base))
    results = []
    for d in all_dirs:
        run_dir = os.path.join(base, d)
        label = _window_label_from_syslog(run_dir)
        if label is None:
            continue
        # Find matching window definition
        win = next((w for w in WINDOWS if w["label"] == label), None)
        if win is None:
            continue
        metrics = _load_test_metrics(run_dir)
        results.append({"label": label, "win": win, "metrics": metrics, "run_dir": run_dir})

    if not results:
        print("No walk-forward run directories found in logs/. Run without --summarize first.")
        return

    # De-duplicate: keep latest run per label
    seen = {}
    for r in results:
        seen[r["label"]] = r  # later dirs overwrite earlier ones (dirs are sorted)
    ordered = [seen[w["label"]] for w in WINDOWS if w["label"] in seen]
    print_summary(ordered)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Walk-forward validation orchestrator")
    parser.add_argument("--windows", nargs="+", default=None,
                        help="Subset of window labels to run, e.g. W1_2019 W3_2021")
    parser.add_argument("--summarize", action="store_true",
                        help="Only print summary from existing run logs (no training)")
    parser.add_argument("--smoke", action="store_true",
                        help="Run in smoke mode (fast, just checks plumbing)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.summarize:
        summarize_existing()
        return

    # Select windows
    selected = WINDOWS
    if args.windows:
        selected = [w for w in WINDOWS if w["label"] in args.windows]
        missing = set(args.windows) - {w["label"] for w in selected}
        if missing:
            print(f"Unknown window labels: {missing}")
            print(f"Valid labels: {[w['label'] for w in WINDOWS]}")
            return

    if args.smoke:
        # Smoke: run only first window in smoke mode to check plumbing
        selected = selected[:1]
        mode = "smoke"
        print("SMOKE MODE: running only first window.")
    else:
        mode = "train"

    results = []
    for win in selected:
        run_dir = run_window(win, mode=mode, seed=args.seed)
        metrics = _load_test_metrics(run_dir) if run_dir else None
        results.append({"label": win["label"], "win": win, "metrics": metrics, "run_dir": run_dir or ""})

    if not args.smoke:
        print_summary(results)


if __name__ == "__main__":
    main()
