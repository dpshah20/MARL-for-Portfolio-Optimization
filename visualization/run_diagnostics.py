import argparse
import csv
import json
import os
from collections import Counter, defaultdict


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


def _safe_float(v, default=0.0):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _safe_int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def analyze_selection(run_dir):
    exec_path = os.path.join(run_dir, "execution_logs.jsonl")
    logs = _read_jsonl(exec_path)

    counts = Counter()
    first_seen = {}
    last_seen = {}

    for row in logs:
        date = row.get("date", "")
        selected = row.get("selected_tickers", [])
        if not selected:
            alloc = row.get("allocations", {})
            selected = [t for t, w in alloc.items() if _safe_float(w) > 0.0]

        for t in selected:
            counts[t] += 1
            if t not in first_seen:
                first_seen[t] = date
            last_seen[t] = date

    out_path = os.path.join(run_dir, "5_stock_selection_frequency.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["ticker", "selected_days", "first_selected", "last_selected"])
        for ticker, c in counts.most_common():
            writer.writerow([ticker, c, first_seen.get(ticker, ""), last_seen.get(ticker, "")])

    print(f"[Diagnostics] Wrote selection frequency: {out_path}")
    print("[Diagnostics] Top 20 selected tickers:")
    for t, c in counts.most_common(20):
        print(f"  {t}: {c}")


def analyze_trades(run_dir):
    trades_path = os.path.join(run_dir, "3_trade_history.csv")
    if not os.path.exists(trades_path):
        print(f"[Diagnostics] Missing trade file: {trades_path}")
        return

    ticker_stats = defaultdict(lambda: {
        "buy_trades": 0,
        "sell_trades": 0,
        "buy_shares": 0,
        "sell_shares": 0,
        "gross_buy_value": 0.0,
        "gross_sell_value": 0.0,
        "commission": 0.0,
        "realized_pnl": 0.0,
    })

    with open(trades_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        has_new_cols = reader.fieldnames is not None and "buy_shares" in reader.fieldnames

        for row in reader:
            t = row.get("ticker", "")
            if not t:
                continue

            action = (row.get("action", "") or "").upper()
            shares_signed = _safe_int(row.get("shares", 0))
            buy_shares = _safe_int(row.get("buy_shares", 0)) if has_new_cols else max(shares_signed, 0)
            sell_shares = _safe_int(row.get("sell_shares", 0)) if has_new_cols else max(-shares_signed, 0)
            price = _safe_float(row.get("price", 0.0))
            comm = _safe_float(row.get("commission", row.get("comm", 0.0)))
            realized_pnl = _safe_float(row.get("realized_pnl", 0.0)) if has_new_cols else 0.0

            st = ticker_stats[t]
            st["commission"] += comm
            st["realized_pnl"] += realized_pnl

            if action == "BUY":
                st["buy_trades"] += 1
                st["buy_shares"] += buy_shares
                st["gross_buy_value"] += buy_shares * price
            elif action == "SELL":
                st["sell_trades"] += 1
                st["sell_shares"] += sell_shares
                st["gross_sell_value"] += sell_shares * price

    out_ticker = os.path.join(run_dir, "6_ticker_trade_stats.csv")
    with open(out_ticker, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "ticker",
            "buy_trades",
            "sell_trades",
            "buy_shares",
            "sell_shares",
            "net_shares",
            "gross_buy_value",
            "gross_sell_value",
            "commission",
            "realized_pnl",
        ])
        for t, st in sorted(ticker_stats.items()):
            writer.writerow([
                t,
                st["buy_trades"],
                st["sell_trades"],
                st["buy_shares"],
                st["sell_shares"],
                st["buy_shares"] - st["sell_shares"],
                f"{st['gross_buy_value']:.2f}",
                f"{st['gross_sell_value']:.2f}",
                f"{st['commission']:.2f}",
                f"{st['realized_pnl']:.2f}",
            ])

    total_realized = sum(x["realized_pnl"] for x in ticker_stats.values())
    total_comm = sum(x["commission"] for x in ticker_stats.values())

    out_overall = os.path.join(run_dir, "7_trade_summary.csv")
    with open(out_overall, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        writer.writerow(["num_tickers_traded", len(ticker_stats)])
        writer.writerow(["total_realized_pnl", f"{total_realized:.2f}"])
        writer.writerow(["total_commission", f"{total_comm:.2f}"])

    print(f"[Diagnostics] Wrote ticker trade stats: {out_ticker}")
    print(f"[Diagnostics] Wrote trade summary: {out_overall}")


def main():
    parser = argparse.ArgumentParser(description="Run post-training diagnostics for MARL portfolio logs")
    parser.add_argument("--run_dir", required=True, help="Path to logs/run_YYYYMMDD_HHMMSS")
    args = parser.parse_args()

    run_dir = args.run_dir
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    analyze_selection(run_dir)
    analyze_trades(run_dir)
    print("[Diagnostics] Completed.")


if __name__ == "__main__":
    main()
