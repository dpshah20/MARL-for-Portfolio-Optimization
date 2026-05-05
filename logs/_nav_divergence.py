import csv, os, statistics

OLD = 'run_20260312_192504'
NEW = 'run_20260502_135314'

def load_perf(run):
    with open(os.path.join('logs', run, '2_daily_performance.csv')) as f:
        return list(csv.DictReader(f))

old_rows = load_perf(OLD)
new_rows = load_perf(NEW)

# ---- NAV divergence: find first date where gap > 5% ----
print("NAV DIVERGENCE ANALYSIS")
print("-" * 50)
for i, (o, n) in enumerate(zip(old_rows, new_rows)):
    o_nav = float(o['nav'])
    n_nav = float(n['nav'])
    gap_pct = (o_nav - n_nav) / o_nav * 100
    if abs(gap_pct) > 5:
        print(f"  First >5% gap  @ step {i}: date={o['date'][:10]}  OLD={o_nav:.2f}  NEW={n_nav:.2f}  gap={gap_pct:+.1f}%")
        break

# Show NAV at key milestones
milestones = [250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2547]
print(f"\n{'Step':<6} {'Date':<12} {'OLD NAV':>10} {'NEW NAV':>10} {'Gap%':>8}")
print("-" * 50)
for m in milestones:
    idx = min(m, len(old_rows)-1)
    o = old_rows[idx]; n = new_rows[idx]
    o_nav = float(o['nav']); n_nav = float(n['nav'])
    gap = (o_nav - n_nav) / o_nav * 100
    print(f"{idx:<6} {o['date'][:10]:<12} {o_nav:>10.2f} {n_nav:>10.2f} {gap:>+8.1f}%")

# ---- Stock selection comparison ----
print("\n\nSTOCK SELECTION FREQUENCY COMPARISON")
print("-" * 60)

def load_trades(run):
    path = os.path.join('logs', run, '3_trade_history.csv')
    tickers = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            t = row.get('ticker', row.get('symbol', ''))
            tickers[t] = tickers.get(t, 0) + 1
    return tickers

old_trades = load_trades(OLD)
new_trades = load_trades(NEW)

all_tickers = sorted(set(list(old_trades.keys()) + list(new_trades.keys())))
print(f"{'Ticker':<16} {'OLD trades':>12} {'NEW trades':>12} {'Diff':>8}")
print("-" * 50)
diffs = []
for t in all_tickers:
    o = old_trades.get(t, 0)
    n = new_trades.get(t, 0)
    diffs.append((abs(o - n), t, o, n))

diffs.sort(reverse=True)
for _, t, o, n in diffs[:25]:
    print(f"{t:<16} {o:>12} {n:>12} {n-o:>+8}")

# ---- Phase breakdown (train vs test) ----
print("\n\nPHASE BREAKDOWN")
print("-" * 50)
for run, rows in [(OLD, old_rows), (NEW, new_rows)]:
    train = [r for r in rows if r['phase'] == 'train']
    test  = [r for r in rows if r['phase'] == 'test']
    if train:
        t_nav = float(train[-1]['nav'])
        t_ret = statistics.mean([float(r['daily_return']) for r in train])
        print(f"{run}  TRAIN: end_nav={t_nav:.2f}  avg_ret={t_ret*100:.4f}%")
    if test:
        s_nav = float(test[-1]['nav'])
        s_ret = statistics.mean([float(r['daily_return']) for r in test])
        print(f"{run}  TEST : end_nav={s_nav:.2f}  avg_ret={s_ret*100:.4f}%")
