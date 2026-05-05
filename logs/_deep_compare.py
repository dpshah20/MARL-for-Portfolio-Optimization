import csv, os, statistics

runs = ['run_20260312_192504', 'run_20260502_135314']

# ---- 2_daily_performance stats ----
print("=" * 60)
print("DAILY PERFORMANCE COMPARISON")
print("=" * 60)
for run in runs:
    path = os.path.join('logs', run, '2_daily_performance.csv')
    with open(path) as f:
        rows = list(csv.DictReader(f))
    navs = [float(r['nav']) for r in rows]
    rets = [float(r['daily_return']) for r in rows if r['daily_return']]
    exposures = [float(r['exposure']) for r in rows if r['exposure']]
    peak = max(navs)
    mdd = min((n - peak) / peak for n, peak in zip(navs, [max(navs[:i+1]) for i in range(len(navs))]))
    print(f"\n{run}")
    print(f"  Final NAV          : {navs[-1]:.4f}")
    print(f"  Peak NAV           : {peak:.4f}")
    print(f"  Max Drawdown       : {mdd*100:.2f}%")
    print(f"  Avg Daily Return   : {statistics.mean(rets)*100:.4f}%")
    print(f"  Std Daily Return   : {statistics.stdev(rets)*100:.4f}%")
    print(f"  Avg Exposure       : {statistics.mean(exposures)*100:.2f}%")

# ---- 4_training_internals ----
print("\n" + "=" * 60)
print("TRAINING INTERNALS COMPARISON")
print("=" * 60)
for run in runs:
    path = os.path.join('logs', run, '4_training_internals.csv')
    if not os.path.exists(path):
        print(f"\n{run}: No training internals file")
        continue
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"\n{run}: Empty file")
        continue
    keys = list(rows[0].keys())
    print(f"\n{run}  ({len(rows)} rows)")
    print(f"  Columns: {keys}")
    # sample last row
    last = rows[-1]
    for k, v in last.items():
        print(f"  {k}: {v}")

# ---- 1_meta_strategy ----
print("\n" + "=" * 60)
print("META STRATEGY COMPARISON")
print("=" * 60)
for run in runs:
    path = os.path.join('logs', run, '1_meta_strategy.csv')
    with open(path) as f:
        rows = list(csv.DictReader(f))
    keys = list(rows[0].keys())
    # look for rho / weight columns
    num_cols = [k for k in keys if k not in ('date', 'step', 'phase')]
    print(f"\n{run}  ({len(rows)} rows)  cols={keys}")
    for col in num_cols:
        try:
            vals = [float(r[col]) for r in rows if r[col]]
        except ValueError:
            continue
        if vals:
            print(f"  {col}: mean={statistics.mean(vals):.4f}  std={statistics.stdev(vals):.4f}  min={min(vals):.4f}  max={max(vals):.4f}")

# ---- 5_stock_selection_frequency (old run only) ----
print("\n" + "=" * 60)
print("STOCK SELECTION FREQUENCY (old run)")
print("=" * 60)
path = os.path.join('logs', 'run_20260312_192504', '5_stock_selection_frequency.csv')
if os.path.exists(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    for r in rows[:20]:
        print(r)
