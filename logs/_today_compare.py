import csv
runs = ['run_20260502_135314', 'run_20260502_225304']
for run in runs:
    with open(f'logs/{run}/2_daily_performance.csv') as f:
        rows = list(csv.DictReader(f))
    last = rows[-1]
    print(run)
    print("  rows      :", len(rows))
    print("  final_nav :", last['nav'])
    print("  last_date :", last['date'][:10])
    # check which system log mentions seed
    import os
    slog = f'logs/{run}/system.log'
    if os.path.exists(slog):
        for line in open(slog):
            if 'seed' in line.lower() or 'Seed' in line:
                print("  seed_line :", line.strip())
    print()
