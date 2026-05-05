import csv, os

runs = ['run_20260312_192504', 'run_20260502_135314']

for run in runs:
    path = os.path.join('logs', run, '2_daily_performance.csv')
    with open(path) as f:
        rows = list(csv.DictReader(f))
    first, last = rows[0], rows[-1]
    keys = list(first.keys())
    date_col = keys[0]
    nav_col = next((k for k in keys if 'nav' in k.lower() or 'portfolio' in k.lower()), keys[-1])
    print(f'--- {run} ---')
    print(f'  Rows      : {len(rows)}')
    print(f'  First date: {first[date_col]}')
    print(f'  Last date : {last[date_col]}')
    print(f'  Final NAV : {last[nav_col]}')
    print(f'  Columns   : {keys}')
    print()
