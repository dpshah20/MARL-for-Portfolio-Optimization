import pandas as pd
import numpy as np
from scipy import stats

ALPHA = 0.05
RISK_FREE_RATE = 0.06  # Annual risk-free rate (6%)


def compute_max_drawdown(returns: pd.Series) -> float:
    cumulative = (1 + returns).cumprod()
    rolling_peak = cumulative.cummax()
    drawdown = cumulative / rolling_peak - 1
    return float(drawdown.min())


def compute_sharpe_ratio(returns: pd.Series, rf_rate: float = RISK_FREE_RATE) -> float:
    """Compute annualized Sharpe ratio."""
    excess_returns = returns - (rf_rate / 252)  # Daily risk-free rate
    return float(np.sqrt(252) * excess_returns.mean() / returns.std())


def compute_sortino_ratio(returns: pd.Series, rf_rate: float = RISK_FREE_RATE) -> float:
    """Compute annualized Sortino ratio (only downside volatility)."""
    excess_returns = returns - (rf_rate / 252)
    downside_returns = returns[returns < 0]
    downside_vol = downside_returns.std()
    if downside_vol == 0:
        return 0.0
    return float(np.sqrt(252) * excess_returns.mean() / downside_vol)


def compute_calmar_ratio(returns: pd.Series) -> float:
    """Compute annualized Calmar ratio (return / max drawdown)."""
    annual_return = (1 + returns.mean()) ** 252 - 1
    mdd = compute_max_drawdown(returns)
    if mdd == 0:
        return 0.0
    return float(annual_return / abs(mdd))


def compute_win_rate(marl_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Percentage of days MARL outperforms benchmark."""
    wins = (marl_returns > benchmark_returns).sum()
    total = len(marl_returns)
    return float(wins / total * 100)


def compute_upside_capture(marl_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Percentage of benchmark upside MARL captures."""
    up_days = benchmark_returns > 0
    if up_days.sum() == 0:
        return 0.0
    marl_up_return = marl_returns[up_days].sum()
    bench_up_return = benchmark_returns[up_days].sum()
    if bench_up_return == 0:
        return 0.0
    return float(marl_up_return / bench_up_return * 100)


def compute_downside_capture(marl_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Percentage of benchmark downside MARL captures."""
    down_days = benchmark_returns < 0
    if down_days.sum() == 0:
        return 0.0
    marl_down_return = marl_returns[down_days].sum()
    bench_down_return = benchmark_returns[down_days].sum()
    if bench_down_return == 0:
        return 0.0
    return float(marl_down_return / bench_down_return * 100)


def compute_beta(marl_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Compute beta relative to benchmark."""
    covariance = np.cov(marl_returns, benchmark_returns)[0, 1]
    variance = np.var(benchmark_returns, ddof=1)
    if variance == 0:
        return 0.0
    return float(covariance / variance)


def compute_alpha(marl_returns: pd.Series, benchmark_returns: pd.Series, beta: float, rf_rate: float = RISK_FREE_RATE) -> float:
    """Compute Jensen's alpha."""
    marl_annual_return = (1 + marl_returns.mean()) ** 252 - 1
    bench_annual_return = (1 + benchmark_returns.mean()) ** 252 - 1
    expected_return = rf_rate + beta * (bench_annual_return - rf_rate)
    return float(marl_annual_return - expected_return)


def run_hypothesis_tests(marl_df: pd.DataFrame, benchmark_df: pd.DataFrame, benchmark_name: str) -> None:
    benchmark = benchmark_df.copy()
    benchmark = benchmark[benchmark["Date"].notna()]
    benchmark["Date"] = pd.to_datetime(benchmark["Date"])
    benchmark["Close"] = pd.to_numeric(benchmark["Close"], errors="coerce")
    benchmark["return"] = benchmark["Close"].pct_change()

    merged = pd.merge(
        marl_df[["date", "daily_return"]],
        benchmark[["Date", "return"]],
        left_on="date",
        right_on="Date",
        how="inner",
    ).dropna(subset=["daily_return", "return"])

    marl_returns = merged["daily_return"]
    benchmark_returns = merged["return"]

    print(f"\n{'=' * 80}")
    print(f"Benchmark: {benchmark_name}")
    print(f"Aligned observations: {len(merged)}")
    print(f"{'=' * 80}")

    if len(merged) < 5:
        print("Not enough aligned observations to run robust hypothesis tests.")
        return

    # =====================================================================
    # HYPOTHESIS 1: MARL mean return > benchmark mean return
    # =====================================================================
    t_stat, p_two_sided = stats.ttest_ind(marl_returns, benchmark_returns, equal_var=False)
    if t_stat > 0:
        p_one_sided = p_two_sided / 2
    else:
        p_one_sided = 1 - (p_two_sided / 2)

    print("\n[H1] Return Improvement (MARL > Benchmark)")
    print(f"  MARL mean return:       {marl_returns.mean():.6f}")
    print(f"  Benchmark mean return:  {benchmark_returns.mean():.6f}")
    print(f"  t-statistic:            {t_stat:.6f}")
    print(f"  p-value (one-sided):    {p_one_sided:.6f}")
    print(f"  Result:                 {'✓ REJECT H0' if (t_stat > 0) and (p_one_sided < ALPHA) else '✗ FAIL TO REJECT H0'}")

    # =====================================================================
    # HYPOTHESIS 2: MARL volatility < benchmark volatility
    # =====================================================================
    var_marl = np.var(marl_returns, ddof=1)
    var_benchmark = np.var(benchmark_returns, ddof=1)
    f_stat = var_marl / var_benchmark

    df1 = len(marl_returns) - 1
    df2 = len(benchmark_returns) - 1
    p_left_tail = stats.f.cdf(f_stat, df1, df2)

    print("\n[H2] Volatility Reduction (MARL < Benchmark)")
    print(f"  MARL volatility:        {np.sqrt(var_marl):.6f}")
    print(f"  Benchmark volatility:   {np.sqrt(var_benchmark):.6f}")
    print(f"  F-statistic:            {f_stat:.6f}")
    print(f"  p-value (left-tail):    {p_left_tail:.6f}")
    print(f"  Result:                 {'✓ REJECT H0' if (f_stat < 1) and (p_left_tail < ALPHA) else '✗ FAIL TO REJECT H0'}")

    # =====================================================================
    # HYPOTHESIS 3: Max drawdown comparison
    # =====================================================================
    marl_mdd = compute_max_drawdown(marl_returns)
    benchmark_mdd = compute_max_drawdown(benchmark_returns)

    print("\n[H3] Maximum Drawdown Reduction")
    print(f"  MARL max drawdown:      {marl_mdd:.6f}")
    print(f"  Benchmark max drawdown: {benchmark_mdd:.6f}")
    print(f"  Drawdown reduction:     {(benchmark_mdd - marl_mdd)*100:.2f}%")
    print(f"  Result:                 {'✓ MARL REDUCES DOWNSIDE RISK' if marl_mdd > benchmark_mdd else '✗ NO IMPROVEMENT'}")

    # =====================================================================
    # HYPOTHESIS 4: Sharpe Ratio Comparison
    # =====================================================================
    marl_sharpe = compute_sharpe_ratio(marl_returns)
    bench_sharpe = compute_sharpe_ratio(benchmark_returns)
    sharpe_diff = marl_sharpe - bench_sharpe

    print("\n[H4] Sharpe Ratio Comparison (Risk-adjusted returns)")
    print(f"  MARL Sharpe ratio:      {marl_sharpe:.6f}")
    print(f"  Benchmark Sharpe ratio: {bench_sharpe:.6f}")
    print(f"  Difference:             {sharpe_diff:.6f}")
    print(f"  Result:                 {'✓ MARL BETTER' if sharpe_diff > 0 else '✗ BENCHMARK BETTER'}")

    # =====================================================================
    # HYPOTHESIS 5: Sortino Ratio Comparison (downside risk)
    # =====================================================================
    marl_sortino = compute_sortino_ratio(marl_returns)
    bench_sortino = compute_sortino_ratio(benchmark_returns)
    sortino_diff = marl_sortino - bench_sortino

    print("\n[H5] Sortino Ratio Comparison (Downside risk-adjusted)")
    print(f"  MARL Sortino ratio:     {marl_sortino:.6f}")
    print(f"  Benchmark Sortino ratio:{bench_sortino:.6f}")
    print(f"  Difference:             {sortino_diff:.6f}")
    print(f"  Result:                 {'✓ MARL BETTER' if sortino_diff > 0 else '✗ BENCHMARK BETTER'}")

    # =====================================================================
    # HYPOTHESIS 6: Calmar Ratio (return / max drawdown)
    # =====================================================================
    marl_calmar = compute_calmar_ratio(marl_returns)
    bench_calmar = compute_calmar_ratio(benchmark_returns)

    print("\n[H6] Calmar Ratio Comparison (Return per unit drawdown)")
    print(f"  MARL Calmar ratio:      {marl_calmar:.6f}")
    print(f"  Benchmark Calmar ratio: {bench_calmar:.6f}")
    print(f"  Result:                 {'✓ MARL BETTER' if marl_calmar > bench_calmar else '✗ BENCHMARK BETTER'}")

    # =====================================================================
    # HYPOTHESIS 7: Win Rate (Hit Ratio)
    # =====================================================================
    win_rate = compute_win_rate(marl_returns, benchmark_returns)

    print("\n[H7] Win Rate / Hit Ratio")
    print(f"  Days MARL > Benchmark:  {win_rate:.2f}%")
    print(f"  Expected (random):      50.00%")
    print(f"  Result:                 {'✓ MARL OUTPERFORMS MORE OFTEN' if win_rate > 50 else '✗ BENCHMARK OUTPERFORMS MORE OFTEN'}")

    # =====================================================================
    # HYPOTHESIS 8: Non-parametric Sign Test (Win Rate significance)
    # =====================================================================
    wins = (marl_returns > benchmark_returns).sum()
    losses = (marl_returns < benchmark_returns).sum()
    ties = (marl_returns == benchmark_returns).sum()
    
    # Binomial test: is win rate significantly different from 50%?
    p_sign_test = stats.binomtest(wins, wins + losses, 0.5, alternative='two-sided').pvalue

    print("\n[H8] Non-parametric Sign Test (Win rate significance)")
    print(f"  Wins:                   {wins} days")
    print(f"  Losses:                 {losses} days")
    print(f"  Ties:                   {ties} days")
    print(f"  p-value (two-sided):    {p_sign_test:.6f}")
    print(f"  Result:                 {'✓ WIN RATE IS SIGNIFICANT' if p_sign_test < ALPHA else '✗ WIN RATE NOT SIGNIFICANT'}")

    # =====================================================================
    # HYPOTHESIS 9: Wilcoxon Signed-Rank Test (non-parametric t-test)
    # =====================================================================
    diff = marl_returns - benchmark_returns
    wilcoxon_stat, wilcoxon_p = stats.wilcoxon(diff.dropna())

    print("\n[H9] Wilcoxon Signed-Rank Test (non-parametric)")
    print(f"  Test statistic:         {wilcoxon_stat:.1f}")
    print(f"  p-value (two-sided):    {wilcoxon_p:.6f}")
    print(f"  Result:                 {'✓ MARL SIGNIFICANTLY DIFFERENT' if wilcoxon_p < ALPHA else '✗ NO SIGNIFICANT DIFFERENCE'}")

    # =====================================================================
    # HYPOTHESIS 10: Beta (Systematic Risk)
    # =====================================================================
    beta = compute_beta(marl_returns, benchmark_returns)
    correlation = marl_returns.corr(benchmark_returns)

    print("\n[H10] Beta & Correlation (Systematic Risk)")
    print(f"  Beta:                   {beta:.6f}")
    print(f"  Correlation:            {correlation:.6f}")
    print(f"  Interpretation:         {'Defensive (moves less)' if beta < 1 else 'Aggressive (moves more)'}")

    # =====================================================================
    # HYPOTHESIS 11: Alpha (Risk-adjusted Excess Return)
    # =====================================================================
    alpha = compute_alpha(marl_returns, benchmark_returns, beta)

    print("\n[H11] Jensen's Alpha (Risk-adjusted Excess Return)")
    print(f"  Alpha (annualized):     {alpha:.6f}")
    print(f"  Beta:                   {beta:.6f}")
    print(f"  Interpretation:         {'✓ MARL OUTPERFORMS' if alpha > 0 else '✗ MARL UNDERPERFORMS'}")

    # =====================================================================
    # HYPOTHESIS 12: Upside Capture Ratio
    # =====================================================================
    upside_capture = compute_upside_capture(marl_returns, benchmark_returns)

    print("\n[H12] Upside Capture Ratio")
    print(f"  Upside capture:         {upside_capture:.2f}%")
    print(f"  Interpretation:         MARL captures {upside_capture:.2f}% of benchmark upside")

    # =====================================================================
    # HYPOTHESIS 13: Downside Capture Ratio
    # =====================================================================
    downside_capture = compute_downside_capture(marl_returns, benchmark_returns)

    print("\n[H13] Downside Capture Ratio")
    print(f"  Downside capture:       {downside_capture:.2f}%")
    print(f"  Interpretation:         MARL captures {downside_capture:.2f}% of benchmark downside")
    print(f"  Risk profile:           {'✓ DEFENSIVE' if downside_capture < 100 else '✗ AGGRESSIVE'}")

    print(f"\n{'=' * 80}")

# ----------------------------
# Load data
# ----------------------------

marl = pd.read_csv("logs/run_20260312_192504/2_daily_performance.csv")
nifty50 = pd.read_csv("nifty50_ohlcv_2015_2025.csv")
nifty100 = pd.read_csv("nifty100_ohlcv_2015_2025.csv")

marl["date"] = pd.to_datetime(marl["date"])

run_hypothesis_tests(marl, nifty50, "Nifty 50")
run_hypothesis_tests(marl, nifty100, "Nifty 100")