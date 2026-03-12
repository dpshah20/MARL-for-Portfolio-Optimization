import matplotlib.pyplot as plt
import pandas as pd


def plot_meta_logs(meta_csv: str, out_png: str = "meta.png"):
    df = pd.read_csv(meta_csv)
    if "week_start_date" not in df.columns:
        raise ValueError("Expected 'week_start_date' in meta CSV")

    df["week_start_date"] = pd.to_datetime(df["week_start_date"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(df["week_start_date"], df["rho_cash"], label="rho_cash", color="tab:red")
    ax1.set_ylabel("rho")
    ax1.set_title("Meta Agent Cash Policy")
    ax1.grid(alpha=0.3)
    ax1.legend()

    for col in ["w_ret", "w_vol", "w_cvar", "w_mdd"]:
        if col in df.columns:
            ax2.plot(df["week_start_date"], df[col], label=col)
    ax2.set_xlabel("Week")
    ax2.set_ylabel("Weight")
    ax2.set_title("Meta Reward Weights")
    ax2.grid(alpha=0.3)
    ax2.legend(ncol=4)

    fig.tight_layout()
    fig.savefig(out_png, dpi=140)
    plt.close(fig)
