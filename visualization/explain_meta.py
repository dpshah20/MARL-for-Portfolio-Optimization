# visualization/explain_meta.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_meta_logs(meta_csv: str, out_png: str = "meta.png"):
    df = pd.read_csv(meta_csv, parse_dates=["Date"])
    # columns expected: Date, rho, w0, w1, w2, w3
    plt.figure(figsize=(10,6))
    plt.plot(df["Date"], df["rho"], label="rho")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
