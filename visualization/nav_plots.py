# visualization/nav_plot.py
import matplotlib.pyplot as plt
import pandas as pd

def plot_nav(csv_path: str, date_col: str = "date", nav_col: str = "nav", out_png: str = "nav.png"):
    df = pd.read_csv(csv_path, parse_dates=[date_col])
    plt.figure(figsize=(10,5))
    plt.plot(df[date_col], df[nav_col], label="Strategy NAV")
    plt.title("NAV Curve")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
