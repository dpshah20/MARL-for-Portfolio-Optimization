# training/run_trainer.py
import argparse, yaml
from training.train_rl_agents import TrainerRL

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/params.yaml")
    parser.add_argument("--mode", type=str, default="smoke", choices=["smoke","train"])
    args = parser.parse_args()
    cfg = yaml.safe_load(open(args.config))
    trainer = TrainerRL(cfg)
    if args.mode=="smoke":
        parquet_paths = [f"processed/{t}_merged.parquet" for t in cfg["tickers"]]
        trainer.run_smoke(parquet_paths, min_date=cfg.get("min_date", None), max_days=cfg.get("smoke_days", 20))
    else:
        raise NotImplementedError("Full train mode not implemented in this quick runner. Use run_smoke for initial tests.")

if __name__ == "__main__":
    main()
