"""Main training orchestrator"""

import argparse
import yaml
import os
import torch
from datetime import datetime

from training.train_rl_agents import TrainerRL 
from training.train_meta_agent import MetaTrainer
from training.checkpoints import CheckpointManager
from dataset.dataset_windows import windows_generator_from_paths
from dataset.dataset_meta import build_meta_dataset
from proj_logging.logger import setup_logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/params.yaml")
    parser.add_argument("--mode", type=str, default="train", choices=["smoke","train","resume"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Load configuration
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    cfg = yaml.safe_load(open(args.config))
    
    # Ensure required paths exist
    os.makedirs(cfg.get("logs_dir", "logging"), exist_ok=True)
    os.makedirs(cfg.get("checkpoint_dir", "checkpoints"), exist_ok=True)
    os.makedirs("processed", exist_ok=True)

    # Setup components
    logger = setup_logger("training")
    ckpt_mgr = CheckpointManager(save_dir=cfg.get("checkpoint_dir", "checkpoints"))
    
    # Initialize trainers
    rl_trainer = TrainerRL(cfg, logger=logger, ckpt_mgr=ckpt_mgr, device=args.device)
    meta_trainer = MetaTrainer(cfg, logger=logger, ckpt_mgr=ckpt_mgr, device=args.device)

    # Load data
    parquet_paths = [os.path.join("processed", f"{t}_merged.parquet") for t in cfg["tickers"]]
    missing = [p for p in parquet_paths if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Missing processed files: {missing}")

    # Run modes
    if args.mode == "smoke":
        rl_trainer.run_smoke(parquet_paths, min_date=min_date, max_days=cfg.get("smoke_days", 20))
        return

    elif args.mode == "train":
        logger.info("Starting full training loop...")
        for date, X in windows_generator_from_paths(parquet_paths, feature_cols, W=cfg["window"]):
            rl_trainer.step_daily(X, date)       # daily actor/critic update
            if rl_trainer.is_weekly_update(date):  # end of week
                meta_states, reward_stats = build_meta_dataset(
                    macros_path="data/macros/combined_macros_weekly.csv",
                    rl_logs_path="logging/training_logs.csv"
                )
                meta_trainer.step_weekly(meta_states, reward_stats)
        rl_trainer.save_all()
        meta_trainer.save_all()
        logger.info("Training complete.")

    elif args.mode == "resume":
        ckpt_mgr.load_all()
        logger.info("Resuming from last checkpoint...")
        # continue same as "train" block

if __name__ == "__main__":
    main()
