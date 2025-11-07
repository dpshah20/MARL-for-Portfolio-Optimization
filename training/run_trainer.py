# training/run_trainer.py
"""
Main orchestrator for training:
 - Loads configs
 - Builds datasets (daily + weekly)
 - Runs daily RL updates and weekly meta-agent updates
 - Handles checkpointing and logging
"""

import argparse, yaml, os
from training.train_rl_agents import TrainerRL
from training.train_meta_agent import TrainerMeta
from training.checkpoints import CheckpointManager
from dataset.dataset_windows import windows_generator_from_paths
from dataset.dataset_meta import build_meta_dataset
from logging.logger import setup_logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/params.yaml")
    parser.add_argument("--mode", type=str, default="train", choices=["smoke","train","resume"])
    args = parser.parse_args()

    # Load configuration
    cfg = yaml.safe_load(open(args.config))

    # Setup logger and checkpoint manager
    logger = setup_logger("training")
    ckpt_mgr = CheckpointManager(cfg)

    # Initialize RL and meta trainers
    rl_trainer = TrainerRL(cfg, logger=logger, ckpt_mgr=ckpt_mgr)
    meta_trainer = TrainerMeta(cfg, logger=logger, ckpt_mgr=ckpt_mgr)

    # Load data
    parquet_paths = [f"processed/{t}_merged.parquet" for t in cfg["tickers"]]
    feature_cols = cfg["feature_cols"]
    min_date = cfg.get("min_date", None)

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
