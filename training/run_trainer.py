# training/run_trainer.py
"""
Main orchestrator for training:
 - Loads configs
 - Streams sliding windows
 - Runs daily RL updates and weekly meta-agent updates
 - Handles checkpointing and logging
"""
import argparse
import yaml
import os
import pandas as pd

from training.train_rl_agents import TrainerRL
from training.train_meta_agent import MetaTrainer
from training.checkpoints import CheckpointManager
from dataset.dataset_windows import windows_generator_from_paths
from dataset.dataset_meta import build_meta_dataset
from proj_logging.logger import setup_logger

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/params.yaml")
    parser.add_argument("--mode", type=str, default="train", choices=["smoke", "train", "resume"])
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    logger = setup_logger("training")
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    ckpt_mgr = CheckpointManager(ckpt_dir)

    logger.info("Logger initialized for training")

    # Initialize trainers
    rl_trainer = TrainerRL(cfg, logger=logger, ckpt_mgr=ckpt_mgr, device=cfg.get("device", "cpu"))
    meta_trainer = MetaTrainer(cfg, logger=logger, ckpt_mgr=ckpt_mgr, device=cfg.get("device", "cpu"))

    # prepare parquet paths (expects processed/{TICKER}_merged.parquet)
    parquet_paths = []
    for t in cfg["tickers"]:
        p = f"processed/{t}_merged.parquet"
        if os.path.exists(p):
            parquet_paths.append(p)
        else:
            # try to find similar filename
            import glob
            matches = glob.glob(f"processed/{t}*merged.parquet") + glob.glob(f"processed/{t}*.parquet")
            if matches:
                parquet_paths.append(matches[0])
            else:
                logger.warning(f"No parquet found for ticker {t}; expected {p}")

    feature_cols = cfg["feature_cols"]
    window = cfg.get("window_length", cfg.get("encoder", {}).get("W", 126))

    if args.mode == "smoke":
        logger.info("Running smoke test")
        rl_trainer.run_smoke(parquet_paths, min_date=cfg.get("min_date", None), max_days=cfg.get("smoke_days", 20))
        return

    if args.mode == "resume":
        logger.info("Resuming from latest checkpoint (if any)")
        ckpt_mgr.load_latest()

    # Training loop (stream windows)
    logger.info("Starting full training loop...")
    step = 0
    for date, X in windows_generator_from_paths(parquet_paths, feature_cols, W=window, min_date=cfg.get("min_date", None)):
        rl_trainer.step_daily(X, date)
        step += 1

        # weekly meta update (trigger by days count or custom rule)
        if step % cfg.get("weekly_update_every_days", 5) == 0:
            try:
                macros_path = cfg.get("macros_weekly_path", "data/macros/combined_macros_weekly.csv")
                rl_logs_path = cfg.get("rl_logs_path", "logging/training_logs.csv")
                meta_states, reward_stats = build_meta_dataset(macros_path=macros_path, rl_logs_path=rl_logs_path)
                if len(meta_states) > 0:
                    meta_trainer.train(meta_states, reward_stats, epochs=cfg.get("weekly_update_steps", 10))
                    logger.info(f"[MetaTrainer] weekly update performed at step {step}")
            except Exception as e:
                logger.exception(f"Meta update failed: {e}")

        # periodic checkpointing
        if step % cfg.get("checkpoint_every_steps", 50) == 0:
            ckpt_mgr.save(step, actor=rl_trainer.agent.actor, critic=rl_trainer.agent.critic, meta_agent=meta_trainer.meta_agent)
            logger.info(f"Checkpoint saved at step {step}")

    # final save
    ckpt_mgr.save(step, actor=rl_trainer.agent.actor, critic=rl_trainer.agent.critic, meta_agent=meta_trainer.meta_agent)
    logger.info("Training loop finished.")


if __name__ == "__main__":
    main()
