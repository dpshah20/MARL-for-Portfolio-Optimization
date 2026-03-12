import argparse
import yaml
import os
import glob
import sys
import torch
import pandas as pd

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# Import our new components
from training.train_rl_agents import TrainerRL
from training.train_meta_agent import MetaTrainer
from training.checkpoints import CheckpointManager
from dataset.dataset_windows import windows_generator_from_paths
from proj_logging.logger import RunLogger
from rl_layer.meta_agent import MetaAgent

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/params.yaml")
    parser.add_argument("--mode", type=str, default="train", choices=["smoke", "train", "resume"])
    args = parser.parse_args()

    # --- 1. Load Configurations ---
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        return

    cfg = yaml.safe_load(open(args.config))
    
    # Attempt to load meta-agent specific config if it exists
    meta_config_path = "configs/meta_agent.yaml"
    if os.path.exists(meta_config_path):
        meta_cfg = yaml.safe_load(open(meta_config_path))
        cfg.update(meta_cfg) # Merge settings
    
    # Override defaults for specific modes
    if args.mode == "smoke":
        cfg["batch_size"] = 32 # Ensure learning happens quickly in smoke test
        cfg["replay_capacity"] = 1000

    # --- 2. Setup Infrastructure ---
    # Initialize the new hierarchical logger
    logger = RunLogger(base_dir="logs")
    logger.info(f"Starting Session in Mode: {args.mode}")
    
    ckpt_dir = cfg.get("checkpoint_dir", "checkpoints")
    ckpt_mgr = CheckpointManager(ckpt_dir, logger=logger)

    # --- 3. Initialize The Brain (Meta-Agent) ---
    # Input Dim = Count(Macro Features) + 4 (Prev Week Stats) + 5 (Prev Action Logits)
    # We need to know how many macro columns we are using
    macro_cols = cfg.get("meta_input", {}).get("macros", [])
    # If not in config, default to 6 standard macro features
    if not macro_cols: 
        macro_cols = ["macro_1", "macro_2", "macro_3", "macro_4", "macro_5", "macro_6"]
        
    meta_input_dim = len(macro_cols) + 4 + 5
    
    device = cfg.get("device", "cpu")
    meta_agent = MetaAgent(
        input_dim=meta_input_dim,
        hidden_dim=cfg.get("meta_hidden_dim", 128),
        rho_min=cfg.get("meta_rho_min", 0.05),
        rho_max=cfg.get("meta_rho_max", 0.30),
        init_std_w=cfg.get("meta_init_std_w", 0.35),
        init_std_rho=cfg.get("meta_init_std_rho", 0.20),
    ).to(device)
    meta_trainer = MetaTrainer(
        meta_agent,
        lr=cfg.get("meta_lr", 1e-3),
        entropy_coef=cfg.get("meta_entropy_coef", 1e-3),
        baseline_momentum=cfg.get("meta_baseline_momentum", 0.95),
    )
    
    logger.info(f"Meta-Agent initialized. Input Dim: {meta_input_dim} (Macros={len(macro_cols)})")

    # --- 4. Initialize The Body (RL Trainer) ---
    rl_trainer = TrainerRL(cfg, logger=logger, ckpt_mgr=ckpt_mgr, device=device)
    
    # CONNECT THE BRAIN TO THE BODY
    rl_trainer.attach_meta_trainer(meta_trainer)

    # --- 5. Resume Logic ---
    start_step = 0
    if args.mode == "resume":
        start_step = ckpt_mgr.load_latest(
            actor=rl_trainer.agent.actor, 
            critic=rl_trainer.agent.critic, 
            meta_agent=meta_agent
        )
        logger.info(f"Resumed training from step {start_step}")

    # --- 6. Data Preparation ---
    # Expects parquet files in 'nifty100/' folder
    parquet_paths = []
    for t in cfg["tickers"]:
        # Try exact match first
        p = f"nifty100/{t}_merged.parquet"
        if os.path.exists(p):
            parquet_paths.append(p)
            logger.info(f"Found data for ticker: {t}")
        else:
            # Fallback search
            matches = glob.glob(f"nifty100/{t}*.parquet")
            if matches:
                parquet_paths.append(matches[0])
            else:
                logger.info(f"[Warning] No data found for ticker: {t}")

    if not parquet_paths:
        logger.error("No valid parquet files found in 'nifty100/'. Exiting.")
        return

    # --- 7. Training Loop ---
    logger.info("Starting Data Stream...")
    
    # Define run limits
    max_steps = float('inf')
    if args.mode == "smoke":
        max_steps = 50 # Run enough steps to trigger at least one batch update
    
    feature_cols = cfg["feature_cols"]
    window_len = cfg.get("window_length", 126)
    min_date = cfg.get("min_date", "2015-01-01")
    train_end_date = cfg.get("train_end_date")
    test_start_date = cfg.get("test_start_date")
    test_end_date = cfg.get("test_end_date")

    train_end_ts = pd.to_datetime(train_end_date) if train_end_date else None
    test_start_ts = pd.to_datetime(test_start_date) if test_start_date else None
    test_end_ts = pd.to_datetime(test_end_date) if test_end_date else None
    current_phase = None

    # Create Generator
    data_gen = windows_generator_from_paths(
        parquet_paths,
        feature_cols,
        W=window_len,
        min_date=min_date,
        return_valid_mask=True,
    )
    
    step = start_step
    try:
        for date, X, valid_asset_mask in data_gen:
            # Stop if we exceeded max steps (for smoke test)
            if step >= start_step + max_steps:
                logger.info("Max steps reached. Stopping.")
                break

            # --- THE MAIN STEP ---
            # 1. Meta-Update (if Monday)
            # 2. Trade Execution (using Meta constraints)
            # 3. RL Update (using Meta rewards)
            ts = pd.to_datetime(date)
            if test_end_ts is not None and ts > test_end_ts:
                logger.info(f"Reached configured test_end_date ({test_end_ts.date()}). Stopping.")
                break

            phase = "train"
            if test_start_ts is not None and ts >= test_start_ts:
                phase = "test"
            elif train_end_ts is not None and ts > train_end_ts:
                phase = "test"

            if current_phase != phase:
                logger.info(f"[Phase Switch] {phase.upper()} phase starting at {ts.date()}")
                current_phase = phase
                rl_trainer.reset_recurrent_memory(clear_transition=True)

            rl_trainer.step_daily(
                X,
                str(date),
                step,
                valid_asset_mask=valid_asset_mask,
                allow_learning=(phase == "train"),
                phase=phase,
            )
            
            step += 1

            # Periodic Checkpoint
            if step % cfg.get("checkpoint_every_steps", 500) == 0:
                ckpt_mgr.save(step, actor=rl_trainer.agent.actor, critic=rl_trainer.agent.critic, meta_agent=meta_agent)

    except KeyboardInterrupt:
        logger.info("Training interrupted by user.")
    except Exception as e:
        logger.error(f"Training crashed: {e}")
        raise e
    finally:
        # Final Save on Exit
        ckpt_mgr.save(step, actor=rl_trainer.agent.actor, critic=rl_trainer.agent.critic, meta_agent=meta_agent)
        logger.info("Session Saved. Exiting.")

if __name__ == "__main__":
    main()