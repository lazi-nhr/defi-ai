CONFIG = {
    "DATA": {
        "forward_fill": True,
        "drop_na_after_ffill": True,
        "cache_dir": "./data_cache",
        "timestamp_format": "%Y-%m-%d %H:%M:%S",
        "asset_price_format": "{ASSET}_{FEATURE}",
        "pair_feature_format": "{ASSET1}_{ASSET2}_{FEATURE}",
        "timestamp_col": "timestamp",
        "sampling": "1m",
        "features": {
            "file_id": "1OCqEkOWV73Z8e-67fpqVL3r3ugVcfml8",
            "file_name": "bin_futures_full_features",
            "type": "csv",
            "separator": ",",
            "index": "datetime",
            "start": "2024-05-01 00:00:00",
            "end": "2025-05-01 00:00:00",
            "individual_identifier": "close",
            "pair_identifier": "beta",
        },
    },
    "ENV": {
        "include_cash": True,
        "action_space_type": "continuous",  # "continuous" or "discrete" - MUST match algorithm!
        "discrete_actions": [-1.0, -0.9, -0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # 21 discrete actions with 0.1 steps
        "trading_window_days": "1D",
        "sliding_window_step": "1D",
        "lookback_window": 24, # context
        "normalize_observations": True, # Normalize features for better learning
        "transaction_costs": {
            "taker_bps": 1.44, # fee based on tier system (0.0144% to 0.045%)
            "maker_bps": 1.0, # rebates (not implemented)
            "slippage_bps": 0, # slippage
        },
        "reward": {
            "risk_lambda": 0.0001,  # Further reduced to encourage more trading
            "lambda_utility": 1.5,  # Reduced for more aggressive trading
            "action_penalty": 0.0001,  # Small penalty to discourage rapid changes
            "action_smoothness_penalty": 0.0002,  # Quadratic penalty for large action changes
            "inaction_penalty": 0.0,  # Disabled - conflicts with action penalty
            "position_taking_reward": 0.0003,  # Reduced from 0.001 to avoid extreme positions
            "spread_direction_reward": 0.0002,  # Reduced from 0.0005 to moderate signal strength
            "min_action_change": 0.05,  # Only penalize changes larger than this threshold
        },
        "seed": 42,
    },
    "SPLITS": {
        "data_start": "2024-05-01",
        "data_end": "2025-04-30",
        #"train": ["2024-05-01 00:00:00", "2024-12-31 23:59:59"],  # 8 months for training
        #"val": ["2025-01-01 00:00:00", "2025-02-28 23:59:59"],    # 2 months for validation
        #"test": ["2025-03-01 00:00:00", "2025-04-30 23:59:59"],   # 2 months for testing
        "test": ["2024-05-01 00:00:00", "2024-06-30 23:59:59"],   # 2 months for testing
        "train": ["2024-07-01 00:00:00", "2025-02-28 23:59:59"],  # 8 months for training
        "val": ["2025-03-01 00:00:00", "2025-04-30 23:59:59"],    # 2 months for validation
    },
    "RL": {
        # ===== General (All Algorithms) =====
        "algorithm": "SAC",  # "SAC", "PPO", or "DQN"
        "timesteps": 2e6,  # Total training timesteps (1e6 - 3e6)
        "policy": "MlpPolicy",  # Policy network architecture
        "gamma": 0.99,  # Discount factor for future rewards
        "learning_rate": 3e-4,  # Learning rate for optimizer
        "batch_size": 64,  # Batch size for training updates
        
        # ===== PPO (On-Policy) =====
        "gae_lambda": 0.95,  # GAE lambda for advantage estimation
        "clip_range": 0.2,  # PPO clipping range for policy updates
        "n_steps": 2048,  # Number of steps to collect before update
        "n_epochs": 10,  # Number of epochs for policy optimization
        #"ent_coef": 0.01,  # Entropy coefficient for exploration (PPO-specific, conflicts with SAC)
        "vf_coef": 0.5,  # Value function loss coefficient
        "max_grad_norm": 0.5,  # Maximum gradient norm for clipping
        
        # ===== SAC (Off-Policy, Continuous) =====
        "ent_coef": "auto",  # Automatic entropy tuning for better exploration
        "buffer_size": 200000,  # Larger replay buffer for more diverse experiences
        "learning_starts": 5000,  # Collect more random experiences before learning
        "tau": 0.005,  # Soft update coefficient for target networks
        "train_freq": 1,  # Update model every N steps
        "gradient_steps": 1,  # Gradient steps per environment step
        
        # ===== DQN (Off-Policy, Discrete) =====
        # "buffer_size": 100000,  # Replay buffer size (shared with SAC)
        # "learning_starts": 1000,  # Steps before learning starts (shared with SAC)
        # "tau": 0.005,  # Soft update coefficient (shared with SAC)
        # "train_freq": 1,  # Update frequency (shared with SAC)
        # "gradient_steps": 1,  # Gradient steps (shared with SAC)
        "exploration_fraction": 0.1,  # Fraction of training for epsilon decay
        "exploration_initial_eps": 1.0,  # Initial epsilon for exploration
        "exploration_final_eps": 0.05,  # Final epsilon after decay
        "target_update_interval": 1,  # Update target network every N steps
    },
    "EVAL": {
        "plots": True,
        "reports_dir": "./reports",
        "frequency": 14400,  # Evaluate every 14,400 steps
        "n_eval_episodes": 1,  # Number of episodes to run during evaluation (default: 5)
        "save_freq": 100000,
    },
    "IO": {
        "models_dir": "./models",
        "tb_logdir": "./tb_logs",
    },
}