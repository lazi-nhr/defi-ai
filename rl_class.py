from config import CONFIG
import os
import json
import pandas as pd
import numpy as np
import pytz
from datetime import datetime
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback, CallbackList

# Import functions from rl_utils module
from rl_utils import (
    identify_assets_features_pairs,
    build_state_tensor_for_interval,
    PortfolioWeightsEnvUtility,
    ensure_dir,
)

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def load_csv_to_df(
    path: str,
    sep: str = ",",
    timestamp_index_col: str | None = "datetime",
    encoding: str = "utf-8-sig",
    **read_csv_kwargs,
) -> pd.DataFrame:
    """
    Load a CSV into a pandas DataFrame.

    Parameters
    ----------
    path : str
        Filesystem path to the CSV.
    parse_timestamp_col : str | None
        If provided and present in the CSV, this column will be parsed to datetime.
        Set to None to skip datetime parsing.
    **read_csv_kwargs :
        Extra arguments passed to `pd.read_csv` (e.g., sep, dtype, usecols).

    Returns
    -------
    pd.DataFrame
    """

    # Parse header-only to check for timestamp col presence
    head = pd.read_csv(path, sep=sep, encoding=encoding, nrows=0)
    if timestamp_index_col and timestamp_index_col in head.columns:
        read_csv_kwargs = {
            **read_csv_kwargs,
            "parse_dates": [timestamp_index_col],
        }

    df = pd.read_csv(path, sep=sep, encoding=encoding, engine="pyarrow", **read_csv_kwargs)

    df = df.set_index("datetime")

    return df


# load features
file_name = CONFIG["DATA"]["features"]["file_name"]
cache_dir = CONFIG["DATA"]["cache_dir"]
# Convert to absolute path if relative
if not os.path.isabs(cache_dir):
    cache_dir = os.path.join(SCRIPT_DIR, cache_dir)
    
index = CONFIG["DATA"]["features"]["index"]
sep = CONFIG["DATA"]["features"].get("seperator", ",")
file_path = os.path.join(cache_dir, f"{file_name}.csv")

# Check if file exists
if not os.path.exists(file_path):
    print(f"\n⚠ ERROR: Data file not found: {file_path}")
    print("\nPlease ensure the data file exists or run the data preparation notebook first.")
    print(f"Expected location: {os.path.abspath(file_path)}")
    print("\nYou can either:")
    print("  1. Download the data using the notebook cells")
    print("  2. Update CONFIG['DATA']['cache_dir'] to point to the correct location")
    print("  3. Update CONFIG['DATA']['features']['file_name'] if using a different file\n")
    exit(1)

features_df = load_csv_to_df(file_path, sep, timestamp_index_col=index)

# print dataframe info
print("Features DataFrame Info:")
print(features_df.info())




#######################################################################################




class PPOAgentManager:
    """
    Unified class for training, fine-tuning, and inference with PPO agent.
    
    Features:
    - Load raw CSV data and handle feature engineering
    - Train new models or fine-tune existing ones
    - Single timestep deterministic prediction
    - Automatic model saving/loading
    - Export predictions to JSON
    
    Example usage:
    ```python
    # Initialize manager
    manager = PPOAgentManager(config=CONFIG)
    
    # Train new model
    manager.train(
        csv_path="data/features.csv",
        train_period=["2024-01-01", "2024-06-30"],
        val_period=["2024-07-01", "2024-07-31"],
        timesteps=1000000
    )
    
    # Fine-tune on new data
    manager.fine_tune(
        csv_path="data/features_new.csv",
        model_path="models/best_model.zip",
        train_period=["2024-08-01", "2024-09-30"],
        timesteps=100000
    )
    
    # Predict single timestep
    result = manager.predict(
        csv_path="data/features_latest.csv",
        model_path="models/best_model.zip",
        timestamp="2024-10-01 12:00:00"
    )
    print(result)  # {"timestamp": "...", "action": 0.85, "confidence": 0.92}
    ```
    """
    
    def __init__(self, config: dict):
        """
        Initialize the PPO Agent Manager.
        
        Parameters
        ----------
        config : dict
            Configuration dictionary containing all settings
        """
        self.config = config
        self.model = None
        self.env = None
        self.feature_columns = None
        self.assets = None
        
    def _load_and_prepare_data(self, csv_path: str) -> tuple:
        """
        Load CSV data and prepare features.
        
        Parameters
        ----------
        csv_path : str
            Path to the CSV file containing features
            
        Returns
        -------
        tuple
            (features_df, X_tensor, R_tensor, VOL_tensor, timestamps, ticker_order)
        """
        print(f"Loading data from: {csv_path}")
        
        # Load CSV
        features_df = load_csv_to_df(
            csv_path,
            sep=self.config["DATA"]["features"].get("separator", ","),
            timestamp_index_col=self.config["DATA"]["features"]["index"]
        )
        
        print(f"Loaded {len(features_df)} rows with {len(features_df.columns)} columns")
        
        # Identify assets and features
        assets, single_features, pair_features, pairs = identify_assets_features_pairs(
            features_df,
            self.config["DATA"]["asset_price_format"],
            self.config["DATA"]["pair_feature_format"]
        )
        
        self.assets = sorted(list(assets))
        print(f"Detected {len(self.assets)} assets: {self.assets}")
        print(f"Single features: {sorted(single_features)}")
        print(f"Pair features: {sorted(pair_features)}")
        
        # Build state tensor
        X_tensor, R_tensor, VOL_tensor, timestamps, ticker_order = build_state_tensor_for_interval(
            features_df,
            self.assets,
            sorted(single_features),
            sorted(pair_features),
            lookback=self.config["ENV"]["lookback_window"]
        )
        
        print(f"Built tensors: X{X_tensor.shape}, R{R_tensor.shape}, VOL{VOL_tensor.shape}")
        print(f"Timestamps: {len(timestamps)} samples from {timestamps.min()} to {timestamps.max()}")
        
        return features_df, X_tensor, R_tensor, VOL_tensor, timestamps, ticker_order
    
    def _create_time_mask(self, timestamps: pd.DatetimeIndex, period: list) -> np.ndarray:
        """
        Create boolean mask for time period.
        
        Parameters
        ----------
        timestamps : pd.DatetimeIndex
            All available timestamps
        period : list
            [start_date, end_date] as strings
            
        Returns
        -------
        np.ndarray
            Boolean mask
        """
        start = pd.to_datetime(period[0]).tz_localize('UTC')
        end = pd.to_datetime(period[1]).tz_localize('UTC')
        
        # Ensure timestamps are UTC
        if timestamps.tz is None:
            timestamps = timestamps.tz_localize('UTC')
        elif timestamps.tz != pytz.UTC:
            timestamps = timestamps.tz_convert('UTC')
        
        mask = (timestamps >= start) & (timestamps <= end)
        print(f"Time mask: {mask.sum()} / {len(mask)} samples in period {period[0]} to {period[1]}")
        
        return mask
    
    def _create_env(self, X: np.ndarray, R: np.ndarray, VOL: np.ndarray, 
                    ticker_order: list, name: str = "env") -> gym.Env:
        """
        Create environment instance.
        
        Parameters
        ----------
        X, R, VOL : np.ndarray
            State tensors
        ticker_order : list
            List of asset tickers
        name : str
            Environment name for monitoring
            
        Returns
        -------
        gym.Env
            Wrapped environment
        """
        env = PortfolioWeightsEnvUtility(
            X, R, VOL, ticker_order,
            self.config["ENV"]["lookback_window"],
            self.config["ENV"]
        )
        env = Monitor(env, filename=None)
        return env
    
    def train(self, csv_path: str, train_period: list, val_period: list,
              timesteps: int = None, save_path: str = None) -> dict:
        """
        Train a new PPO model from scratch.
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file with features
        train_period : list
            [start_date, end_date] for training
        val_period : list
            [start_date, end_date] for validation
        timesteps : int, optional
            Number of training timesteps (uses config if None)
        save_path : str, optional
            Where to save the model (uses config if None)
            
        Returns
        -------
        dict
            Training metrics and paths
        """
        print("\n" + "="*70)
        print("TRAINING NEW MODEL")
        print("="*70)
        
        # Load and prepare data
        _, X_all, R_all, VOL_all, timestamps, ticker_order = self._load_and_prepare_data(csv_path)
        
        # Create time masks
        train_mask = self._create_time_mask(timestamps, train_period)
        val_mask = self._create_time_mask(timestamps, val_period)
        
        # Slice data
        X_train = X_all[train_mask]
        R_train = R_all[train_mask]
        VOL_train = VOL_all[train_mask]
        
        X_val = X_all[val_mask]
        R_val = R_all[val_mask]
        VOL_val = VOL_all[val_mask]
        
        # Create environments
        train_env = self._create_env(X_train, R_train, VOL_train, ticker_order, "train")
        val_env = self._create_env(X_val, R_val, VOL_val, ticker_order, "val")
        
        vec_train = DummyVecEnv([lambda: train_env])
        vec_val = DummyVecEnv([lambda: val_env])
        
        # Create model
        print("\nInitializing PPO model...")
        self.model = PPO(
            policy=self.config["RL"]["policy"],
            env=vec_train,
            gamma=self.config["RL"]["gamma"],
            gae_lambda=self.config["RL"]["gae_lambda"],
            clip_range=self.config["RL"]["clip_range"],
            n_steps=self.config["RL"]["n_steps"],
            batch_size=self.config["RL"]["batch_size"],
            learning_rate=self.config["RL"]["learning_rate"],
            ent_coef=self.config["RL"]["ent_coef"],
            vf_coef=self.config["RL"]["vf_coef"],
            max_grad_norm=self.config["RL"]["max_grad_norm"],
            tensorboard_log=self.config["IO"]["tb_logdir"],
            device="cpu",
            verbose=0
        )
        
        # Setup callbacks
        save_path = save_path or self.config["IO"]["models_dir"]
        ensure_dir(save_path)
        
        eval_callback = EvalCallback(
            vec_val,
            best_model_save_path=save_path,
            log_path=save_path,
            eval_freq=self.config["EVAL"]["frequency"],
            n_eval_episodes=self.config["EVAL"]["n_eval_episodes"],
            deterministic=True,
            render=False,
            verbose=0
        )
        
        # Train
        timesteps = timesteps or int(self.config["RL"]["timesteps"])
        print(f"\nStarting training for {timesteps:,} timesteps...")
        print("Monitor progress: tensorboard --logdir=./tb_logs\n")
        
        self.model.learn(total_timesteps=timesteps, callback=eval_callback, progress_bar=True)
        
        # Save final model
        final_path = os.path.join(save_path, "final_model.zip")
        self.model.save(final_path)
        print(f"\n✓ Training complete! Model saved to: {final_path}")
        
        return {
            "final_model_path": final_path,
            "best_model_path": os.path.join(save_path, "best_model.zip"),
            "timesteps": timesteps,
            "train_period": train_period,
            "val_period": val_period
        }
    
    def fine_tune(self, csv_path: str, model_path: str, train_period: list,
                  val_period: list = None, timesteps: int = None, 
                  save_path: str = None) -> dict:
        """
        Fine-tune an existing model on new data (transfer learning).
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file with new features
        model_path : str
            Path to pre-trained model
        train_period : list
            [start_date, end_date] for fine-tuning
        val_period : list, optional
            [start_date, end_date] for validation
        timesteps : int, optional
            Number of fine-tuning timesteps (default: 10% of original training)
        save_path : str, optional
            Where to save fine-tuned model
            
        Returns
        -------
        dict
            Fine-tuning metrics and paths
        """
        print("\n" + "="*70)
        print("FINE-TUNING EXISTING MODEL")
        print("="*70)
        
        # Load pre-trained model
        print(f"Loading pre-trained model from: {model_path}")
        self.model = PPO.load(model_path)
        
        # Load and prepare new data
        _, X_all, R_all, VOL_all, timestamps, ticker_order = self._load_and_prepare_data(csv_path)
        
        # Create time masks
        train_mask = self._create_time_mask(timestamps, train_period)
        
        # Slice data
        X_train = X_all[train_mask]
        R_train = R_all[train_mask]
        VOL_train = VOL_all[train_mask]
        
        # Create training environment
        train_env = self._create_env(X_train, R_train, VOL_train, ticker_order, "finetune_train")
        vec_train = DummyVecEnv([lambda: train_env])
        
        # Update model's environment
        self.model.set_env(vec_train)
        
        # Setup validation if provided
        vec_val = None
        if val_period:
            val_mask = self._create_time_mask(timestamps, val_period)
            X_val = X_all[val_mask]
            R_val = R_all[val_mask]
            VOL_val = VOL_all[val_mask]
            val_env = self._create_env(X_val, R_val, VOL_val, ticker_order, "finetune_val")
            vec_val = DummyVecEnv([lambda: val_env])
        
        # Setup callbacks
        save_path = save_path or self.config["IO"]["models_dir"]
        ensure_dir(save_path)
        
        callbacks = []
        if vec_val:
            eval_callback = EvalCallback(
                vec_val,
                best_model_save_path=save_path,
                log_path=save_path,
                eval_freq=self.config["EVAL"]["frequency"],
                n_eval_episodes=self.config["EVAL"]["n_eval_episodes"],
                deterministic=True,
                render=False,
                verbose=0
            )
            callbacks.append(eval_callback)
        
        # Fine-tune
        timesteps = timesteps or int(self.config["RL"]["timesteps"] * 0.1)  # 10% of original
        print(f"\nFine-tuning for {timesteps:,} timesteps...")
        
        callback = CallbackList(callbacks) if callbacks else None
        self.model.learn(total_timesteps=timesteps, callback=callback, progress_bar=True, reset_num_timesteps=False)
        
        # Save fine-tuned model
        finetuned_path = os.path.join(save_path, "finetuned_model.zip")
        self.model.save(finetuned_path)
        print(f"\n✓ Fine-tuning complete! Model saved to: {finetuned_path}")
        
        return {
            "finetuned_model_path": finetuned_path,
            "timesteps": timesteps,
            "train_period": train_period,
            "val_period": val_period
        }
    
    def predict(self, csv_path: str, model_path: str, timestamp: str,
                output_json: str = None) -> dict:
        """
        Predict action for a single timestep (deterministic).
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file with features
        model_path : str
            Path to trained model
        timestamp : str
            Timestamp for prediction (e.g., "2024-10-01 12:00:00")
        output_json : str, optional
            Path to save prediction as JSON
            
        Returns
        -------
        dict
            Prediction result with timestamp, action, and metadata
        """
        print("\n" + "="*70)
        print("SINGLE TIMESTEP PREDICTION")
        print("="*70)
        
        # Load model
        if self.model is None or model_path:
            print(f"Loading model from: {model_path}")
            self.model = PPO.load(model_path)
        
        # Load and prepare data
        features_df, X_all, R_all, VOL_all, timestamps, ticker_order = self._load_and_prepare_data(csv_path)
        
        # Find the timestamp
        target_time = pd.to_datetime(timestamp).tz_localize('UTC') if pd.to_datetime(timestamp).tz is None else pd.to_datetime(timestamp)
        
        if timestamps.tz is None:
            timestamps = timestamps.tz_localize('UTC')
        elif timestamps.tz != pytz.UTC:
            timestamps = timestamps.tz_convert('UTC')
        
        # Find closest timestamp
        time_diffs = np.abs((timestamps - target_time).total_seconds())
        idx = np.argmin(time_diffs)
        actual_time = timestamps[idx]
        
        if time_diffs[idx] > 3600:  # More than 1 hour difference
            print(f"⚠ Warning: Requested time {timestamp} not found exactly.")
            print(f"  Using closest timestamp: {actual_time} (diff: {time_diffs[idx]/60:.1f} minutes)")
        else:
            print(f"✓ Found timestamp: {actual_time}")
        
        # Get state
        state = X_all[idx]
        
        # Predict (deterministic)
        action, _states = self.model.predict(state, deterministic=True)
        action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        
        # Get additional context
        returns = R_all[idx]
        volatility = VOL_all[idx]
        
        # Build result
        result = {
            "timestamp": str(actual_time),
            "requested_timestamp": timestamp,
            "action": action_value,
            "model_path": model_path,
            "assets": ticker_order,
            "returns": returns.tolist() if isinstance(returns, np.ndarray) else [float(returns)],
            "volatility": volatility.tolist() if isinstance(volatility, np.ndarray) else [float(volatility)],
            "state_shape": list(state.shape),
            "prediction_time": datetime.now().isoformat()
        }
        
        # Save to JSON if requested
        if output_json:
            ensure_dir(os.path.dirname(output_json))
            with open(output_json, 'w') as f:
                json.dump(result, f, indent=2)
            print(f"✓ Prediction saved to: {output_json}")
        
        print("\n📊 Prediction Result:")
        print(f"   Action: {action_value:+.4f}")
        print(f"   Assets: {', '.join(ticker_order)}")
        
        return result
    
    def batch_predict(self, csv_path: str, model_path: str, 
                     start_time: str, end_time: str,
                     output_json: str = None) -> list:
        """
        Predict actions for multiple timesteps.
        
        Parameters
        ----------
        csv_path : str
            Path to CSV file with features
        model_path : str
            Path to trained model
        start_time, end_time : str
            Time range for predictions
        output_json : str, optional
            Path to save predictions as JSON
            
        Returns
        -------
        list
            List of prediction dictionaries
        """
        print("\n" + "="*70)
        print("BATCH PREDICTION")
        print("="*70)
        
        # Load model
        if self.model is None or model_path:
            print(f"Loading model from: {model_path}")
            self.model = PPO.load(model_path)
        
        # Load and prepare data
        _, X_all, R_all, VOL_all, timestamps, ticker_order = self._load_and_prepare_data(csv_path)
        
        # Create time mask
        period = [start_time, end_time]
        time_mask = self._create_time_mask(timestamps, period)
        
        # Get indices
        indices = np.where(time_mask)[0]
        print(f"Predicting for {len(indices)} timesteps...")
        
        results = []
        for idx in indices:
            state = X_all[idx]
            action, _ = self.model.predict(state, deterministic=True)
            action_value = float(action[0]) if isinstance(action, np.ndarray) else float(action)
            
            result = {
                "timestamp": str(timestamps[idx]),
                "action": action_value,
                "returns": R_all[idx].tolist() if isinstance(R_all[idx], np.ndarray) else [float(R_all[idx])],
                "volatility": VOL_all[idx].tolist() if isinstance(VOL_all[idx], np.ndarray) else [float(VOL_all[idx])]
            }
            results.append(result)
        
        # Save to JSON if requested
        if output_json:
            ensure_dir(os.path.dirname(output_json))
            output = {
                "model_path": model_path,
                "period": period,
                "assets": ticker_order,
                "num_predictions": len(results),
                "predictions": results
            }
            with open(output_json, 'w') as f:
                json.dump(output, f, indent=2)
            print(f"✓ Batch predictions saved to: {output_json}")
        
        return results

print("✓ PPOAgentManager class defined successfully!")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("PPOAgentManager is ready to use!")
    print("="*70)
    print("\nNote: The build_state_tensor_for_interval function needs to be")
    print("properly extracted from the notebook with the correct signature.")
    print("\nTo use this class, import it in your scripts:")
    print("  from rl_class import PPOAgentManager")
    print("\nOr uncomment the example code below after fixing the tensor builder.")
    print("="*70)
    
    # TODO: Uncomment after implementing proper build_state_tensor_for_interval
    # that handles multiple pairs and returns (X, R, VOL, timestamps, ticker_order)
    
    """
    # Example usage
    manager = PPOAgentManager(config=CONFIG)
    
    # Construct absolute paths
    cache_dir = CONFIG["DATA"]["cache_dir"]
    if not os.path.isabs(cache_dir):
        cache_dir = os.path.join(SCRIPT_DIR, cache_dir)
    
    csv_path = os.path.join(cache_dir, CONFIG["DATA"]["features"]["file_name"] + ".csv")
    
    models_dir = "./models/ppo_example"
    if not os.path.isabs(models_dir):
        models_dir = os.path.join(SCRIPT_DIR, models_dir)
    
    # Train a new model
    train_results = manager.train(
        csv_path=csv_path,
        train_period=CONFIG["SPLITS"]["train"],
        val_period=CONFIG["SPLITS"]["val"],
        timesteps=1_000_000,
        save_path=models_dir
    )
    
    # Predict for a single timestamp
    prediction = manager.predict(
        csv_path=csv_path,
        model_path=train_results["best_model_path"],
        timestamp="2024-10-01 12:00:00",
        output_json="./predictions/single_prediction.json"
    )
    
    print("\nSingle Prediction Output:")
    print(prediction)
    """