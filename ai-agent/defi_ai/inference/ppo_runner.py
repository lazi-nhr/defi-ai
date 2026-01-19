from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from stable_baselines3 import PPO

import numpy as np

try:
    from stable_baselines3 import PPO
except Exception as e:  # pragma: no cover
    PPO = None  # type: ignore


PAIR_FEATURES_ORDER = [
    "alpha",
    "beta",
    "corr",
    "pval",
    "spreadNorm",
    "spreadNormKalman",
    "spreadNormMa",
    "spreadNormVol",
]


def continuous_action_to_weights(action: float) -> tuple[float, float]:
    action = float(np.clip(action, -1.0, 1.0))
    position_size = action * 0.5
    return position_size, -position_size


@dataclass
class PPOModelRunner:
    model_path: Path
    _model: Optional[object] = None

    

    def _normalize_sb3_path(self, p: Path) -> str:
        s = str(p)
        # SB3 sometimes expects the base path and appends ".zip" internally in some contexts.
        # Also guard against accidental ".zip.zip".
        if s.endswith(".zip.zip"):
            s = s[:-4]  # remove one ".zip"
        return s

    def load(self):
        if self._model is None:
            path = self._normalize_sb3_path(Path(self.model_path))
            self._model = PPO.load(path)


    def predict_weights(self, pair_features: dict[str, list[float]], lookback: int) -> tuple[float, float, float]:
        """Returns (w1, w2, raw_action)."""
        self.load()
        obs = np.zeros((len(PAIR_FEATURES_ORDER), lookback), dtype=np.float32)
        for i, feat in enumerate(PAIR_FEATURES_ORDER):
            obs[i, :] = np.asarray(pair_features[feat], dtype=np.float32)
        obs = obs.reshape(-1).astype(np.float32)
        obs = np.clip(obs, -5.0, 5.0)

        # Align observation length to the trained policy's observation space
        expected = int(self._model.observation_space.shape[0])  # type: ignore[attr-defined]
        if obs.shape[0] < expected:
            obs = np.pad(obs, (0, expected - obs.shape[0]), mode="constant")
        elif obs.shape[0] > expected:
            obs = obs[:expected]

        action, _ = self._model.predict(obs, deterministic=True)  # type: ignore[attr-defined]
        a = float(action[0]) if isinstance(action, np.ndarray) else float(action)
        w1, w2 = continuous_action_to_weights(a)
        return float(w1), float(w2), a
