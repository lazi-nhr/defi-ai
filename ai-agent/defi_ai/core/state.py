from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class RuntimeConfig:
    # Existing config (kept as-is)
    interval: str = "1h"
    candle_limit: int = 250
    lookback: int = 30
    notional_usd: float = 100.0
    tick_seconds: int = 60
    default_model_path: Path = Path("models/best_model.zip")

    # PoC additions (safe defaults)
    slow_tick_seconds: int = 1800
    default_pair: str = "BTC-USDT"
    artifacts_dir: Path = Path("artifacts")
    signal_path: Path = Path("signal.json")


@dataclass
class AgentState:
    """
    LangGraph state container (per-run/per-tick).
    This is intentionally separate from RuntimeConfig so you do not break existing code.
    """
    # Tick metadata
    tick_type: str = "fast"
    tick_started_ms: int = 0

    # Config (embedded)
    config: RuntimeConfig = field(default_factory=RuntimeConfig)

    # Control/config loaded at runtime (optional)
    control: Dict[str, Any] = field(default_factory=dict)

    # Market context
    pair: str = ""
    market_candles: List[Dict[str, Any]] = field(default_factory=list)
    features: Dict[str, Any] = field(default_factory=dict)
    market: Dict[str, Any] = field(default_factory=dict)


    # Position state (PoC)
    position: str = "flat"

    # LLM metadata (do NOT override your actual LLaMA wiring; this is for trace context)
    llm_backend: str = "local"
    llm_model_id: str = "local-llama"

    # Inference artifacts
    raw_llm_output: str = ""
    decision: Dict[str, Any] = field(default_factory=dict)

    # Risk
    risk_passed: bool = False
    risk_reason: str = ""

    # Output artifacts
    last_signal: Dict[str, Any] = field(default_factory=dict)
