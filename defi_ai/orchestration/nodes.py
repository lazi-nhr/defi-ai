from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from langsmith import traceable

from defi_ai.core import state
from defi_ai.core.state import AgentState
from defi_ai.features.pair_features import compute_pair_features
from defi_ai.utils.env import get_env, ensure_dirs

from defi_ai.pairs.selector import PairSelectionConfig, select_top_pairs
from defi_ai.marketdata.hyperliquid import CandleConfig, HyperliquidClient

from defi_ai.inference.ppo_runner import PPOModelRunner
from defi_ai.inference.signal import generate_signal, SignalConfig


ARTIFACTS_DIR = Path(get_env("ARTIFACTS_DIR", "artifacts"))
SIGNAL_PATH = Path(get_env("DEFI_AI_SIGNAL_PATH", "signal.json"))
ACTIVE_PAIRS_PATH = ARTIFACTS_DIR / "pairs" / "active_pairs.json"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")

def _json_safe(obj: Any) -> Any:
    """Convert common non-JSON-safe objects (pandas/numpy) into JSON-safe structures."""
    try:
        import numpy as np
    except Exception:  # pragma: no cover
        np = None  # type: ignore

    # primitives
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # pandas Timestamp / datetime-like
    if hasattr(obj, "isoformat") and "pandas" in str(type(obj)).lower():
        return obj.isoformat()

    # numpy scalars
    if np is not None and isinstance(obj, getattr(np, "generic", ())):
        return obj.item()

    # dict -> force string keys
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    # list/tuple
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]

    # numpy arrays
    if np is not None and isinstance(obj, getattr(np, "ndarray", ())):
        return [_json_safe(x) for x in obj.tolist()]

    # fallback
    return str(obj)



def _get_active_pair_fallback(default_pair: str) -> str:
    data = _read_json(ACTIVE_PAIRS_PATH)
    if not data:
        return default_pair
    return data.get("selected_pair") or default_pair


# ---------------------------
# Node implementations
# ---------------------------

@traceable(name="load_control")
def load_control(state: AgentState) -> AgentState:
    """
    Loads runtime control/config and ensures artifact directories exist.
    This keeps the PoC deterministic and traceable.
    """
    ensure_dirs([ARTIFACTS_DIR, ACTIVE_PAIRS_PATH.parent])

    # A small control file is optional; you can extend this later.
    control_path = Path(get_env("CONTROL_PATH", str(ARTIFACTS_DIR / "control.json")))
    control = _read_json(control_path) or {}

    state.control = control
    state.tick_started_ms = _now_ms()
    return state

def _get_active_pair(default_sel: Dict[str, str]) -> Dict[str, str]:
    data = _read_json(ACTIVE_PAIRS_PATH) or {}
    sel = data.get("selected_pair")
    if isinstance(sel, dict) and "asset1" in sel and "asset2" in sel:
        return {"asset1": sel["asset1"], "asset2": sel["asset2"]}
    return default_sel


@traceable(name="resolve_pair_and_model")
def resolve_pair_and_model(state: AgentState) -> AgentState:
    default_sel = state.control.get("selected_pair") or {"asset1": "BTC", "asset2": "ETH"}
    sel = _get_active_pair(default_sel)

    state.control["selected_pair"] = sel
    state.pair = f"{sel['asset1']}-{sel['asset2']}"  # for display/tracing only

    state.llm_backend = get_env("LLM_BACKEND", getattr(state, "llm_backend", "local"))
    state.llm_model_id = get_env("LLM_MODEL_ID", getattr(state, "llm_model_id", "local-llama"))
    return state


@traceable(name="fetch_candles")
def fetch_candles(state: AgentState) -> AgentState:
    sel = state.control.get("selected_pair") or {"asset1": "BTC", "asset2": "ETH"}
    a1, a2 = sel["asset1"], sel["asset2"]

    client = HyperliquidClient()
    cfg = CandleConfig(interval=state.config.interval, limit=state.config.candle_limit)

    df1 = client.fetch_candles(a1, cfg)
    df2 = client.fetch_candles(a2, cfg)

    if df1 is None or df2 is None or df1.empty or df2.empty:
        raise RuntimeError(f"No candles returned for {a1} or {a2} (interval={cfg.interval}, limit={cfg.limit})")

    # IMPORTANT: only store JSON-safe primitives in state (no Series/DataFrames/Timestamps)
    p1_close = df1["close"].astype(float).tolist()
    p2_close = df2["close"].astype(float).tolist()

    # Align lengths (use the last common window)
    n = min(len(p1_close), len(p2_close))
    p1_close = p1_close[-n:]
    p2_close = p2_close[-n:]

    state.market_candles = None  # optional: avoid stale single-series usage elsewhere
    state.control["selected_pair"] = sel
    state.control["asset1"] = a1
    state.control["asset2"] = a2
    state.control["p1_close"] = p1_close
    state.control["p2_close"] = p2_close

    return state



@traceable(name="compute_features")
def compute_features(state: AgentState) -> AgentState:
    # Read JSON-safe closes from control
    p1_close = state.control.get("p1_close")
    p2_close = state.control.get("p2_close")
    if not p1_close or not p2_close:
        raise RuntimeError("Missing p1_close/p2_close. fetch_candles must run before compute_features.")

    # Build Series with RangeIndex (no Timestamp keys)
    import pandas as pd
    p1 = pd.Series(p1_close)
    p2 = pd.Series(p2_close)

    # Optional hedge params from selector (if you persist them); otherwise compute_pair_features estimates.
    alpha = None
    beta = None
    sel_meta = state.control.get("selected_pair_meta") or {}
    if "alpha" in sel_meta and "beta" in sel_meta:
        alpha = float(sel_meta["alpha"])
        beta = float(sel_meta["beta"])

    feats = compute_pair_features(
        p1=p1,
        p2=p2,
        lookback=state.config.lookback,
        alpha=alpha,
        beta=beta,
    )

    # Ensure JSON-safe for tracing
    state.features = _json_safe(feats)
    return state


def _get_llm():
    from langchain_ollama import ChatOllama
    import os

    return ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "llama3"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        temperature=0.1,
    )




def _build_prompt(state: AgentState) -> str:
    """
    Simple prompt. If you already have your own prompt templates, use them here.
    """
    return (
        "You are a risk-aware trading agent.\n"
        "Return a single-line JSON object with keys: action, confidence, rationale.\n"
        "Allowed actions: buy, sell, hold.\n\n"
        f"Pair: {state.pair}\n"
        f"Features: {json.dumps(state.features, sort_keys=True)}\n"
        f"Position: {state.position}\n"
    )



_PPO_RUNNER = None

@traceable(name="rl_inference")
def rl_inference(state):
    global _PPO_RUNNER

    if _PPO_RUNNER is None:
        _PPO_RUNNER = PPOModelRunner(
            model_path=state.config.default_model_path
        )

    sel = state.control.get("selected_pair") or _get_active_pair({"asset1": "BTC", "asset2": "ETH"})
    state.control["selected_pair"] = sel

    asset1, asset2 = sel["asset1"], sel["asset2"]

    signal = generate_signal(
        runner=_PPO_RUNNER,
        model_path=state.config.default_model_path,
        asset1=asset1,
        asset2=asset2,
        pair_features=state.features,
        lookback=state.config.lookback,
        notional_usd=state.config.notional_usd,
        cfg=SignalConfig(bar_timeframe=state.config.interval),
    )

    # RL is the source of truth
    state.decision = _json_safe(signal)
    return state



@traceable(name="llm_assist")
def llm_assist(state: AgentState) -> AgentState:
    llm = _get_llm()

    prompt = (
        "You are assisting an RL trading system.\n"
        "Do NOT change the RL decision.\n"
        "Return ONLY JSON: {note: string, risk_flag: boolean}.\n\n"
        f"RL decision:\n{json.dumps(state.decision, sort_keys=True)}\n"
    )

    raw = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
    txt = raw if isinstance(raw, str) else str(raw)

    advice = {"note": txt, "risk_flag": False}
    try:
        parsed = json.loads(txt.strip())
        if isinstance(parsed, dict):
            advice["note"] = str(parsed.get("note", advice["note"]))
            advice["risk_flag"] = bool(parsed.get("risk_flag", False))
    except Exception:
        pass

    # Attach without overriding the RL decision
    state.llm_advice = advice
    if isinstance(state.decision, dict):
        state.decision.setdefault("debug", {})
        state.decision["debug"]["llm"] = advice

    return state



@traceable(name="risk_checks")
def risk_checks(state: AgentState) -> AgentState:
    d = state.decision if isinstance(state.decision, dict) else {}

    # Example: expect targets dict for rebalance
    targets = d.get("targets") or d.get("weights")
    if targets is None:
        # If your signal schema differs, adapt this check.
        state.risk_passed = False
        state.risk_reason = "missing_targets"
        return state

    state.risk_passed = True
    state.risk_reason = ""
    return state



@traceable(name="write_signal")
def write_signal(state: AgentState) -> AgentState:
    """
    Writes a single signal.json artifact for Hummingbot paper-mode or the local simulator.
    """
    out = {
        "ts_ms": _now_ms(),
        "pair": state.pair,
        "decision": state.decision,
        "risk_passed": bool(state.risk_passed),
        "risk_reason": state.risk_reason,
        "llm_backend": state.llm_backend,
        "llm_model_id": state.llm_model_id,
    }
    _write_json(SIGNAL_PATH, out)
    state.last_signal = out
    return state




@traceable(name="pair_selection")
def pair_selection(state):
    universe = state.control.get("universe", ["BTC", "ETH", "SOL", "AVAX", "ARB"])

    cfg = PairSelectionConfig(
        assets=universe,
        interval=state.config.interval,
        limit=state.config.candle_limit,
        top_k=state.control.get("pairs_top_k", 5),
    )

    client = HyperliquidClient()
    ranked = select_top_pairs(cfg, client)

    best = ranked[0]
    payload = {
        "selected_pair": {
            "asset1": best.asset1,
            "asset2": best.asset2,
        },
        "selected_pair_meta": {"alpha": best.alpha, "beta": best.beta},

        "ranked_pairs": [
            {
                "asset1": p.asset1,
                "asset2": p.asset2,
                "score": p.score,
                "corr": p.corr,
                "p_value": p.p_value,
                "alpha": p.alpha,
                "beta": p.beta,
            }
            for p in ranked
        ],
    }

    _write_json(ACTIVE_PAIRS_PATH, payload)
    state.control["selected_pair"] = payload["selected_pair"]
    state.control["selected_pair_meta"] = payload["selected_pair_meta"]
    return state



