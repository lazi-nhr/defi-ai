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


# ---------------------------
# Control plane nodes (no webhooks)
# ---------------------------

from typing import Literal, Tuple

WatcherDecision = Literal[
    "KEEP_TRADING",
    "TRIGGER_SLOW_TICK_RESELECT",
    "TRIGGER_RL_FINETUNE",
    "PAUSE_PAIR",
]


def _get_control_int(state: AgentState, key: str, default: int) -> int:
    try:
        return int((state.control or {}).get(key, default))
    except Exception:
        return default


def _get_control_float(state: AgentState, key: str, default: float) -> float:
    try:
        return float((state.control or {}).get(key, default))
    except Exception:
        return default


def _control_set(state: AgentState, key: str, value: Any) -> None:
    if state.control is None:
        state.control = {}
    state.control[key] = value


def _control_get(state: AgentState, key: str, default: Any = None) -> Any:
    if state.control is None:
        state.control = {}
    return state.control.get(key, default)


def _load_perf_snapshot() -> Dict[str, Any]:
    """
    Minimal performance input interface.

    If you already have a simulator / paper engine, have it write a JSON file that this node reads.

    Default path: artifacts/perf/perf_snapshot.json
    You can override via PERF_SNAPSHOT_PATH env var.
    """
    perf_path = Path(get_env("PERF_SNAPSHOT_PATH", str(ARTIFACTS_DIR / "perf" / "perf_snapshot.json")))
    return _read_json(perf_path) or {}


@traceable(name="metrics_rollup")
def metrics_rollup(state: AgentState) -> AgentState:
    """
    Reads a lightweight performance snapshot and stores a "rollup" in state.control.

    Expected (example) perf snapshot JSON shape:
      {
        "ts_ms": 123,
        "pair": "BTC-ETH",
        "window": "6h",
        "pnl_usd": -12.3,
        "pnl_pct": -0.8,
        "trades": 42,
        "winrate": 0.43,
        "max_drawdown_pct": 1.7,
        "avg_slippage_bps": 9.2,
        "fill_rate": 0.98,
        "avg_latency_ms": 120
      }

    If the file is absent, the node is a safe no-op (rollup defaults).
    """
    snap = _load_perf_snapshot()

    # Default rollup if no perf exists yet (keeps pipeline stable)
    rollup = {
        "ts_ms": int(snap.get("ts_ms", _now_ms())),
        "pair": str(snap.get("pair", getattr(state, "pair", "") or "")),
        "window": str(snap.get("window", "6h")),
        "pnl_usd": float(snap.get("pnl_usd", 0.0)),
        "pnl_pct": float(snap.get("pnl_pct", 0.0)),
        "trades": int(snap.get("trades", 0)),
        "winrate": float(snap.get("winrate", 0.0)),
        "max_drawdown_pct": float(snap.get("max_drawdown_pct", 0.0)),
        "avg_slippage_bps": float(snap.get("avg_slippage_bps", 0.0)),
        "fill_rate": float(snap.get("fill_rate", 0.0)),
        "avg_latency_ms": float(snap.get("avg_latency_ms", 0.0)),
    }

    _control_set(state, "perf_rollup", rollup)
    return state


@traceable(name="watcher_eval")
def watcher_eval(state: AgentState) -> AgentState:
    """
    ChatOllama watcher that recommends whether to keep trading,
    re-run slow tick selection, request RL finetuning, or pause the pair.

    Output written to state.control["watcher_verdict"] as JSON-safe dict:
      {
        "decision": "...",
        "confidence": 0.0-1.0,
        "reasons": [ ... ],
      }
    """
    llm = _get_llm()

    rollup = _control_get(state, "perf_rollup", {}) or {}
    sel = _control_get(state, "selected_pair", {"asset1": "BTC", "asset2": "ETH"})
    pair = getattr(state, "pair", f"{sel.get('asset1','')}-{sel.get('asset2','')}")

    # Guardrails: enforce schema; watcher must choose from fixed decisions.
    prompt = (
        "You are a trading performance watchdog for a pairs trading system.\n"
        "You do NOT execute trades. You only recommend the next control action.\n\n"
        "Return ONLY a single-line JSON object with keys:\n"
        "  decision: one of [KEEP_TRADING, TRIGGER_SLOW_TICK_RESELECT, TRIGGER_RL_FINETUNE, PAUSE_PAIR]\n"
        "  confidence: number between 0 and 1\n"
        "  reasons: array of short strings\n\n"
        f"PAIR: {pair}\n"
        f"ROLLUP: {json.dumps(rollup, sort_keys=True)}\n"
        f"RISK_FLAG_FROM_LLM_ASSIST: {bool((getattr(state, 'llm_advice', {}) or {}).get('risk_flag', False))}\n"
    )

    raw = llm.invoke(prompt) if hasattr(llm, "invoke") else llm(prompt)
    txt = raw if isinstance(raw, str) else str(raw)

    verdict = {
        "decision": "KEEP_TRADING",
        "confidence": 0.0,
        "reasons": ["default_no_data"],
        "raw": txt,
    }

    try:
        parsed = json.loads(txt.strip())
        if isinstance(parsed, dict):
            decision = str(parsed.get("decision", "KEEP_TRADING")).strip()
            confidence = float(parsed.get("confidence", 0.0))
            reasons = parsed.get("reasons", [])
            if decision in (
                "KEEP_TRADING",
                "TRIGGER_SLOW_TICK_RESELECT",
                "TRIGGER_RL_FINETUNE",
                "PAUSE_PAIR",
            ):
                verdict["decision"] = decision
            verdict["confidence"] = max(0.0, min(1.0, confidence))
            verdict["reasons"] = reasons if isinstance(reasons, list) else [str(reasons)]
    except Exception:
        # keep defaults, preserve raw
        pass

    _control_set(state, "watcher_verdict", verdict)
    return state


@traceable(name="governor")
def governor(state: AgentState) -> AgentState:
    """
    Applies hysteresis + cooldown and outputs an authorized action in:
      state.control["governor_action"]

    Configuration keys (all optional, via control.json or env injected into state.control):
      watcher_interval_s (default 600)
      reselection_cooldown_s (default 21600)  # 6h
      finetune_cooldown_s (default 86400)     # 24h
      pause_on_confidence_ge (default 0.85)
      min_confidence_to_act (default 0.60)
      consecutive_breach_required (default 2)

    State keys used (stored in control):
      governor_last_eval_ms
      governor_last_reselect_ms
      governor_last_finetune_ms
      governor_consecutive_breaches
    """
    now = _now_ms()

    watcher_interval_s = _get_control_int(state, "watcher_interval_s", 600)
    last_eval = _get_control_int(state, "governor_last_eval_ms", 0)
    if last_eval and (now - last_eval) < watcher_interval_s * 1000:
        # Not time yet: explicitly set KEEP_TRADING and return
        _control_set(state, "governor_action", "KEEP_TRADING")
        _control_set(state, "governor_skipped", True)
        return state

    _control_set(state, "governor_last_eval_ms", now)
    _control_set(state, "governor_skipped", False)

    verdict = _control_get(state, "watcher_verdict", {}) or {}
    decision = str(verdict.get("decision", "KEEP_TRADING"))
    conf = float(verdict.get("confidence", 0.0))

    min_conf = _get_control_float(state, "min_confidence_to_act", 0.60)
    pause_on_conf = _get_control_float(state, "pause_on_confidence_ge", 0.85)

    reselection_cooldown_s = _get_control_int(state, "reselection_cooldown_s", 6 * 3600)
    finetune_cooldown_s = _get_control_int(state, "finetune_cooldown_s", 24 * 3600)
    last_reselect = _get_control_int(state, "governor_last_reselect_ms", 0)
    last_finetune = _get_control_int(state, "governor_last_finetune_ms", 0)

    consecutive_required = _get_control_int(state, "consecutive_breach_required", 2)
    consecutive = _get_control_int(state, "governor_consecutive_breaches", 0)

    # Hard override: PAUSE if watcher says pause with high confidence
    if decision == "PAUSE_PAIR" and conf >= pause_on_conf:
        _control_set(state, "governor_action", "PAUSE_PAIR")
        _control_set(state, "governor_consecutive_breaches", 0)
        return state

    # If confidence too low, do nothing
    if conf < min_conf:
        _control_set(state, "governor_action", "KEEP_TRADING")
        _control_set(state, "governor_consecutive_breaches", 0)
        return state

    # Hysteresis: count breaches for non-KEEP decisions
    if decision in ("TRIGGER_SLOW_TICK_RESELECT", "TRIGGER_RL_FINETUNE", "PAUSE_PAIR"):
        consecutive += 1
    else:
        consecutive = 0
    _control_set(state, "governor_consecutive_breaches", consecutive)

    if consecutive < consecutive_required:
        _control_set(state, "governor_action", "KEEP_TRADING")
        return state

    # Cooldowns for reselection/fine-tune
    if decision == "TRIGGER_SLOW_TICK_RESELECT":
        if last_reselect and (now - last_reselect) < reselection_cooldown_s * 1000:
            _control_set(state, "governor_action", "KEEP_TRADING")
            return state
        _control_set(state, "governor_last_reselect_ms", now)
        _control_set(state, "governor_action", "TRIGGER_SLOW_TICK_RESELECT")
        _control_set(state, "governor_consecutive_breaches", 0)
        return state

    if decision == "TRIGGER_RL_FINETUNE":
        if last_finetune and (now - last_finetune) < finetune_cooldown_s * 1000:
            _control_set(state, "governor_action", "KEEP_TRADING")
            return state
        _control_set(state, "governor_last_finetune_ms", now)
        _control_set(state, "governor_action", "TRIGGER_RL_FINETUNE")
        _control_set(state, "governor_consecutive_breaches", 0)
        return state

    if decision == "PAUSE_PAIR":
        # Lower-confidence pause (not "hard") still allowed after hysteresis
        _control_set(state, "governor_action", "PAUSE_PAIR")
        _control_set(state, "governor_consecutive_breaches", 0)
        return state

    _control_set(state, "governor_action", "KEEP_TRADING")
    return state


@traceable(name="apply_governor_action")
def apply_governor_action(state: AgentState) -> AgentState:
    """
    Executes the authorized action locally (no webhooks):
      - RESELECT -> runs pair_selection() and updates ACTIVE_PAIRS_PATH (your existing node)
      - FINETUNE -> records a finetune request artifact flag (you wire training separately)
      - PAUSE    -> sets control flag and writes a pause artifact

    This node is safe to call every tick; it only performs side effects when governor_action != KEEP_TRADING.
    """
    action = str(_control_get(state, "governor_action", "KEEP_TRADING"))

    if action == "TRIGGER_SLOW_TICK_RESELECT":
        # Re-use your existing slow tick implementation
        state = pair_selection(state)
        _control_set(state, "last_control_action", {"ts_ms": _now_ms(), "action": action})
        return state

    if action == "TRIGGER_RL_FINETUNE":
        # Minimal PoC: emit a training request artifact; your trainer picks it up.
        req_path = ARTIFACTS_DIR / "rl" / "finetune_request.json"
        payload = {
            "ts_ms": _now_ms(),
            "pair": getattr(state, "pair", ""),
            "selected_pair": _control_get(state, "selected_pair", None),
            "selected_pair_meta": _control_get(state, "selected_pair_meta", None),
            "perf_rollup": _control_get(state, "perf_rollup", None),
            "watcher_verdict": _control_get(state, "watcher_verdict", None),
        }
        _write_json(req_path, _json_safe(payload))
        _control_set(state, "finetune_requested", True)
        _control_set(state, "last_control_action", {"ts_ms": _now_ms(), "action": action})
        return state

    if action == "PAUSE_PAIR":
        pause_path = ARTIFACTS_DIR / "control" / "pause.json"
        payload = {
            "ts_ms": _now_ms(),
            "pair": getattr(state, "pair", ""),
            "reason": _control_get(state, "watcher_verdict", {}),
        }
        _write_json(pause_path, _json_safe(payload))
        _control_set(state, "paused", True)
        _control_set(state, "last_control_action", {"ts_ms": _now_ms(), "action": action})
        return state

    return state




