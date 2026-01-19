from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from defi_ai.core.state import AgentState
from defi_ai.orchestration import nodes


def build_trading_graph() -> Any:
    """
    Fast tick graph + control plane (no webhooks):
      load_control
        -> resolve_pair_and_model
        -> metrics_rollup
        -> watcher_eval
        -> governor
        -> apply_governor_action
        -> fetch_candles
        -> compute_features
        -> rl_inference
        -> llm_assist
        -> risk_checks
        -> write_signal
        -> END
    """
    g = StateGraph(AgentState)

    # Existing
    g.add_node("load_control", nodes.load_control)
    g.add_node("resolve_pair_and_model", nodes.resolve_pair_and_model)
    g.add_node("fetch_candles", nodes.fetch_candles)
    g.add_node("compute_features", nodes.compute_features)
    g.add_node("rl_inference", nodes.rl_inference)
    g.add_node("llm_assist", nodes.llm_assist)
    g.add_node("risk_checks", nodes.risk_checks)
    g.add_node("write_signal", nodes.write_signal)

    # New control-plane nodes
    g.add_node("metrics_rollup", nodes.metrics_rollup)
    g.add_node("watcher_eval", nodes.watcher_eval)
    g.add_node("governor", nodes.governor)
    g.add_node("apply_governor_action", nodes.apply_governor_action)

    g.set_entry_point("load_control")

    # Control-plane first (so it can reselect/pause BEFORE trading tick)
    g.add_edge("load_control", "resolve_pair_and_model")
    g.add_edge("resolve_pair_and_model", "metrics_rollup")
    g.add_edge("metrics_rollup", "watcher_eval")
    g.add_edge("watcher_eval", "governor")
    g.add_edge("governor", "apply_governor_action")

    # Then proceed with your original fast tick
    g.add_edge("apply_governor_action", "fetch_candles")
    g.add_edge("fetch_candles", "compute_features")
    g.add_edge("compute_features", "rl_inference")
    g.add_edge("rl_inference", "llm_assist")
    g.add_edge("llm_assist", "risk_checks")
    g.add_edge("risk_checks", "write_signal")
    g.add_edge("write_signal", END)

    return g.compile()



def build_pair_selection_graph() -> Any:
    """
    Slow tick graph:
      pair_selection -> END
    """
    g = StateGraph(AgentState)
    g.add_node("pair_selection", nodes.pair_selection)
    g.set_entry_point("pair_selection")
    g.add_edge("pair_selection", END)
    return g.compile()
