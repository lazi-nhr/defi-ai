from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


class DefiAiSignal(ScriptStrategyBase):
    """
    Minimal script strategy:
      - polls signal.json
      - places market orders based on action buy/sell/hold
    """

    markets = {}  # set dynamically in __init__

    def __init__(self, connectors: Dict[str, Any]):
        super().__init__(connectors)
        self.signal_path = Path(os.getenv("DEFI_AI_SIGNAL_PATH", "signal.json"))
        self.last_ts = None

        # You MUST set these for your environment.
        # Example: connector="binance_paper_trade" and trading_pair="BTC-USDT"
        self.connector_name = os.getenv("HB_CONNECTOR", "")
        self.trading_pair = os.getenv("HB_TRADING_PAIR", "")

        if not self.connector_name or not self.trading_pair:
            self.logger().warning(
                "HB_CONNECTOR and HB_TRADING_PAIR must be set. No orders will be placed."
            )

        self.markets = {self.connector_name: {self.trading_pair}} if self.connector_name and self.trading_pair else {}

    def on_tick(self):
        sig = _read_json(self.signal_path)
        if not sig:
            return

        ts = sig.get("ts_ms")
        if ts == self.last_ts:
            return
        self.last_ts = ts

        decision = sig.get("decision") or {}
        action = str(decision.get("action", "hold")).lower()
        risk_passed = bool(sig.get("risk_passed", False))

        if not self.connector_name or not self.trading_pair:
            return

        if not risk_passed:
            self.logger().info(f"Signal rejected by risk gate: {sig.get('risk_reason', '')}")
            return

        connector = self.connectors.get(self.connector_name)
        if connector is None:
            self.logger().warning(f"Connector not found: {self.connector_name}")
            return

        # VERY simple sizing: use a fixed quote amount (paper PoC)
        quote_amount = float(os.getenv("HB_QUOTE_AMOUNT", "50"))

        if action == "buy":
            self.logger().info(f"Placing BUY market order {self.trading_pair} quote={quote_amount}")
            self.buy(self.connector_name, self.trading_pair, quote_amount)

        elif action == "sell":
            # Sell uses base amount, not quote amount. For PoC, use a fixed base amount.
            base_amount = float(os.getenv("HB_BASE_AMOUNT", "0.001"))
            self.logger().info(f"Placing SELL market order {self.trading_pair} base={base_amount}")
            self.sell(self.connector_name, self.trading_pair, base_amount)

        else:
            self.logger().info("HOLD - no action")
