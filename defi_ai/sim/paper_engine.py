from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class PaperAccount:
    usd: float = 100.0
    base: float = 0.0  # coin amount


class PaperEngine:
    """
    Minimal paper engine:
    - Reads a decision (buy/sell/hold) + a price
    - Executes market-style trades with full notional
    - Tracks balances
    """

    def __init__(self, account: Optional[PaperAccount] = None):
        self.account = account or PaperAccount()

    def step(self, decision: Dict[str, Any], price: float) -> Dict[str, Any]:
        action = (decision or {}).get("action", "hold")
        action = str(action).lower()

        if price <= 0:
            return {"executed": False, "reason": "invalid_price", "account": self.snapshot()}

        if action == "buy" and self.account.usd > 0:
            # all-in buy
            qty = self.account.usd / price
            self.account.base += qty
            self.account.usd = 0.0
            return {"executed": True, "action": "buy", "qty": qty, "price": price, "account": self.snapshot()}

        if action == "sell" and self.account.base > 0:
            # all-in sell
            usd = self.account.base * price
            qty = self.account.base
            self.account.base = 0.0
            self.account.usd += usd
            return {"executed": True, "action": "sell", "qty": qty, "price": price, "account": self.snapshot()}

        return {"executed": False, "reason": "hold_or_no_balance", "account": self.snapshot()}

    def snapshot(self) -> Dict[str, float]:
        return {"usd": float(self.account.usd), "base": float(self.account.base)}
