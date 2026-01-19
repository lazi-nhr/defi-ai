from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Optional

from defi_ai.sim.paper_engine import PaperEngine, PaperAccount
from defi_ai.utils.env import load_env, get_env


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--notional-usd", type=float, default=100.0)
    args = parser.parse_args()

    load_env()

    signal_path = Path(get_env("DEFI_AI_SIGNAL_PATH", "signal.json"))
    price_path = Path(get_env("DEFI_AI_LAST_PRICE_PATH", "artifacts/last_price.json"))

    engine = PaperEngine(PaperAccount(usd=args.notional_usd, base=0.0))
    last_ts = None

    while True:
        sig = _read_json(signal_path)
        if sig and sig.get("ts_ms") != last_ts:
            last_ts = sig.get("ts_ms")

            # price source:
            # - if you already write last price from your feed, point DEFI_AI_LAST_PRICE_PATH to it
            # - else fall back to 1.0
            px = 1.0
            px_data = _read_json(price_path)
            if px_data and "price" in px_data:
                try:
                    px = float(px_data["price"])
                except Exception:
                    pass

            decision = sig.get("decision") or {}
            result = engine.step(decision, px)
            print(f"[paper] signal={decision} price={px} result={result}")

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
