DeFi-AI Signal Script (Paper Mode)

Purpose:
- Minimal Hummingbot script strategy that reads signal.json and places market orders.
- Intended as a PoC integration path (simple and auditable).

Usage:
1) Copy hummingbot_scripts/defi_ai_signal.py into your Hummingbot "scripts/" directory.
2) Set environment variable DEFI_AI_SIGNAL_PATH to the absolute path of your repo's signal.json
3) Start Hummingbot in paper mode.
4) Run:
   script defi_ai_signal

Notes:
- This script is intentionally minimal. It does not do advanced position sizing, slippage modelling, or retries.
- It assumes a connector is configured and that market order placement is supported.
