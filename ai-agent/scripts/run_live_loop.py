from __future__ import annotations

# IMPORTANT: load env before importing traceable / building graphs
from defi_ai.utils.env import load_env
load_env()

from scripts.langsmith_force_simple_ingest import force_simple_langsmith_ingest
force_simple_langsmith_ingest()


import argparse
import asyncio
import time

from langsmith import traceable
from defi_ai.core.state import AgentState
from defi_ai.orchestration.graph import build_trading_graph, build_pair_selection_graph


@traceable(name="fast_tick_run")
async def run_fast_tick(tick_seconds: int, slow_tick_seconds: int) -> None:
    g = build_trading_graph()
    state = AgentState()
    state.tick_type = "fast"
    state.config.tick_seconds = tick_seconds
    state.config.slow_tick_seconds = slow_tick_seconds

    while True:
        start = time.time()
        _ = g.invoke(state)
        elapsed = time.time() - start
        await asyncio.sleep(max(0.0, tick_seconds - elapsed))


@traceable(name="slow_tick_run")
async def run_slow_tick(slow_tick_seconds: int, tick_seconds: int) -> None:
    g = build_pair_selection_graph()
    state = AgentState()
    state.tick_type = "slow"
    state.config.tick_seconds = tick_seconds
    state.config.slow_tick_seconds = slow_tick_seconds

    g.invoke(state)  # run once immediately

    while True:
        start = time.time()
        g.invoke(state)
        elapsed = time.time() - start
        await asyncio.sleep(max(0.0, slow_tick_seconds - elapsed))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tick-seconds", type=int, default=30)
    parser.add_argument("--slow-tick-seconds", type=int, default=1800)
    args = parser.parse_args()

    await asyncio.gather(
        run_fast_tick(args.tick_seconds, args.slow_tick_seconds),
        run_slow_tick(args.slow_tick_seconds, args.tick_seconds),
    )


if __name__ == "__main__":
    asyncio.run(main())
