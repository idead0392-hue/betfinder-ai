"""
Launches BetFinder agents as HTTP API services using openai-agentserve.

Each agent runs on its own port and exposes:
  - POST /invoke {"input": "...", "stream": bool}
  - Auto docs at /docs and /redoc

Usage:
  python agentserve_run.py           # start default set
  python agentserve_run.py --only csgo,lol  # start subset

The processes are independent; Ctrl+C will terminate all.
"""
from __future__ import annotations

import argparse
import multiprocessing as mp
import signal
import sys
import time

from typing import Dict, Callable

try:
    from agentserve import serve  # type: ignore
    HAVE_AGENTSERVE = True
except Exception:
    HAVE_AGENTSERVE = False
    serve = None  # will define fallback below


def _import_agent_factories() -> Dict[str, Callable[[], object]]:
    """Return mapping of agent keys to zero-arg constructors.

    We adapt to this repo's structure: agents are in sport_agents.py and expose concrete classes.
    """
    from sport_agents import (
        BasketballAgent,
        FootballAgent,
        HockeyAgent,
        CSGOAgent,
        LeagueOfLegendsAgent,
        Dota2Agent,
        VALORANTAgent,
        ApexAgent,
        BaseballAgent,
        TennisAgent,
        SoccerAgent,
        CollegeFootballAgent,
    )

    return {
        'nba': BasketballAgent,
        'nfl': FootballAgent,
        'nhl': HockeyAgent,
        'csgo': CSGOAgent,
        'lol': LeagueOfLegendsAgent,
        'dota2': Dota2Agent,
        'valorant': VALORANTAgent,
        'apex': ApexAgent,
        'mlb': BaseballAgent,
        'tennis': TennisAgent,
        'soccer': SoccerAgent,
        'cfb': CollegeFootballAgent,
    }


DEFAULT_PORTS: Dict[str, int] = {
    'nba': 9001,
    'nfl': 9002,
    'nhl': 9003,
    'csgo': 9004,
    'lol': 9005,
    'dota2': 9006,
    'valorant': 9007,
    'apex': 9008,
    'mlb': 9009,
    'tennis': 9010,
    'soccer': 9011,
    'cfb': 9012,
}


class AgentServeAdapter:
    """Adapter to expose our SportAgent via a simple .invoke(input) interface.

    Input can be a string prompt or a dict with optional keys:
      - props: list of prop dicts to analyze
      - cap: int, limit to N props
    Returns a JSON-serializable dict with picks and meta.
    """

    def __init__(self, agent_obj):
        self._agent = agent_obj

    def invoke(self, input):  # agentserve convention
        try:
            # Accept either string or dict input
            if isinstance(input, str):
                prompt = input
                props = None
                cap = 25
            elif isinstance(input, dict):
                prompt = str(input.get('input') or input.get('prompt') or '')
                props = input.get('props')
                cap = int(input.get('cap', 25))
            else:
                prompt = ''
                props = None
                cap = 25

            # If props are not provided, fetch current board for this sport
            if not props:
                try:
                    props = self._agent.fetch_props(max_props=cap)
                except Exception:
                    props = []

            picks = self._agent.make_picks(props_data=props, log_to_ledger=False)

            return {
                'success': True,
                'sport': getattr(self._agent, 'sport_name', 'unknown'),
                'count': len(picks),
                'picks': picks,
                'note': 'Development/testing endpoint. For production, use main BetFinder UI. Set PRIZEPICKS_CSV for live data.'
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}


def _serve_agent(agent_key: str, port: int):
    factories = _import_agent_factories()
    if agent_key not in factories:
        raise SystemExit(f"Unknown agent key: {agent_key}")
    AgentClass = factories[agent_key]
    agent = AgentClass()
    adapter = AgentServeAdapter(agent)
    # Start service (blocks current process)
    # agentserve will spin up a FastAPI/uvicorn server and expose /invoke
    print(f"Starting {agent_key} on port {port}...")

    if HAVE_AGENTSERVE and serve is not None:
        serve(agent=adapter, port=port)
        return

    # Fallback: lightweight FastAPI server that mimics agentserve /invoke
    try:
        from fastapi import FastAPI
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
        from typing import Any, Optional
        import uvicorn
        from starlette.responses import JSONResponse, StreamingResponse

        app = FastAPI(title=f"{agent_key.upper()} Agent API", version="0.1")
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        class InvokeBody(BaseModel):
            input: Optional[Any] = None
            stream: Optional[bool] = False
            props: Optional[list] = None
            cap: Optional[int] = 25

        @app.post("/invoke")
        def invoke(body: InvokeBody):
            payload = body.dict()
            # Merge into a single input dict for adapter
            input_data = {
                'input': payload.get('input'),
                'props': payload.get('props'),
                'cap': payload.get('cap') or 25,
            }
            if body.stream:
                # Stream picks as newline-delimited JSON lines
                def _gen():
                    result = adapter.invoke(input_data)
                    if not isinstance(result, dict) or not result.get('success'):
                        yield (JSONResponse(result).body or b"{}") + b"\n"
                        return
                    picks = result.get('picks') or []
                    # Emit a header with summary
                    header = {k: v for k, v in result.items() if k != 'picks'}
                    yield (JSONResponse(header).body or b"{}") + b"\n"
                    for p in picks:
                        yield (JSONResponse(p).body or b"{}") + b"\n"
                return StreamingResponse(_gen(), media_type="application/json")
            else:
                return adapter.invoke(input_data)

        uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
    except Exception as e:
        print(f"Failed to start server for {agent_key} on {port}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Serve agents via openai-agentserve")
    parser.add_argument("--only", type=str, default="",
                        help="Comma-separated keys to run (e.g., nba,nfl,csgo). Defaults to all.")
    parser.add_argument("--base-port", type=int, default=9001,
                        help="Base port to assign sequentially if not using defaults.")
    parser.add_argument("--use-default-ports", action="store_true",
                        help="Use hardcoded DEFAULT_PORTS mapping.")
    args = parser.parse_args()

    factories = _import_agent_factories()
    selected = list(factories.keys())
    if args.only:
        requested = [s.strip() for s in args.only.split(',') if s.strip()]
        selected = [s for s in requested if s in factories]
        unknown = [s for s in requested if s not in factories]
        if unknown:
            print(f"Ignoring unknown agent keys: {', '.join(unknown)}")

    # Assign ports
    if args.use_default_ports:
        port_map = {k: DEFAULT_PORTS[k] for k in selected}
    else:
        port_map = {k: args.base_port + i for i, k in enumerate(selected)}

    procs: Dict[str, mp.Process] = {}

    def shutdown(*_):
        print("Shutting down agent processes...")
        for k, p in procs.items():
            if p.is_alive():
                p.terminate()
        for k, p in procs.items():
            p.join(timeout=5)
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    for key in selected:
        port = port_map[key]
        p = mp.Process(target=_serve_agent, args=(key, port), daemon=False)
        p.start()
        procs[key] = p
        print(f"✅ {key} listening on http://localhost:{port}")

    # Keep parent alive while children run
    try:
        while any(p.is_alive() for p in procs.values()):
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown()


if __name__ == "__main__":
    main()
