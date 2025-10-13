"""Minimal client to test agentserve endpoints for NBA and others."""
from __future__ import annotations

import argparse
import requests


def call_invoke(url: str, text: str, stream: bool = False):
    payload = {"input": text}
    if stream:
        payload["stream"] = True
    if stream:
        with requests.post(url, json=payload, stream=True) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if line:
                    print(line.decode("utf-8"))
    else:
        resp = requests.post(url, json=payload)
        resp.raise_for_status()
        print(resp.json())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:9001/invoke", help="Agent endpoint URL")
    ap.add_argument("--text", default="Show today's top NBA prop picks")
    ap.add_argument("--stream", action="store_true")
    args = ap.parse_args()

    call_invoke(args.url, args.text, stream=args.stream)


if __name__ == "__main__":
    main()
