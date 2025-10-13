"""
Demo client for testing agentserve endpoints - NOT for production prop analysis.

This is a development/testing tool only. For real prop analysis, use the 
main BetFinder UI or integrate directly with your production systems.
"""
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
    ap = argparse.ArgumentParser(description="Demo client for testing agent endpoints - NOT for production use")
    ap.add_argument("--url", default="http://localhost:9001/invoke", help="Agent endpoint URL")
    ap.add_argument("--text", default="Test query for development purposes only")
    ap.add_argument("--stream", action="store_true")
    args = ap.parse_args()

    print("⚠️  DEMO ONLY - Not for real prop analysis or production use")
    call_invoke(args.url, args.text, stream=args.stream)


if __name__ == "__main__":
    main()
