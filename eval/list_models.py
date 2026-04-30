"""List models available on an OpenAI-compatible endpoint.

Usage:
    # Defaults to LAS LiteLLM. Reads LAS_API_TOKEN or OPENAI_API_KEY from env.
    python eval/list_models.py

    # Other endpoints
    python eval/list_models.py --base-url https://api.openai.com/v1
"""

from __future__ import annotations

import argparse
import os
import sys


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="https://llm-west.ncsu-las.net/v1")
    ap.add_argument("--filter", default="",
                     help="Substring filter (case-insensitive).")
    args = ap.parse_args(argv)

    key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("LAS_API_TOKEN")
        or os.environ.get("LAS_API_KEY")
    )
    if not key:
        print("ERROR: Set OPENAI_API_KEY or LAS_API_TOKEN before running this.",
              file=sys.stderr)
        return 1

    from openai import OpenAI

    client = OpenAI(api_key=key, base_url=args.base_url)
    try:
        page = client.models.list()
    except Exception as e:
        print(f"ERROR: {type(e).__name__}: {e}", file=sys.stderr)
        return 2

    needle = args.filter.lower()
    ids = sorted({m.id for m in page.data})
    if needle:
        ids = [m for m in ids if needle in m.lower()]

    print(f"# {len(ids)} models on {args.base_url}")
    for m in ids:
        print(m)
    return 0


if __name__ == "__main__":
    sys.exit(main())
