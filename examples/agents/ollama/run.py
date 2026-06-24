#!/usr/bin/env python3
# Copyright 2026 The Kubeflow Authors.
#
# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper around :class:`kubeflow_mcp.agents.ollama.OllamaProvider`."""

from __future__ import annotations

import argparse

from kubeflow_mcp.agents.ollama import DEFAULT_MODEL, DEFAULT_URL, OllamaProvider


def main() -> None:
    p = argparse.ArgumentParser(description="Run the Ollama Kubeflow agent")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--url", default=DEFAULT_URL)
    p.add_argument(
        "--mode",
        default="full",
        choices=["full", "progressive", "semantic", "static", "mcp"],
    )
    p.add_argument("--thinking", action="store_true")
    args = p.parse_args()
    OllamaProvider().run(
        model=args.model,
        mode=args.mode,
        url=args.url,
        thinking=args.thinking,
    )


if __name__ == "__main__":
    main()
