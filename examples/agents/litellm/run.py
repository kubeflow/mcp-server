#!/usr/bin/env python3
# Copyright 2026 The Kubeflow Authors.
#
# SPDX-License-Identifier: Apache-2.0
"""Thin wrapper around :class:`kubeflow_mcp.agents.litellm_provider.LiteLLMProvider`."""

from __future__ import annotations

import argparse

from kubeflow_mcp.agents.litellm_provider import LiteLLMProvider


def main() -> None:
    p = argparse.ArgumentParser(description="Run the LiteLLM chat loop")
    p.add_argument("--model", default=LiteLLMProvider.default_model)
    args = p.parse_args()
    LiteLLMProvider().run(model=args.model, mode="full")


if __name__ == "__main__":
    main()
