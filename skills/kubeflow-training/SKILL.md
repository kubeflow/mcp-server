---
name: kubeflow-training
description: >-
  Train and fine-tune AI models on Kubernetes with Kubeflow Trainer. Use when
  the user wants to run distributed training (PyTorch DDP/FSDP, DeepSpeed),
  fine-tune a HuggingFace model (LoRA/QLoRA), size GPU resources for a model,
  or debug stuck/failed TrainJobs on any Kubernetes platform (vanilla K8s,
  Kind, Minikube, OpenShift, EKS, GKE).
license: Apache-2.0
metadata:
  homepage: https://github.com/kubeflow/mcp-server
---

# Kubeflow Training

Train and fine-tune AI models on Kubernetes through the Kubeflow MCP server
(`kubeflow-mcp`). This skill pairs with that server: the tools do the work,
and the server publishes detailed guides as MCP resources you read on demand.

## Prerequisites

- Kubeflow Trainer installed on the cluster (TrainJob CRD must exist). Supported
  Trainer, SDK, Python, and Kubernetes versions are listed in the
  [compatibility matrix](https://github.com/kubeflow/mcp-server#requirements)
  of the server README.
- The `kubeflow-mcp` MCP server connected (see `mcp.json` in this skill). The
  server's default `readonly` persona cannot submit jobs; `mcp.json` starts it
  with `--persona data-scientist`, which the Training step requires.

## Workflow

Always follow the phases in order:

1. **Planning** — call `pre_flight()` first (pass `model=` for GPU sizing).
   It checks compatibility, cluster resources, and available runtimes in one
   shot.
2. **Discovery** — list runtimes and existing jobs before creating anything.
3. **Training** — preview before submitting: call training tools with
   `confirmed=False` first, inspect the plan, then confirm. Requires the
   `data-scientist` persona or higher; `readonly` has no `fine_tune` or
   `run_custom_training`.
4. **Monitoring** — watch job status; use `get_training_events()` to debug
   stuck or failed jobs.

## Detailed guides (progressive disclosure)

Read these MCP resources from the connected `kubeflow-mcp` server only when
the task needs them — do not preload all of them:

| Resource | When to read |
| --- | --- |
| `skill://kubeflow/training-patterns` | Writing training code: distributed training (DDP/FSDP/DeepSpeed) and LoRA/QLoRA patterns |
| `skill://kubeflow/platform-fixes` | Platform-specific issues: volumes, tolerations, OpenShift/EKS/GKE quirks |
| `skill://kubeflow/troubleshooting` | A job is stuck, failing, or erroring: error-to-fix tables and diagnostics |

The same content is also available under the server's legacy
`trainer://guides/*` URIs. `skill://index.json` lists every skill this server
serves.

## Safety

- Never submit a training job without a confirmed preview.
- Treat cluster mutations (create/delete jobs) as actions requiring explicit
  user intent.
