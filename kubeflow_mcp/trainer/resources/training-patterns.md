# Training Patterns

Code patterns for distributed training and LoRA fine-tuning via `run_custom_training()`.

---

## Process Group Initialization

`torchrun` sets env vars but does NOT auto-initialize the process group.

```python
import os, torch, torch.distributed as dist
from datetime import timedelta

if "WORLD_SIZE" in os.environ:
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, timeout=timedelta(minutes=30))
```

Use long timeouts (30-60 min) — large model downloads on rank 0 block other ranks.

**Frameworks that handle init internally** (do NOT call `init_process_group`):
- HuggingFace `Trainer` / `SFTTrainer`
- DeepSpeed (`deepspeed.init_distributed()`)

## Backend Selection

| Backend | When to use |
|---------|------------|
| `gloo` | CPU training, or clusters without NCCL. Works everywhere. |
| `nccl` | GPU training with NVIDIA GPUs. Faster but requires NCCL libraries. |

## Rank-Gated Operations

Only rank 0 should perform shared I/O:

```python
rank = int(os.environ.get("RANK", "0"))

if rank == 0:
    model = download_model(...)
dist.barrier()  # all ranks wait
```

Guard logging, checkpointing, and evaluation with `if rank == 0`.

## Data Sharding

```python
from torch.utils.data import DataLoader, DistributedSampler

sampler = DistributedSampler(dataset) if dist.is_initialized() else None
loader = DataLoader(dataset, sampler=sampler, shuffle=(sampler is None))
```

## Cleanup

```python
try:
    train(...)
finally:
    if dist.is_initialized():
        dist.destroy_process_group()
```

## Device Placement

- Use `LOCAL_RANK` (not `RANK`) for device: `torch.device(f"cuda:{local_rank}")`
- Do NOT set `device_map="auto"` — breaks DDP
- Do NOT hardcode `cuda:0`

## torchrun Environment Variables

| Variable | Meaning |
|----------|---------|
| `RANK` | Global rank across all nodes |
| `LOCAL_RANK` | Rank within the current node |
| `WORLD_SIZE` | Total number of processes |
| `MASTER_ADDR` | Address of rank-0 node |
| `MASTER_PORT` | Port for rank-0 rendezvous |

---

## LoRA via run_custom_training()

Use when no torchtune runtime exists or on CPU-only clusters.

### Required Packages

`transformers`, `peft`, `trl`, `datasets`, `accelerate`

Call `get_runtime("<runtime>", include_packages=True)` first — do NOT duplicate already-installed packages.

### Key Rules

- HF `SFTTrainer` handles `init_process_group` internally — do NOT call it manually
- Set `ddp_backend="gloo"` in `SFTConfig` for CPU or when NCCL is unavailable
- Always check `tokenizer.pad_token`; set to `tokenizer.eos_token` if None
- Use `torch_dtype=torch.bfloat16` when CUDA is available
- Use `gradient_accumulation_steps=4+` on small GPUs
- For QLoRA: add `bitsandbytes` and use `BitsAndBytesConfig` for 4-bit quantization
- `task_type="CAUSAL_LM"` for decoder-only models
- Typical `target_modules=["q_proj", "v_proj"]` — adjust per architecture

### Common Pitfalls

- Forgetting `token=hf_token` for gated models (Llama, Mistral, Gemma)
- Missing `pad_token` causes silent training failures
- `device_map="auto"` breaks DDP multi-GPU training
- Not adding emptyDir volumes on read-only filesystems (see trainer://guides/platform-fixes)
