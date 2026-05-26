# Copyright The Kubeflow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training submission tools: fine_tune, run_custom_training, run_container_training."""

import ast
import logging
import os
import tempfile
import uuid
from collections.abc import Callable
from typing import Any

from kubeflow.trainer.constants.constants import DEFAULT_PIP_INDEX_URLS

from kubeflow_mcp.common.constants import ErrorCode
from kubeflow_mcp.common.types import PreviewResponse, ToolError, ToolResponse
from kubeflow_mcp.common.utils import (
    MCP_MANAGED_LABEL,
    MCP_MANAGED_VALUE,
    get_trainer_client,
    get_trainer_client_for_namespace,
)
from kubeflow_mcp.core.security import (
    check_namespace_allowed,
    is_safe_python_code,
    mask_sensitive_data,
    validate_k8s_name,
    validate_training_bounds,
)

logger = logging.getLogger(__name__)

# Import types at module level to avoid import deadlocks
# when tools are called in rapid succession
try:
    from kubeflow.trainer.options import (  # type: ignore[attr-defined]
        Annotations,
        ContainerPatch,
        JobSetSpecPatch,
        JobSetTemplatePatch,
        JobSpecPatch,
        JobTemplatePatch,
        Labels,
        Name,
        PodSpecPatch,
        PodTemplatePatch,
        ReplicatedJobPatch,
        RuntimePatch,
        TrainerArgs,
        TrainerCommand,
        TrainingRuntimeSpecPatch,
    )
    from kubeflow.trainer.types.types import (
        BuiltinTrainer,
        CustomTrainer,
        CustomTrainerContainer,
        DataFormat,
        DataType,
        HuggingFaceDatasetInitializer,
        HuggingFaceModelInitializer,
        Initializer,
        LoraConfig,
        Loss,
        S3DatasetInitializer,
        S3ModelInitializer,
        TorchTuneConfig,
        TorchTuneInstructDataset,
    )

    _SDK_AVAILABLE = True
except ImportError:
    _SDK_AVAILABLE = False

_TRAINING_SCRIPT_DIR: str | None = None


def _get_script_dir() -> str:
    """Lazily create a temp directory for training script files."""
    global _TRAINING_SCRIPT_DIR
    if _TRAINING_SCRIPT_DIR is None or not os.path.isdir(_TRAINING_SCRIPT_DIR):
        _TRAINING_SCRIPT_DIR = tempfile.mkdtemp(prefix="kubeflow_mcp_scripts_")
    return _TRAINING_SCRIPT_DIR


def _make_train_func(script: str, func_args: dict[str, Any] | None = None) -> Callable:
    """Convert a script string into a function whose source is inspectable.

    ``inspect.getsource(func)`` requires the function's source to exist in a
    real file.  We write a ``def train(): ...`` wrapper to a temp file
    and compile against that path so ``inspect`` can read it back.

    When *func_args* is provided, the generated function signature includes matching
    keyword parameters so the trainer can pass them at runtime.
    """
    tree = ast.parse(script)

    func_name = "train"
    lines = script.strip().splitlines()

    has_train = False
    train_node = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "train":
            has_train = True
            train_node = node
            break
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "train":
            has_train = True
            train_node = node
            break

    if not has_train:
        if func_args:
            params = ", ".join(f"{k}=None" for k in func_args)
            wrapped = f"def {func_name}({params}):\n"
        else:
            wrapped = f"def {func_name}():\n"
        for line in lines:
            wrapped += f"    {line}\n"
    else:
        param_names = [arg.arg for arg in train_node.args.args]
        if func_args:
            expected_params = list(func_args)
            missing = [p for p in expected_params if p not in param_names]
            if missing:
                raise ValueError(f"train() missing required params: {missing}")
        wrapped = script

    script_dir = _get_script_dir()
    script_path = os.path.join(script_dir, f"_mcp_train_{uuid.uuid4().hex[:8]}.py")
    with open(script_path, "w") as f:
        f.write(wrapped)

    code = compile(wrapped, script_path, "exec")
    ns: dict[str, Any] = {}
    exec(code, ns)  # noqa: S102
    return ns[func_name]


def _get_client(namespace: str | None = None) -> Any:
    """Return a client targeting the given namespace.

    When *namespace* is ``None`` the shared singleton is returned via the
    module-level ``get_trainer_client`` (mockable in tests).  When a
    namespace is explicitly provided a cached scoped client is returned.
    """
    if namespace is None:
        return get_trainer_client()
    return get_trainer_client_for_namespace(namespace)


def _inject_ownership_label(options: list) -> list:
    """Merge the MCP ownership label into existing Labels or append a new one."""
    for opt in options:
        if isinstance(opt, Labels):
            opt.labels[MCP_MANAGED_LABEL] = MCP_MANAGED_VALUE
            return options
    options.append(Labels(labels={MCP_MANAGED_LABEL: MCP_MANAGED_VALUE}))
    return options


def _sdk_error(e: Exception, hint: str | None = None) -> dict[str, Any]:
    """Convert an exception into a ToolError dict with optional K8s response detail."""
    details: dict[str, Any] | None = None
    if e.__cause__:
        details = {"cause": str(e.__cause__)}
    elif hasattr(e, "response"):
        try:
            details = {"response": e.response.text}  # type: ignore[union-attr]
        except Exception:
            pass
    return ToolError(
        error=str(e),
        error_code=ErrorCode.SDK_ERROR,
        details=details,
        hint=hint,
    ).model_dump()


def _build_initializer(
    model: str,
    dataset: str,
    hf_token: str | None = None,
    s3_endpoint: str | None = None,
    s3_access_key_id: str | None = None,
    s3_secret_access_key: str | None = None,
    s3_region: str | None = None,
    s3_role_arn: str | None = None,
    model_ignore_patterns: list[str] | None = None,
    dataset_ignore_patterns: list[str] | None = None,
) -> "Initializer":
    """Build an Initializer from model/dataset URIs.

    Auto-detects the storage backend from the URI prefix:
    - ``hf://`` → HuggingFace initializers (access_token used for gated models)
    - ``s3://`` → S3 initializers (endpoint/key/secret/region/role_arn used)
    """
    s3_kwargs: dict[str, Any] = {}
    if s3_endpoint:
        s3_kwargs["endpoint"] = s3_endpoint
    if s3_access_key_id:
        s3_kwargs["access_key_id"] = s3_access_key_id
    if s3_secret_access_key:
        s3_kwargs["secret_access_key"] = s3_secret_access_key
    if s3_region:
        s3_kwargs["region"] = s3_region
    if s3_role_arn:
        s3_kwargs["role_arn"] = s3_role_arn

    if model.startswith("s3://"):
        model_init: Any = S3ModelInitializer(
            storage_uri=model, ignore_patterns=model_ignore_patterns, **s3_kwargs
        )
    else:
        model_init = HuggingFaceModelInitializer(
            storage_uri=model, ignore_patterns=model_ignore_patterns, access_token=hf_token
        )

    if dataset.startswith("s3://"):
        dataset_init: Any = S3DatasetInitializer(
            storage_uri=dataset, ignore_patterns=dataset_ignore_patterns, **s3_kwargs
        )
    else:
        dataset_init = HuggingFaceDatasetInitializer(
            storage_uri=dataset, ignore_patterns=dataset_ignore_patterns, access_token=hf_token
        )

    return Initializer(model=model_init, dataset=dataset_init)


def _build_runtime_patch(
    node_selector: dict[str, str] | None = None,
    tolerations: list[dict[str, Any]] | None = None,
    volumes: list[dict[str, Any]] | None = None,
    volume_mounts: list[dict[str, Any]] | None = None,
    affinity: dict[str, Any] | None = None,
    service_account_name: str | None = None,
    image_pull_secrets: list[dict[str, Any]] | None = None,
    env: list[dict[str, Any]] | None = None,
    labels: dict[str, str] | None = None,
    annotations: dict[str, str] | None = None,
    has_initializers: bool = False,
) -> list[Any]:
    """Build runtime patch options from JSON dicts.

    Returns a list of options to pass to ``client.train(options=...)``.
    Returns an empty list if no patches are specified.

    The ``env`` parameter sets environment variables on the ``node`` container
    via ``ContainerPatch``.  This is the mechanism used by ``fine_tune()``
    (BuiltinTrainer) where env cannot be set through ``spec.trainer.env``.

    Top-level options (``Labels``, ``Annotations``) are appended alongside
    the ``RuntimePatch`` in the returned list so they are applied independently.
    """
    has_pod_patch = any(
        [
            node_selector,
            tolerations,
            volumes,
            volume_mounts,
            affinity,
            service_account_name,
            image_pull_secrets,
            env,
        ]
    )
    has_meta = labels or annotations

    if not has_pod_patch and not has_meta:
        return []

    if not _SDK_AVAILABLE:
        return []

    options: list[Any] = []

    if has_pod_patch:
        node_containers = None
        if volume_mounts or env:
            node_containers = [
                ContainerPatch(
                    name="node",
                    volume_mounts=volume_mounts,
                    env=env,
                )
            ]

        node_pod_spec = PodSpecPatch(
            node_selector=node_selector,
            tolerations=tolerations,
            volumes=volumes,
            containers=node_containers,
            affinity=affinity,
            service_account_name=service_account_name,
            image_pull_secrets=image_pull_secrets,
        )

        replicated_jobs = [
            ReplicatedJobPatch(
                name="node",
                template=JobTemplatePatch(
                    spec=JobSpecPatch(
                        template=PodTemplatePatch(spec=node_pod_spec),
                    ),
                ),
            ),
        ]

        if has_initializers and (volumes or volume_mounts):
            for init_name in ("dataset-initializer", "model-initializer"):
                init_containers = None
                if volume_mounts:
                    init_containers = [ContainerPatch(name=init_name, volume_mounts=volume_mounts)]
                init_pod_spec = PodSpecPatch(volumes=volumes, containers=init_containers)
                replicated_jobs.append(
                    ReplicatedJobPatch(
                        name=init_name,
                        template=JobTemplatePatch(
                            spec=JobSpecPatch(
                                template=PodTemplatePatch(spec=init_pod_spec),
                            ),
                        ),
                    ),
                )

        options.append(
            RuntimePatch(
                training_runtime_spec=TrainingRuntimeSpecPatch(
                    template=JobSetTemplatePatch(
                        spec=JobSetSpecPatch(replicated_jobs=replicated_jobs),
                    ),
                ),
            )
        )

    if labels:
        options.append(Labels(labels=labels))
    if annotations:
        options.append(Annotations(annotations=annotations))

    return options


def _validate_fine_tune_params(
    namespace: str | None,
    name: str | None,
    dtype: str | None,
    loss: str | None,
    dataset_source: str | None,
    batch_size: int,
    epochs: int,
    num_nodes: int,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float | None,
    confirmed: bool,
) -> dict[str, Any] | None:
    """Run all fine_tune validations. Returns error dict on failure, None on success."""
    if not _SDK_AVAILABLE:
        return ToolError(
            error="Required training libraries not available", error_code=ErrorCode.SDK_ERROR
        ).model_dump()

    ns_err = check_namespace_allowed(namespace)
    if ns_err:
        return ns_err.model_dump()

    if name:
        err = validate_k8s_name(name)
        if err:
            return err.model_dump()

    if dtype and dtype not in ("bf16", "fp32"):
        return ToolError(
            error=f"Invalid dtype '{dtype}'. Must be 'bf16' or 'fp32'.",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    valid_losses = {m.name: m for m in Loss} if _SDK_AVAILABLE else {}
    if loss and loss not in valid_losses:
        return ToolError(
            error=f"Invalid loss '{loss}'. Must be one of: {list(valid_losses.keys())}.",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    valid_formats = {m.name.lower(): m for m in DataFormat} if _SDK_AVAILABLE else {}
    if dataset_source and dataset_source.lower() not in valid_formats:
        return ToolError(
            error=f"Invalid dataset_source '{dataset_source}'. Must be one of: {list(valid_formats.keys())}.",
            error_code=ErrorCode.VALIDATION_ERROR,
        ).model_dump()

    if not confirmed:
        gpu_err = _check_gpu_available()
        if gpu_err:
            return gpu_err

    bounds_err = validate_training_bounds(
        batch_size=batch_size,
        epochs=epochs,
        num_nodes=num_nodes,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
    )
    if bounds_err:
        return bounds_err.model_dump()
    return None


def _check_gpu_available() -> dict[str, Any] | None:
    """Return error dict if cluster has zero GPUs, None otherwise."""
    try:
        from kubeflow_mcp.trainer.api.planning import get_cluster_resources

        cluster = get_cluster_resources()
        cluster_data = cluster.get("data", cluster)
        if cluster_data.get("gpu_total", -1) == 0:
            return ToolError(
                error=(
                    "fine_tune() requires GPUs (torchtune uses NCCL backend). "
                    "No GPUs detected in cluster. Use run_custom_training() "
                    "with gloo backend instead."
                ),
                error_code=ErrorCode.VALIDATION_ERROR,
                details={
                    "tool_selection": {
                        "recommended": "run_custom_training",
                        "reason": "CPU-only cluster, torchtune requires GPUs",
                    },
                    "resource": "trainer://guides/training-patterns",
                },
            ).model_dump()
    except Exception:
        logger.debug("GPU pre-check skipped (cluster query failed)")
    return None


def _build_fine_tune_config(
    *,
    model: str,
    dataset: str,
    runtime: str,
    name: str | None,
    hf_token: str | None,
    batch_size: int,
    epochs: int,
    num_nodes: int,
    dtype: str | None,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float | None,
    use_dora: bool | None,
    quantize_base: bool | None,
    s3_access_key_id: str | None,
    s3_secret_access_key: str | None,
    optional_fields: list[tuple[str, Any]],
) -> dict[str, Any]:
    """Build the preview config dict for fine_tune, masking secrets."""
    config: dict[str, Any] = {
        "model": model,
        "dataset": dataset,
        "runtime": runtime,
        "name": name,
        "hf_token": "***" if hf_token else None,
        "batch_size": batch_size,
        "epochs": epochs,
        "num_nodes": num_nodes,
        "dtype": dtype,
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "use_dora": use_dora,
        "quantize_base": quantize_base,
    }
    if s3_access_key_id:
        config["s3_access_key_id"] = "***"
    if s3_secret_access_key:
        config["s3_secret_access_key"] = "***"
    for k, v in optional_fields:
        if v is not None:
            config[k] = v
    return config


def _build_lora_cfg(
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float | None,
    use_dora: bool | None,
    quantize_base: bool | None,
    apply_lora_to_mlp: bool | None,
    apply_lora_to_output: bool | None,
    lora_attn_modules: list[str] | None,
) -> "LoraConfig":
    """Build a LoraConfig from fine_tune parameters."""
    lora_kwargs: dict[str, Any] = {
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "use_dora": use_dora,
        "quantize_base": quantize_base,
    }
    if apply_lora_to_mlp is not None:
        lora_kwargs["apply_lora_to_mlp"] = apply_lora_to_mlp
    if apply_lora_to_output is not None:
        lora_kwargs["apply_lora_to_output"] = apply_lora_to_output
    if lora_attn_modules is not None:
        lora_kwargs["lora_attn_modules"] = lora_attn_modules
    return LoraConfig(**lora_kwargs)


def _build_dataset_preprocess_cfg(
    dataset_source: str | None,
    dataset_split: str | None,
    dataset_train_on_input: bool | None,
    dataset_system_prompt: str | None,
    dataset_column_map: dict[str, str] | None,
) -> "TorchTuneInstructDataset | None":
    """Build dataset preprocess config if any dataset option is set."""
    if not any(
        v is not None
        for v in [
            dataset_source,
            dataset_split,
            dataset_train_on_input,
            dataset_system_prompt,
            dataset_column_map,
        ]
    ):
        return None
    valid_formats = {m.name.lower(): m for m in DataFormat}
    ds_kwargs: dict[str, Any] = {}
    if dataset_source:
        ds_kwargs["source"] = valid_formats[dataset_source.lower()]
    if dataset_split is not None:
        ds_kwargs["split"] = dataset_split
    if dataset_train_on_input is not None:
        ds_kwargs["train_on_input"] = dataset_train_on_input
    if dataset_system_prompt is not None:
        ds_kwargs["new_system_prompt"] = dataset_system_prompt
    if dataset_column_map is not None:
        ds_kwargs["column_map"] = dataset_column_map
    return TorchTuneInstructDataset(**ds_kwargs)


def fine_tune(
    model: str,
    dataset: str,
    runtime: str = "torchtune",
    name: str | None = None,
    namespace: str | None = None,
    hf_token: str | None = None,
    batch_size: int = 4,
    epochs: int = 1,
    num_nodes: int = 1,
    dtype: str | None = None,
    loss: str | None = None,
    resources_per_node: dict[str, Any] | None = None,
    lora_rank: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float | None = None,
    use_dora: bool | None = None,
    quantize_base: bool | None = None,
    apply_lora_to_mlp: bool | None = None,
    apply_lora_to_output: bool | None = None,
    lora_attn_modules: list[str] | None = None,
    dataset_source: str | None = None,
    dataset_split: str | None = None,
    dataset_train_on_input: bool | None = None,
    dataset_system_prompt: str | None = None,
    dataset_column_map: dict[str, str] | None = None,
    model_ignore_patterns: list[str] | None = None,
    dataset_ignore_patterns: list[str] | None = None,
    s3_endpoint: str | None = None,
    s3_access_key_id: str | None = None,
    s3_secret_access_key: str | None = None,
    s3_region: str | None = None,
    s3_role_arn: str | None = None,
    node_selector: dict[str, str] | None = None,
    tolerations: list[dict[str, Any]] | None = None,
    volumes: list[dict[str, Any]] | None = None,
    volume_mounts: list[dict[str, Any]] | None = None,
    affinity: dict[str, Any] | None = None,
    service_account_name: str | None = None,
    image_pull_secrets: list[dict[str, Any]] | None = None,
    labels: dict[str, str] | None = None,
    annotations: dict[str, str] | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Fine-tune a model using LoRA/QLoRA via BuiltinTrainer + torchtune runtime.

    Requires a torchtune ClusterTrainingRuntime (e.g. ``torchtune-llama3.2-1b``).
    Returns an error for non-torchtune runtimes — use ``run_custom_training()``
    with a LoRA script instead (see ``trainer://guides/lora-script-template`` for best practices).

    Supports HuggingFace (``hf://``) and S3 (``s3://``) model/dataset sources.
    Requires ``confirmed=True`` to submit. First call returns a preview.

    Note: ``env`` is NOT supported for fine_tune. Use ``run_custom_training()``
    with a LoRA script if you need custom environment variables.

    Args:
        model: Model URI. Use ``hf://`` prefix for HuggingFace (e.g.,
            ``hf://google/gemma-2b``) or ``s3://`` prefix for S3.
        dataset: Dataset URI. Same prefix rules as ``model``.
        runtime: ClusterTrainingRuntime name. Must be a torchtune runtime.
            Run ``list_runtimes()`` to see what is installed in your cluster.
        name: Custom TrainJob name. Auto-generated if omitted.
        namespace: K8s namespace for the TrainJob. Uses default from kubeconfig
            when omitted. Must be allowed by policy.
        hf_token: HuggingFace access token for gated models (Llama, Mistral).
        batch_size: Per-GPU batch size. Defaults to 4.
        epochs: Number of training epochs. Defaults to 1.
        num_nodes: Distributed training nodes. Defaults to 1.
        dtype: Training precision — ``"bf16"`` or ``"fp32"``. Uses runtime
            default if not specified.
        loss: Loss function name. Currently only ``"CEWithChunkedOutputLoss"``
            is supported. Uses runtime default if omitted.
        resources_per_node: Resource dict for the training pod (e.g.,
            ``{"cpu": "8", "memory": "32Gi", "gpu": 2}``).
        lora_rank: LoRA rank. Defaults to 8.
        lora_alpha: LoRA alpha scaling. Defaults to 16.
        lora_dropout: LoRA dropout probability (0.0–1.0). Default if omitted.
        use_dora: Enable DoRA (weight-decomposed LoRA). Default if omitted.
        quantize_base: Quantize base model weights for QLoRA. Default if omitted.
        apply_lora_to_mlp: Apply LoRA to MLP layers. Default if omitted.
        apply_lora_to_output: Apply LoRA to output projection. Default if omitted.
        lora_attn_modules: Attention modules to apply LoRA to (e.g.,
            ``["q_proj", "v_proj", "output_proj"]``). Default if omitted.
        dataset_source: Instruct dataset format — one of ``"json"``, ``"csv"``,
            ``"parquet"``, ``"arrow"``, ``"text"``, ``"xml"``.
        dataset_split: Dataset split name (e.g., ``"train"``).
        dataset_train_on_input: Whether to train on the input portion.
        dataset_system_prompt: Override system prompt for instruct datasets.
        dataset_column_map: Column name mapping (e.g.,
            ``{"input": "question", "output": "answer"}``).
        model_ignore_patterns: Glob patterns for files to skip when downloading
            the model (e.g., ``["*.bin", "*.safetensors"]``).
        dataset_ignore_patterns: Glob patterns for files to skip when downloading
            the dataset.
        s3_endpoint: S3-compatible endpoint URL (MinIO, Ceph, etc.).
        s3_access_key_id: S3 access key ID.
        s3_secret_access_key: S3 secret access key.
        s3_region: S3 region.
        s3_role_arn: IAM role ARN for S3 access.
        node_selector: K8s node selector (e.g., ``{"gpu-type": "a100"}``).
        tolerations: K8s tolerations for tainted nodes.
        volumes: K8s volume definitions.
        volume_mounts: K8s volume mounts.
        affinity: K8s pod affinity/anti-affinity rules.
        service_account_name: K8s service account for the training pod.
        image_pull_secrets: K8s image pull secrets.
        labels: Extra labels to apply to the TrainJob.
        annotations: Extra annotations to apply to the TrainJob.
        confirmed: Set ``True`` to submit job. ``False`` returns preview only.

    Returns:
        dict: If ``confirmed=False``: preview with ``config`` dict.
            If ``confirmed=True``: ``job_name``, ``status``, ``message``.

    Example:
        >>> fine_tune("hf://google/gemma-2b", "hf://tatsu-lab/alpaca", confirmed=True)
        {"data": {"job_name": "train-gemma-abc", "status": "Created"}}

    Note:
        Call ``get_cluster_resources()`` first to verify GPU availability.
        For QLoRA set ``quantize_base=True``. For DoRA set ``use_dora=True``.
    """
    try:
        validation_err = _validate_fine_tune_params(
            namespace=namespace,
            name=name,
            dtype=dtype,
            loss=loss,
            dataset_source=dataset_source,
            batch_size=batch_size,
            epochs=epochs,
            num_nodes=num_nodes,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            confirmed=confirmed,
        )
        if validation_err:
            return validation_err

        optional_fields = [
            ("loss", loss),
            ("resources_per_node", resources_per_node),
            ("apply_lora_to_mlp", apply_lora_to_mlp),
            ("apply_lora_to_output", apply_lora_to_output),
            ("lora_attn_modules", lora_attn_modules),
            ("dataset_source", dataset_source),
            ("dataset_split", dataset_split),
            ("dataset_train_on_input", dataset_train_on_input),
            ("dataset_system_prompt", dataset_system_prompt),
            ("dataset_column_map", dataset_column_map),
            ("model_ignore_patterns", model_ignore_patterns),
            ("dataset_ignore_patterns", dataset_ignore_patterns),
            ("s3_endpoint", s3_endpoint),
            ("s3_region", s3_region),
            ("s3_role_arn", s3_role_arn),
            ("node_selector", node_selector),
            ("tolerations", tolerations),
            ("volumes", volumes),
            ("volume_mounts", volume_mounts),
            ("affinity", affinity),
            ("service_account_name", service_account_name),
            ("image_pull_secrets", image_pull_secrets),
            ("labels", labels),
            ("annotations", annotations),
        ]
        config = _build_fine_tune_config(
            model=model,
            dataset=dataset,
            runtime=runtime,
            name=name,
            hf_token=hf_token,
            batch_size=batch_size,
            epochs=epochs,
            num_nodes=num_nodes,
            dtype=dtype,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            use_dora=use_dora,
            quantize_base=quantize_base,
            s3_access_key_id=s3_access_key_id,
            s3_secret_access_key=s3_secret_access_key,
            optional_fields=optional_fields,
        )

        if not (runtime.startswith("torchtune") or runtime == "torch-tune"):
            return ToolError(
                error=(
                    f"Runtime '{runtime}' does not support BuiltinTrainer. "
                    "fine_tune() requires a torchtune runtime (e.g. torchtune-llama3.2-1b). "
                    "Run list_runtimes() to find available torchtune runtimes. "
                    "For non-torchtune runtimes, use run_custom_training() with a LoRA "
                    "fine-tuning script instead — see the lora-script-template resource."
                ),
                error_code=ErrorCode.VALIDATION_ERROR,
            ).model_dump()

        config["mode"] = "builtin_trainer"

        if not confirmed:
            return PreviewResponse(
                message="Review config and set confirmed=True to submit job", config=config
            ).model_dump()

        options = _build_runtime_patch(
            node_selector=node_selector,
            tolerations=tolerations,
            volumes=volumes,
            volume_mounts=volume_mounts,
            affinity=affinity,
            service_account_name=service_account_name,
            image_pull_secrets=image_pull_secrets,
            labels=labels,
            annotations=annotations,
            has_initializers=True,
        )
        if name:
            options.append(Name(name=name))

        client = _get_client(namespace)
        initializer = _build_initializer(
            model=model,
            dataset=dataset,
            hf_token=hf_token,
            s3_endpoint=s3_endpoint,
            s3_access_key_id=s3_access_key_id,
            s3_secret_access_key=s3_secret_access_key,
            s3_region=s3_region,
            s3_role_arn=s3_role_arn,
            model_ignore_patterns=model_ignore_patterns,
            dataset_ignore_patterns=dataset_ignore_patterns,
        )
        lora_cfg = _build_lora_cfg(
            lora_rank,
            lora_alpha,
            lora_dropout,
            use_dora,
            quantize_base,
            apply_lora_to_mlp,
            apply_lora_to_output,
            lora_attn_modules,
        )
        dataset_preprocess = _build_dataset_preprocess_cfg(
            dataset_source,
            dataset_split,
            dataset_train_on_input,
            dataset_system_prompt,
            dataset_column_map,
        )

        valid_losses = {m.name: m for m in Loss} if _SDK_AVAILABLE else {}
        torch_cfg = TorchTuneConfig(
            batch_size=batch_size,
            epochs=epochs,
            num_nodes=num_nodes,
            dtype=DataType(dtype) if dtype else None,
            loss=valid_losses[loss] if loss else None,
            resources_per_node=resources_per_node,
            peft_config=lora_cfg,
            dataset_preprocess_config=dataset_preprocess,
        )

        trainer = BuiltinTrainer(config=torch_cfg)
        _inject_ownership_label(options)
        job_name = client.train(
            runtime=runtime, initializer=initializer, trainer=trainer, options=options
        )

        return ToolResponse(
            data={
                "job_name": job_name,
                "status": "Created",
                "message": f"Training job '{job_name}' submitted successfully",
                "next_steps": [
                    f"get_training_logs(name='{job_name}') — monitor progress",
                    f"get_training_job(name='{job_name}') — check status",
                ],
            }
        ).model_dump()

    except Exception as e:
        return _sdk_error(
            e,
            hint="Use troubleshooting_guide prompt for diagnosis, or resource_planning to check requirements",
        )


def run_custom_training(
    script: str,
    runtime: str | None = None,
    name: str | None = None,
    namespace: str | None = None,
    num_nodes: int = 1,
    gpu_per_node: int = 1,
    func_args: dict[str, Any] | None = None,
    packages: list[str] | None = None,
    image: str | None = None,
    pip_index_urls: list[str] | None = None,
    resources_per_node: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
    node_selector: dict[str, str] | None = None,
    tolerations: list[dict[str, Any]] | None = None,
    volumes: list[dict[str, Any]] | None = None,
    volume_mounts: list[dict[str, Any]] | None = None,
    affinity: dict[str, Any] | None = None,
    service_account_name: str | None = None,
    image_pull_secrets: list[dict[str, Any]] | None = None,
    labels: dict[str, str] | None = None,
    annotations: dict[str, str] | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Run a custom Python training script on the cluster.

    Thin wrapper around ``CustomTrainer``. The script is converted into an
    inspectable function and passed directly to ``client.train()``.

    Args:
        script: Python code string. Heuristically validated; not sandboxed.
        runtime: ClusterTrainingRuntime name. Defaults to ``torch-distributed``.
            Run ``list_runtimes()`` to see available
            runtimes.
        name: TrainJob name. Auto-generated if not provided.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
            Must be allowed by policy.
        num_nodes: Distributed training nodes. Defaults to 1.
        gpu_per_node: GPUs per node. Set 0 for CPU-only. Defaults to 1.
        func_args: Dict of keyword arguments passed to the training function
            at runtime (e.g., ``{"lr": 0.001, "epochs": 5}``). The script
            function signature must accept matching parameter names.
        packages: Pip packages to install (e.g., ``["torch", "transformers"]``).
        image: Custom base container image for the training pod. Uses runtime
            default if omitted.
        pip_index_urls: Custom PyPI mirror URLs (e.g., internal Nexus/Artifactory).
        resources_per_node: Full resource dict for the training pod (e.g.,
            ``{"cpu": "8", "memory": "32Gi", "gpu": 2}``). When provided,
            overrides ``gpu_per_node``. When omitted, defaults to
            ``{"gpu": gpu_per_node}`` (or no resource constraints if
            ``gpu_per_node=0``).
        env: Environment variables set on the trainer container in the TrainJob
            spec. Dict format: ``{"KEY": "VALUE"}``.
        node_selector: K8s node selector applied to the pod.
        tolerations: K8s tolerations for tainted nodes.
        volumes: K8s volume definitions.
        volume_mounts: K8s volume mounts.
        affinity: K8s pod affinity/anti-affinity rules.
        service_account_name: K8s service account for the training pod.
        image_pull_secrets: K8s image pull secrets.
        labels: Extra labels to apply to the TrainJob.
        annotations: Extra annotations to apply to the TrainJob.
        confirmed: Set ``True`` to submit. ``False`` returns preview.

    Returns:
        dict: If ``confirmed=False``: preview with truncated script.
            If ``confirmed=True``: ``job_name``, ``status``, ``message``.
    """
    try:
        if not _SDK_AVAILABLE:
            return ToolError(
                error="Required training libraries not available",
                error_code=ErrorCode.SDK_ERROR,
            ).model_dump()

        ns_err = check_namespace_allowed(namespace)
        if ns_err:
            return ns_err.model_dump()

        bounds_err = validate_training_bounds(
            num_nodes=num_nodes,
            gpu_per_node=gpu_per_node,
            script=script,
            packages=packages,
        )
        if bounds_err:
            return bounds_err.model_dump()

        if name:
            err = validate_k8s_name(name)
            if err:
                return err.model_dump()

        effective_resources = resources_per_node or (
            {"gpu": gpu_per_node} if gpu_per_node > 0 else None
        )

        config: dict[str, Any] = {
            "script": script[:200] + "..." if len(script) > 200 else script,
            "runtime": runtime or "(default: torch-distributed)",
            "name": name,
            "num_nodes": num_nodes,
            "gpu_per_node": gpu_per_node,
            "func_args": func_args,
            "packages": packages or [],
            "image": image,
            "pip_index_urls": pip_index_urls or [],
            "resources_per_node": effective_resources,
        }
        for k, v in [
            ("env", mask_sensitive_data(env) if env else None),
            ("node_selector", node_selector),
            ("tolerations", tolerations),
            ("volumes", volumes),
            ("volume_mounts", volume_mounts),
            ("affinity", affinity),
            ("service_account_name", service_account_name),
            ("image_pull_secrets", image_pull_secrets),
            ("labels", labels),
            ("annotations", annotations),
        ]:
            if v:
                config[k] = v

        safe, reason = is_safe_python_code(script)
        safety_warnings: list[str] = [] if safe else [reason]

        if not confirmed:
            return PreviewResponse(
                message="Review config and set confirmed=True to submit job",
                config={**config, "safety_warnings": safety_warnings},
            ).model_dump()

        if safety_warnings:
            allow_unsafe = os.environ.get("KUBEFLOW_MCP_UNSAFE_SCRIPTS", "").lower() in (
                "1",
                "true",
                "yes",
            )
            if not allow_unsafe:
                return ToolError(
                    error="Script blocked: unsafe patterns detected. "
                    "Set KUBEFLOW_MCP_UNSAFE_SCRIPTS=true to override.",
                    error_code=ErrorCode.VALIDATION_ERROR,
                    details={"safety_warnings": safety_warnings},
                ).model_dump()
            logger.warning(
                "Submitting script with safety warnings (override enabled): %s",
                safety_warnings,
            )

        train_func = _make_train_func(script, func_args=func_args)

        if volumes and not any(m.get("mountPath") == "/workspace" for m in (volume_mounts or [])):
            volumes = [*volumes, {"name": "workspace", "emptyDir": {}}]
            volume_mounts = [
                *(volume_mounts or []),
                {"name": "workspace", "mountPath": "/workspace"},
            ]

        trainer = CustomTrainer(
            func=train_func,
            func_args=func_args,
            packages_to_install=packages,
            image=image,
            pip_index_urls=pip_index_urls or list(DEFAULT_PIP_INDEX_URLS),
            num_nodes=num_nodes,
            resources_per_node=effective_resources,
            env=env,
        )

        options = _build_runtime_patch(
            node_selector=node_selector,
            tolerations=tolerations,
            volumes=volumes,
            volume_mounts=volume_mounts,
            affinity=affinity,
            service_account_name=service_account_name,
            image_pull_secrets=image_pull_secrets,
            labels=labels,
            annotations=annotations,
        )

        if name:
            options.append(Name(name=name))

        _inject_ownership_label(options)
        client = _get_client(namespace)
        job_name = client.train(
            runtime=runtime,
            trainer=trainer,
            options=options,
        )

        return ToolResponse(
            data={
                "job_name": job_name,
                "status": "Created",
                "message": f"Custom training job '{job_name}' submitted",
                "next_steps": [
                    f"get_training_logs(name='{job_name}') — monitor progress",
                    f"get_training_job(name='{job_name}') — check status",
                ],
            }
        ).model_dump()

    except Exception as e:
        return _sdk_error(e, hint="Use troubleshooting_guide prompt for diagnosis")


def run_container_training(
    image: str,
    command: list[str] | None = None,
    args: list[str] | None = None,
    name: str | None = None,
    runtime: str | None = None,
    namespace: str | None = None,
    num_nodes: int = 1,
    gpu_per_node: int = 1,
    resources_per_node: dict[str, Any] | None = None,
    env: dict[str, str] | None = None,
    node_selector: dict[str, str] | None = None,
    tolerations: list[dict[str, Any]] | None = None,
    volumes: list[dict[str, Any]] | None = None,
    volume_mounts: list[dict[str, Any]] | None = None,
    affinity: dict[str, Any] | None = None,
    service_account_name: str | None = None,
    image_pull_secrets: list[dict[str, Any]] | None = None,
    labels: dict[str, str] | None = None,
    annotations: dict[str, str] | None = None,
    confirmed: bool = False,
) -> dict[str, Any]:
    """Run training with a pre-built container image.

    No script validation — full control via container ENTRYPOINT/CMD.
    Use ``command`` to override the default entrypoint. Use ``args`` to override
    the container arguments.

    Args:
        image: Container image (e.g., ``pytorch/pytorch:2.0-cuda11.8``).
        command: Override the trainer container command (sets ``.spec.trainer.command``
            on the TrainJob). When omitted the container's ENTRYPOINT/CMD is used.
        args: Override the trainer container arguments (sets ``.spec.trainer.args`` on the
            TrainJob. When omitted the container's default CMD args are used.
        name: Custom name for the TrainJob. Auto-generated if not provided.
        runtime: ClusterTrainingRuntime name. Defaults to ``torch-distributed``.
            Run ``list_runtimes()`` to see available
            runtimes.
        namespace: K8s namespace. Uses default from kubeconfig when omitted.
            Must be allowed by policy.
        num_nodes: Distributed training nodes. Defaults to 1.
        gpu_per_node: GPUs per node. Set 0 for CPU-only. Defaults to 1.
        resources_per_node: Full resource dict (e.g.,
            ``{"cpu": "8", "memory": "32Gi", "gpu": 2}``). Overrides
            ``gpu_per_node`` when provided.
        env: Environment variables as dict (e.g., ``{"HF_TOKEN": "xxx"}``).
        node_selector: K8s node selector.
        tolerations: K8s tolerations.
        volumes: K8s volume definitions.
        volume_mounts: K8s volume mounts.
        affinity: K8s pod affinity/anti-affinity rules.
        service_account_name: K8s service account for the training pod.
        image_pull_secrets: K8s image pull secrets.
        labels: Extra labels to apply to the TrainJob.
        annotations: Extra annotations to apply to the TrainJob.
        confirmed: Set ``True`` to submit. ``False`` returns preview.

    Returns:
        dict: If ``confirmed=False``: preview with config.
            If ``confirmed=True``: ``job_name``, ``status``, ``message``.
    """
    try:
        if not _SDK_AVAILABLE:
            return ToolError(
                error="Required training libraries not available",
                error_code=ErrorCode.SDK_ERROR,
            ).model_dump()

        ns_err = check_namespace_allowed(namespace)
        if ns_err:
            return ns_err.model_dump()

        bounds_err = validate_training_bounds(
            num_nodes=num_nodes,
            gpu_per_node=gpu_per_node,
        )
        if bounds_err:
            return bounds_err.model_dump()

        if name:
            err = validate_k8s_name(name)
            if err:
                return err.model_dump()

        effective_resources = resources_per_node or (
            {"gpu": gpu_per_node} if gpu_per_node > 0 else None
        )

        config: dict[str, Any] = {
            "image": image,
            "command": command,
            "args": args,
            "name": name,
            "runtime": runtime or "(default: torch-distributed)",
            "num_nodes": num_nodes,
            "gpu_per_node": gpu_per_node,
            "resources_per_node": effective_resources,
            "env": mask_sensitive_data(env) if env else None,
        }
        for k, v in [
            ("node_selector", node_selector),
            ("tolerations", tolerations),
            ("volumes", volumes),
            ("volume_mounts", volume_mounts),
            ("affinity", affinity),
            ("service_account_name", service_account_name),
            ("image_pull_secrets", image_pull_secrets),
            ("labels", labels),
            ("annotations", annotations),
        ]:
            if v:
                config[k] = v

        if not confirmed:
            return PreviewResponse(
                message="Review config and set confirmed=True to submit job",
                config=config,
            ).model_dump()

        trainer = CustomTrainerContainer(
            image=image,
            num_nodes=num_nodes,
            resources_per_node=effective_resources,
            env=env,
        )

        options = _build_runtime_patch(
            node_selector=node_selector,
            tolerations=tolerations,
            volumes=volumes,
            volume_mounts=volume_mounts,
            affinity=affinity,
            service_account_name=service_account_name,
            image_pull_secrets=image_pull_secrets,
            labels=labels,
            annotations=annotations,
        )

        if name:
            options.append(Name(name=name))
        if command:
            options.append(TrainerCommand(command=command))
        if args:
            options.append(TrainerArgs(args=args))

        _inject_ownership_label(options)
        client = _get_client(namespace)
        job_name = client.train(
            runtime=runtime,
            trainer=trainer,
            options=options,
        )

        return ToolResponse(
            data={
                "job_name": job_name,
                "status": "Created",
                "message": f"Container training job '{job_name}' submitted",
                "next_steps": [
                    f"get_training_logs(name='{job_name}') — monitor progress",
                    f"get_training_job(name='{job_name}') — check status",
                ],
            }
        ).model_dump()

    except Exception as e:
        return _sdk_error(e, hint="Use troubleshooting_guide prompt for diagnosis")
