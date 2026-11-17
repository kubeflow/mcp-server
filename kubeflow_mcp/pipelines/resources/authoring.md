# Writing pipelines for `create_pipeline`

`create_pipeline` registers a Kubeflow pipeline from KFP v2 Python code. Use it when the
user asks for a pipeline ("train this model on this dataset"): write the code, preview it,
then confirm. The server compiles the code to pipeline YAML and uploads it unchanged;
`create_pipeline_run` then runs it by name.

## Where code runs

- **`@dsl.component` bodies run in the cluster**, one pod per step. Put all real work there:
  imports, data loading, training, evaluation. Any package is fine inside a component.
- **The `@dsl.pipeline` body runs on the MCP server** when the pipeline is compiled. It may
  only wire steps together, so the server accepts a small set of statements there.

## Accepted shape

At the top level:

- `from kfp import dsl` and `from kfp import kubernetes`
- `from kfp.dsl import ...` (types such as `Dataset`, `Model`, `Input`, `Output`)
- `from typing import NamedTuple, List, Dict, Optional`
- constants with literal values, e.g. `BASE_IMAGE = "python:3.11-slim"`
- `@dsl.component` functions and exactly one `@dsl.pipeline` function

Inside the `@dsl.pipeline` function:

- call components and assign the task: `train_task = train(dataset=load_task.outputs["dataset"])`
- task options: `.after`, `.set_caching_options`, `.set_cpu_request`, `.set_cpu_limit`,
  `.set_memory_request`, `.set_memory_limit`, `.set_accelerator_type`,
  `.set_accelerator_limit`, `.set_retry`, `.set_display_name`, `.set_env_variable`,
  `.set_container_image`, `.ignore_upstream_failure`
- `kubernetes.*` helpers such as `kubernetes.use_secret_as_env(task, ...)`
- `with dsl.If(...)`, `dsl.Elif`, `dsl.Else`, `with dsl.ParallelFor(...) as item`,
  `with dsl.ExitHandler(task)`, and `dsl.Collected`, `dsl.OneOf`, `dsl.importer`
- task results through `.output` and `.outputs["name"]`

Not accepted, with the reason in the error: other top-level imports, `import x`, loops,
comprehensions, builtins such as `open` or `print` in the pipeline body, string type
annotations and `@dsl.container_component`. Component parameter types must be `str`,
`int`, `float`, `bool`, `list`, `dict`, `dsl.Input[...]`, `dsl.Output[...]` or a
`NamedTuple`. Decorator options must be literals or top-level constants.

`@dsl.component` options: `base_image`, `packages_to_install`, `pip_index_urls`,
`pip_trusted_hosts`, `install_kfp_package`, `use_venv`.
`@dsl.pipeline` options: `name`, `description`, `display_name`, `pipeline_root`.

## Images and packages

`packages_to_install` is installed with pip every time the step starts, which can take
minutes for large packages such as `torch`. Pick a `base_image` that already has the heavy
packages and keep `packages_to_install` for small extras:

| Step | Suggested `base_image` |
|------|------------------------|
| Data loading, light preprocessing | `python:3.11-slim` |
| PyTorch training | `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` |

Without `base_image`, KFP uses `python:3.11`. Use images the cluster can pull; air-gapped
clusters need images from their own registry.

## Example: train a model on a dataset

```python
from typing import NamedTuple

from kfp import dsl
from kfp.dsl import Dataset, Input, Metrics, Model, Output

LIGHT_IMAGE = "python:3.11-slim"
TRAIN_IMAGE = "pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"


@dsl.component(base_image=LIGHT_IMAGE, packages_to_install=["datasets==3.2.0"])
def load_data(dataset_name: str, split: str, dataset: Output[Dataset]):
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    if len(ds) == 0:
        raise ValueError(f"dataset {dataset_name} split {split} is empty")
    ds.save_to_disk(dataset.path)


@dsl.component(base_image=TRAIN_IMAGE, packages_to_install=["datasets==3.2.0", "transformers==4.47.1"])
def train(
    dataset: Input[Dataset],
    model_name: str,
    epochs: int,
    model: Output[Model],
    metrics: Output[Metrics],
) -> NamedTuple("Outputs", [("loss", float)]):
    from datasets import load_from_disk

    ds = load_from_disk(dataset.path)
    # ... fine-tune model_name on ds for `epochs` epochs and save it to model.path ...
    loss = 0.0
    metrics.log_metric("loss", loss)
    return (loss,)


@dsl.component(base_image=LIGHT_IMAGE)
def evaluate(model: Input[Model], loss: float) -> float:
    return 1.0 - loss


@dsl.pipeline(name="train-on-dataset", description="Load a dataset, train and evaluate.")
def train_on_dataset(
    dataset_name: str = "stanfordnlp/imdb",
    model_name: str = "distilbert-base-uncased",
    epochs: int = 1,
):
    load_task = load_data(dataset_name=dataset_name, split="train")
    train_task = train(
        dataset=load_task.outputs["dataset"], model_name=model_name, epochs=epochs
    )
    train_task.set_accelerator_type("nvidia.com/gpu").set_accelerator_limit(1)
    train_task.set_memory_limit("16G")
    evaluate(model=train_task.outputs["model"], loss=train_task.outputs["loss"])
```

Then:

1. `create_pipeline(name="train-on-dataset", pipeline_code=..., dataset="hf://stanfordnlp/imdb")`:
   the preview lists components, images and parameters. Show it to the user.
2. The same call with `confirmed=True` compiles and uploads it.
3. `create_experiment(name=...)` (preview, then confirm), or pick one from `list_experiments()`.
4. `create_pipeline_run(pipeline="train-on-dataset", experiment=..., params={...})`.
5. Follow it with `get_pipeline_run(run_id)` and `get_pipeline_logs(run_id)`.

## Dataset check

`dataset=` takes the dataset URI the pipeline reads. For `hf://<org>/<dataset>` the format
is checked in the preview and the Hub is asked on submit; a dataset the Hub reports
missing is rejected. Other schemes, such as `s3://`, are not checked: let the first
component fail with a clear message if the data cannot be read.

## Distributed training with Kubeflow Trainer

A component can create a `TrainJob` (`trainer.kubeflow.org/v1alpha1`) with the
`kubernetes` client and wait for it. The pipeline's service account (usually
`pipeline-runner`) then needs RBAC to create and get `trainjobs` in the namespace, and
Kubeflow Trainer must be installed. Without that the step fails with `403 Forbidden`.

## Errors

- A rejected shape names the line and the rule; fix it and preview again.
- A compile error is KFP's own message with the line of your code when there is one,
  such as a wrong argument name or type.
- The safety scan flags patterns such as `subprocess` or `os.system` anywhere in the code.
  The preview lists them, and submission is blocked unless the server operator sets
  `KUBEFLOW_MCP_UNSAFE_SCRIPTS=true`.
