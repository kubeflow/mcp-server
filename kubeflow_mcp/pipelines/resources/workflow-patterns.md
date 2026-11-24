# Kubeflow Pipelines workflow patterns

How to use the `pipelines` client tools end to end. All tools take an optional
`namespace`; without it the server's configured default is used.

A **pipeline** is the registered blueprint, found by name (`list_pipelines`,
`get_pipeline`). A **run** is one execution of it, found by its `run_id`
(`list_pipeline_runs`, `get_pipeline_run`). Every response carries the IDs
(`pipeline_id`, `run_id`, `experiment_id`), and every tool call is written to the
server's audit log with sensitive values masked.

## Build a pipeline from a request

When the user asks for a pipeline ("train this model on this dataset"), write KFP v2
code as described in `pipelines://guides/authoring`, then:

```
create_pipeline(name, pipeline_code, dataset="hf://org/ds")   # preview: checks the code
create_pipeline(name, pipeline_code, dataset=..., confirmed=True)   # compile and upload
create_experiment(name="...")                                 # preview, then confirmed=True
create_pipeline_run(pipeline=name, experiment="...", params={...})   # preview, then confirm
get_pipeline_run(run_id)                                      # follow it
```

- When the preview or the compile reports an error, fix the code and preview again.
- An existing pipeline name gets a new version; the preview says which it will be.
- `list_experiments()` shows existing experiments to reuse.

## Run a registered pipeline

```
list_pipelines()                                   # find the pipeline name
get_pipeline(name)                                 # check its versions
create_pipeline_run(pipeline=name, params={...})   # preview: what will run, where
create_pipeline_run(pipeline=name, params={...}, confirmed=True)   # after the user approves
get_pipeline_run(run_id)                           # follow the state until it finishes
```

- `version=` picks a pipeline version; the latest is used when it is omitted.
- `experiment=` must name an existing experiment (`create_experiment` makes one); the
  server default is used when omitted.
- Only registered pipelines can be run, by name. Register one with `create_pipeline`
  or outside MCP (KFP UI, `kfp` CLI or SDK).

## Find and debug a failed run

```
list_pipeline_runs(pipeline=name, status="failed")
get_pipeline_run(run_id)                  # which task failed, its error
get_pipeline_logs(run_id, task="train")   # that task's pod output
retry_pipeline_run(run_id)                # preview, then confirmed=True to retry
```

- `get_pipeline_run` lists task names; pass one as `task=` to read only its logs.
- Retry re-runs the tasks that did not succeed. Only `failed` or `canceled` runs can be retried.

## Stop a run

```
stop_pipeline_run(run_id)                  # preview
stop_pipeline_run(run_id, confirmed=True)  # after the user approves
```

KFP runs carry no MCP ownership label, so stop and retry act on any run in the
namespace. Always show the preview and confirm the run with the user first.

## Experiments

```
list_experiments()                        # experiments in the namespace
get_experiment(name)
create_experiment(name)                   # preview, then confirmed=True
delete_experiment(name)                   # platform-admin only; preview, then confirmed=True
```

`create_experiment` returns the existing experiment when the name is taken.
`delete_experiment` keeps the experiment's runs.

## Delete a pipeline

```
delete_pipeline(name)                        # preview: lists every version it removes
delete_pipeline(name, version="v1")          # only that version
delete_pipeline(name, confirmed=True)        # platform-admin only
```

Runs are kept, but a run whose version is gone can no longer be retried. Pipelines
carry no MCP ownership label, so only `platform-admin` can delete them.

## Run states

`pending` → `running` → `succeeded` | `failed` | `skipped`.
A stopped run goes `canceling` → a terminal state (`canceled` or `failed`). `paused` runs wait
for an action.

## Paging

List tools return `next_page_token` and `has_more`. Pass the token as `page_token`
to read the next page. `list_pipeline_runs(pipeline=...)` filters each page after the
server returns it, so a page can hold fewer runs than `limit` while `has_more` is true.
