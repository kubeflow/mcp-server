# Kubeflow Pipelines workflow patterns

How to use the `pipelines` client tools end to end. All tools take an optional
`namespace`; without it the server's configured default is used.

## Run a registered pipeline

```
list_pipelines()                                   # find the pipeline name
get_pipeline(name)                                 # check its versions
create_pipeline_run(pipeline=name, params={...})   # preview: what will run, where
create_pipeline_run(pipeline=name, params={...}, confirmed=True)   # after the user approves
get_pipeline_run(run_id)                           # follow the state until it finishes
```

- `version=` picks a pipeline version; the latest is used when it is omitted.
- `experiment=` must name an existing experiment; the server default is used when omitted.
- Only registered pipelines can be run. Compile and upload a pipeline outside MCP
  (KFP UI, `kfp` CLI or SDK), then run it here by name.

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

## Run states

`pending` → `running` → `succeeded` | `failed` | `skipped`.
A stopped run goes `canceling` → `canceled`. `paused` runs wait for an action.

## Paging

List tools return `next_page_token` and `has_more`. Pass the token as `page_token`
to read the next page. `list_pipeline_runs(pipeline=...)` filters each page after the
server returns it, so a page can hold fewer runs than `limit` while `has_more` is true.
