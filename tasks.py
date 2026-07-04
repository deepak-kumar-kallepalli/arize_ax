"""Create Arize AX tasks that run evaluators on a project or dataset.

Mirrors the UI "New Task" configuration:
  pick datasource (project / dataset), filter spans, variable mapping.
"""

from arize import ArizeClient
from arize.tasks.types import BaseEvaluationTaskRequestEvaluatorsInner, Task, TaskRun

from config import SPACE_ID


def _evaluator_bindings(
    evaluators: list[dict],
) -> list[BaseEvaluationTaskRequestEvaluatorsInner]:
    """evaluators: [{"evaluator_id": ..., "column_mappings": {...},
    "query_filter": ...}] - column_mappings is the UI's variable mapping,
    e.g. {"input": "attributes.input.value", "output": "attributes.output.value"}"""
    return [
        BaseEvaluationTaskRequestEvaluatorsInner(
            evaluator_id=e["evaluator_id"],
            column_mappings=e.get("column_mappings"),
            query_filter=e.get("query_filter"),
        )
        for e in evaluators
    ]


def create_project_task(
    client: ArizeClient,
    *,
    name: str,
    evaluators: list[dict],
    project: str,
    task_type: str = "template_evaluation",  # or "code_evaluation"
    query_filter: str | None = None,
    sampling_rate: float = 1.0,
    is_continuous: bool = True,
) -> Task:
    """Datasource = project. `query_filter` is the UI span filter,
    e.g. "attributes.openinference.span.kind == 'LLM'"."""
    return client.tasks.create_evaluation_task(
        space=SPACE_ID,
        name=name,
        task_type=task_type,
        evaluators=_evaluator_bindings(evaluators),
        project=project,
        query_filter=query_filter,
        sampling_rate=sampling_rate,
        is_continuous=is_continuous,
    )


def create_dataset_task(
    client: ArizeClient,
    *,
    name: str,
    evaluators: list[dict],
    dataset: str,
    experiment_ids: list[str],
    task_type: str = "template_evaluation",
) -> Task:
    """Datasource = dataset. Dataset tasks grade experiment outputs (an
    experiment = one run of your app over the dataset), so at least one
    experiment ID is required. Variable mapping references dataset/
    experiment columns (e.g. {"input": "input", "output": "output"})."""
    return client.tasks.create_evaluation_task(
        space=SPACE_ID,
        name=name,
        task_type=task_type,
        evaluators=_evaluator_bindings(evaluators),
        dataset=dataset,
        experiment_ids=experiment_ids,
    )


def run_task_now(
    client: ArizeClient,
    *,
    task: str,
    wait: bool = True,
    timeout: float = 600.0,
    **trigger_kwargs,
) -> TaskRun:
    """On-demand run (project tasks accept data_start_time/data_end_time/
    max_spans; dataset tasks accept max_examples/example_ids/...)."""
    run = client.tasks.trigger_run(space=SPACE_ID, task=task, **trigger_kwargs)
    if wait:
        run = client.tasks.wait_for_run(run_id=run.id, timeout=timeout)
    return run
