"""Demo: create LLM-as-judge + code evaluators, version them, and run
them as tasks on a project and a dataset.

Usage:
  python main.py --project <project-id-or-name>
  python main.py --dataset <dataset-id-or-name> [--experiment <experiment-id>]
  python main.py --create-golden          # golden dataset + experiment + eval
  (without --experiment, a demo experiment is run on the dataset first)
"""

import argparse
import time
from datetime import datetime, timedelta, timezone

from config import get_client
from evaluators import (
    add_llm_judge_version,
    create_custom_code_evaluator,
    create_llm_judge_evaluator,
    find_evaluator,
    list_versions,
)
from experiments import create_golden_dataset, run_experiment
from tasks import create_dataset_task, create_project_task, run_task_now
from spans import list_spans

HALLUCINATION_TEMPLATE_V1 = """\
You are an evaluation assistant. Given the input and output below, decide
whether the output contains hallucinated content.

Input: {input}
Output: {output}

Answer with exactly one label: "factual" or "hallucinated"."""

HALLUCINATION_TEMPLATE_V2 = """\
You are a strict fact-checking judge. Evaluate whether the output is fully
grounded in the input. Any claim not supported by the input counts as
hallucination.

Input: {input}
Output: {output}

Answer with exactly one label: "factual" or "hallucinated"."""

RESPONSE_LENGTH_CODE = '''\
def evaluate(output):
    """Pass if the response is non-empty and under 2000 chars."""
    text = output or ""
    ok = 0 < len(text) < 2000
    return "pass" if ok else "fail"
'''


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", help="Project ID or name to attach a task to")
    parser.add_argument("--dataset", help="Dataset ID or name to attach a task to")
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Experiment ID whose outputs the dataset task grades "
        "(repeatable; if omitted, a demo experiment is run first)",
    )
    parser.add_argument(
        "--create-golden",
        action="store_true",
        help="Create a small golden QA dataset and use it as the datasource",
    )
    args = parser.parse_args()

    client = get_client()

    # 1. LLM-as-judge evaluator (scope=span, maximize, explanations on).
    # Get-or-create: names are unique per space, and a first run may have
    # created it already.
    e_time = datetime.now(timezone.utc)
    s_time = e_time - timedelta(days=20)

    judge = find_evaluator(client, name="hallucination-judge")
    list_spans(client, "arize-final-test", start_time=s_time, end_time=e_time)

    # if judge:
    #     print(f"Reusing LLM judge evaluator: {judge.id}")
    # else:
    #     judge = create_llm_judge_evaluator(
    #         client,
    #         name="hallucination-judge",
    #         description="Detects hallucinated content in LLM responses",
    #         template=HALLUCINATION_TEMPLATE_V1,
    #         choices={"factual": 1, "hallucinated": 0},
    #         scope="span",
    #         optimization_direction="maximize",
    #         enable_explanations=True,
    #         enable_structured_output=False,
    #         enable_function_calling=True,
    #     )
    #     print(f"Created LLM judge evaluator: {judge.id} (version {judge.version.id})")

    # 2. Custom code evaluator
    # code_eval = create_custom_code_evaluator(
    #     client,
    #     name="response-length-check",
    #     description="Fails empty or overly long responses",
    #     code=RESPONSE_LENGTH_CODE,
    #     variables=["output"],
    #     scope="span",
    # )
    # print(f"Created code evaluator: {code_eval.id} (version {code_eval.version.id})")

    # 3. Add a v2 to the judge evaluator
    # v2 = add_llm_judge_version(
    #     client,
    #     evaluator=judge.id,
    #     eval_name="hallucination-judge",
    #     commit_message="Stricter grounding criteria",
    #     template=HALLUCINATION_TEMPLATE_V2,
    #     choices={"factual": 1, "hallucinated": 0},
    # )
    # versions = list_versions(client, evaluator=judge.id)
    # print(f"Added version {v2.id}; evaluator now has {len(versions)} versions")

    # 4a. Continuous task on a project with span filter + variable mapping
    if args.project:
        task = create_project_task(
            client,
            name="prod-hallucination-check",
            task_type="template_evaluation",
            evaluators=[
                {
                    "evaluator_id": judge.id,
                    "column_mappings": {
                        "input": "attributes.input.value",
                        "output": "attributes.output.value",
                    },
                }
            ],
            project=args.project,
            query_filter="attributes.openinference.span.kind = 'AGENT'",
            sampling_rate=1.0,
            is_continuous=False,
        )
        print(f"Created continuous project task: {task.id}")

        end_time = datetime.now(timezone.utc)

        run = run_task_now(
            client,
            task=task.id,
            data_start_time=end_time - timedelta(days=10),
            data_end_time=end_time,
        )
        print(f"Run {run.id} finished with status: {run.status}")


    # 4b. On-demand task on a dataset (golden dataset -> experiment -> eval)
    if args.create_golden:
        stamp = int(time.time())
        golden = create_golden_dataset(client, name=f"golden-qa-{stamp}")
        args.dataset = golden.id
        print(f"Created golden dataset: {golden.id}")

    if args.dataset and not args.experiment:
        # Dataset tasks grade experiment outputs, so produce some first.
        experiment = run_experiment(
            client,
            name=f"demo-experiment-{int(time.time())}",
            dataset=args.dataset,
        )
        args.experiment = [experiment.id]
        print(f"Ran experiment: {experiment.id}")

    if args.dataset:
        task = create_dataset_task(
            client,
            name="dataset-hallucination-check",
            task_type="template_evaluation",
            evaluators=[
                {
                    "evaluator_id": judge.id,
                    "column_mappings": {"input": "input", "output": "output"},
                }
            ],
            dataset=args.dataset,
            experiment_ids=args.experiment,
        )
        print(f"Created dataset task: {task.id}")
        run = run_task_now(client, task=task.id)
        print(f"Run {run.id} finished with status: {run.status}")

    if not args.project and not args.dataset:
        print("No --project/--dataset given; skipped task creation.")


if __name__ == "__main__":
    main()
