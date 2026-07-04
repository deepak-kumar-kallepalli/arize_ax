"""Golden datasets and experiments.

A golden dataset is a curated set of examples with trusted reference
answers (ground truth). An experiment is one run of your app/LLM over a
dataset - each example gets an output. Dataset evaluation tasks then
grade those experiment outputs (hence experiment_ids is required).
"""

from arize import ArizeClient
from arize.datasets.types import Dataset
from arize.experiments.types import Example, Experiment

from config import SPACE_ID

GOLDEN_QA_EXAMPLES = [
    {
        "input": "What is the capital of France?",
        "reference": "Paris",
    },
    {
        "input": "Who wrote Pride and Prejudice?",
        "reference": "Jane Austen",
    },
    {
        "input": "What is the boiling point of water at sea level in Celsius?",
        "reference": "100",
    },
]


def create_golden_dataset(
    client: ArizeClient,
    *,
    name: str,
    examples: list[dict] | None = None,
) -> Dataset:
    """Create a golden dataset. Each example holds an `input` and a
    trusted `reference` answer that evaluators compare outputs against."""
    return client.datasets.create(
        space=SPACE_ID,
        name=name,
        examples=examples or GOLDEN_QA_EXAMPLES,
    )


def demo_task(example: Example) -> str:
    """Stand-in for your real app: answer the example's input.
    Replace the body with a call to your LLM/agent."""
    question = str(example.input.get("input", ""))
    canned = {
        "What is the capital of France?": "Paris",
        "Who wrote Pride and Prejudice?": "Jane Austen",
    }
    return canned.get(question, "I am not sure.")


def run_experiment(
    client: ArizeClient,
    *,
    name: str,
    dataset: str,
    task=demo_task,
    concurrency: int = 3,
) -> Experiment:
    """Run `task` over every dataset example and upload the outputs to
    Arize as an experiment. Returns the experiment (its .id is what
    dataset evaluation tasks take as experiment_ids)."""
    experiment, _results_df = client.experiments.run(
        space=SPACE_ID,
        name=name,
        dataset=dataset,
        task=task,
        concurrency=concurrency,
    )
    return experiment
