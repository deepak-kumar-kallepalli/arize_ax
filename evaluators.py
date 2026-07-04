"""Create and version Arize AX evaluators (LLM-as-judge and code).

Mirrors the UI "New Evaluator" form:
  name, scope (span/trace/session), template, choices,
  optimization_direction, enable explanations,
  enable_structured_output, enable_function_calling.
"""

from arize import ArizeClient
from arize._generated.api_client.models.invocation_params import InvocationParams
from arize._generated.api_client.models.provider_params import ProviderParams
from arize._generated.api_client.models.response_format import ResponseFormat
from arize.evaluators.types import (
    CustomCodeConfig,
    EvaluatorLlmConfig,
    EvaluatorVersionCode,
    EvaluatorVersionTemplate,
    EvaluatorWithVersion,
    ManagedCodeConfig,
    ManagedCodeEvaluator,
    TemplateConfig,
)

from config import AI_INTEGRATION_ID, JUDGE_MODEL_NAME, SPACE_ID


def _template_config(
    *,
    eval_name: str,
    template: str,
    choices: dict[str, float],
    scope: str = "span",
    optimization_direction: str = "maximize",
    enable_explanations: bool = True,
    enable_structured_output: bool = False,
    enable_function_calling: bool = True,
    temperature: float = 0.0,
) -> TemplateConfig:
    """Build the config shared by evaluator creation and new versions."""
    return TemplateConfig(
        name=eval_name,
        template=template,
        classification_choices=choices,
        data_granularity=scope,
        direction=optimization_direction,
        include_explanations=enable_explanations,
        use_function_calling_if_available=enable_function_calling,
        llm_config=EvaluatorLlmConfig(
            ai_integration_id=AI_INTEGRATION_ID,
            model_name=JUDGE_MODEL_NAME,
            invocation_parameters=InvocationParams(
                temperature=temperature,
                response_format=(
                    ResponseFormat(type="json_object")
                    if enable_structured_output
                    else None
                ),
            ),
            provider_parameters=ProviderParams(),
        ),
    )


def create_llm_judge_evaluator(
    client: ArizeClient,
    *,
    name: str,
    template: str,
    choices: dict[str, float],
    scope: str = "span",
    optimization_direction: str = "maximize",
    enable_explanations: bool = True,
    enable_structured_output: bool = False,
    enable_function_calling: bool = True,
    description: str | None = None,
    commit_message: str = "Initial version",
) -> EvaluatorWithVersion:
    """LLM-as-judge evaluator. `choices` maps labels to scores,
    e.g. {"factual": 1, "hallucinated": 0}. `template` uses {variable}
    placeholders that tasks bind via column mappings."""
    return client.evaluators.create_template_evaluator(
        space=SPACE_ID,
        name=name,
        description=description,
        commit_message=commit_message,
        template_config=_template_config(
            eval_name=name,
            template=template,
            choices=choices,
            scope=scope,
            optimization_direction=optimization_direction,
            enable_explanations=enable_explanations,
            enable_structured_output=enable_structured_output,
            enable_function_calling=enable_function_calling,
        ),
    )


def add_llm_judge_version(
    client: ArizeClient,
    *,
    evaluator: str,
    eval_name: str,
    commit_message: str,
    template: str,
    choices: dict[str, float],
    scope: str = "span",
    optimization_direction: str = "maximize",
    enable_explanations: bool = True,
    enable_structured_output: bool = False,
    enable_function_calling: bool = True,
) -> EvaluatorVersionTemplate:
    """Add a new immutable version to an existing template evaluator.
    `evaluator` is the evaluator ID (or name, resolved within the space)."""
    return client.evaluators.create_template_version(
        space=SPACE_ID,
        evaluator=evaluator,
        commit_message=commit_message,
        template_config=_template_config(
            eval_name=eval_name,
            template=template,
            choices=choices,
            scope=scope,
            optimization_direction=optimization_direction,
            enable_explanations=enable_explanations,
            enable_structured_output=enable_structured_output,
            enable_function_calling=enable_function_calling,
        ),
    )


def create_custom_code_evaluator(
    client: ArizeClient,
    *,
    name: str,
    code: str,
    variables: list[str],
    imports: str | None = None,
    scope: str = "span",
    description: str | None = None,
    commit_message: str = "Initial version",
) -> EvaluatorWithVersion:
    """Code evaluator running your Python. `code` must define
    evaluate(<variables>) returning a label/score; `variables` are bound
    to span/dataset columns by the task's variable mapping."""
    return client.evaluators.create_code_evaluator(
        space=SPACE_ID,
        name=name,
        description=description,
        commit_message=commit_message,
        code_config=CustomCodeConfig(
            type="custom",
            name=name,
            code=code,
            imports=imports,
            variables=variables,
            data_granularity=scope,
        ),
    )


def create_managed_code_evaluator(
    client: ArizeClient,
    *,
    name: str,
    managed_evaluator: ManagedCodeEvaluator,
    variables: list[str],
    scope: str = "span",
    description: str | None = None,
    commit_message: str = "Initial version",
) -> EvaluatorWithVersion:
    """Built-in code evaluator (MatchesRegex, JSONParseable,
    ContainsAnyKeyword, ContainsAllKeywords, ExactMatch)."""
    return client.evaluators.create_code_evaluator(
        space=SPACE_ID,
        name=name,
        description=description,
        commit_message=commit_message,
        code_config=ManagedCodeConfig(
            type="managed",
            name=name,
            managed_evaluator=managed_evaluator,
            variables=variables,
            data_granularity=scope,
        ),
    )


def add_code_version(
    client: ArizeClient,
    *,
    evaluator: str,
    commit_message: str,
    code_config: CustomCodeConfig | ManagedCodeConfig,
) -> EvaluatorVersionCode:
    return client.evaluators.create_code_version(
        space=SPACE_ID,
        evaluator=evaluator,
        commit_message=commit_message,
        code_config=code_config,
    )


def list_versions(client: ArizeClient, *, evaluator: str):
    return client.evaluators.list_versions(
        space=SPACE_ID, evaluator=evaluator
    ).evaluator_versions
