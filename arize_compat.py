"""Forward-compatibility patch for arize 8.38.0.

The Arize API now returns `use_structured_output_if_available` inside
`template_config`, but the generated TemplateConfig model in the newest
published SDK (8.38.0) doesn't define it, so response deserialization
raises "Error due to additional fields ...". Registering the field in the
model's known-properties list makes from_dict skip the strict check (the
value is ignored until an SDK release adds the field for real).

Import this module before making evaluator API calls. Delete once the
SDK ships the field.
"""

from arize._generated.api_client.models.template_config import TemplateConfig

_EXTRA_RESPONSE_FIELDS = ["use_structured_output_if_available"]

_props = getattr(TemplateConfig, "_TemplateConfig__properties")
for _field in _EXTRA_RESPONSE_FIELDS:
    if _field not in _props:
        _props.append(_field)
