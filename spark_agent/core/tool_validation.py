from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from spark_agent.core.types import JsonObject


class ToolArgumentValidationError(ValueError):
    """Raised when model-provided tool arguments do not match the tool schema."""


def validate_tool_arguments(tool_name: str, tool_definition: Mapping[str, Any], arguments: JsonObject) -> JsonObject:
    function = tool_definition.get("function")
    if not isinstance(function, Mapping):
        raise ToolArgumentValidationError(f"{tool_name}: tool definition is missing function")
    parameters = function.get("parameters")
    if parameters is None:
        return dict(arguments)
    if not isinstance(parameters, Mapping):
        raise ToolArgumentValidationError(f"{tool_name}: parameters schema must be an object")
    _validate_schema(arguments, parameters, path=tool_name)
    return dict(arguments)


def _validate_schema(value: Any, schema: Mapping[str, Any], *, path: str) -> None:
    expected_type = schema.get("type")
    if expected_type is not None:
        _validate_type(value, expected_type, path=path)

    if expected_type == "object" or isinstance(value, Mapping):
        _validate_object(value, schema, path=path)
    elif expected_type == "array" or (
        isinstance(value, Sequence) and not isinstance(value, str)
    ):
        _validate_array(value, schema, path=path)
    elif expected_type in {"integer", "number"}:
        _validate_number_bounds(value, schema, path=path)
    elif expected_type == "string":
        _validate_string_bounds(value, schema, path=path)


def _validate_object(value: Any, schema: Mapping[str, Any], *, path: str) -> None:
    if not isinstance(value, Mapping):
        raise ToolArgumentValidationError(f"{path}: expected object")
    properties = schema.get("properties", {})
    if not isinstance(properties, Mapping):
        raise ToolArgumentValidationError(f"{path}: properties must be an object")
    required = schema.get("required", [])
    if not isinstance(required, Sequence) or isinstance(required, str):
        raise ToolArgumentValidationError(f"{path}: required must be a list")
    for key in required:
        if not isinstance(key, str):
            raise ToolArgumentValidationError(f"{path}: required keys must be strings")
        if key not in value:
            raise ToolArgumentValidationError(f"{path}.{key}: missing required argument")
    additional = schema.get("additionalProperties", True)
    if additional is False:
        extra = sorted(str(key) for key in value if key not in properties)
        if extra:
            raise ToolArgumentValidationError(f"{path}: unexpected arguments: {', '.join(extra)}")
    for key, item in value.items():
        if key not in properties:
            continue
        subschema = properties[key]
        if not isinstance(subschema, Mapping):
            raise ToolArgumentValidationError(f"{path}.{key}: property schema must be an object")
        _validate_schema(item, subschema, path=f"{path}.{key}")


def _validate_array(value: Any, schema: Mapping[str, Any], *, path: str) -> None:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ToolArgumentValidationError(f"{path}: expected array")
    min_items = schema.get("minItems")
    max_items = schema.get("maxItems")
    if min_items is not None and len(value) < int(min_items):
        raise ToolArgumentValidationError(f"{path}: expected at least {min_items} items")
    if max_items is not None and len(value) > int(max_items):
        raise ToolArgumentValidationError(f"{path}: expected at most {max_items} items")
    items = schema.get("items")
    if items is None:
        return
    if not isinstance(items, Mapping):
        raise ToolArgumentValidationError(f"{path}: items schema must be an object")
    for index, item in enumerate(value):
        _validate_schema(item, items, path=f"{path}[{index}]")


def _validate_type(value: Any, expected_type: Any, *, path: str) -> None:
    if isinstance(expected_type, Sequence) and not isinstance(expected_type, str):
        if not any(_matches_type(value, item) for item in expected_type):
            expected = ", ".join(str(item) for item in expected_type)
            raise ToolArgumentValidationError(f"{path}: expected one of {expected}")
        return
    if not _matches_type(value, expected_type):
        raise ToolArgumentValidationError(f"{path}: expected {expected_type}")


def _matches_type(value: Any, expected_type: Any) -> bool:
    if expected_type == "object":
        return isinstance(value, Mapping)
    if expected_type == "array":
        return isinstance(value, Sequence) and not isinstance(value, str | bytes | bytearray)
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return isinstance(value, int | float) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "null":
        return value is None
    return True


def _validate_number_bounds(value: Any, schema: Mapping[str, Any], *, path: str) -> None:
    if not isinstance(value, int | float) or isinstance(value, bool):
        return
    minimum = schema.get("minimum")
    maximum = schema.get("maximum")
    if minimum is not None and value < minimum:
        raise ToolArgumentValidationError(f"{path}: expected >= {minimum}")
    if maximum is not None and value > maximum:
        raise ToolArgumentValidationError(f"{path}: expected <= {maximum}")


def _validate_string_bounds(value: Any, schema: Mapping[str, Any], *, path: str) -> None:
    if not isinstance(value, str):
        return
    min_length = schema.get("minLength")
    max_length = schema.get("maxLength")
    if min_length is not None and len(value) < int(min_length):
        raise ToolArgumentValidationError(f"{path}: expected at least {min_length} characters")
    if max_length is not None and len(value) > int(max_length):
        raise ToolArgumentValidationError(f"{path}: expected at most {max_length} characters")
