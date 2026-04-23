from __future__ import annotations

from typing import Any


async def run(args: dict, settings: Any, redis_client: Any = None) -> str:
    data = args.get("data")
    schema = args.get("schema")
    if data is None:
        return "error: missing 'data' argument"
    if not isinstance(schema, dict):
        return "error: 'schema' must be a dict"
    try:
        import jsonschema  # type: ignore[import]
        jsonschema.validate(instance=data, schema=schema)
        return "valid"
    except jsonschema.ValidationError as exc:
        return f"validation error: {exc.message}"
    except jsonschema.SchemaError as exc:
        return f"schema error: {exc.message}"
    except ImportError:
        return "error: jsonschema package not installed"
