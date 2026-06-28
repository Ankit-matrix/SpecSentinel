"""
spec_parser.py — Parses an OpenAPI 3.x spec (dict) into a compact representation
that fits comfortably into an LLM prompt without hitting token limits.
"""

from __future__ import annotations

import json
from typing import Any


def _resolve_ref(spec: dict, ref: str) -> dict:
    """Follow a $ref pointer inside the spec."""
    parts = ref.lstrip("#/").split("/")
    node = spec
    for p in parts:
        node = node[p]
    return node


def _schema_summary(spec: dict, schema: dict, depth: int = 0) -> dict:
    """Recursively produce a concise schema summary (type + required keys)."""
    if "$ref" in schema:
        schema = _resolve_ref(spec, schema["$ref"])

    s_type = schema.get("type", "object")
    summary: dict[str, Any] = {"type": s_type}

    if s_type == "object":
        props = schema.get("properties", {})
        required = schema.get("required", [])
        summary["required"] = required
        if depth < 2:  # don't recurse infinitely
            summary["properties"] = {
                k: _schema_summary(spec, v, depth + 1) for k, v in props.items()
            }
    elif s_type == "array":
        items = schema.get("items", {})
        summary["items"] = _schema_summary(spec, items, depth + 1)

    if "enum" in schema:
        summary["enum"] = schema["enum"]

    return summary


def _extract_request_body(spec: dict, operation: dict) -> dict | None:
    rb = operation.get("requestBody")
    if not rb:
        return None
    if "$ref" in rb:
        rb = _resolve_ref(spec, rb["$ref"])
    content = rb.get("content", {})
    for media_type in ("application/json", "text/json", "*/*"):
        if media_type in content:
            schema = content[media_type].get("schema", {})
            return _schema_summary(spec, schema)
    return None


def _extract_responses(spec: dict, operation: dict) -> dict:
    out = {}
    for status_code, resp in operation.get("responses", {}).items():
        if "$ref" in resp:
            resp = _resolve_ref(spec, resp["$ref"])
        content = resp.get("content", {})
        schema_summary = None
        for media_type in ("application/json", "text/json", "*/*"):
            if media_type in content:
                schema = content[media_type].get("schema", {})
                schema_summary = _schema_summary(spec, schema)
                break
        out[status_code] = {
            "description": resp.get("description", ""),
            "schema": schema_summary,
        }
    return out


def _extract_parameters(spec: dict, operation: dict, path_level_params: list) -> dict:
    params: dict[str, list] = {"path": [], "query": [], "header": []}
    all_params = list(path_level_params) + list(operation.get("parameters", []))
    for p in all_params:
        if "$ref" in p:
            p = _resolve_ref(spec, p["$ref"])
        loc = p.get("in", "query")
        entry = {
            "name": p["name"],
            "required": p.get("required", False),
            "schema": _schema_summary(spec, p.get("schema", {})),
            "description": p.get("description", ""),
        }
        if loc in params:
            params[loc].append(entry)
    return params


def parse_openapi_spec(spec: dict) -> dict:
    """
    Returns a compact spec summary suitable for LLM prompting:
    {
      "info": { "title": ..., "version": ... },
      "base_path": "/",
      "endpoints": [
        {
          "path": "/posts",
          "method": "POST",
          "summary": "...",
          "operation_id": "...",
          "parameters": { "path": [...], "query": [...] },
          "request_body_schema": {...} | null,
          "responses": { "201": {...}, "422": {...} }
        },
        ...
      ]
    }
    """
    info = spec.get("info", {})
    servers = spec.get("servers", [{}])
    base_path = servers[0].get("url", "/") if servers else "/"

    endpoints = []
    for path, path_item in spec.get("paths", {}).items():
        path_level_params = path_item.get("parameters", [])
        for method in ("get", "post", "put", "patch", "delete", "head", "options"):
            operation = path_item.get(method)
            if not operation:
                continue
            endpoints.append(
                {
                    "path": path,
                    "method": method.upper(),
                    "summary": operation.get("summary", operation.get("description", "")),
                    "operation_id": operation.get("operationId", ""),
                    "tags": operation.get("tags", []),
                    "parameters": _extract_parameters(spec, operation, path_level_params),
                    "request_body_schema": _extract_request_body(spec, operation),
                    "responses": _extract_responses(spec, operation),
                }
            )

    return {
        "info": {"title": info.get("title", ""), "version": info.get("version", "")},
        "base_path": base_path,
        "endpoints": endpoints,
    }


def load_openapi_from_url(url: str) -> dict:
    import requests
    resp = requests.get(url, timeout=10)
    resp.raise_for_status()
    if url.endswith((".yaml", ".yml")):
        import yaml
        return yaml.safe_load(resp.text)
    return resp.json()


def load_openapi_from_file(path: str) -> dict:
    with open(path) as f:
        if path.endswith((".yaml", ".yml")):
            import yaml
            return yaml.safe_load(f)
        return json.load(f)