#!/usr/bin/env python3
"""Fetch latest X prompts via APIPRO chat completion and save as JSON."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List
from urllib import error, request


def env_int(name: str, default: int, min_value: int, max_value: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        value = int(raw)
    except ValueError:
        return default
    return max(min_value, min(max_value, value))


def iso_utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def normalize_base_url(value: str) -> str:
    base = value.strip() or "http://apipro.maynor1024.live/v1"
    return base.rstrip("/")


def post_chat_completion(
    base_url: str,
    api_key: str,
    model: str,
    query: str,
    lookback_hours: int,
    max_items: int,
    timeout_seconds: int,
) -> Dict[str, Any]:
    endpoint = f"{base_url}/chat/completions"

    system_prompt = (
        "You are a structured data collector. "
        "Directly fetch latest public posts from X and return JSON only."
    )

    user_prompt = (
        "Task: collect latest X posts related to GPT-Image-2 prompts.\n"
        f"Lookback hours: {lookback_hours}\n"
        f"Search query intent: {query}\n"
        f"Max items: {max_items}\n"
        "Rules:\n"
        "1) Deduplicate by URL or text similarity.\n"
        "2) Keep newest first.\n"
        "3) For each item include prompt when available.\n"
        "Output strictly as JSON object:\n"
        "{"
        "\"meta\":{"
        "\"source\":\"x\","
        "\"lookback_hours\":24,"
        "\"query\":\"...\","
        "\"count\":0"
        "},"
        "\"items\":["
        "{"
        "\"url\":\"\","
        "\"author\":\"\","
        "\"created_at\":\"\","
        "\"text\":\"\","
        "\"prompt\":\"\","
        "\"reason\":\"\""
        "}"
        "]"
        "}"
    )

    payload = {
        "model": model,
        "temperature": 0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    req = request.Request(
        endpoint,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "awesome-gptimage2-apipro/1.0",
        },
        data=json.dumps(payload).encode("utf-8"),
    )

    with request.urlopen(req, timeout=timeout_seconds) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


def extract_message_content(resp: Dict[str, Any]) -> str:
    choices = resp.get("choices", [])
    if not choices:
        raise ValueError("No choices in completion response")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if not isinstance(content, str) or not content.strip():
        raise ValueError("No message content in completion response")
    return content.strip()


def strip_code_fence(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines:
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return text


def normalize_item(item: Dict[str, Any]) -> Dict[str, str]:
    return {
        "url": str(item.get("url", "")).strip(),
        "author": str(item.get("author", "")).strip(),
        "created_at": str(item.get("created_at", "")).strip(),
        "text": str(item.get("text", "")).strip(),
        "prompt": str(item.get("prompt", "")).strip(),
        "reason": str(item.get("reason", "")).strip(),
    }


def normalize_output(
    parsed: Dict[str, Any],
    base_url: str,
    model: str,
    query: str,
    lookback_hours: int,
) -> Dict[str, Any]:
    raw_items = parsed.get("items", [])
    items: List[Dict[str, str]] = []
    if isinstance(raw_items, list):
        for obj in raw_items:
            if isinstance(obj, dict):
                items.append(normalize_item(obj))

    meta_raw = parsed.get("meta", {})
    output = {
        "meta": {
            "generated_at_utc": iso_utc_now(),
            "source": str(meta_raw.get("source", "x")),
            "query": str(meta_raw.get("query", query)),
            "lookback_hours": int(meta_raw.get("lookback_hours", lookback_hours)),
            "count": len(items),
            "provider": "apipro.maynor1024.live",
            "base_url": base_url,
            "model": model,
        },
        "items": items,
    }
    return output


def write_json(path: str, payload: Dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")


def main() -> int:
    api_key = os.getenv("APIPRO_API_KEY", "").strip()
    if not api_key:
        print("Missing required env: APIPRO_API_KEY", file=sys.stderr)
        return 2

    base_url = normalize_base_url(os.getenv("APIPRO_BASE_URL", "http://apipro.maynor1024.live/v1"))
    model = os.getenv("APIPRO_MODEL", "grok-4.1-fast").strip() or "grok-4.1-fast"
    query = os.getenv(
        "APIPRO_QUERY",
        "(gptimage2 OR gpt-image-2 OR \"gpt image 2\") (prompt OR 提示词)",
    ).strip()
    lookback_hours = env_int("APIPRO_LOOKBACK_HOURS", 24, 1, 168)
    max_items = env_int("APIPRO_MAX_ITEMS", 60, 1, 200)
    timeout_seconds = env_int("APIPRO_TIMEOUT_SECONDS", 90, 10, 600)
    output_file = os.getenv("APIPRO_OUTPUT_FILE", "data/latest-prompts.json").strip()

    try:
        resp = post_chat_completion(
            base_url=base_url,
            api_key=api_key,
            model=model,
            query=query,
            lookback_hours=lookback_hours,
            max_items=max_items,
            timeout_seconds=timeout_seconds,
        )
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="ignore")
        print(f"HTTP {exc.code}: {body[:500]}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
        print(str(exc), file=sys.stderr)
        return 1

    try:
        content = extract_message_content(resp)
        parsed = json.loads(strip_code_fence(content))
        if not isinstance(parsed, dict):
            raise ValueError("Model content is not a JSON object")
    except Exception as exc:
        print(f"Invalid model output JSON: {exc}", file=sys.stderr)
        return 1

    output = normalize_output(
        parsed=parsed,
        base_url=base_url,
        model=model,
        query=query,
        lookback_hours=lookback_hours,
    )
    write_json(output_file, output)
    print(f"Saved {output['meta']['count']} items to {output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
