#!/usr/bin/env python3
"""ShadowWrite local API chat CLI.

Features:
- Interactive terminal chat loop.
- Supports OpenAI-compatible Chat Completions APIs.
- Supports Gemini REST API (generateContent).
- Streams model output to terminal and Markdown file.
- Persists user messages as quote blocks.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, TextIO
from urllib import error, parse, request


Message = Dict[str, str]


@dataclass
class AppConfig:
    provider: str
    model: str
    api_key: str
    base_url: str
    output_path: Path
    temperature: float
    timeout_sec: int
    system_prompt: str
    stream: bool
    default_section: str


def parse_bool(value: str | None, default: bool) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    return default


def load_dotenv(path: Path) -> None:
    """Load KEY=VALUE lines from a .env file into os.environ.

    Existing environment variables are not overwritten.
    """
    if not path.exists():
        return
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive local writing chat that appends AI output to Markdown."
    )
    parser.add_argument(
        "--provider",
        default=os.getenv("SHADOWWRITE_PROVIDER", "openai_compat"),
        choices=["openai_compat", "gemini"],
        help="API provider adapter.",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("SHADOWWRITE_MODEL", ""),
        help="Model name (for example: gpt-4o-mini, deepseek-chat, gemini-2.0-flash).",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("SHADOWWRITE_API_KEY", ""),
        help="API key. If omitted, provider-specific env vars are used.",
    )
    parser.add_argument(
        "--base-url",
        default=os.getenv("SHADOWWRITE_BASE_URL", ""),
        help="Provider base URL override.",
    )
    parser.add_argument(
        "--output",
        default=os.getenv("SHADOWWRITE_OUTPUT", "novel_draft.md"),
        help="Markdown file path for appended output.",
    )
    parser.add_argument(
        "--system",
        default=os.getenv("SHADOWWRITE_SYSTEM_PROMPT", ""),
        help="Optional system prompt.",
    )
    parser.add_argument(
        "--section",
        default=os.getenv("SHADOWWRITE_SECTION", ""),
        help="Manual section label used in Markdown headings.",
    )
    parser.add_argument(
        "--stream",
        action=argparse.BooleanOptionalAction,
        default=parse_bool(os.getenv("SHADOWWRITE_STREAM"), True),
        help="Enable streaming output/write (default: true).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=float(os.getenv("SHADOWWRITE_TEMPERATURE", "0.7")),
        help="Sampling temperature.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=int(os.getenv("SHADOWWRITE_TIMEOUT_SEC", "120")),
        help="HTTP timeout in seconds.",
    )
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> AppConfig:
    provider = args.provider

    if provider == "gemini":
        api_key = (
            args.api_key
            or os.getenv("GEMINI_API_KEY", "")
            or os.getenv("GOOGLE_API_KEY", "")
        )
        model = args.model or "gemini-2.0-flash"
        base_url = args.base_url or "https://generativelanguage.googleapis.com/v1beta"
    else:
        api_key = (
            args.api_key
            or os.getenv("OPENAI_API_KEY", "")
            or os.getenv("DEEPSEEK_API_KEY", "")
        )
        model = args.model or "gpt-4o-mini"
        base_url = (
            args.base_url
            or os.getenv("OPENAI_BASE_URL", "")
            or os.getenv("DEEPSEEK_BASE_URL", "")
            or "https://api.openai.com/v1"
        )

    if not api_key:
        raise ValueError(
            "Missing API key. Set SHADOWWRITE_API_KEY or provider-specific keys."
        )

    return AppConfig(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        output_path=Path(args.output),
        temperature=args.temperature,
        timeout_sec=args.timeout,
        system_prompt=args.system.strip(),
        stream=bool(args.stream),
        default_section=args.section.strip(),
    )


def http_post_json(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout_sec: int,
) -> Dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_sec) as resp:
            raw = resp.read().decode("utf-8")
            return json.loads(raw)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"HTTP {exc.code} from {url}\n{detail}"
        ) from exc
    except error.URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON response from {url}") from exc


def open_stream_request(
    url: str,
    payload: Dict[str, Any],
    headers: Dict[str, str],
    timeout_sec: int,
):
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(url, data=body, headers=headers, method="POST")
    try:
        return request.urlopen(req, timeout=timeout_sec)
    except error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from {url}\n{detail}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Network error for {url}: {exc.reason}") from exc


def iter_sse_payloads(resp) -> Iterator[str]:
    data_lines: List[str] = []
    while True:
        raw = resp.readline()
        if not raw:
            break
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")

        if not line:
            if data_lines:
                yield "\n".join(data_lines)
                data_lines = []
            continue

        if line.startswith(":"):
            continue
        if line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if data_lines:
        yield "\n".join(data_lines)


def extract_text_fragments(content: Any) -> List[str]:
    if isinstance(content, str):
        return [content]
    if isinstance(content, list):
        parts: List[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return parts
    return []


def extract_openai_text(response: Dict[str, Any]) -> str:
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError("OpenAI-compatible response missing 'choices'.")
    message = choices[0].get("message") or {}
    parts = extract_text_fragments(message.get("content"))
    if parts:
        return "".join(parts).strip()
    raise RuntimeError("OpenAI-compatible response has no text content.")


def extract_openai_delta_text(event: Dict[str, Any]) -> str:
    choices = event.get("choices") or []
    if not choices:
        return ""
    delta = choices[0].get("delta") or {}

    parts = extract_text_fragments(delta.get("content"))
    if parts:
        return "".join(parts)

    direct_text = delta.get("text")
    if isinstance(direct_text, str):
        return direct_text
    reasoning_text = delta.get("reasoning_content")
    if isinstance(reasoning_text, str):
        return reasoning_text
    return ""


def to_gemini_request(messages: List[Message], temperature: float) -> Dict[str, Any]:
    system_chunks: List[str] = []
    contents: List[Dict[str, Any]] = []

    for msg in messages:
        role = msg["role"]
        text = msg["content"]
        if role == "system":
            system_chunks.append(text)
            continue
        if role == "assistant":
            gemini_role = "model"
        else:
            gemini_role = "user"
        contents.append({"role": gemini_role, "parts": [{"text": text}]})

    payload: Dict[str, Any] = {
        "contents": contents,
        "generationConfig": {"temperature": temperature},
    }
    if system_chunks:
        payload["systemInstruction"] = {
            "parts": [{"text": "\n\n".join(system_chunks)}]
        }
    return payload


def extract_gemini_text_or_empty(response: Dict[str, Any]) -> str:
    candidates = response.get("candidates") or []
    if not candidates:
        return ""
    first = candidates[0]
    content = first.get("content") or {}
    parts = content.get("parts") or []
    chunks: List[str] = []
    for part in parts:
        if isinstance(part, dict):
            text = part.get("text")
            if isinstance(text, str):
                chunks.append(text)
    return "".join(chunks)


def extract_gemini_text(response: Dict[str, Any]) -> str:
    text = extract_gemini_text_or_empty(response)
    if text:
        return text.strip()

    if not (response.get("candidates") or []):
        feedback = response.get("promptFeedback")
        if feedback:
            raise RuntimeError(f"Gemini blocked request: {feedback}")
        raise RuntimeError("Gemini response missing 'candidates'.")
    raise RuntimeError("Gemini response has no text parts.")


def call_model(config: AppConfig, messages: List[Message]) -> str:
    if config.provider == "gemini":
        endpoint = (
            f"{config.base_url.rstrip('/')}/models/{parse.quote(config.model)}:generateContent"
            f"?key={parse.quote(config.api_key)}"
        )
        payload = to_gemini_request(messages, config.temperature)
        response = http_post_json(
            endpoint,
            payload,
            {"Content-Type": "application/json"},
            config.timeout_sec,
        )
        return extract_gemini_text(response)

    endpoint = f"{config.base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
    }
    response = http_post_json(
        endpoint,
        payload,
        {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {config.api_key}",
        },
        config.timeout_sec,
    )
    return extract_openai_text(response)


def iter_openai_stream(config: AppConfig, messages: List[Message]) -> Iterator[str]:
    endpoint = f"{config.base_url.rstrip('/')}/chat/completions"
    payload = {
        "model": config.model,
        "messages": messages,
        "temperature": config.temperature,
        "stream": True,
    }
    headers = {
        "Content-Type": "application/json",
        "Accept": "text/event-stream",
        "Authorization": f"Bearer {config.api_key}",
    }

    with open_stream_request(endpoint, payload, headers, config.timeout_sec) as resp:
        for payload_text in iter_sse_payloads(resp):
            if payload_text == "[DONE]":
                break
            try:
                event = json.loads(payload_text)
            except json.JSONDecodeError:
                continue
            chunk = extract_openai_delta_text(event)
            if chunk:
                yield chunk


def iter_gemini_stream(config: AppConfig, messages: List[Message]) -> Iterator[str]:
    endpoint = (
        f"{config.base_url.rstrip('/')}/models/{parse.quote(config.model)}:streamGenerateContent"
        f"?alt=sse&key={parse.quote(config.api_key)}"
    )
    payload = to_gemini_request(messages, config.temperature)
    headers = {"Content-Type": "application/json", "Accept": "text/event-stream"}

    previous_full_text = ""
    with open_stream_request(endpoint, payload, headers, config.timeout_sec) as resp:
        for payload_text in iter_sse_payloads(resp):
            try:
                event = json.loads(payload_text)
            except json.JSONDecodeError:
                continue

            current_full_text = extract_gemini_text_or_empty(event)
            if not current_full_text:
                continue

            if current_full_text.startswith(previous_full_text):
                delta = current_full_text[len(previous_full_text):]
                previous_full_text = current_full_text
            elif previous_full_text and current_full_text in previous_full_text:
                delta = ""
            else:
                delta = current_full_text
                previous_full_text = previous_full_text + current_full_text

            if delta:
                yield delta


def stream_model(config: AppConfig, messages: List[Message]) -> Iterator[str]:
    if not config.stream:
        full_text = call_model(config, messages)
        if full_text:
            yield full_text
        return

    try:
        if config.provider == "gemini":
            yield from iter_gemini_stream(config, messages)
        else:
            yield from iter_openai_stream(config, messages)
    except Exception as exc:
        print(f"\n[Warn] Streaming failed, fallback to non-stream: {exc}")
        full_text = call_model(config, messages)
        if full_text:
            yield full_text


def ensure_output_file(path: Path, config: AppConfig) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = (
        "# ShadowWrite Session\n\n"
        f"- Created: {ts}\n"
        f"- Provider: {config.provider}\n"
        f"- Model: {config.model}\n\n"
        "---\n"
    )
    path.write_text(header, encoding="utf-8")


def section_suffix(section: str) -> str:
    return f" - {section}" if section else ""


def quote_markdown(text: str) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return "> "
    return "\n".join(f"> {line}" if line else ">" for line in lines)


def append_user_turn(path: Path, text: str, section: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = (
        f"\n\n#### User Input{section_suffix(section)} ({stamp})\n\n"
        f"{quote_markdown(text)}\n"
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(block)


def append_section_marker(path: Path, section: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    marker = (
        f"\n\n---\n\n"
        f"## Section: {section}\n\n"
        f"_switched at {stamp}_\n"
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(marker)


def append_note(path: Path, note: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = f"\n\n> [Note {stamp}] {note}\n"
    with path.open("a", encoding="utf-8") as fh:
        fh.write(block)


@contextmanager
def open_assistant_block(path: Path, section: str) -> Iterator[TextIO]:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n\n### Assistant{section_suffix(section)} ({stamp})\n\n"
        )
        yield fh
        fh.write("\n")


def write_snapshot(
    output_path: Path,
    history: List[Message],
    provider: str,
    model: str,
    target_path: str = "",
) -> Path:
    now = datetime.now()
    if target_path.strip():
        snapshot_path = Path(target_path.strip())
    else:
        snapshot_path = output_path.with_name(
            f"{output_path.stem}_snapshot_{now.strftime('%Y%m%d_%H%M%S')}.md"
        )
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)

    lines: List[str] = [
        "# ShadowWrite Snapshot",
        "",
        f"- Created: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Provider: {provider}",
        f"- Model: {model}",
        "",
        "---",
        "",
    ]

    for msg in history:
        if msg["role"] == "system":
            lines.append("### System")
            lines.append("")
            lines.append(msg["content"].strip())
            lines.append("")
            continue
        if msg["role"] == "user":
            lines.append("#### User Input")
            lines.append("")
            lines.append(quote_markdown(msg["content"]))
            lines.append("")
            continue
        lines.append("### Assistant")
        lines.append("")
        lines.append(msg["content"].strip())
        lines.append("")

    snapshot_path.write_text("\n".join(lines), encoding="utf-8")
    return snapshot_path


def print_help() -> None:
    print("Commands:")
    print("  /help              Show commands")
    print("  /clear             Clear in-memory chat history (keeps file content)")
    print("  /system <text>     Set/replace system prompt for next turns")
    print("  /section <name>    Set manual section label for subsequent turns")
    print("  /note <markdown>   Append a manual note line to Markdown")
    print("  /snapshot [path]   Save current in-memory conversation snapshot")
    print("  /exit              Exit")


def apply_system_prompt(history: List[Message], prompt: str) -> None:
    if history and history[0]["role"] == "system":
        if prompt:
            history[0]["content"] = prompt
        else:
            history.pop(0)
        return
    if prompt:
        history.insert(0, {"role": "system", "content": prompt})


def run_chat(config: AppConfig) -> int:
    ensure_output_file(config.output_path, config)

    history: List[Message] = []
    if config.system_prompt:
        history.append({"role": "system", "content": config.system_prompt})

    print(f"Provider: {config.provider}")
    print(f"Model:    {config.model}")
    print(f"Output:   {config.output_path}")
    print(f"Stream:   {config.stream}")
    print("Type /help for commands.")

    current_section = config.default_section
    if current_section:
        append_section_marker(config.output_path, current_section)
        print(f"Section:  {current_section}")

    while True:
        try:
            user_text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            return 0

        if not user_text:
            continue
        if user_text in {"/exit", "/quit"}:
            print("Exiting.")
            return 0
        if user_text == "/help":
            print_help()
            continue
        if user_text == "/clear":
            history = history[:1] if history and history[0]["role"] == "system" else []
            print("In-memory history cleared.")
            continue
        if user_text.startswith("/system"):
            new_prompt = user_text[len("/system") :].strip()
            apply_system_prompt(history, new_prompt)
            if new_prompt:
                print("System prompt updated.")
            else:
                print("System prompt removed.")
            continue
        if user_text.startswith("/section"):
            new_section = user_text[len("/section") :].strip()
            current_section = new_section
            if new_section:
                append_section_marker(config.output_path, new_section)
                print(f"Section switched: {new_section}")
            else:
                print("Section cleared.")
            continue
        if user_text.startswith("/note"):
            note = user_text[len("/note") :].strip()
            if not note:
                print("Usage: /note <markdown>")
                continue
            append_note(config.output_path, note)
            print("Note appended.")
            continue
        if user_text.startswith("/snapshot"):
            target = user_text[len("/snapshot") :].strip()
            snapshot_path = write_snapshot(
                output_path=config.output_path,
                history=history,
                provider=config.provider,
                model=config.model,
                target_path=target,
            )
            print(f"Snapshot saved: {snapshot_path}")
            continue

        append_user_turn(config.output_path, user_text, current_section)
        history.append({"role": "user", "content": user_text})

        answer_parts: List[str] = []
        print("\nAI> ", end="", flush=True)
        try:
            with open_assistant_block(config.output_path, current_section) as writer:
                for chunk in stream_model(config, history):
                    if not chunk:
                        continue
                    print(chunk, end="", flush=True)
                    writer.write(chunk)
                    writer.flush()
                    answer_parts.append(chunk)
        except Exception as exc:  # noqa: BLE001
            print(f"\n[Error] {exc}")
            continue

        print("")
        answer = "".join(answer_parts).strip()
        if not answer:
            print("[Warn] Empty model response.")
            continue
        history.append({"role": "assistant", "content": answer})


def main() -> int:
    load_dotenv(Path(".env"))
    args = parse_args()
    try:
        config = resolve_config(args)
    except ValueError as exc:
        print(f"Config error: {exc}")
        return 2
    return run_chat(config)


if __name__ == "__main__":
    sys.exit(main())
