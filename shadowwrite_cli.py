#!/usr/bin/env python3
"""ShadowWrite local API chat CLI.

Features:
- Interactive terminal chat loop.
- Supports OpenAI-compatible Chat Completions APIs.
- Supports Gemini REST API (generateContent).
- Streams model output to terminal and Markdown file.
- Persists user messages as quote blocks.

python shadowwrite_cli.py
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from html import escape as html_escape
import json
import os
import re
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
    chat_html: bool
    chat_html_path: Path
    show_timestamp: bool
    user_input_mode: str


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
        "--chat-html",
        action=argparse.BooleanOptionalAction,
        default=parse_bool(os.getenv("SHADOWWRITE_CHAT_HTML"), True),
        help="Write an additional chat-style HTML transcript.",
    )
    parser.add_argument(
        "--chat-html-output",
        default=os.getenv("SHADOWWRITE_CHAT_HTML_OUTPUT", ""),
        help="Path of chat-style HTML transcript. Filename-only values are placed next to --output.",
    )
    parser.add_argument(
        "--show-time",
        action=argparse.BooleanOptionalAction,
        default=parse_bool(os.getenv("SHADOWWRITE_SHOW_TIMESTAMP"), False),
        help="Show visible timestamps in markdown/html headings.",
    )
    parser.add_argument(
        "--user-input-mode",
        default=os.getenv("SHADOWWRITE_USER_INPUT_MODE", "details"),
        choices=["details", "blockquote"],
        help="How to format user input in markdown.",
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


def resolve_chat_html_path(output_path: Path, chat_html_output: str) -> Path:
    raw = chat_html_output.strip()
    if not raw:
        return output_path.with_suffix(".chat.html")

    candidate = Path(raw)

    # If the user points to a directory, write a sibling-named html file there.
    if raw.endswith(("/", "\\")):
        if candidate.is_absolute():
            directory = candidate
        else:
            directory = output_path.parent / candidate
        return directory / f"{output_path.stem}.chat.html"

    # If the user only provides a filename, keep html next to markdown output.
    if not candidate.is_absolute() and candidate.parent == Path("."):
        return output_path.parent / candidate.name

    return candidate


def resolve_config(args: argparse.Namespace) -> AppConfig:
    provider = args.provider
    output_path = Path(args.output)

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
        output_path=output_path,
        temperature=args.temperature,
        timeout_sec=args.timeout,
        system_prompt=args.system.strip(),
        stream=bool(args.stream),
        default_section=args.section.strip(),
        chat_html=bool(args.chat_html),
        chat_html_path=resolve_chat_html_path(output_path, args.chat_html_output),
        show_timestamp=bool(args.show_time),
        user_input_mode=args.user_input_mode,
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


def quote_markdown(text: str) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return "> "
    return "\n".join(f"> {line}" if line else ">" for line in lines)


def format_turn_summary(role: str, stamp: str, section: str, show_timestamp: bool) -> str:
    sec = f" - {section}" if section else ""
    if show_timestamp:
        return f"{role}{sec} - {stamp}"
    return f"{role}{sec}"


def format_turn_anchor(turn_id: int) -> str:
    return f"sw-turn-{turn_id}"


def metadata_comment(role: str, stamp: str, section: str, turn_id: int) -> str:
    section_part = f' section="{section}"' if section else ""
    return f'<!-- sw: turn="{turn_id}" role="{role}" ts="{stamp}"{section_part} -->'


def format_turn_tooltip(role: str, stamp: str, section: str) -> str:
    normalized = role.strip().lower()
    if normalized in {"user", "user input"}:
        role_slug = "user"
    elif normalized in {"assistant", "model"}:
        role_slug = "assistant"
    elif normalized == "system":
        role_slug = "system"
    else:
        role_slug = normalized.replace(" ", "_")
    section_part = f" | section={section}" if section else ""
    return f"sw: role={role_slug} | ts={stamp}{section_part}".replace('"', "'")


def format_turn_link(
    role: str,
    stamp: str,
    section: str,
    show_timestamp: bool,
    anchor_id: str | None = None,
) -> str:
    label = format_turn_summary(role, stamp, section, show_timestamp)
    tooltip = format_turn_tooltip(role, stamp, section)
    safe_label = label.replace("[", r"\[").replace("]", r"\]")
    target = f"#{anchor_id}" if anchor_id else "#"
    return f'[{safe_label}]({target} "{tooltip}")'


def format_details_body_text(text: str) -> str:
    """Typora-friendly details body: no blank markdown lines, use <br> for newlines."""
    lines = text.strip().splitlines()
    if not lines:
        return ""
    return "<br>\n".join(html_escape(line) for line in lines)


def detect_next_turn_id(path: Path) -> int:
    if not path.exists():
        return 1
    text = path.read_text(encoding="utf-8", errors="ignore")
    hits = re.findall(r'<!--\s*sw:\s*turn="(\d+)"', text)
    if not hits:
        return 1
    return max(int(x) for x in hits) + 1


def ensure_chat_html_file(path: Path, config: AppConfig) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    title = f"ShadowWrite Chat View - {config.model}"
    template = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #f6f7fb;
    --card: #ffffff;
    --user: #eef4ff;
    --assistant: #ffffff;
    --line: #dce2ee;
    --text: #1f2937;
    --muted: #6b7280;
  }
  body {
    margin: 0;
    background: linear-gradient(120deg, #f7f9ff 0%, #f4f7f1 100%);
    color: var(--text);
    font-family: "Segoe UI", "PingFang SC", "Microsoft YaHei", sans-serif;
  }
  main {
    max-width: 980px;
    margin: 0 auto;
    padding: 20px 16px 80px;
  }
  .meta {
    margin: 0 0 16px;
    color: var(--muted);
    font-size: 13px;
  }
  .turn {
    margin: 14px 0;
    border: 1px solid var(--line);
    border-radius: 12px;
    overflow: hidden;
    background: var(--card);
  }
  .turn.user { background: var(--user); }
  .turn.assistant { background: var(--assistant); }
  .turn .head {
    padding: 10px 12px;
    font-size: 12px;
    color: var(--muted);
    border-bottom: 1px solid var(--line);
    background: rgba(255, 255, 255, 0.6);
  }
  .turn .body {
    margin: 0;
    padding: 12px;
    white-space: pre-wrap;
    line-height: 1.6;
    font-size: 14px;
  }
  .turn .body.md {
    white-space: pre-wrap;
  }
  .turn .body.rendered {
    white-space: normal;
  }
  .turn .body.rendered > :first-child {
    margin-top: 0;
  }
  .turn .body.rendered > :last-child {
    margin-bottom: 0;
  }
  .turn .body.rendered pre,
  .turn .body.rendered code {
    font-family: Consolas, "Courier New", monospace;
  }
  .turn .body.rendered pre {
    border: 1px solid var(--line);
    border-radius: 8px;
    background: #f8fafc;
    padding: 10px;
    overflow-x: auto;
  }
  details.turn.user > summary {
    cursor: pointer;
    list-style: none;
    padding: 10px 12px;
    font-size: 12px;
    color: var(--muted);
    border-bottom: 1px solid var(--line);
    background: rgba(255, 255, 255, 0.6);
  }
  details.turn.user > summary::-webkit-details-marker { display: none; }
  .section-marker {
    margin: 20px 0 8px;
    color: #4b5563;
    font-size: 13px;
  }
  .time {
    display: none;
  }
  body.show-time .time {
    display: inline;
  }
  .toolbar {
    display: flex;
    gap: 10px;
    margin: 0 0 10px;
    align-items: center;
    flex-wrap: wrap;
  }
  .toolbar button {
    border: 1px solid var(--line);
    background: #fff;
    color: #374151;
    font-size: 12px;
    border-radius: 8px;
    padding: 6px 10px;
    cursor: pointer;
  }
  .toolbar .status {
    color: var(--muted);
    font-size: 12px;
  }
  .toolbar button.follow-paused {
    border-color: #f59e0b;
    color: #92400e;
    background: #fffbeb;
  }
</style>
</head>
<body>
<main>
  <h1>__TITLE__</h1>
  <p class="meta">Created __CREATED__ - Provider __PROVIDER__ - Model __MODEL__</p>
  <div class="toolbar">
    <button id="toggle-time">Toggle Time</button>
    <button id="toggle-md">Toggle Markdown Render</button>
    <button id="toggle-follow">Following Latest</button>
    <span id="md-status" class="status"></span>
  </div>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.1.6/dist/purify.min.js"></script>
<script>
(() => {
  const body = document.body;
  const main = document.querySelector("main");
  if (!main) return;
  let mdEnabled = true;
  const byId = (id) => document.getElementById(id);
  const timeBtn = byId("toggle-time");
  const mdBtn = byId("toggle-md");
  const followBtn = byId("toggle-follow");
  const mdStatus = byId("md-status");
  let followLatest = true;
  let mutateFrame = 0;
  const followThreshold = 80;

  if (__SHOW_TIMESTAMP__) {
    body.classList.add("show-time");
  }

  const escapeHtml = (value) => value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");

  const renderInline = (text) => {
    let safe = escapeHtml(text);
    safe = safe.replace(/`([^`]+)`/g, "<code>$1</code>");
    safe = safe.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
    safe = safe.replace(/\*([^*]+)\*/g, "<em>$1</em>");
    safe = safe.replace(/\[([^\]]+)\]\((https?:\/\/[^\s)]+)\)/g, "<a href=\"$2\" target=\"_blank\" rel=\"noopener noreferrer\">$1</a>");
    return safe;
  };

  const fallbackMarkdownToHtml = (src) => {
    const lines = src.replace(/\r\n/g, "\n").split("\n");
    const out = [];
    let paragraph = [];
    let quote = [];
    let listTag = "";
    let inCode = false;
    let codeLang = "";
    let codeLines = [];

    const flushParagraph = () => {
      if (!paragraph.length) return;
      out.push("<p>" + renderInline(paragraph.join(" ")) + "</p>");
      paragraph = [];
    };
    const flushQuote = () => {
      if (!quote.length) return;
      out.push("<blockquote>" + quote.map((line) => renderInline(line)).join("<br>") + "</blockquote>");
      quote = [];
    };
    const closeList = () => {
      if (!listTag) return;
      out.push("</" + listTag + ">");
      listTag = "";
    };
    const flushCode = () => {
      const klass = codeLang ? " class=\"language-" + escapeHtml(codeLang) + "\"" : "";
      out.push("<pre><code" + klass + ">" + escapeHtml(codeLines.join("\n")) + "</code></pre>");
      codeLines = [];
      codeLang = "";
    };

    for (const rawLine of lines) {
      const line = rawLine || "";
      const fence = line.match(/^```([\w-]*)\s*$/);
      if (fence) {
        flushParagraph();
        flushQuote();
        closeList();
        if (!inCode) {
          inCode = true;
          codeLang = fence[1] || "";
          codeLines = [];
        } else {
          flushCode();
          inCode = false;
        }
        continue;
      }

      if (inCode) {
        codeLines.push(line);
        continue;
      }

      if (!line.trim()) {
        flushParagraph();
        flushQuote();
        closeList();
        continue;
      }

      const heading = line.match(/^(#{1,6})\s+(.*)$/);
      if (heading) {
        flushParagraph();
        flushQuote();
        closeList();
        const level = Math.min(6, heading[1].length);
        out.push("<h" + level + ">" + renderInline(heading[2]) + "</h" + level + ">");
        continue;
      }

      const ul = line.match(/^\s*[-*+]\s+(.*)$/);
      if (ul) {
        flushParagraph();
        flushQuote();
        if (listTag && listTag !== "ul") closeList();
        if (!listTag) {
          listTag = "ul";
          out.push("<ul>");
        }
        out.push("<li>" + renderInline(ul[1]) + "</li>");
        continue;
      }

      const ol = line.match(/^\s*\d+\.\s+(.*)$/);
      if (ol) {
        flushParagraph();
        flushQuote();
        if (listTag && listTag !== "ol") closeList();
        if (!listTag) {
          listTag = "ol";
          out.push("<ol>");
        }
        out.push("<li>" + renderInline(ol[1]) + "</li>");
        continue;
      }

      const quoteLine = line.match(/^>\s?(.*)$/);
      if (quoteLine) {
        flushParagraph();
        closeList();
        quote.push(quoteLine[1]);
        continue;
      }

      flushQuote();
      closeList();
      paragraph.push(line.trim());
    }

    if (inCode) flushCode();
    flushParagraph();
    flushQuote();
    closeList();

    return out.join("\n");
  };

  const canUseMarked = () => Boolean(window.marked && window.DOMPurify);
  const markdownToHtml = (src) => {
    if (canUseMarked()) {
      const raw = window.marked.parse(src, { headerIds: false, mangle: false });
      return window.DOMPurify.sanitize(raw);
    }
    return fallbackMarkdownToHtml(src);
  };

  const updateStatus = () => {
    if (!mdStatus) return;
    if (!mdEnabled) {
      mdStatus.textContent = "Markdown render: off";
      return;
    }
    mdStatus.textContent = canUseMarked()
      ? "Markdown render: full (marked)"
      : "Markdown render: fallback";
  };

  const latestTurn = () => {
    const turns = main.querySelectorAll(".turn, details.turn");
    return turns.length ? turns[turns.length - 1] : null;
  };

  const isNearBottom = () => {
    const root = document.documentElement;
    const remain = root.scrollHeight - window.innerHeight - window.scrollY;
    return remain <= followThreshold;
  };

  const updateFollowButton = () => {
    if (!followBtn) return;
    if (followLatest) {
      followBtn.textContent = "Following Latest";
      followBtn.classList.remove("follow-paused");
    } else {
      followBtn.textContent = "Resume Follow";
      followBtn.classList.add("follow-paused");
    }
  };

  const scrollToLatest = (force = false) => {
    if (!force && !followLatest) return;
    const node = latestTurn();
    if (!node) return;
    node.scrollIntoView({ block: "end", behavior: "auto" });
  };

  const renderMarkdownBlocks = () => {
    if (!mdEnabled) return;
    main.querySelectorAll("pre.body.md").forEach((pre) => {
      if (pre.dataset.rendered === "1") return;
      const src = pre.textContent || "";
      const div = document.createElement("div");
      div.className = "body rendered";
      div.innerHTML = markdownToHtml(src);
      pre.dataset.rendered = "1";
      pre.replaceWith(div);
    });
    updateStatus();
  };

  if (timeBtn) {
    timeBtn.addEventListener("click", () => {
      body.classList.toggle("show-time");
    });
  }
  if (mdBtn) {
    mdBtn.addEventListener("click", () => {
      mdEnabled = !mdEnabled;
      if (mdEnabled) {
        renderMarkdownBlocks();
      } else {
        updateStatus();
        window.location.reload();
      }
    });
  }

  if (followBtn) {
    followBtn.addEventListener("click", () => {
      followLatest = true;
      updateFollowButton();
      scrollToLatest(true);
    });
  }

  window.addEventListener(
    "scroll",
    () => {
      const nearBottom = isNearBottom();
      if (nearBottom !== followLatest) {
        followLatest = nearBottom;
        updateFollowButton();
      }
    },
    { passive: true }
  );

  const observer = new MutationObserver(() => {
    if (mutateFrame) return;
    mutateFrame = window.requestAnimationFrame(() => {
      mutateFrame = 0;
      renderMarkdownBlocks();
      scrollToLatest();
    });
  });
  observer.observe(main, { childList: true, subtree: true });
  updateFollowButton();
  updateStatus();
  renderMarkdownBlocks();
  scrollToLatest(true);
})();
</script>
"""
    content = (
        template
        .replace("__TITLE__", html_escape(title))
        .replace("__CREATED__", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        .replace("__PROVIDER__", html_escape(config.provider))
        .replace("__MODEL__", html_escape(config.model))
        .replace("__SHOW_TIMESTAMP__", str(config.show_timestamp).lower())
    )
    path.write_text(content, encoding="utf-8")


def append_chat_section_marker(path: Path, section: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fragment = (
        f'\n<p class="section-marker">Section switched to <strong>{html_escape(section)}</strong> '
        f'<span class="time">at {html_escape(stamp)}</span></p>\n'
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(fragment)


def append_chat_note(path: Path, note: str) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    fragment = (
        '\n<div class="turn user">'
        f'<div class="head">Note <span class="time"> - {html_escape(stamp)}</span></div>'
        f'<pre class="body">{html_escape(note)}</pre>'
        '</div>\n'
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(fragment)


def append_user_turn(
    path: Path,
    text: str,
    section: str,
    show_timestamp: bool,
    mode: str,
    turn_id: int,
) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    anchor_id = format_turn_anchor(turn_id)
    summary = format_turn_summary("User Input", stamp, section, show_timestamp)
    summary_link = format_turn_link(
        "User Input", stamp, section, show_timestamp, anchor_id=anchor_id
    )
    tooltip = format_turn_tooltip("User Input", stamp, section)
    comment = metadata_comment("user", stamp, section, turn_id)
    anchor = f'<a id="{anchor_id}"></a>'

    if mode == "blockquote":
        block = (
            f"\n\n{comment}\n"
            f"{anchor}\n"
            f"{summary_link}\n\n"
            f"{quote_markdown(text)}\n"
        )
    else:
        details_body = format_details_body_text(text)
        block = (
            f"\n\n{comment}\n"
            f"{anchor}\n"
            "<details class=\"sw-user-turn\"><summary>"
            f"<strong><a href=\"#{anchor_id}\" title=\"{html_escape(tooltip)}\">{html_escape(summary)}</a></strong>"
            f"</summary>{details_body}</details>\n"
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
def open_assistant_block(
    path: Path,
    section: str,
    show_timestamp: bool,
    turn_id: int,
) -> Iterator[TextIO]:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    anchor_id = format_turn_anchor(turn_id)
    summary_link = format_turn_link(
        "Assistant", stamp, section, show_timestamp, anchor_id=anchor_id
    )
    comment = metadata_comment("assistant", stamp, section, turn_id)
    anchor = f'<a id="{anchor_id}"></a>'
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            f"\n\n{comment}\n"
            f"{anchor}\n"
            f"{summary_link}\n\n"
        )
        yield fh
        fh.write("\n")


@contextmanager
def open_assistant_chat_block(path: Path, section: str, turn_id: int) -> Iterator[TextIO]:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = format_turn_summary("Assistant", stamp, section, False)
    anchor_id = format_turn_anchor(turn_id)
    time_html = f'<span class="time"> - {html_escape(stamp)}</span>'
    with path.open("a", encoding="utf-8") as fh:
        fh.write(
            f'\n<div class="turn assistant" id="{html_escape(anchor_id)}">'
            f'<div class="head">{html_escape(summary)}{time_html}</div>'
            '<pre class="body md">'
        )
        yield fh
        fh.write("</pre></div>\n")


def append_chat_user_turn(path: Path, text: str, section: str, turn_id: int) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    summary = format_turn_summary("User Input", stamp, section, False)
    anchor_id = format_turn_anchor(turn_id)
    time_html = f'<span class="time"> - {html_escape(stamp)}</span>'
    body = html_escape(text)
    fragment = (
        f'\n<details class="turn user" id="{html_escape(anchor_id)}">'
        f"<summary>{html_escape(summary)}{time_html}</summary>"
        f'<pre class="body">{body}</pre>'
        "</details>\n"
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(fragment)


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


def print_help(startup: bool = False) -> None:
    if startup:
        print("")
        print("Quick Start:")
        print("  - Type normal text, press Enter -> send to model.")
        print("  - Use commands below at any time:")
    else:
        print("Commands:")
    print("  /help              Show commands")
    print("  /clear             Clear in-memory chat history (keeps file content)")
    print("  /system <text>     Set/replace system prompt for next turns")
    print("  /section <name>    Set section tag (example: /section Plot)")
    print("  /section           Clear current section tag")
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
    if config.chat_html:
        ensure_chat_html_file(config.chat_html_path, config)
    next_turn_id = detect_next_turn_id(config.output_path)

    history: List[Message] = []
    if config.system_prompt:
        history.append({"role": "system", "content": config.system_prompt})

    print(f"Provider: {config.provider}")
    print(f"Model:    {config.model}")
    print(f"Output:   {config.output_path}")
    print(f"Stream:   {config.stream}")
    if config.chat_html:
        print(f"ChatView: {config.chat_html_path}")
    print_help(startup=True)

    current_section = config.default_section
    if current_section:
        append_section_marker(config.output_path, current_section)
        if config.chat_html:
            append_chat_section_marker(config.chat_html_path, current_section)
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
                if config.chat_html:
                    append_chat_section_marker(config.chat_html_path, new_section)
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
            if config.chat_html:
                append_chat_note(config.chat_html_path, note)
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

        user_turn_id = next_turn_id
        next_turn_id += 1
        append_user_turn(
            config.output_path,
            user_text,
            current_section,
            config.show_timestamp,
            config.user_input_mode,
            user_turn_id,
        )
        if config.chat_html:
            append_chat_user_turn(
                config.chat_html_path,
                user_text,
                current_section,
                user_turn_id,
            )
        history.append({"role": "user", "content": user_text})

        answer_parts: List[str] = []
        print("\nAI> ", end="", flush=True)
        try:
            assistant_turn_id = next_turn_id
            next_turn_id += 1
            html_ctx = (
                open_assistant_chat_block(
                    config.chat_html_path,
                    current_section,
                    assistant_turn_id,
                )
                if config.chat_html
                else nullcontext(None)
            )
            with open_assistant_block(
                config.output_path,
                current_section,
                config.show_timestamp,
                assistant_turn_id,
            ) as writer, html_ctx as html_writer:
                for chunk in stream_model(config, history):
                    if not chunk:
                        continue
                    print(chunk, end="", flush=True)
                    writer.write(chunk)
                    writer.flush()
                    if html_writer:
                        html_writer.write(html_escape(chunk))
                        html_writer.flush()
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


