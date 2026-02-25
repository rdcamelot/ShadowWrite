#!/usr/bin/env python3
"""ShadowWrite Local HTTP Server.

Receives real-time chat messages from the ShadowWrite Chrome extension
via HTTP POST and persists them to local Markdown files.

Usage:
    python shadowwrite_server.py                        # default port 24601
    python shadowwrite_server.py --port 24602           # custom port
    python shadowwrite_server.py --output-dir ./chats   # custom output dir

Environment variables (or .env file in CWD / script directory):
    SHADOWWRITE_SERVER_PORT      Port to listen on (default: 24601)
    SHADOWWRITE_SERVER_HOST      Host to bind (default: 127.0.0.1)
    SHADOWWRITE_OUTPUT_DIR       Output directory (default: ./outputs)
    SHADOWWRITE_SERVER_CHAT_HTML  Whether to generate .chat.html files (default: true)

Requires: Python 3.10+, stdlib only

python shadowwrite_server.py
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import sys
import threading
from datetime import datetime
from html import escape as html_escape
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Any, Dict, List
from urllib.parse import parse_qs, urlparse

# Pre-import tkinter so the browse-directory dialog opens faster.
# If tkinter is unavailable (rare), we simply skip — the browse
# endpoint will return an error at call time.
try:
    import tkinter as tk
    from tkinter import filedialog as _tk_filedialog

    # Windows high-DPI fix — makes the dialog render at native resolution
    # instead of being blurry due to DPI virtualisation.
    if sys.platform == "win32":
        try:
            import ctypes
            ctypes.windll.shcore.SetProcessDpiAwareness(1)   # PROCESS_SYSTEM_DPI_AWARE
        except Exception:
            pass  # older Windows or no shcore
except ImportError:
    tk = None  # type: ignore
    _tk_filedialog = None  # type: ignore


# ---------------------------------------------------------------------------
# .env loader (same as shadowwrite_cli.py)
# ---------------------------------------------------------------------------

def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in ('"', "'"):
                value = value[1:-1]
            if key and key not in os.environ:
                os.environ[key] = value


# ---------------------------------------------------------------------------
# Conversation state — in-memory per-session tracking
# ---------------------------------------------------------------------------

class ConversationState:
    """Track per-conversation state for idempotent writes."""

    def __init__(self):
        self._lock = threading.Lock()
        # { conversationId: { messageId: {content, thinking, md_offset, html_offset, turn_id} } }
        self._written: Dict[str, Dict[str, Dict[str, Any]]] = {}
        # { conversationId: next_turn_id }
        self._turn_ids: Dict[str, int] = {}
        # { conversationId: { platform, title, output_path, html_path } }
        self._meta: Dict[str, Dict[str, Any]] = {}

    def get_written(self, conversation_id: str, message_id: str) -> Dict[str, Any] | None:
        """Return stored write-record or None if not yet written."""
        with self._lock:
            return self._written.get(conversation_id, {}).get(message_id)

    def is_written(self, conversation_id: str, message_id: str) -> bool:
        with self._lock:
            return message_id in self._written.get(conversation_id, {})

    def mark_written(
        self,
        conversation_id: str,
        message_id: str,
        content: str,
        thinking: str,
        md_offset: int,
        html_offset: int,
        turn_id: int,
    ) -> None:
        with self._lock:
            self._written.setdefault(conversation_id, {})[message_id] = {
                "content": content,
                "thinking": thinking,
                "md_offset": md_offset,
                "html_offset": html_offset,
                "turn_id": turn_id,
            }

    def next_turn_id(self, conversation_id: str) -> int:
        with self._lock:
            tid = self._turn_ids.get(conversation_id, 1)
            self._turn_ids[conversation_id] = tid + 1
            return tid

    def get_meta(self, conversation_id: str) -> Dict[str, Any] | None:
        with self._lock:
            return self._meta.get(conversation_id)

    def set_meta(self, conversation_id: str, meta: Dict[str, Any]) -> None:
        with self._lock:
            self._meta[conversation_id] = meta

    def clear_conversation(self, conversation_id: str) -> None:
        """Reset written-IDs and turn counter for a conversation (keep meta)."""
        with self._lock:
            self._written.pop(conversation_id, None)
            self._turn_ids.pop(conversation_id, None)


# ---------------------------------------------------------------------------
# Markdown writer functions (adapted from shadowwrite_cli.py)
# ---------------------------------------------------------------------------

def sanitize_filename(name: str) -> str:
    """Remove characters unsafe for file names."""
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip()[:120]


def ensure_output_file(path: Path, platform: str, title: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header = (
        f"# {title or 'ShadowWrite Capture'}\n\n"
        f"- Created: {ts}\n"
        f"- Platform: {platform}\n"
        f"- Source: Chrome Extension (Route 3)\n\n"
        "---\n"
    )
    path.write_text(header, encoding="utf-8")


def ensure_html_file(path: Path, title: str) -> None:
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    html = (
        "<!DOCTYPE html>\n<html><head>\n"
        '<meta charset="UTF-8">\n'
        f"<title>{html_escape(title or 'ShadowWrite Capture')}</title>\n"
        "<style>\n"
        "body{font-family:sans-serif;max-width:800px;margin:2em auto;padding:0 1em;"
        "background:#1a1a2e;color:#e0e0e0}\n"
        ".turn{margin:1em 0;padding:0.75em 1em;border-radius:8px}\n"
        ".user{background:#2a2a3e}\n"
        ".assistant{background:#1e3a2e}\n"
        ".head{font-weight:bold;margin-bottom:0.5em}\n"
        ".time{color:#888;font-size:0.85em}\n"
        "pre.body{white-space:pre-wrap;font-family:inherit}\n"
        "details summary{cursor:pointer}\n"
        "hr{border:none;border-top:1px solid #333;margin:1.5em 0}\n"
        "</style>\n"
        "</head><body>\n"
        f"<h1>{html_escape(title or 'ShadowWrite Capture')}</h1>\n"
        f"<p>Created: {ts}</p>\n"
    )
    path.write_text(html, encoding="utf-8")


def quote_markdown(text: str) -> str:
    lines = text.strip().splitlines()
    if not lines:
        return "> "
    return "\n".join(f"> {line}" if line else ">" for line in lines)


def truncate_file(path: Path, offset: int) -> None:
    """Truncate *path* back to *offset* bytes (for streaming upsert)."""
    if not path.exists():
        return
    with path.open("r+b") as fh:
        fh.truncate(offset)


def file_end(path: Path) -> int:
    """Return current EOF byte offset, 0 if file does not exist."""
    try:
        return path.stat().st_size
    except FileNotFoundError:
        return 0


def format_details_body(text: str) -> str:
    """Typora-friendly details body: escape HTML, use <br> for newlines."""
    lines = text.strip().splitlines()
    if not lines:
        return ""
    return "<br>\n".join(html_escape(line) for line in lines)


def append_user_turn(path: Path, content: str, turn_id: int) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = (
        f'<!-- sw: turn="{turn_id}" role="user" ts="{stamp}" -->\n'
        f'<a id="sw-turn-{turn_id}"></a>\n'
    )
    anchor_href = f"#sw-turn-{turn_id}"
    tooltip = f"sw: role=user | ts={stamp}"
    summary = "User Input"
    details_body = format_details_body(content)
    block = (
        "\n\n---\n\n"
        f"{meta}"
        '<details class="sw-user-turn"><summary>'
        f'<strong><a href="{anchor_href}" title="{html_escape(tooltip)}">{html_escape(summary)}</a></strong>'
        f'</summary>{details_body}</details>\n'
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(block)


def append_assistant_turn(path: Path, content: str, thinking: str, turn_id: int) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    meta = (
        f'<!-- sw: turn="{turn_id}" role="assistant" ts="{stamp}" -->\n'
        f'<a id="sw-turn-{turn_id}"></a>\n'
    )

    parts = [
        "\n\n",
        meta,
        f'[Assistant](#sw-turn-{turn_id} '
        f'"sw: role=assistant | ts={stamp}")\n\n',
    ]

    if thinking:
        parts.append("<details><summary>Thinking</summary>\n\n")
        parts.append(thinking.strip())
        parts.append("\n\n</details>\n\n")

    parts.append(content.strip())
    parts.append("\n")

    with path.open("a", encoding="utf-8") as fh:
        fh.write("".join(parts))


def append_html_user_turn(path: Path, content: str, turn_id: int) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    anchor_id = f"sw-turn-{turn_id}"
    time_html = f'<span class="time"> - {html_escape(stamp)}</span>'
    body = html_escape(content)
    fragment = (
        f'\n<hr>\n<details class="turn user" id="{html_escape(anchor_id)}">'
        f"<summary>User Input{time_html}</summary>"
        f'<pre class="body">{body}</pre>'
        "</details>\n"
    )
    with path.open("a", encoding="utf-8") as fh:
        fh.write(fragment)


def append_html_assistant_turn(path: Path, content: str, thinking: str, turn_id: int) -> None:
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    anchor_id = f"sw-turn-{turn_id}"
    time_html = f'<span class="time"> - {html_escape(stamp)}</span>'

    parts = [
        f'\n<div class="turn assistant" id="{html_escape(anchor_id)}">',
        f'<div class="head">Assistant{time_html}</div>',
    ]

    if thinking:
        parts.append(
            '<details><summary>Thinking</summary>'
            f'<pre class="body">{html_escape(thinking)}</pre></details>'
        )

    parts.append(f'<pre class="body md">{html_escape(content)}</pre></div>\n')

    with path.open("a", encoding="utf-8") as fh:
        fh.write("".join(parts))


# ---------------------------------------------------------------------------
# HTTP Request Handler
# ---------------------------------------------------------------------------

class ShadowWriteHandler(BaseHTTPRequestHandler):
    """Handle incoming requests from the Chrome extension."""

    state: ConversationState
    output_dir: Path

    def log_message(self, format, *args):
        ts = datetime.now().strftime("%H:%M:%S")
        sys.stderr.write(f"[{ts}] {format % args}\n")

    # ── CORS ──────────────────────────────────────────────────────
    def _set_cors_headers(self):
        origin = self.headers.get("Origin", "")
        # Allow any localhost origin or known AI platform origins
        allowed_patterns = [
            "chrome-extension://",
            "http://127.0.0.1",
            "http://localhost",
        ]
        if any(origin.startswith(p) for p in allowed_patterns):
            self.send_header("Access-Control-Allow-Origin", origin)
        else:
            # Also allow known AI platforms for direct fetch
            self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
        self.send_header("Access-Control-Max-Age", "86400")

    def _send_json(self, status: int, data: Any) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self._set_cors_headers()
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ── OPTIONS (CORS preflight) ──────────────────────────────────
    def do_OPTIONS(self):
        self.send_response(204)
        self._set_cors_headers()
        self.end_headers()

    # ── GET endpoints ─────────────────────────────────────────────
    def do_GET(self):
        path = urlparse(self.path).path

        if path == "/api/health":
            self._send_json(200, {
                "status": "ok",
                "service": "ShadowWrite",
                "version": "0.1.0",
            })
            return

        if path == "/api/conversations":
            convs = []
            if self.state:
                with self.state._lock:
                    for cid, meta in self.state._meta.items():
                        convs.append({
                            "conversationId": cid,
                            "platform": meta.get("platform", ""),
                            "title": meta.get("title", ""),
                            "messageCount": len(self.state._written.get(cid, {})),
                        })
            self._send_json(200, {"conversations": convs})
            return

        if path == "/api/config":
            query = parse_qs(urlparse(self.path).query)
            conv_id = query.get("conversationId", [None])[0]
            response = {
                "outputDir": str(self.output_dir),
                "globalDir": str(self.output_dir),
                "chatHtml": self.chat_html,
                "isCustom": False,
            }
            if conv_id:
                meta = self.state.get_meta(conv_id)
                if meta:
                    conv_base = meta.get("base_dir", str(self.output_dir))
                    response["outputDir"] = conv_base
                    response["isCustom"] = conv_base != str(self.output_dir)
            self._send_json(200, response)
            return

        self._send_json(404, {"error": "Not found"})

    # ── POST endpoints ────────────────────────────────────────────
    def do_POST(self):
        path = urlparse(self.path).path

        if path == "/api/messages":
            self._handle_messages()
            return

        if path == "/api/config":
            self._handle_config_update()
            return

        if path == "/api/browse-directory":
            self._handle_browse_directory()
            return

        self._send_json(404, {"error": "Not found"})

    def _handle_browse_directory(self):
        """Open a native directory-picker dialog and return the selected path."""
        if tk is None:
            self._send_json(500, {"error": "tkinter not available in this Python installation"})
            return

        # Read optional initialdir from request body
        initial_dir = str(self.output_dir)
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length > 0:
            try:
                raw = self.rfile.read(content_length)
                payload = json.loads(raw.decode("utf-8"))
                if "initialDir" in payload:
                    candidate = Path(payload["initialDir"]).resolve()
                    if candidate.is_dir():
                        initial_dir = str(candidate)
            except Exception:
                pass  # fall back to output_dir

        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            folder = _tk_filedialog.askdirectory(
                title="ShadowWrite — 选择输出目录",
                initialdir=initial_dir,
            )
            root.destroy()

            if folder:
                self._send_json(200, {"selected": folder})
            else:
                self._send_json(200, {"selected": None, "cancelled": True})
        except Exception as e:
            self._send_json(500, {"error": f"Failed to open dialog: {e}"})

    def _rename_conversation_files(
        self,
        md_path: Path,
        html_path: Path,
        new_display: str,
        platform: str,
        base_dir: Path | None = None,
    ) -> tuple[Path | None, Path | None]:
        """Rename conversation files + directory when title changes.

        Returns (new_md_path, new_html_path) on success, (None, None) on failure.
        """
        try:
            old_dir = md_path.parent
            dir_name = sanitize_filename(platform)
            new_dir = (base_dir or self.output_dir) / dir_name / sanitize_filename(new_display)

            if new_dir.exists() and new_dir != old_dir:
                self.log_message("  [rename] Target dir already exists, skipping: %s", new_dir)
                return (None, None)

            new_md   = new_dir / f"{new_display}.md"
            new_html = new_dir / f"{new_display}.chat.html"

            if old_dir == new_dir:
                if md_path.exists():
                    md_path.rename(new_md)
                if html_path.exists():
                    html_path.rename(new_html)
            else:
                new_dir.parent.mkdir(parents=True, exist_ok=True)
                old_dir.rename(new_dir)
                old_md_in_new   = new_dir / md_path.name
                old_html_in_new = new_dir / html_path.name
                if old_md_in_new.exists() and old_md_in_new != new_md:
                    old_md_in_new.rename(new_md)
                if old_html_in_new.exists() and old_html_in_new != new_html:
                    old_html_in_new.rename(new_html)

            self.log_message("  [rename] %s -> %s", old_dir.name, new_dir.name)
            return (new_md, new_html)
        except Exception as e:
            self.log_message("  [rename] Failed: %s", e)
            return (None, None)

    def _move_conversation_files(
        self,
        md_path: Path,
        html_path: Path,
        new_base_dir: Path,
        platform: str,
    ) -> tuple[Path | None, Path | None]:
        """Move conversation files to a different base output directory.

        Returns (new_md_path, new_html_path) on success, (None, None) on failure.
        """
        try:
            old_dir = md_path.parent
            dir_name = sanitize_filename(platform)
            display_name = md_path.stem
            new_dir = new_base_dir / dir_name / sanitize_filename(display_name)

            if new_dir == old_dir:
                return (md_path, html_path)

            if new_dir.exists():
                self.log_message("  [move] Target dir already exists, skipping: %s", new_dir)
                return (None, None)

            new_dir.parent.mkdir(parents=True, exist_ok=True)
            new_md = new_dir / md_path.name
            new_html = new_dir / html_path.name

            shutil.move(str(old_dir), str(new_dir))

            self.log_message("  [move] %s \u2192 %s", old_dir, new_dir)
            return (new_md, new_html)
        except Exception as e:
            self.log_message("  [move] Failed: %s", e)
            return (None, None)

    def _handle_config_update(self):
        """Update server configuration at runtime."""
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json(400, {"error": "Empty body"})
            return
        try:
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._send_json(400, {"error": f"Invalid JSON: {e}"})
            return

        conversation_id = payload.get("conversationId")
        if conversation_id:
            self._handle_conversation_config(conversation_id, payload)
            return

        # Global config update
        changed = []

        if "outputDir" in payload:
            new_dir = Path(payload["outputDir"]).resolve()
            new_dir.mkdir(parents=True, exist_ok=True)
            type(self).output_dir = new_dir
            changed.append("outputDir")

        if "chatHtml" in payload:
            type(self).chat_html = bool(payload["chatHtml"])
            changed.append("chatHtml")

        self.log_message("  [config] updated: %s", ", ".join(changed) or "(none)")
        self._send_json(200, {
            "status": "ok",
            "changed": changed,
            "outputDir": str(self.output_dir),
            "globalDir": str(self.output_dir),
            "chatHtml": self.chat_html,
        })

    def _handle_conversation_config(self, conversation_id: str, payload: dict):
        """Update per-conversation configuration (output directory)."""
        meta = self.state.get_meta(conversation_id)

        if "outputDir" not in payload:
            effective_dir = meta.get("base_dir", str(self.output_dir)) if meta else str(self.output_dir)
        elif payload["outputDir"]:
            # Set custom directory
            new_base = Path(payload["outputDir"]).resolve()
            new_base.mkdir(parents=True, exist_ok=True)
            effective_dir = str(new_base)
            if meta:
                old_base = meta.get("base_dir", str(self.output_dir))
                if str(new_base) != old_base:
                    md_path = Path(meta["md_path"])
                    html_path = Path(meta["html_path"])
                    if md_path.exists():
                        new_md, new_html = self._move_conversation_files(
                            md_path, html_path, new_base, meta["platform"],
                        )
                        if new_md:
                            meta["md_path"] = str(new_md)
                            meta["html_path"] = str(new_html)
                    meta["base_dir"] = str(new_base)
                    self.state.set_meta(conversation_id, meta)
        else:
            # Reset to global (outputDir is null/empty)
            effective_dir = str(self.output_dir)
            if meta:
                old_base = meta.get("base_dir", str(self.output_dir))
                if old_base != str(self.output_dir):
                    md_path = Path(meta["md_path"])
                    html_path = Path(meta["html_path"])
                    if md_path.exists():
                        new_md, new_html = self._move_conversation_files(
                            md_path, html_path, self.output_dir, meta["platform"],
                        )
                        if new_md:
                            meta["md_path"] = str(new_md)
                            meta["html_path"] = str(new_html)
                    meta["base_dir"] = str(self.output_dir)
                    self.state.set_meta(conversation_id, meta)

        self.log_message("  [config] conv %s \u2192 outputDir: %s", conversation_id[:16], effective_dir)
        self._send_json(200, {
            "status": "ok",
            "conversationId": conversation_id,
            "outputDir": effective_dir,
            "globalDir": str(self.output_dir),
            "isCustom": effective_dir != str(self.output_dir),
            "chatHtml": self.chat_html,
        })

    def _handle_messages(self):
        # Parse request body
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            self._send_json(400, {"error": "Empty body"})
            return

        try:
            raw = self.rfile.read(content_length)
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self._send_json(400, {"error": f"Invalid JSON: {e}"})
            return

        # Validate required fields
        platform = payload.get("platform", "")
        conversation_id = payload.get("conversationId", "")
        messages = payload.get("messages", [])
        title = payload.get("title", "")
        url = payload.get("url", "")
        custom_output_dir = payload.get("outputDir")  # per-conversation override

        if not platform or not conversation_id:
            self._send_json(400, {
                "error": "Missing required fields: platform, conversationId"
            })
            return

        if not isinstance(messages, list) or not messages:
            self._send_json(400, {"error": "messages must be a non-empty array"})
            return

        # Determine base output directory (per-conversation override or global)
        if custom_output_dir:
            base_dir = Path(custom_output_dir).resolve()
            base_dir.mkdir(parents=True, exist_ok=True)
        else:
            base_dir = self.output_dir

        # Resolve output paths
        meta = self.state.get_meta(conversation_id)
        if meta is None:
            # Prefer conversation title for file naming, fall back to platform_id
            if title and title.strip():
                display_name = sanitize_filename(title.strip())
            else:
                display_name = sanitize_filename(f"{platform}_{conversation_id}")
            # Directory: {base_dir}/{platform}/{display_name}/
            dir_name = sanitize_filename(platform)
            sub_name = sanitize_filename(display_name)
            md_path = base_dir / dir_name / sub_name / f"{display_name}.md"
            html_path = base_dir / dir_name / sub_name / f"{display_name}.chat.html"

            ensure_output_file(md_path, platform, title)
            if self.chat_html:
                ensure_html_file(html_path, title)

            meta = {
                "platform": platform,
                "title": title,
                "url": url,
                "md_path": str(md_path),
                "html_path": str(html_path),
                "base_dir": str(base_dir),
            }
            self.state.set_meta(conversation_id, meta)
        else:
            md_path = Path(meta["md_path"])
            html_path = Path(meta["html_path"])

            # Move files if per-conversation output directory changed
            old_base = meta.get("base_dir", str(self.output_dir))
            if str(base_dir) != old_base:
                if md_path.exists():
                    new_md, new_html = self._move_conversation_files(
                        md_path, html_path, base_dir, platform,
                    )
                    if new_md:
                        md_path = new_md
                        html_path = new_html
                        meta["md_path"] = str(md_path)
                        meta["html_path"] = str(html_path)
                meta["base_dir"] = str(base_dir)

            # Rename files / directory when title changes
            if title and title.strip() and title != meta.get("title"):
                old_title = meta.get("title", "")
                meta["title"] = title
                new_display = sanitize_filename(title.strip())
                old_display = sanitize_filename(old_title.strip()) if old_title and old_title.strip() else None
                current_display = md_path.stem

                if new_display and new_display != current_display:
                    new_md, new_html = self._rename_conversation_files(
                        md_path, html_path, new_display, platform, base_dir,
                    )
                    if new_md:
                        md_path = new_md
                        html_path = new_html
                        meta["md_path"] = str(md_path)
                        meta["html_path"] = str(html_path)

                self.state.set_meta(conversation_id, meta)

            # If output files were deleted externally, reset state so
            # messages are re-written from scratch on next snapshot.
            if not md_path.exists():
                self.log_message(
                    "  [!] Output file missing for %s — resetting state",
                    conversation_id[:16],
                )
                self.state.clear_conversation(conversation_id)
                ensure_output_file(md_path, platform, title)
                if self.chat_html:
                    ensure_html_file(html_path, title)

        # Process messages — only write new ones; upsert on streaming updates
        written_count = 0
        skipped_count = 0
        updated_count = 0

        for msg in messages:
            msg_id = msg.get("messageId", "")
            if not msg_id:
                continue

            sender  = msg.get("sender", "")
            content = msg.get("content", "")
            thinking = msg.get("thinking", "")

            if not content:
                continue

            prev = self.state.get_written(conversation_id, msg_id)

            if prev is not None:
                # Already written — check whether content changed (streaming update)
                if prev["content"] == content and prev["thinking"] == thinking:
                    skipped_count += 1
                    continue
                # Content changed: truncate files back to the saved offset and rewrite
                turn_id = prev["turn_id"]
                truncate_file(md_path, prev["md_offset"])
                if self.chat_html:
                    truncate_file(html_path, prev["html_offset"])
                updated_count += 1
            else:
                turn_id = self.state.next_turn_id(conversation_id)
                written_count += 1

            md_off   = file_end(md_path)
            html_off = file_end(html_path) if self.chat_html else 0

            if sender.lower() in ("user",):
                append_user_turn(md_path, content, turn_id)
                if self.chat_html:
                    append_html_user_turn(html_path, content, turn_id)
            else:
                append_assistant_turn(md_path, content, thinking, turn_id)
                if self.chat_html:
                    append_html_assistant_turn(html_path, content, thinking, turn_id)

            self.state.mark_written(
                conversation_id, msg_id, content, thinking,
                md_off, html_off, turn_id,
            )
            self.log_message(
                "  %s turn %d %s: %s (%d chars)",
                sender, turn_id,
                "(update)" if prev is not None else "",
                conversation_id[:16], len(content),
            )

        self._send_json(200, {
            "status": "ok",
            "written": written_count,
            "updated": updated_count,
            "skipped": skipped_count,
            "conversationId": conversation_id,
        })


# ---------------------------------------------------------------------------
# Server setup
# ---------------------------------------------------------------------------

def create_handler_class(state: ConversationState, output_dir: Path, chat_html: bool = True):
    """Create handler class with shared state."""

    class Handler(ShadowWriteHandler):
        pass

    Handler.state = state
    Handler.output_dir = output_dir
    Handler.chat_html = chat_html
    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ShadowWrite local HTTP service for Chrome extension",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("SHADOWWRITE_SERVER_HOST", "127.0.0.1"),
        help="Host to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.environ.get("SHADOWWRITE_SERVER_PORT", "24601")),
        help="Port to listen on (default: 24601)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(os.environ.get("SHADOWWRITE_OUTPUT_DIR", "./outputs")),
        help="Directory for output files (default: ./outputs)",
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        default=os.environ.get("SHADOWWRITE_SERVER_CHAT_HTML", "true").lower() == "false",
        help="Disable .chat.html file generation",
    )
    return parser.parse_args()


def main() -> int:
    # Load .env
    for env_path in [Path.cwd() / ".env", Path(__file__).resolve().parent / ".env"]:
        load_dotenv(env_path)

    args = parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    state = ConversationState()
    chat_html = not args.no_html
    handler_class = create_handler_class(state, output_dir, chat_html)

    server = HTTPServer((args.host, args.port), handler_class)

    # Graceful shutdown — must call server.shutdown() from a *different* thread.
    # Calling it directly in the signal handler (main thread) deadlocks because
    # serve_forever() is also running in the main thread and can't advance.
    def shutdown_handler(sig, frame):
        print("\n[ShadowWrite] Shutting down...")
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown_handler)
    signal.signal(signal.SIGTERM, shutdown_handler)

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ShadowWrite Local Server                                    ║
║                                                              ║
║  Listening:   http://{args.host}:{args.port}               ║
║  Output dir:  {str(output_dir):<45s}║
║  Chat HTML:   {'ON' if chat_html else 'OFF':<45s}║
║  Health:      GET  /api/health                               ║
║  Messages:    POST /api/messages                             ║
║                                                              ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
""")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        print("[ShadowWrite] Server stopped.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
