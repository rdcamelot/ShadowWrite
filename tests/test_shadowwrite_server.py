import json
import os
import tempfile
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.request import Request, urlopen

from shadowwrite_server import (
    ConversationState,
    create_handler_class,
    prepare_extension_release,
)


class ShadowWriteServerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.state_path = Path(self.tmp.name) / ".shadowwrite-state.json"
        self.start_server(ConversationState(self.state_path))

    def start_server(self, state):
        self.handler_class = create_handler_class(state, Path(self.tmp.name), chat_html=True)
        self.server = HTTPServer(("127.0.0.1", 0), self.handler_class)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        host, port = self.server.server_address
        self.base_url = f"http://{host}:{port}"

    def restart_server(self, drop_state=False):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        if drop_state:
            self.state_path.unlink(missing_ok=True)
        self.start_server(ConversationState(self.state_path))

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.tmp.cleanup()

    def post_messages(self, messages):
        payload = {
            "platform": "test",
            "conversationId": "conv-1",
            "title": "Regression Chat",
            "url": "https://example.test/chat/conv-1",
            "messages": messages,
        }
        return self.request_json("/api/messages", payload)

    def request_json(self, path, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        req = Request(
            f"{self.base_url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST" if data is not None else "GET",
        )
        with urlopen(req, timeout=5) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def read_markdown(self):
        files = [
            path for path in Path(self.tmp.name).rglob("*.md")
            if ".context.md" not in path.name
            and ".pre-state-backup-" not in path.name
        ]
        self.assertEqual(len(files), 1)
        return files[0].read_text(encoding="utf-8")

    def test_health_reports_extension_release(self):
        health = self.request_json("/api/health")
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["service"], "ShadowWrite")
        self.assertTrue(health["extensionVersion"])
        self.assertTrue(health["extensionVersionName"])

    def test_extension_release_changes_with_source(self):
        extension_dir = Path(self.tmp.name) / "extension-fixture"
        extension_dir.mkdir()
        manifest_path = extension_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps({"manifest_version": 3, "name": "Fixture", "version": "1.2.3"}),
            encoding="utf-8",
        )
        script_path = extension_dir / "worker.js"
        script_path.write_text("const build = 1;\n", encoding="utf-8")

        first = prepare_extension_release(extension_dir)
        second = prepare_extension_release(extension_dir)
        self.assertEqual(first, second)
        self.assertRegex(first["versionName"], r"^1\.2\.3\+local\.[0-9a-f]{12}$")

        script_path.write_text("const build = 2;\n", encoding="utf-8")
        changed = prepare_extension_release(extension_dir)
        self.assertNotEqual(changed["versionName"], first["versionName"])

    def test_mid_conversation_update_rewrites_following_turns(self):
        initial = [
            {"messageId": "m1", "sender": "user", "content": "hello", "thinking": ""},
            {"messageId": "m2", "sender": "AI", "content": "draft answer", "thinking": ""},
            {"messageId": "m3", "sender": "user", "content": "follow up", "thinking": ""},
        ]
        self.assertEqual(self.post_messages(initial)["written"], 3)

        updated = [
            {"messageId": "m1", "sender": "user", "content": "hello", "thinking": ""},
            {"messageId": "m2", "sender": "AI", "content": "final answer", "thinking": ""},
            {"messageId": "m3", "sender": "user", "content": "follow up", "thinking": ""},
        ]
        result = self.post_messages(updated)
        self.assertEqual(result["updated"], 1)
        self.assertEqual(result["written"], 1)

        markdown = self.read_markdown()
        self.assertIn("final answer", markdown)
        self.assertIn("follow up", markdown)
        self.assertNotIn("draft answer", markdown)
        self.assertEqual(markdown.count("follow up"), 1)

    def test_server_restart_does_not_duplicate_existing_snapshot(self):
        messages = [
            {"messageId": "m1", "sender": "user", "content": "restart user", "thinking": ""},
            {"messageId": "m2", "sender": "AI", "content": "restart assistant", "thinking": ""},
        ]
        self.assertEqual(self.post_messages(messages)["written"], 2)

        self.restart_server()
        result = self.post_messages(messages)
        self.assertEqual(result["written"], 0)
        self.assertEqual(result["updated"], 0)
        self.assertEqual(result["skipped"], 2)

        markdown = self.read_markdown()
        self.assertEqual(markdown.count("restart user"), 1)
        self.assertEqual(markdown.count("restart assistant"), 1)
        state_content = self.state_path.read_text(encoding="utf-8")
        self.assertIn("content_hash", state_content)
        self.assertNotIn("restart assistant", state_content)

    def test_pre_state_output_is_backed_up_before_migration(self):
        messages = [
            {"messageId": "m1", "sender": "user", "content": "legacy user", "thinking": ""},
            {"messageId": "m2", "sender": "AI", "content": "legacy assistant", "thinking": ""},
        ]
        self.post_messages(messages)

        self.restart_server(drop_state=True)
        result = self.post_messages(messages)
        self.assertEqual(result["written"], 2)

        backups = list(Path(self.tmp.name).rglob("*.pre-state-backup-*.md"))
        self.assertEqual(len(backups), 1)
        self.assertIn("legacy assistant", backups[0].read_text(encoding="utf-8"))
        markdown = self.read_markdown()
        self.assertEqual(markdown.count("legacy user"), 1)
        self.assertEqual(markdown.count("legacy assistant"), 1)

    def test_context_incremental_updates_round_trip(self):
        self.post_messages([
            {"messageId": "m1", "sender": "user", "content": "hello", "thinking": ""},
        ])

        result = self.request_json("/api/context", {
            "conversationId": "conv-1",
            "blocks": ["## Canon\n- Persistent fact"],
            "inlines": ["decision recorded"],
        })
        self.assertEqual(result["count"], 2)

        context = self.request_json("/api/context?conversationId=conv-1")
        self.assertTrue(context["exists"])
        self.assertIn("Persistent fact", context["content"])
        self.assertIn("decision recorded", context["content"])
        self.assertTrue(context["path"].endswith(".context.md"))

    def test_context_summary_uses_configured_api_and_writes_file(self):
        self.post_messages([
            {"messageId": "m1", "sender": "user", "content": "summarize this", "thinking": ""},
            {"messageId": "m2", "sender": "AI", "content": "important answer", "thinking": ""},
        ])
        self.request_json("/api/context", {
            "conversationId": "conv-1",
            "blocks": ["## Existing memory\n- Early stable fact"],
        })
        prompts = []
        self.handler_class._call_openai_api = staticmethod(
            lambda api_key, model, base_url, prompt: (
                prompts.append(prompt) or "## Summary\n- Stable decision"
            )
        )

        with patch.dict(os.environ, {
            "SHADOWWRITE_PROVIDER": "openai_compat",
            "SHADOWWRITE_API_KEY": "test-key",
            "SHADOWWRITE_BASE_URL": "https://api.example.test/v1",
            "SHADOWWRITE_MODEL": "test-model",
        }):
            result = self.request_json("/api/context/summarize", {
                "conversationId": "conv-1",
            })

        self.assertEqual(result["status"], "ok")
        self.assertEqual(len(prompts), 1)
        self.assertIn("Early stable fact", prompts[0])
        self.assertIn("important answer", prompts[0])
        context = self.request_json("/api/context?conversationId=conv-1")
        self.assertIn("Auto-summarized by ShadowWrite", context["content"])
        self.assertIn("Stable decision", context["content"])
        self.assertNotIn("\\n", context["content"])

    def test_web_clip_creates_then_appends_same_page(self):
        payload = {
            "title": "Chapter One",
            "url": "https://example.test/book/1",
            "domain": "example.test",
            "category": "clips",
            "group": "Example Book",
            "content": "First captured body.",
        }
        created = self.request_json("/api/clip", payload)
        self.assertEqual(created["mode"], "created")

        payload["content"] = "Second captured body."
        appended = self.request_json("/api/clip", payload)
        self.assertEqual(appended["mode"], "appended")

        clip_path = Path(appended["path"])
        content = clip_path.read_text(encoding="utf-8")
        self.assertIn("Source: https://example.test/book/1", content)
        self.assertEqual(content.count("First captured body."), 1)
        self.assertEqual(content.count("Second captured body."), 1)


if __name__ == "__main__":
    unittest.main()
