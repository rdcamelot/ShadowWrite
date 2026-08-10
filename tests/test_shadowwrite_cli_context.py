import tempfile
import unittest
from pathlib import Path

from shadowwrite_cli import (
    append_context_updates,
    build_system_prompt,
    extract_context_updates,
)


class ShadowWriteCliContextTests(unittest.TestCase):
    def test_context_markers_are_extracted_and_persisted(self):
        answer = (
            "Visible answer\n"
            "<!-- context-update-start -->\n"
            "## Character\n- Name: Ada\n"
            "<!-- context-update-end -->\n"
            "<!-- context-update: chapter one completed -->"
        )
        blocks, inlines = extract_context_updates(answer)
        self.assertEqual(blocks, ["## Character\n- Name: Ada"])
        self.assertEqual(inlines, ["chapter one completed"])

        with tempfile.TemporaryDirectory() as tmp:
            context_path = Path(tmp) / "story_context.md"
            count = append_context_updates(
                context_path,
                blocks=blocks,
                inlines=inlines,
                turn_id=3,
            )
            self.assertEqual(count, 2)
            content = context_path.read_text(encoding="utf-8")
            self.assertIn("Name: Ada", content)
            self.assertIn("Turn 3", content)
            self.assertIn("chapter one completed", content)

    def test_system_prompt_includes_existing_context_and_instructions(self):
        with tempfile.TemporaryDirectory() as tmp:
            context_path = Path(tmp) / "project_context.md"
            context_path.write_text("## Decision\n- Use local files", encoding="utf-8")
            prompt = build_system_prompt("Answer concisely", context_path)

        self.assertIn("PROJECT CONTEXT", prompt)
        self.assertIn("Use local files", prompt)
        self.assertIn("context-update-start", prompt)
        self.assertIn("Answer concisely", prompt)


if __name__ == "__main__":
    unittest.main()
