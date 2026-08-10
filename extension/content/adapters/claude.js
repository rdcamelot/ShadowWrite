/**
 * ShadowWrite — Claude Adapter
 *
 * Selectors based on claude.ai DOM structure.
 */

"use strict";

(() => {
  if (!window.BaseShadowWriteAdapter) {
    console.error("[ShadowWrite] BaseShadowWriteAdapter not loaded.");
    return;
  }

  class ClaudeAdapter extends window.BaseShadowWriteAdapter {
    constructor() {
      super("claude");
    }

    isValidConversationUrl(url) {
      return /claude\.ai\/chat\/[a-f0-9-]+/.test(url);
    }

    extractConversationInfo(url) {
      const match = url.match(/\/chat\/([a-f0-9-]+)/);
      const id = match ? match[1] : url;
      return {
        conversationId: `claude_${id}`,
        isNewConversation: false,
      };
    }

    extractTitle() {
      // Claude sometimes leaves document.title as "Claude"; try visible heading first.
      const domTitle =
        document.querySelector('[data-testid="conversation-title"]')?.textContent ||
        document.querySelector('[data-testid="chat-title"]')?.textContent ||
        document.querySelector("main h1")?.textContent ||
        document.title || "";

      const clean = (domTitle || "")
        .replace(/\s*\|\s*Claude\s*$/i, "")
        .replace(/\s*-\s*Claude\s*$/i, "")
        .trim();
      return clean || null;
    }

    /**
     * Best-effort project extraction for Claude.
     * Priority:
     * 1) Dedicated project/breadcrumb DOM nodes (if present)
     * 2) Title pattern: "project / conversation"
     */
    extractProject() {
      const title = this.extractTitle() || "";

      // 1) DOM-based project name candidates (Claude UI may change over time)
      const projectCandidates = [
        '[data-testid="project-name"]',
        '[data-testid="breadcrumb-project"]',
        'nav a[href*="/project/"]',
        'nav a[href*="/projects/"]',
        'a[href*="/project/"]',
        'a[href*="/projects/"]',
      ];

      for (const selector of projectCandidates) {
        const el = document.querySelector(selector);
        const txt = (el?.textContent || "").trim();
        if (txt && txt.toLowerCase() !== "projects") {
          const escaped = txt.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
          const convTitle = title
            .replace(new RegExp(`^${escaped}\\s*[/｜-]\\s*`), "")
            .trim() || title;
          return { project: txt, title: convTitle };
        }
      }

      // 2) Fallback: parse title split by slash-like separators
      const slashMatch = title.match(/^(.+?)\s*[\/｜]\s*(.+)$/);
      if (slashMatch) {
        const project = slashMatch[1].trim();
        const convTitle = slashMatch[2].trim();
        if (project && convTitle) {
          return { project, title: convTitle };
        }
      }

      // No reliable project signal — keep current flat layout.
      return null;
    }

    extractMessages() {
      const messages = [];
      let position = 0;

      // querySelectorAll returns elements in DOM order — this correctly
      // interleaves user and AI messages.
      const allEls = document.querySelectorAll(
        '[data-testid="user-message"], [data-testid="assistant-message"], .font-claude-response'
      );

      allEls.forEach((el) => {
        if (this.isInEditMode(el)) return;

        if (el.matches('[data-testid="user-message"]')) {
          messages.push({
            messageId: this.generateMessageId("user", position),
            sender: "user",
            content: this.extractFormattedContent(el),
            thinking: "",
            position: position++,
          });
        } else {
          const content = this._extractFormalResponse(el);
          if (content) {
            messages.push({
              messageId: this.generateMessageId("AI", position),
              sender: "AI",
              content,
              thinking: "",
              position: position++,
            });
          }
        }
      });

      return messages;
    }

    /**
     * Extract only the formal response, skipping thinking/reasoning blocks.
     */
    _extractFormalResponse(el) {
      const clone = el.cloneNode(true);
      // Remove thinking blocks (Claude wraps them in collapsible sections
      // with transition-all and rounded-lg classes)
      clone.querySelectorAll(".transition-all.rounded-lg").forEach((block) => {
        // Check if it has a collapse/expand button — that's the thinking block
        if (block.querySelector("button")) {
          block.remove();
        }
      });
      return this.extractFormattedContent(clone);
    }

    isMessageElement(node) {
      if (!node || node.nodeType !== Node.ELEMENT_NODE) return false;
      return (
        node.matches?.('[data-testid="user-message"]') ||
        node.matches?.('[data-testid="assistant-message"]') ||
        node.matches?.(".font-claude-response") ||
        node.closest?.('[data-testid="user-message"]') !== null ||
        node.closest?.('[data-testid="assistant-message"]') !== null ||
        node.closest?.(".font-claude-response") !== null ||
        node.matches?.("[data-test-render-count]") ||
        node.closest?.("[data-test-render-count]") !== null
      );
    }
  }

  const adapter = new ClaudeAdapter();
  adapter.start();
  window.__shadowWriteAdapter = adapter;
})();
