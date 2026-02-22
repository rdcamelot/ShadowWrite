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

    extractMessages() {
      if (this.isInEditMode(document.body)) return [];

      const messages = [];
      let position = 0;

      // querySelectorAll returns elements in DOM order — this correctly
      // interleaves user and AI messages.
      const allEls = document.querySelectorAll(
        '[data-testid="user-message"], .font-claude-response'
      );

      allEls.forEach((el) => {
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
        node.matches?.(".font-claude-response") ||
        node.closest?.('[data-testid="user-message"]') !== null ||
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
