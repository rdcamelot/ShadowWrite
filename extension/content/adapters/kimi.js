/**
 * ShadowWrite — Kimi Adapter
 *
 * Selectors based on kimi.moonshot.cn / kimi.com DOM structure.
 */

"use strict";

(() => {
  if (!window.BaseShadowWriteAdapter) {
    console.error("[ShadowWrite] BaseShadowWriteAdapter not loaded.");
    return;
  }

  class KimiAdapter extends window.BaseShadowWriteAdapter {
    constructor() {
      super("kimi");
    }

    isValidConversationUrl(url) {
      return /kimi\.(moonshot\.cn|com)\/chat\/[a-zA-Z0-9]+/.test(url);
    }

    extractConversationInfo(url) {
      const match = url.match(/\/chat\/([a-zA-Z0-9_-]+)/);
      const id = match ? match[1] : url;
      return {
        conversationId: `kimi_${id}`,
        isNewConversation: false,
      };
    }

    extractMessages() {
      const messages = [];
      const legacyItems = document.querySelectorAll(".chat-content-item");
      const roleSelector = [
        '[data-message-author-role="user"]',
        '[data-message-author-role="assistant"]',
        '[data-role="user"]',
        '[data-role="assistant"]',
        '[data-testid="user-message"]',
        '[data-testid="assistant-message"]',
      ].join(", ");
      const items = legacyItems.length > 0
        ? Array.from(legacyItems)
        : Array.from(document.querySelectorAll(roleSelector))
            .filter((element) => !element.parentElement?.closest(roleSelector));

      items.forEach((item, index) => {
        if (this.isInEditMode(item)) return;
        const marker = [
          item.getAttribute("data-message-author-role"),
          item.getAttribute("data-role"),
          item.getAttribute("data-testid"),
        ].filter(Boolean).join(" ").toLowerCase();
        const isUser = item.classList.contains("chat-content-item-user")
          || /user|human/.test(marker);
        const isAssistant = item.classList.contains("chat-content-item-assistant")
          || /assistant|model/.test(marker);

        if (isUser) {
          const textEl = item.querySelector(
            '.user-content, [data-testid="message-content"], .whitespace-pre-wrap'
          ) || item;
          if (textEl) {
            messages.push({
              messageId: this.generateMessageId("user", index),
              sender: "user",
              content: this.extractFormattedContent(textEl),
              thinking: "",
              position: index,
            });
          }
        }

        if (isAssistant) {
          // Exclude thinking stage content
          const mdEls = item.querySelectorAll(
            '.markdown-container, .editor-content, [data-testid="message-content"], .markdown, .prose'
          );
          let content = "";
          for (const md of mdEls) {
            if (!md.closest(".think-stage")) {
              content += this.extractFormattedContent(md) + "\n";
            }
          }
          content = content.trim();

          if (!content) {
            const clone = item.cloneNode(true);
            clone.querySelectorAll(
              '.think-stage, [data-testid="thinking-content"], button, [role="toolbar"]'
            ).forEach((node) => node.remove());
            content = this.extractFormattedContent(clone).trim();
          }

          if (content) {
            messages.push({
              messageId: this.generateMessageId("AI", index),
              sender: "AI",
              content,
              thinking: "",
              position: index,
            });
          }
        }
      });

      return messages;
    }

    isMessageElement(node) {
      if (!node || node.nodeType !== Node.ELEMENT_NODE) return false;
      return (
        node.matches?.(".chat-content-item") ||
        node.matches?.('[data-message-author-role="user"], [data-message-author-role="assistant"], [data-role="user"], [data-role="assistant"], [data-testid="user-message"], [data-testid="assistant-message"]') ||
        node.closest?.(".chat-content-item") !== null ||
        node.closest?.('[data-message-author-role="user"], [data-message-author-role="assistant"], [data-role="user"], [data-role="assistant"], [data-testid="user-message"], [data-testid="assistant-message"]') !== null
      );
    }
  }

  const adapter = new KimiAdapter();
  adapter.start();
  window.__shadowWriteAdapter = adapter;
})();
