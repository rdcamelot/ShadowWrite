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
      if (this.isInEditMode(document.body)) return [];

      const messages = [];
      const items = document.querySelectorAll(".chat-content-item");

      items.forEach((item, index) => {
        if (item.classList.contains("chat-content-item-user")) {
          const textEl = item.querySelector(".user-content");
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

        if (item.classList.contains("chat-content-item-assistant")) {
          // Exclude thinking stage content
          const mdEls = item.querySelectorAll(
            ".markdown-container, .editor-content"
          );
          let content = "";
          for (const md of mdEls) {
            if (!md.closest(".think-stage")) {
              content += this.extractFormattedContent(md) + "\n";
            }
          }
          content = content.trim();

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
        node.closest?.(".chat-content-item") !== null
      );
    }
  }

  const adapter = new KimiAdapter();
  adapter.start();
  window.__shadowWriteAdapter = adapter;
})();
