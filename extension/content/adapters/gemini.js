/**
 * ShadowWrite — Gemini Adapter
 *
 * Selectors based on gemini.google.com DOM structure.
 */

"use strict";

(() => {
  if (!window.BaseShadowWriteAdapter) {
    console.error("[ShadowWrite] BaseShadowWriteAdapter not loaded.");
    return;
  }

  class GeminiAdapter extends window.BaseShadowWriteAdapter {
    constructor() {
      super("gemini");
    }

    isValidConversationUrl(url) {
      return /gemini\.google\.com\/(gem\/|app\/)/.test(url);
    }

    extractConversationInfo(url) {
      // /gem/{type}/{id}  or  /app/{id}
      const match = url.match(/\/(gem\/[^/]+\/|app\/)([a-f0-9]+)/);
      const id = match ? match[2] : url.split("/").pop();
      return {
        conversationId: `gemini_${id}`,
        isNewConversation: false,
      };
    }

    extractMessages() {
      if (this.isInEditMode(document.body)) return [];

      const messages = [];
      let position = 0;

      // User queries
      const userEls = document.querySelectorAll("user-query .query-text");
      userEls.forEach((el) => {
        messages.push({
          messageId: this.generateMessageId("user", position),
          sender: "user",
          content: this.extractFormattedContent(el),
          thinking: "",
          position: position++,
        });
      });

      // Model responses
      const aiEls = document.querySelectorAll("model-response .model-response-text");
      aiEls.forEach((el) => {
        const content = this.extractFormattedContent(el);
        if (content) {
          messages.push({
            messageId: this.generateMessageId("AI", position),
            sender: "AI",
            content,
            thinking: "",
            position: position++,
          });
        }
      });

      return messages;
    }

    isMessageElement(node) {
      if (!node || node.nodeType !== Node.ELEMENT_NODE) return false;
      return (
        node.matches?.("user-query") ||
        node.matches?.("model-response") ||
        node.closest?.("user-query") !== null ||
        node.closest?.("model-response") !== null
      );
    }
  }

  const adapter = new GeminiAdapter();
  adapter.start();
  window.__shadowWriteAdapter = adapter;
})();
