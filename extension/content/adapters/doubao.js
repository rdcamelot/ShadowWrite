/**
 * ShadowWrite — Doubao (豆包) Adapter
 *
 * Selectors based on www.doubao.com DOM structure.
 */

"use strict";

(() => {
  if (!window.BaseShadowWriteAdapter) {
    console.error("[ShadowWrite] BaseShadowWriteAdapter not loaded.");
    return;
  }

  class DoubaoAdapter extends window.BaseShadowWriteAdapter {
    constructor() {
      super("doubao");
    }

    isValidConversationUrl(url) {
      // Exclude /chat/local (local model page)
      return /www\.doubao\.com\/chat\/[a-zA-Z0-9]/.test(url)
          && !/\/chat\/local/.test(url);
    }

    extractConversationInfo(url) {
      const match = url.match(/\/chat\/([a-zA-Z0-9_-]+)/);
      const id = match ? match[1] : url;
      return {
        conversationId: `doubao_${id}`,
        isNewConversation: false,
      };
    }

    extractMessages() {
      if (this.isInEditMode(document.body)) return [];

      const messages = [];
      const items = document.querySelectorAll('[data-testid="union_message"]');

      items.forEach((item, index) => {
        // User message
        const sendEl = item.querySelector('[data-testid="send_message"]');
        if (sendEl) {
          const textEl = sendEl.querySelector(
            '[data-testid="message_text_content"]'
          );
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

        // AI message
        const recvEl = item.querySelector('[data-testid="receive_message"]');
        if (recvEl) {
          // Thinking content
          let thinking = "";
          const thinkBlock = recvEl.querySelector(
            '[data-testid="think_block_collapse"]'
          );
          if (thinkBlock) {
            const thinkText = thinkBlock.querySelector(
              '[data-testid="message_text_content"]'
            );
            if (thinkText) {
              thinking = this.extractFormattedContent(thinkText);
            }
          }

          // Main content (outside think block)
          const allText = recvEl.querySelectorAll(
            '[data-testid="message_text_content"]'
          );
          let content = "";
          for (const el of allText) {
            if (!el.closest('[data-testid="think_block_collapse"]')) {
              content = this.extractFormattedContent(el);
              break;
            }
          }

          if (content) {
            messages.push({
              messageId: this.generateMessageId("AI", index),
              sender: "AI",
              content,
              thinking,
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
        node.matches?.('[data-testid="union_message"]') ||
        node.closest?.('[data-testid="union_message"]') !== null
      );
    }
  }

  const adapter = new DoubaoAdapter();
  adapter.start();
  window.__shadowWriteAdapter = adapter;
})();
