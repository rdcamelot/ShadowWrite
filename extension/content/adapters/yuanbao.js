/**
 * ShadowWrite — Yuanbao (腾讯元宝) Adapter
 *
 * Selectors based on yuanbao.tencent.com DOM structure.
 */

"use strict";

(() => {
  if (!window.BaseShadowWriteAdapter) {
    console.error("[ShadowWrite] BaseShadowWriteAdapter not loaded.");
    return;
  }

  class YuanbaoAdapter extends window.BaseShadowWriteAdapter {
    constructor() {
      super("yuanbao");
    }

    isValidConversationUrl(url) {
      // yuanbao.tencent.com/chat/{app_id}/{conv_id}
      return /yuanbao\.tencent\.com\/chat\/[^/]+\/[a-zA-Z0-9]/.test(url);
    }

    extractConversationInfo(url) {
      const match = url.match(/\/chat\/([^/]+)\/([a-zA-Z0-9_-]+)/);
      const id = match ? `${match[1]}_${match[2]}` : url;
      return {
        conversationId: `yuanbao_${id}`,
        isNewConversation: false,
      };
    }

    extractMessages() {
      if (this.isInEditMode(document.body)) return [];

      const messages = [];
      let position = 0;

      // querySelectorAll returns in DOM order — correctly interleaves
      const allItems = document.querySelectorAll(
        ".agent-chat__list__item--human, .agent-chat__list__item--ai"
      );

      allItems.forEach((item) => {
        if (item.matches(".agent-chat__list__item--human")) {
          const textEl = item.querySelector(".hyc-content-text");
          if (textEl) {
            messages.push({
              messageId: this.generateMessageId("user", position),
              sender: "user",
              content: this.extractFormattedContent(textEl),
              thinking: "",
              position: position++,
            });
          }
        } else {
          // AI message
          let thinking = "";
          const thinkEl = item.querySelector(
            ".hyc-component-reasoner__think-content"
          );
          if (thinkEl) {
            const thinkText = thinkEl.querySelector(
              ".hyc-component-reasoner__text"
            );
            if (thinkText) {
              thinking = this.extractFormattedContent(thinkText);
            }
          }

          // Main response text
          const responseEl = item.querySelector(
            ".hyc-component-reasoner__text"
          );
          let content = "";
          if (responseEl && !responseEl.closest(".hyc-component-reasoner__think-content")) {
            content = this.extractFormattedContent(responseEl);
          }
          // Fallback: direct extraction from AI item
          if (!content) {
            content = this.extractFormattedContent(item);
          }

          if (content) {
            messages.push({
              messageId: this.generateMessageId("AI", position),
              sender: "AI",
              content,
              thinking,
              position: position++,
            });
          }
        }
      });

      return messages;
    }

    isMessageElement(node) {
      if (!node || node.nodeType !== Node.ELEMENT_NODE) return false;
      return (
        node.matches?.(".agent-chat__list__item--human") ||
        node.matches?.(".agent-chat__list__item--ai") ||
        node.closest?.(".agent-chat__list__item--human") !== null ||
        node.closest?.(".agent-chat__list__item--ai") !== null
      );
    }
  }

  const adapter = new YuanbaoAdapter();
  adapter.start();
  window.__shadowWriteAdapter = adapter;
})();
