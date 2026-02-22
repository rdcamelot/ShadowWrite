/**
 * ShadowWrite — DeepSeek Adapter
 *
 * ⚠️ DeepSeek uses obfuscated CSS class names that may change between versions.
 */

"use strict";

(() => {
  if (!window.BaseShadowWriteAdapter) {
    console.error("[ShadowWrite] BaseShadowWriteAdapter not loaded.");
    return;
  }

  class DeepSeekAdapter extends window.BaseShadowWriteAdapter {
    constructor() {
      super("deepseek");
    }

    isValidConversationUrl(url) {
      return /chat\.deepseek\.com\/a\/chat\/s\//.test(url);
    }

    extractConversationInfo(url) {
      const match = url.match(/\/a\/chat\/s\/([a-zA-Z0-9_-]+)/);
      const id = match ? match[1] : url;
      return {
        conversationId: `deepseek_${id}`,
        isNewConversation: false,
      };
    }

    extractMessages() {
      if (this.isInEditMode(document.body)) return [];

      const messages = [];
      let position = 0;

      // querySelectorAll returns in DOM order — correctly interleaves
      const allEls = document.querySelectorAll("._9663006, ._4f9bf79._43c05b5");

      allEls.forEach((el) => {
        if (el.matches("._9663006")) {
          // User message
          const textEl = el.querySelector(".fbb737a4");
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
          const thinkEl =
            el.querySelector(".ds-think-content") ||
            el.querySelector(".e1675d8b .ds-markdown");
          if (thinkEl) {
            thinking = this.extractFormattedContent(thinkEl);
          }

          // Main content — .ds-markdown not inside think block
          const mdEls = el.querySelectorAll(":scope > .ds-markdown, .ds-message > .ds-markdown");
          let content = "";
          for (const md of mdEls) {
            if (!md.closest(".ds-think-content") && !md.closest(".e1675d8b")) {
              content = this.extractFormattedContent(md);
              break;
            }
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
        node.matches?.("._9663006") ||
        node.matches?.("._4f9bf79") ||
        node.closest?.("._9663006") !== null ||
        node.closest?.("._4f9bf79") !== null
      );
    }
  }

  const adapter = new DeepSeekAdapter();
  adapter.start();
  window.__shadowWriteAdapter = adapter;
})();
