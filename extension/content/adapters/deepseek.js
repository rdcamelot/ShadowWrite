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
      const messages = [];
      let position = 0;

      // querySelectorAll returns in DOM order — correctly interleaves
      const allEls = document.querySelectorAll("._9663006, ._4f9bf79._43c05b5");

      if (allEls.length === 0) {
        return this._extractRoleMarkedMessages();
      }

      allEls.forEach((el) => {
        if (this.isInEditMode(el)) return;

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

    _extractRoleMarkedMessages() {
      const selector = [
        '[data-message-author-role="user"]',
        '[data-message-author-role="assistant"]',
        '[data-role="user"]',
        '[data-role="assistant"]',
        '[data-testid="user-message"]',
        '[data-testid="assistant-message"]',
      ].join(", ");
      const elements = Array.from(document.querySelectorAll(selector))
        .filter((element) => !element.parentElement?.closest(selector));

      return elements.map((element, index) => {
        if (this.isInEditMode(element)) return null;
        const marker = [
          element.getAttribute("data-message-author-role"),
          element.getAttribute("data-role"),
          element.getAttribute("data-testid"),
        ].filter(Boolean).join(" ").toLowerCase();
        const sender = /user|human/.test(marker) ? "user" : "AI";
        const thinkEl = element.querySelector(
          '.ds-think-content, [data-testid="thinking-content"]'
        );
        const thinking = thinkEl
          ? this.extractFormattedContent(thinkEl).trim()
          : "";
        const clone = element.cloneNode(true);
        clone.querySelectorAll(
          '.ds-think-content, [data-testid="thinking-content"], button, [role="toolbar"]'
        ).forEach((node) => node.remove());
        const contentEl = clone.querySelector(
          '.ds-markdown, [data-testid="message-content"], .markdown, .prose, .whitespace-pre-wrap'
        ) || clone;
        const content = this.extractFormattedContent(contentEl).trim();
        if (!content) return null;

        return {
          messageId: this.generateMessageId(sender, index),
          sender,
          content,
          thinking,
          position: index,
        };
      }).filter(Boolean);
    }

    isMessageElement(node) {
      if (!node || node.nodeType !== Node.ELEMENT_NODE) return false;
      return (
        node.matches?.("._9663006") ||
        node.matches?.("._4f9bf79") ||
        node.matches?.('[data-message-author-role="user"], [data-message-author-role="assistant"], [data-role="user"], [data-role="assistant"], [data-testid="user-message"], [data-testid="assistant-message"]') ||
        node.closest?.("._9663006") !== null ||
        node.closest?.("._4f9bf79") !== null ||
        node.closest?.('[data-message-author-role="user"], [data-message-author-role="assistant"], [data-role="user"], [data-role="assistant"], [data-testid="user-message"], [data-testid="assistant-message"]') !== null
      );
    }
  }

  const adapter = new DeepSeekAdapter();
  adapter.start();
  window.__shadowWriteAdapter = adapter;
})();
