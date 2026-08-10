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
      const messages = [];
      let position = 0;

      // querySelectorAll returns in DOM order — correctly interleaves
      const legacyItems = document.querySelectorAll(
        ".agent-chat__list__item--human, .agent-chat__list__item--ai"
      );
      const roleSelector = [
        '[data-message-author-role="user"]',
        '[data-message-author-role="assistant"]',
        '[data-role="user"]',
        '[data-role="assistant"]',
        '[data-testid="user-message"]',
        '[data-testid="assistant-message"]',
      ].join(", ");
      const allItems = legacyItems.length > 0
        ? Array.from(legacyItems)
        : Array.from(document.querySelectorAll(roleSelector))
            .filter((element) => !element.parentElement?.closest(roleSelector));

      allItems.forEach((item) => {
        if (this.isInEditMode(item)) return;
        const marker = [
          item.getAttribute("data-message-author-role"),
          item.getAttribute("data-role"),
          item.getAttribute("data-testid"),
        ].filter(Boolean).join(" ").toLowerCase();
        const isUser = item.matches(".agent-chat__list__item--human")
          || /user|human/.test(marker);

        if (isUser) {
          const textEl = item.querySelector(
            '.hyc-content-text, [data-testid="message-content"], .whitespace-pre-wrap'
          ) || item;
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
            '.hyc-component-reasoner__think-content, [data-testid="thinking-content"]'
          );
          if (thinkEl) {
            const thinkText = thinkEl.querySelector(
              '.hyc-component-reasoner__text, [data-testid="message-content"]'
            ) || thinkEl;
            thinking = this.extractFormattedContent(thinkText);
          }

          // Main response text
          let content = "";
          const responseEls = item.querySelectorAll(
            '.hyc-component-reasoner__text, [data-testid="message-content"], .markdown, .prose'
          );
          for (const responseEl of responseEls) {
            if (
              !responseEl.closest(".hyc-component-reasoner__think-content") &&
              !responseEl.closest('[data-testid="thinking-content"]')
            ) {
              content = this.extractFormattedContent(responseEl);
              break;
            }
          }

          if (!content) {
            const clone = item.cloneNode(true);
            clone.querySelectorAll(
              '.hyc-component-reasoner__think-content, [data-testid="thinking-content"], button, [role="toolbar"]'
            ).forEach((node) => node.remove());
            content = this.extractFormattedContent(clone);
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
        node.matches?.('[data-message-author-role="user"], [data-message-author-role="assistant"], [data-role="user"], [data-role="assistant"], [data-testid="user-message"], [data-testid="assistant-message"]') ||
        node.closest?.(".agent-chat__list__item--human") !== null ||
        node.closest?.(".agent-chat__list__item--ai") !== null ||
        node.closest?.('[data-message-author-role="user"], [data-message-author-role="assistant"], [data-role="user"], [data-role="assistant"], [data-testid="user-message"], [data-testid="assistant-message"]') !== null
      );
    }
  }

  const adapter = new YuanbaoAdapter();
  adapter.start();
  window.__shadowWriteAdapter = adapter;
})();
