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

    /**
     * Extract meaningful conversation title from the DOM title element.
     */
    extractTitle() {
      // Primary: dedicated title element in Gemini's sidebar/header
      const titleEl = document.querySelector('[data-test-id="conversation-title"]');
      if (titleEl) {
        const text = (titleEl.innerText || titleEl.textContent || "").trim();
        if (text && text.length > 2) {
          return text.length > 80 ? text.substring(0, 80) + "…" : text;
        }
      }
      // Fallback: page title (filter generic ones)
      const title = document.title?.trim();
      if (title && title !== "Gemini" && !title.startsWith("Google") && title.length > 2) {
        return title.length > 80 ? title.substring(0, 80) + "…" : title;
      }
      return null;
    }

    extractMessages() {
      if (this.isInEditMode(document.body)) return [];

      const messages = [];

      // Gemini DOM: #chat-history > .conversation-container[]
      // Each container has one user-query + one model-response.
      const chatHistory = document.querySelector("#chat-history");
      if (!chatHistory) {
        console.log("[ShadowWrite] Gemini: #chat-history not found yet.");
        return messages;
      }

      const blocks = chatHistory.querySelectorAll(".conversation-container");
      if (blocks.length === 0) {
        console.log("[ShadowWrite] Gemini: no .conversation-container found.");
        return messages;
      }

      blocks.forEach((block, blockIndex) => {
        // Skip if user is editing in this block
        if (this.isInEditMode(block)) return;

        // User message (even position)
        const userEl = block.querySelector("user-query .query-text");
        if (userEl) {
          let content = this.extractFormattedContent(userEl);
          // Strip Gemini's "你说" / "You said" prefix
          content = content.replace(/^(你说|You said)\s*/i, "");
          if (content) {
            const pos = blockIndex * 2;
            messages.push({
              messageId: this.generateMessageId("user", pos),
              sender: "user",
              content,
              thinking: "",
              position: pos,
            });
          }
        }

        // AI response (odd position)
        const aiEl = block.querySelector("model-response .model-response-text");
        if (aiEl) {
          const content = this.extractFormattedContent(aiEl);
          if (content) {
            const pos = blockIndex * 2 + 1;
            messages.push({
              messageId: this.generateMessageId("AI", pos),
              sender: "AI",
              content,
              thinking: "",
              position: pos,
            });
          }
        }
      });

      console.log(`[ShadowWrite] Gemini: extracted ${messages.length} messages from ${blocks.length} blocks.`);
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
