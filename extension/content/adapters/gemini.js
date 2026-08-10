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
      const match = url.match(/\/(gem\/[^/]+\/|app\/)([^/?#]+)/);
      const id = match ? match[2] : url.split("/").pop();
      return {
        conversationId: `gemini_${id}`,
        isNewConversation: false,
      };
    }

    getObserverRoot() {
      return document.querySelector('#chat-history, chat-history, [data-test-id="chat-history"]')
        || document.querySelector("main")
        || document.body;
    }

    /**
     * Extract the current conversation title without confusing it with a
     * different sidebar item or Gemini's temporary first-prompt title.
     */
    extractTitle() {
      // Gemini renders many conversation-title nodes in the sidebar. Scope the
      // lookup to links whose path exactly matches the current conversation.
      const currentUrl = this.pageUrl || location.href;
      let parsedCurrentUrl;
      try {
        parsedCurrentUrl = new URL(currentUrl, location.href);
      } catch {
        parsedCurrentUrl = null;
      }

      if (parsedCurrentUrl) {
        const currentPath = parsedCurrentUrl.pathname.replace(/\/$/, "");
        const links = Array.from(document.querySelectorAll("a[href]"))
          .filter((link) => {
            try {
              const linkUrl = new URL(link.getAttribute("href"), parsedCurrentUrl);
              return linkUrl.pathname.replace(/\/$/, "") === currentPath;
            } catch {
              return false;
            }
          })
          .sort((a, b) => {
            const aCurrent = a.getAttribute("aria-current") === "page" ? 1 : 0;
            const bCurrent = b.getAttribute("aria-current") === "page" ? 1 : 0;
            return bCurrent - aCurrent;
          });

        for (const link of links) {
          const titleEl = link.querySelector(
            '[data-test-id="conversation-title"], [data-test-id="conversation-title-text"], .conversation-title, [class*="conversation-title"]'
          );
          const candidates = [
            titleEl?.innerText || titleEl?.textContent,
            link.getAttribute("title"),
            link.innerText || link.textContent,
          ];
          for (const candidate of candidates) {
            const title = this._normalizeTitleCandidate(candidate);
            if (title) return title;
          }
        }
      }

      // The browser tab is less precise than the active sidebar item, but is a
      // useful fallback once Gemini has generated a real title.
      return this._normalizeTitleCandidate(document.title, true);
    }

    _normalizeTitleCandidate(value, stripBrand = false) {
      let title = String(value || "").replace(/\s+/g, " ").trim();
      if (stripBrand) {
        title = title
          .replace(/\s*(?:[-–—|]\s*)?(?:Google\s+)?Gemini\s*$/i, "")
          .trim();
      }
      title = title
        .replace(/\s*(?:More options|Open conversation menu|更多选项|打开对话菜单)\s*$/i, "")
        .trim();

      if (
        title.length < 2 ||
        title.length > 80 ||
        /^(?:Google\s+)?Gemini$/i.test(title) ||
        /^(?:New chat|Chats?|新对话|对话)$/i.test(title) ||
        /https?:\/\//i.test(title)
      ) {
        return null;
      }

      const firstQueryEl = document.querySelector(
        "user-query .query-text, user-query [data-test-id='user-query-content'], user-query .user-query-content, user-query"
      );
      const firstQuery = String(firstQueryEl?.innerText || firstQueryEl?.textContent || "")
        .replace(/^(?:你说|You said)\s*/i, "")
        .replace(/\s+/g, " ")
        .trim();
      if (title.length >= 40 && firstQuery.length >= title.length) {
        const comparableTitle = title.replace(/…$/, "").trim();
        if (comparableTitle && firstQuery.startsWith(comparableTitle)) {
          return null;
        }
      }

      return title;
    }

    extractMessages() {
      const messages = [];

      // Gemini DOM: #chat-history > .conversation-container[]
      // Each container has one user-query + one model-response.
      const chatHistory = document.querySelector(
        '#chat-history, chat-history, [data-test-id="chat-history"]'
      ) || document.querySelector("main");
      if (!chatHistory || !chatHistory.querySelector("user-query, model-response")) {
        return messages;
      }

      const blocks = chatHistory.querySelectorAll(".conversation-container");
      if (blocks.length > 0) {
        blocks.forEach((block, blockIndex) => {
          // Skip if user is editing in this block
          if (this.isInEditMode(block)) return;

          const userContent = this._extractUserContent(block);
          if (userContent) {
            const pos = blockIndex * 2;
            messages.push({
              messageId: this.generateMessageId("user", pos),
              sender: "user",
              content: userContent,
              thinking: "",
              position: pos,
            });
          }

          const aiContent = this._extractAiContent(block);
          if (aiContent) {
            const pos = blockIndex * 2 + 1;
            messages.push({
              messageId: this.generateMessageId("AI", pos),
              sender: "AI",
              content: aiContent,
              thinking: "",
              position: pos,
            });
          }
        });

        return messages;
      }

      // Fallback for Gemini DOM variants that no longer wrap turns in
      // .conversation-container, while preserving visible order.
      const turns = chatHistory.querySelectorAll("user-query, model-response");
      turns.forEach((turn, index) => {
        if (this.isInEditMode(turn)) return;
        if (turn.matches("user-query")) {
          const content = this._extractUserContent(turn);
          if (content) {
            messages.push({
              messageId: this.generateMessageId("user", index),
              sender: "user",
              content,
              thinking: "",
              position: index,
            });
          }
        } else if (turn.matches("model-response")) {
          const content = this._extractAiContent(turn);
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

    _extractUserContent(scope) {
      const userEl = scope.matches?.("user-query")
        ? scope
        : scope.querySelector(
            "user-query .query-text, user-query [data-test-id='user-query-content'], user-query .user-query-content, user-query"
          );
      if (!userEl) return "";
      const content = this.extractFormattedContent(userEl)
        .replace(/^(你说|You said)\s*/i, "")
        .trim();
      return this._stripInjectedContextPrefix(content);
    }

    _extractAiContent(scope) {
      const aiEl = scope.matches?.("model-response")
        ? scope.querySelector(
            ".model-response-text, .markdown, [data-test-id='response-content']"
          ) || scope
        : scope.querySelector(
            "model-response .model-response-text, model-response .markdown, model-response [data-test-id='response-content'], model-response"
          );
      return aiEl ? this.extractFormattedContent(aiEl).trim() : "";
    }

    isMessageElement(node) {
      if (!node || node.nodeType !== Node.ELEMENT_NODE) return false;
      return (
        node.matches?.(".conversation-container") ||
        node.matches?.("user-query") ||
        node.matches?.("model-response") ||
        node.querySelector?.("user-query, model-response") !== null ||
        node.closest?.(".conversation-container") !== null ||
        node.closest?.("user-query") !== null ||
        node.closest?.("model-response") !== null
      );
    }

    /* ---- Input element hooks for context injection ---- */

    getInputElement() {
      // Gemini uses a rich text editor
      return document.querySelector('.ql-editor[contenteditable="true"]')
        || document.querySelector('rich-textarea .text-input-field')
        || document.querySelector('div[contenteditable="true"]');
    }

    getSubmitButton() {
      return document.querySelector('button.send-button')
        || document.querySelector('button[aria-label="Send message"]');
    }
  }

  const adapter = new GeminiAdapter();
  adapter.start();
  window.__shadowWriteAdapter = adapter;
})();
