/**
 * ShadowWrite — ChatGPT Adapter
 *
 * Selectors based on chat.openai.com / chatgpt.com DOM structure.
 */

"use strict";

(() => {
  if (!window.BaseShadowWriteAdapter) {
    console.error("[ShadowWrite] BaseShadowWriteAdapter not loaded.");
    return;
  }

  class ChatGPTAdapter extends window.BaseShadowWriteAdapter {
    constructor() {
      super("chatgpt");
    }

    /* ---- Abstract implementations ---- */

    isValidConversationUrl(url) {
      // chatgpt.com/c/{id}  or  chatgpt.com/g/{gpt}/c/{id}
      return /chatgpt\.com\/(c\/|g\/[^/]+\/c\/)/.test(url)
          || /chat\.openai\.com\/(c\/|g\/[^/]+\/c\/)/.test(url);
    }

    extractConversationInfo(url) {
      const match = url.match(/\/c\/([a-f0-9-]+)/);
      const id = match ? match[1] : url;
      return {
        conversationId: `chatgpt_${id}`,
        isNewConversation: false,
      };
    }

    extractMessages() {
      if (this.isInEditMode(document.body)) return [];

      const messages = [];
      const articles = document.querySelectorAll("article");

      articles.forEach((article, index) => {
        const userEl = article.querySelector(
          'div[data-message-author-role="user"]'
        );
        const assistantEl = article.querySelector(
          'div[data-message-author-role="assistant"]'
        );

        if (userEl) {
          const textEl = userEl.querySelector(".whitespace-pre-wrap");
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

        if (assistantEl) {
          const proseEl = assistantEl.querySelector(".markdown.prose");
          const content = proseEl
            ? this.extractFormattedContent(proseEl)
            : this.extractFormattedContent(assistantEl);
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
        node.tagName === "ARTICLE" ||
        node.closest?.("article") !== null ||
        node.matches?.('[data-message-author-role]') ||
        node.closest?.('[data-message-author-role]') !== null
      );
    }

    extractTitle() {
      // ChatGPT puts the conversation title in the page <title>
      const title = document.title?.replace(/ \| ChatGPT$/, "").trim();
      return title || null;
    }

    /**
     * Detect ChatGPT Project conversations and extract the project name.
     * Project URL pattern: /g/g-p-{id}-{slug}/c/{conv_id}
     * Project title format: "project_name - conversation_title"
     * Returns { project, title } when in a project, null otherwise.
     */
    extractProject() {
      const url = this.pageUrl || window.location.href;
      // Only project GPTs have /g/g-p- in the URL
      if (!/\/g\/g-p-/.test(url)) return null;

      const fullTitle = this.extractTitle();
      if (!fullTitle) return null;

      // Split on first " - " separator
      const sep = fullTitle.indexOf(" - ");
      if (sep <= 0) {
        // No separator — use full title as conv title, slug as project
        return { project: "project", title: fullTitle };
      }
      return {
        project: fullTitle.substring(0, sep).trim(),
        title: fullTitle.substring(sep + 3).trim(),
      };
    }

    /* ---- Input element hooks for context injection ---- */

    getInputElement() {
      // ChatGPT uses a ProseMirror contenteditable div inside the composer
      return document.querySelector("#prompt-textarea")
        || document.querySelector('div[contenteditable="true"]');
    }

    getSubmitButton() {
      return document.querySelector('button[data-testid="send-button"]')
        || document.querySelector('form button[type="submit"]');
    }
  }

  /* ---- Bootstrap ---- */
  const adapter = new ChatGPTAdapter();
  adapter.start();
  window.__shadowWriteAdapter = adapter;
})();
