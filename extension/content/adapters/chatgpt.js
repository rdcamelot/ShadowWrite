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
      const messages = [];
      const articles = Array.from(document.querySelectorAll("article"))
        .filter((article) => article.querySelector('[data-message-author-role]'));

      articles.forEach((article, index) => {
        if (this.isInEditMode(article)) return;

        const userEl = article.querySelector(
          '[data-message-author-role="user"]'
        );
        const assistantEl = article.querySelector(
          '[data-message-author-role="assistant"]'
        );

        if (userEl) this._appendRoleMessage(messages, userEl, "user", index);
        if (assistantEl) this._appendRoleMessage(messages, assistantEl, "AI", index);
      });

      if (articles.length === 0) {
        const roleElements = Array.from(document.querySelectorAll(
          '[data-message-author-role="user"], [data-message-author-role="assistant"]'
        )).filter((element) => !element.parentElement?.closest('[data-message-author-role]'));

        roleElements.forEach((element, index) => {
          if (this.isInEditMode(element)) return;
          const role = element.getAttribute("data-message-author-role") === "user"
            ? "user"
            : "AI";
          this._appendRoleMessage(messages, element, role, index);
        });
      }

      return messages;
    }

    _appendRoleMessage(messages, element, role, position) {
      const contentElement = role === "user"
        ? element.querySelector('.whitespace-pre-wrap, .markdown, [data-testid="message-content"]') || element
        : element.querySelector('.markdown.prose, .markdown, [data-testid="message-content"]') || element;
      const extracted = this.extractFormattedContent(contentElement).trim();
      const content = role === "user"
        ? this._stripInjectedContextPrefix(extracted)
        : extracted;
      if (!content) return;

      messages.push({
        messageId: this.generateMessageId(role, position),
        sender: role,
        content,
        thinking: "",
        position,
      });
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
