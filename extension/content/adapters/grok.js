/**
 * ShadowWrite - Grok Adapter
 *
 * Grok ships internal same-origin conversation APIs. We prefer those over
 * fragile DOM scraping so the adapter survives UI churn more reliably.
 */

"use strict";

(() => {
  if (!window.BaseShadowWriteAdapter) {
    console.error("[ShadowWrite] BaseShadowWriteAdapter not loaded.");
    return;
  }

  class GrokAdapter extends window.BaseShadowWriteAdapter {
    constructor() {
      super("grok");
      this._cachedTitle = null;
      this._cachedMessages = [];
      this._cachedSnapshot = null;
      this._lastSnapshotAt = 0;
      this._snapshotPromise = null;
    }

    isValidConversationUrl(url) {
      return /grok\.com\/(?:c|chat|chat-v2|chat-v1)\/[^/?#]+/.test(url);
    }

    extractConversationInfo(url) {
      const id = this._extractNativeConversationId(url);
      return {
        conversationId: id ? `grok_${id}` : url,
        isNewConversation: false,
      };
    }

    getUrlKey(url) {
      const parsed = new URL(url, location.origin);
      const rid = parsed.searchParams.get("rid");
      return `${parsed.origin}${parsed.pathname}${rid ? `?rid=${encodeURIComponent(rid)}` : ""}`;
    }

    extractTitle() {
      if (this._cachedTitle) return this._cachedTitle;
      const title = document.title
        ?.replace(/\s*\|\s*Grok\s*$/i, "")
        .replace(/\s*-\s*Grok\s*$/i, "")
        .trim();
      return title || null;
    }

    extractMessages() {
      return this._cachedMessages;
    }

    isMessageElement(node) {
      if (!node || node.nodeType !== Node.ELEMENT_NODE) return false;
      return (
        node.matches?.(
          "article, [data-testid*=\"message\"], .action-buttons, .thinking-container, .search-results, .inline-media-container, .prose, textarea, [contenteditable=\"true\"]"
        ) ||
        node.querySelector?.(
          "article, [data-testid*=\"message\"], .action-buttons, .thinking-container, .search-results, .inline-media-container, .prose, textarea, [contenteditable=\"true\"]"
        ) !== null ||
        node.closest?.(
          "article, [data-testid*=\"message\"], .action-buttons, .thinking-container, .search-results, .inline-media-container, .prose"
        ) !== null
      );
    }

    _handleUrlChange() {
      this._cachedTitle = null;
      this._cachedMessages = [];
      this._cachedSnapshot = null;
      this._lastSnapshotAt = 0;
      this._snapshotPromise = null;
      super._handleUrlChange();
    }

    async _checkForChanges() {
      if (!this.currentConversationId) return;

      const epoch = this._epoch;
      try {
        const snapshot = await this._getSnapshot();
        if (epoch !== this._epoch) return;
        const messages = snapshot.messages || [];
        if (messages.length === 0) return;

        const signature = this._buildSnapshotSignature(messages);
        if (signature !== this.lastMessagesSignature) {
          this.lastMessagesSignature = signature;
          this._cancelInitialCapture();
          await this._sendToService(messages);
        }
      } catch (err) {
        console.warn("[ShadowWrite] Grok snapshot refresh failed:", err);
      }
    }

    async _captureAndSend(retries = 0, epoch) {
      if (epoch === undefined) epoch = this._epoch;
      if (!this.isTracking || epoch !== this._epoch) return;

      try {
        const snapshot = await this._getSnapshot(true);
        if (epoch !== this._epoch) return;
        const messages = snapshot.messages || [];

        if (messages.length === 0) {
          if (retries < 3) {
            const delay = (retries + 1) * 1500;
            console.log(
              `[ShadowWrite] Grok: no messages yet, retry ${retries + 1}/3 in ${delay}ms...`
            );
            setTimeout(() => this._captureAndSend(retries + 1, epoch), delay);
          } else {
            console.log("[ShadowWrite] Grok: no messages after 3 retries, waiting for MutationObserver.");
          }
          return;
        }

        this.lastMessagesSignature = this._buildSnapshotSignature(messages);
        await this._sendToService(messages);
      } catch (err) {
        console.warn("[ShadowWrite] Grok initial capture failed:", err);
      }
    }

    getInputElement() {
      return (
        document.querySelector('textarea[placeholder]') ||
        document.querySelector('textarea') ||
        document.querySelector('[contenteditable="true"][role="textbox"]') ||
        document.querySelector('[contenteditable="true"]')
      );
    }

    getSubmitButton() {
      return (
        document.querySelector('button[aria-label*="send" i]') ||
        document.querySelector('button[title*="send" i]') ||
        document.querySelector('form button[type="submit"]')
      );
    }

    async _getSnapshot(forceRefresh = false) {
      const nativeConversationId = this._extractNativeConversationId();
      if (!nativeConversationId) {
        return { title: null, messages: [] };
      }

      const now = Date.now();
      if (
        !forceRefresh &&
        this._cachedSnapshot &&
        now - this._lastSnapshotAt < 800
      ) {
        return this._cachedSnapshot;
      }

      if (this._snapshotPromise) {
        return this._snapshotPromise;
      }

      const requestEpoch = this._epoch;
      const requestConversationId = nativeConversationId;
      this._snapshotPromise = this._fetchSnapshot(nativeConversationId)
        .then((snapshot) => {
          if (
            requestEpoch !== this._epoch ||
            requestConversationId !== this._extractNativeConversationId()
          ) {
            return this._cachedSnapshot || { title: null, messages: [] };
          }
          this._cachedSnapshot = snapshot;
          this._cachedTitle = snapshot.title || this._cachedTitle;
          this._cachedMessages = snapshot.messages || [];
          this._lastSnapshotAt = Date.now();
          return snapshot;
        })
        .finally(() => {
          this._snapshotPromise = null;
        });

      return this._snapshotPromise;
    }

    async _fetchSnapshot(nativeConversationId) {
      const encodedId = encodeURIComponent(nativeConversationId);
      const [conversationResult, responsesResult] = await Promise.allSettled([
        this._fetchJson(`/rest/app-chat/conversations_v2/${encodedId}`),
        this._fetchJson(`/rest/app-chat/conversations/${encodedId}/responses`),
      ]);

      const conversationData = conversationResult.status === "fulfilled"
        ? conversationResult.value
        : null;
      if (conversationResult.status === "rejected") {
        console.debug?.("[ShadowWrite] Grok title endpoint failed:", conversationResult.reason);
      }

      const apiMessages = responsesResult.status === "fulfilled"
        ? this._buildMessagesFromResponses(responsesResult.value)
        : [];
      const messages = apiMessages.length > 0
        ? apiMessages
        : this._extractMessagesFromDom();

      if (messages.length === 0 && responsesResult.status === "rejected") {
        throw responsesResult.reason;
      }

      return {
        title: conversationData?.conversation?.title?.trim() || this.extractTitle(),
        messages,
      };
    }

    _extractMessagesFromDom() {
      const selector = [
        '[data-message-author-role="user"]',
        '[data-message-author-role="assistant"]',
        '[data-testid="user-message"]',
        '[data-testid="human-message"]',
        '[data-testid="assistant-message"]',
        '[data-testid="model-message"]',
        '[data-role="user"]',
        '[data-role="assistant"]',
      ].join(", ");
      const elements = Array.from(document.querySelectorAll(selector))
        .filter((element) => !element.parentElement?.closest(selector));

      return elements.map((element, index) => {
        const marker = [
          element.getAttribute("data-message-author-role"),
          element.getAttribute("data-testid"),
          element.getAttribute("data-role"),
        ].filter(Boolean).join(" ").toLowerCase();
        const sender = /user|human/.test(marker) ? "user" : "AI";
        const thinkingElement = element.querySelector(
          '.thinking-container, [data-testid="thinking-content"]'
        );
        const thinking = thinkingElement
          ? this.extractFormattedContent(thinkingElement).trim()
          : "";
        const clone = element.cloneNode(true);
        clone.querySelectorAll(
          'button, [role="toolbar"], .action-buttons, .thinking-container, [data-testid="thinking-content"]'
        ).forEach((node) => node.remove());
        const contentElement = clone.querySelector(
          '.prose, .markdown, [data-testid="message-content"]'
        ) || clone;
        const extracted = this.extractFormattedContent(contentElement).trim();
        const content = sender === "user"
          ? this._stripInjectedContextPrefix(extracted)
          : extracted;
        if (!content && !thinking) return null;

        return {
          messageId: element.id
            ? `dom_${element.id}`
            : this.generateMessageId(sender, index),
          sender,
          content,
          thinking,
          position: index,
        };
      }).filter(Boolean);
    }

    async _fetchJson(path) {
      const resp = await fetch(path, {
        credentials: "same-origin",
        headers: { Accept: "application/json" },
      });

      if (!resp.ok) {
        throw new Error(`HTTP ${resp.status} for ${path}`);
      }
      return resp.json();
    }

    _buildMessagesFromResponses(data) {
      const responses = Array.isArray(data?.responses) ? data.responses : [];
      if (responses.length === 0) return [];

      const mainResponses = responses.filter((response) => !response?.threadParentId);
      if (mainResponses.length === 0) return [];

      const sortedResponses = [...mainResponses].sort((a, b) =>
        this._compareResponses(a, b)
      );
      const responseMap = new Map(
        sortedResponses
          .filter((response) => response?.responseId)
          .map((response) => [response.responseId, response])
      );

      const rid = new URL(location.href).searchParams.get("rid");
      let leaf = rid ? responseMap.get(rid) : null;
      if (!leaf) {
        leaf = sortedResponses[sortedResponses.length - 1] || null;
      }

      const branch = [];
      const seen = new Set();
      let current = leaf;

      while (current && current.responseId && !seen.has(current.responseId)) {
        branch.push(current);
        seen.add(current.responseId);
        current = current.parentResponseId
          ? responseMap.get(current.parentResponseId) || null
          : null;
      }

      const ordered = branch.length > 0 ? branch.reverse() : sortedResponses;

      return ordered
        .map((response, index) => this._normalizeResponse(response, index))
        .filter((message) => message && (message.content || message.thinking));
    }

    _normalizeResponse(response, index) {
      if (!response) return null;

      const sender = response.sender === "human" ? "user" : "AI";
      const extracted = this._buildContent(response).trim();
      const content = sender === "user"
        ? this._stripInjectedContextPrefix(extracted)
        : extracted;
      const thinking = this._normalizeThinking(response.thinkingTrace).trim();

      if (!content && !thinking) return null;

      return {
        messageId: response.responseId
          ? `resp_${response.responseId}`
          : this.generateMessageId(sender, index),
        sender,
        content,
        thinking,
        position: index,
      };
    }

    _buildContent(response) {
      const parts = [];
      if (typeof response.message === "string" && response.message.trim()) {
        parts.push(response.message.trim());
      }

      const imageUrls = Array.isArray(response.generatedImageUrls)
        ? response.generatedImageUrls.filter(Boolean)
        : [];
      if (imageUrls.length > 0) {
        parts.push(imageUrls.map((url) => `![grok image](${url})`).join("\n"));
      }

      return parts.join("\n\n");
    }

    _normalizeThinking(thinkingTrace) {
      if (!thinkingTrace) return "";
      if (typeof thinkingTrace === "string") {
        return thinkingTrace;
      }
      if (Array.isArray(thinkingTrace)) {
        return thinkingTrace
          .map((entry) =>
            typeof entry === "string" ? entry : JSON.stringify(entry, null, 2)
          )
          .join("\n\n");
      }
      if (typeof thinkingTrace === "object") {
        return JSON.stringify(thinkingTrace, null, 2);
      }
      return String(thinkingTrace);
    }

    _compareResponses(a, b) {
      const timeA = Date.parse(a?.createTime || "") || 0;
      const timeB = Date.parse(b?.createTime || "") || 0;
      if (timeA !== timeB) return timeA - timeB;
      return String(a?.responseId || "").localeCompare(String(b?.responseId || ""));
    }

    _extractNativeConversationId(url = location.href) {
      const match = url.match(/\/(?:c|chat|chat-v2|chat-v1)\/([^/?#]+)/);
      return match ? match[1] : null;
    }
  }

  const adapter = new GrokAdapter();
  adapter.start();
  window.__shadowWriteAdapter = adapter;
})();
