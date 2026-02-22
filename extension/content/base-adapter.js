/**
 * ShadowWrite — BaseShadowWriteAdapter
 *
 * Template-method base class for per-platform content scripts.
 * Subclasses implement 4 abstract methods; the base class handles
 * MutationObserver, debounce, URL watching, and HTTP delivery.
 *
 * Architecture inspired by Chat Memo's BasePlatformAdapter,
 * but simplified: no IndexedDB / background messaging — content scripts
 * POST directly to the local ShadowWrite HTTP service.
 */

"use strict";

(() => {
  // Guard against double-injection
  if (window.__shadowWriteAdapter) return;

  /* ------------------------------------------------------------------ */
  /*  Constants                                                          */
  /* ------------------------------------------------------------------ */

  const DEBOUNCE_DELAY = 1000;            // ms after last DOM change
  const URL_POLL_INTERVAL = 1000;         // ms between URL polls
  const DEFAULT_HOST = "http://127.0.0.1";
  const DEFAULT_PORT = 24601;
  const API_PATH = "/api/messages";

  /* ------------------------------------------------------------------ */
  /*  BaseShadowWriteAdapter                                             */
  /* ------------------------------------------------------------------ */

  class BaseShadowWriteAdapter {
    /**
     * @param {string} platform  e.g. "chatgpt", "deepseek", "gemini" …
     */
    constructor(platform) {
      this.platform = platform;
      this.pageUrl = location.href;

      // State
      this.currentConversationId = null;
      this.lastMessagesJson = "";          // serialised snapshot for diff
      this.lastKnownUrl = location.href.split("?")[0];

      // Observers / timers
      this.contentObserver = null;
      this.debounceTimer = null;
      this.urlCheckInterval = null;

      // Settings (loaded from chrome.storage.sync)
      this.settings = {
        host: DEFAULT_HOST,
        port: DEFAULT_PORT,
        autoCapture: true,
        enabled: true,
      };
    }

    /* ==============================================================
     *  Abstract methods — subclasses MUST override
     * ============================================================== */

    /**
     * Is the current URL a valid conversation page for this platform?
     * @param {string} url
     * @returns {boolean}
     */
    isValidConversationUrl(_url) {
      throw new Error("isValidConversationUrl() not implemented");
    }

    /**
     * Extract conversation metadata from the URL.
     * @param {string} url
     * @returns {{ conversationId: string, isNewConversation: boolean }}
     */
    extractConversationInfo(_url) {
      throw new Error("extractConversationInfo() not implemented");
    }

    /**
     * Extract all visible messages from the DOM.
     * @returns {Array<{
     *   messageId: string,
     *   sender: "user" | "AI",
     *   content: string,
     *   thinking: string,
     *   position: number
     * }>}
     */
    extractMessages() {
      throw new Error("extractMessages() not implemented");
    }

    /**
     * Determine whether a DOM node is (or contains) a message element.
     * Used to filter MutationObserver noise.
     * @param {Node} node
     * @returns {boolean}
     */
    isMessageElement(_node) {
      throw new Error("isMessageElement() not implemented");
    }

    /* ==============================================================
     *  Optional overrides
     * ============================================================== */

    /**
     * Try to extract a page/conversation title.
     * @returns {string | null}
     */
    extractTitle() {
      return document.title || null;
    }

    /* ==============================================================
     *  Lifecycle
     * ============================================================== */

    async start() {
      await this._loadSettings();
      if (!this.settings.enabled) {
        console.log(`[ShadowWrite] Adapter ${this.platform} disabled.`);
        return;
      }
      this._init();
      this._setupEventListeners();
      this._startUrlWatcher();
      console.log(`[ShadowWrite] ${this.platform} adapter started.`);
    }

    /* -------------------------------------------------------------- */
    /*  Initialisation                                                 */
    /* -------------------------------------------------------------- */

    _init() {
      const url = location.href;
      if (!this.isValidConversationUrl(url)) {
        console.log("[ShadowWrite] Not a valid conversation page, waiting…");
        return;
      }
      const info = this.extractConversationInfo(url);
      this.currentConversationId = info.conversationId;
      this.pageUrl = url;

      // Initial capture
      if (this.settings.autoCapture) {
        this._captureAndSend();
      }

      // Start observing DOM
      this._setupMutationObserver();
    }

    /* -------------------------------------------------------------- */
    /*  Settings                                                       */
    /* -------------------------------------------------------------- */

    async _loadSettings() {
      try {
        const result = await chrome.storage.sync.get({
          host: DEFAULT_HOST,
          port: DEFAULT_PORT,
          autoCapture: true,
          enabled: true,
        });
        Object.assign(this.settings, result);
      } catch {
        // Defaults already set
      }
    }

    /* -------------------------------------------------------------- */
    /*  MutationObserver                                                */
    /* -------------------------------------------------------------- */

    _setupMutationObserver() {
      if (this.contentObserver) {
        this.contentObserver.disconnect();
      }

      this.contentObserver = new MutationObserver((mutations) => {
        if (!this.settings.autoCapture) return;

        let hasRelevant = false;
        for (const mutation of mutations) {
          if (mutation.type === "childList") {
            for (const node of mutation.addedNodes) {
              if (node.nodeType === Node.ELEMENT_NODE && this.isMessageElement(node)) {
                hasRelevant = true;
                break;
              }
            }
          } else if (mutation.type === "characterData") {
            // Walk parents to check if inside a message
            let el = mutation.target.parentElement;
            while (el) {
              if (this.isMessageElement(el)) {
                hasRelevant = true;
                break;
              }
              el = el.parentElement;
            }
          }
          if (hasRelevant) break;
        }

        if (hasRelevant) {
          this._debouncedCapture();
        }
      });

      this.contentObserver.observe(document.body, {
        childList: true,
        subtree: true,
        characterData: true,
      });
    }

    _debouncedCapture() {
      if (this.debounceTimer) clearTimeout(this.debounceTimer);
      this.debounceTimer = setTimeout(() => {
        this._checkForChanges();
      }, DEBOUNCE_DELAY);
    }

    /**
     * Second-pass verification: re-extract messages, compare JSON snapshot.
     */
    _checkForChanges() {
      if (!this.currentConversationId) return;

      // Skip if user is editing a message
      if (document.querySelector("textarea:focus")) return;

      const messages = this.extractMessages();
      const json = JSON.stringify(messages);

      if (json !== this.lastMessagesJson) {
        this.lastMessagesJson = json;
        this._sendToService(messages);
      }
    }

    /**
     * Capture current messages and send without diff check (initial load).
     */
    _captureAndSend() {
      const messages = this.extractMessages();
      if (messages.length === 0) return;
      this.lastMessagesJson = JSON.stringify(messages);
      this._sendToService(messages);
    }

    /* -------------------------------------------------------------- */
    /*  URL Watcher                                                    */
    /* -------------------------------------------------------------- */

    _startUrlWatcher() {
      if (this.urlCheckInterval) clearInterval(this.urlCheckInterval);

      this.urlCheckInterval = setInterval(() => {
        const current = location.href.split("?")[0];
        if (current !== this.lastKnownUrl) {
          const oldUrl = this.lastKnownUrl;
          this.lastKnownUrl = current;
          console.log(`[ShadowWrite] URL changed: ${oldUrl} → ${current}`);
          this._handleUrlChange();
        }
      }, URL_POLL_INTERVAL);
    }

    _handleUrlChange() {
      // Reset state
      this.lastMessagesJson = "";
      if (this.contentObserver) {
        this.contentObserver.disconnect();
        this.contentObserver = null;
      }

      // Re-initialise for new conversation
      setTimeout(() => this._init(), 500);
    }

    /* -------------------------------------------------------------- */
    /*  Event Listeners                                                 */
    /* -------------------------------------------------------------- */

    _setupEventListeners() {
      // Settings update from popup / background
      chrome.runtime.onMessage.addListener((message) => {
        if (message.type === "settingsUpdated" && message.settings) {
          Object.assign(this.settings, message.settings);
          console.log("[ShadowWrite] Settings updated:", this.settings);
        }
      });
    }

    /* -------------------------------------------------------------- */
    /*  HTTP Delivery                                                   */
    /* -------------------------------------------------------------- */

    async _sendToService(messages) {
      const endpoint =
        `${this.settings.host}:${this.settings.port}${API_PATH}`;
      const payload = {
        platform: this.platform,
        url: this.pageUrl,
        conversationId: this.currentConversationId,
        title: this.extractTitle(),
        messages,
      };

      try {
        const resp = await fetch(endpoint, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        if (!resp.ok) {
          console.warn(
            `[ShadowWrite] Server responded ${resp.status}: ${await resp.text()}`
          );
        } else {
          // Notify UI of success
          window.dispatchEvent(
            new CustomEvent("shadowwrite-save-success", {
              detail: { count: messages.length },
            })
          );
        }
      } catch (err) {
        console.warn("[ShadowWrite] Cannot reach local service:", err.message);
        window.dispatchEvent(
          new CustomEvent("shadowwrite-save-error", {
            detail: { error: err.message },
          })
        );
      }
    }

    /* -------------------------------------------------------------- */
    /*  Utilities                                                       */
    /* -------------------------------------------------------------- */

    /**
     * Generate a stable message ID.
     */
    generateMessageId(sender, index) {
      return `msg_${sender}_position_${index}`;
    }

    /**
     * Extract visible text from an element, cleaning extra whitespace.
     */
    extractFormattedContent(element) {
      if (!element) return "";
      // Clone to avoid side effects
      const clone = element.cloneNode(true);
      // Remove hidden/script elements
      clone.querySelectorAll("script, style, .sr-only").forEach((el) => el.remove());
      return (clone.innerText || clone.textContent || "")
        .replace(/\n{3,}/g, "\n\n")
        .trim();
    }

    /**
     * Check if the user is currently editing inside an element.
     */
    isInEditMode(element) {
      if (!element) return false;
      const focused = element.querySelector("textarea:focus, [contenteditable]:focus");
      return !!focused;
    }

    /**
     * Cleanup on unload.
     */
    destroy() {
      if (this.contentObserver) this.contentObserver.disconnect();
      if (this.debounceTimer) clearTimeout(this.debounceTimer);
      if (this.urlCheckInterval) clearInterval(this.urlCheckInterval);
    }
  }

  // Export to window for adapter scripts
  window.BaseShadowWriteAdapter = BaseShadowWriteAdapter;
})();
