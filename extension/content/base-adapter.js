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
  const THROTTLE_INTERVAL = 3000;         // max wait during continuous DOM changes
  const URL_POLL_INTERVAL = 1000;         // ms between URL polls
  const DEFAULT_HOST = "127.0.0.1";
  const DEFAULT_PORT = 24601;
  const CONTEXT_INVALIDATED_MSG = "Extension context invalidated";

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
      this.isTracking = false;             // per-conversation tracking toggle
      this._epoch = 0;                     // incremented on URL change; stale timers check this

      // Observers / timers
      this.contentObserver = null;
      this.debounceTimer = null;
      this.throttleTimer = null;           // max-wait during streaming
      this.urlCheckInterval = null;

      // Settings (loaded from chrome.storage.sync)
      this.settings = {
        host: DEFAULT_HOST,
        port: DEFAULT_PORT,
        autoCapture: true,
        enabled: true,
      };

      this._contextInvalidated = false;

      // Context injection state
      this._contextMode = "off";         // "off" | "inject" | "auto-summary"
      this._contextContent = "";         // cached context file content
      this._submitHooked = false;        // whether hidden inject is active
      this._origSubmitHandler = null;    // ref for cleanup
    }

    /* ==============================================================
     *  Context validation
     * ============================================================== */

    /**
     * Check if an error indicates the extension context has been invalidated
     * (e.g. user reloaded the extension without refreshing the page).
     */
    _handleContextInvalidated(err) {
      if (err && String(err.message || err).includes(CONTEXT_INVALIDATED_MSG)) {
        if (!this._contextInvalidated) {
          this._contextInvalidated = true;
          console.warn("[ShadowWrite] Extension context invalidated — please refresh the page.");
          this.destroy();
          window.dispatchEvent(new CustomEvent("shadowwrite-context-invalidated"));
        }
        return true;
      }
      return false;
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
        // Notify UI — no active conversation
        window.dispatchEvent(new CustomEvent("shadowwrite-tracking-state", {
          detail: { tracking: false, hasConversation: false },
        }));
        return;
      }
      const info = this.extractConversationInfo(url);
      this.currentConversationId = info.conversationId;
      this.pageUrl = url;

      // Check if this conversation is tracked (persisted per-conversation state)
      this._loadTrackingState();
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
          contextMode: "off",
        });
        Object.assign(this.settings, result);
        this._contextMode = result.contextMode || "off";
      } catch {
        // Defaults already set
      }
    }

    /* -------------------------------------------------------------- */
    /*  Per-conversation Tracking                                      */
    /* -------------------------------------------------------------- */

    /**
     * Load tracking state for current conversation from chrome.storage.local.
     */
    async _loadTrackingState() {
      try {
        const data = await chrome.storage.local.get({ trackedConversations: {} });
        const entry = data.trackedConversations[this.currentConversationId];

        if (entry && !entry.disabled) {
          // Previously tracked — resume
          this.isTracking = true;
        } else if (!entry && this.settings.autoCapture) {
          // Never seen before + autoCapture ON → auto-enable
          console.log(`[ShadowWrite] autoCapture: auto-enabling tracking for ${this.currentConversationId}`);
          this.enableTracking();
          return; // enableTracking handles notification + observer
        } else {
          // Explicitly disabled or autoCapture OFF
          this.isTracking = false;
        }
      } catch {
        this.isTracking = false;
      }

      // Notify UI
      window.dispatchEvent(new CustomEvent("shadowwrite-tracking-state", {
        detail: {
          tracking: this.isTracking,
          hasConversation: true,
          conversationId: this.currentConversationId,
        },
      }));

      if (this.isTracking) {
        console.log(`[ShadowWrite] Tracking ON for ${this.currentConversationId}`);
        this._captureAndSend();
        this._setupMutationObserver();
      } else {
        console.log(`[ShadowWrite] Tracking OFF for ${this.currentConversationId} — click dot to enable`);
      }
    }

    /**
     * Enable tracking for the current conversation.
     */
    async enableTracking() {
      if (!this.currentConversationId || this._contextInvalidated) return;
      try {
        const data = await chrome.storage.local.get({ trackedConversations: {} });
        data.trackedConversations[this.currentConversationId] = {
          title: this.extractTitle(),
          platform: this.platform,
          url: this.pageUrl,
          enabledAt: new Date().toISOString(),
        };
        await chrome.storage.local.set(data);
      } catch (err) {
        if (this._handleContextInvalidated(err)) return;
        console.warn("[ShadowWrite] Failed to save tracking state:", err);
      }

      this.isTracking = true;
      // Clear snapshot cache so _captureAndSend always sends a fresh snapshot
      this.lastMessagesJson = "";
      window.dispatchEvent(new CustomEvent("shadowwrite-tracking-state", {
        detail: { tracking: true, hasConversation: true, conversationId: this.currentConversationId },
      }));

      // Start capturing
      this._captureAndSend();
      this._setupMutationObserver();

      // Setup hidden context inject if mode is "inject"
      if (this._contextMode === "inject") {
        this._setupHiddenInject();
      }
    }

    /**
     * Disable tracking for the current conversation.
     */
    async disableTracking() {
      if (!this.currentConversationId || this._contextInvalidated) return;
      try {
        const data = await chrome.storage.local.get({ trackedConversations: {} });
        // Store disabled marker so autoCapture won't re-enable this conversation
        data.trackedConversations[this.currentConversationId] = { disabled: true };
        await chrome.storage.local.set(data);
      } catch (err) {
        if (this._handleContextInvalidated(err)) return;
        console.warn("[ShadowWrite] Failed to save tracking state:", err);
      }

      this.isTracking = false;
      window.dispatchEvent(new CustomEvent("shadowwrite-tracking-state", {
        detail: { tracking: false, hasConversation: true, conversationId: this.currentConversationId },
      }));

      // Stop observer
      if (this.contentObserver) {
        this.contentObserver.disconnect();
        this.contentObserver = null;
      }
      this._teardownHiddenInject();
    }

    /* -------------------------------------------------------------- */
    /*  MutationObserver                                                */
    /* -------------------------------------------------------------- */

    _setupMutationObserver() {
      if (this.contentObserver) {
        this.contentObserver.disconnect();
      }

      this.contentObserver = new MutationObserver((mutations) => {
        if (!this.settings.autoCapture || !this.isTracking) return;

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
      const epoch = this._epoch;
      // Trailing edge — resets on every mutation
      if (this.debounceTimer) clearTimeout(this.debounceTimer);
      this.debounceTimer = setTimeout(() => this._flushCapture(epoch), DEBOUNCE_DELAY);

      // Throttle — fires at most every THROTTLE_INTERVAL even if
      // mutations keep coming (e.g. during AI streaming output).
      if (!this.throttleTimer) {
        this.throttleTimer = setTimeout(() => this._flushCapture(epoch), THROTTLE_INTERVAL);
      }
    }

    /**
     * Execute capture and clear both timers.
     */
    _flushCapture(epoch) {
      // Stale timer from a previous conversation — discard silently
      if (epoch !== undefined && epoch !== this._epoch) return;
      if (this.debounceTimer)  { clearTimeout(this.debounceTimer);  this.debounceTimer  = null; }
      if (this.throttleTimer) { clearTimeout(this.throttleTimer); this.throttleTimer = null; }
      this._checkForChanges();
    }

    /**
     * Second-pass verification: re-extract messages, compare JSON snapshot.
     */
    _checkForChanges() {
      if (!this.currentConversationId) return;

      // Skip if user is editing a message
      if (document.querySelector("textarea:focus")) return;

      const messages = this.extractMessages();
      if (messages.length === 0) return; // nothing to send

      const json = JSON.stringify(messages);

      if (json !== this.lastMessagesJson) {
        this.lastMessagesJson = json;
        this._sendToService(messages);
      }
    }

    /**
     * Capture current messages and send without diff check (initial load).
     * Retries up to 3 times if DOM is not ready yet.
     */
    _captureAndSend(retries = 0, epoch) {
      // Capture epoch on first call so retries are bound to this conversation
      if (epoch === undefined) epoch = this._epoch;
      // Stale retry from a previous conversation — discard
      if (epoch !== this._epoch) return;

      const messages = this.extractMessages();
      if (messages.length === 0) {
        if (retries < 3) {
          const delay = (retries + 1) * 1500; // 1.5s, 3s, 4.5s
          console.log(
            `[ShadowWrite] No messages found yet, retry ${retries + 1}/3 in ${delay}ms…`
          );
          setTimeout(() => this._captureAndSend(retries + 1, epoch), delay);
        } else {
          console.log("[ShadowWrite] No messages after 3 retries, waiting for MutationObserver.");
        }
        return;
      }
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
      // Bump epoch — all pending timers from the old conversation become stale
      this._epoch++;

      // Reset state
      this.lastMessagesJson = "";
      this.isTracking = false;
      if (this.contentObserver) {
        this.contentObserver.disconnect();
        this.contentObserver = null;
      }
      // Clear pending capture timers
      if (this.debounceTimer)  { clearTimeout(this.debounceTimer);  this.debounceTimer  = null; }
      if (this.throttleTimer) { clearTimeout(this.throttleTimer); this.throttleTimer = null; }

      // Immediately update dot to OFF so user sees instant feedback
      window.dispatchEvent(new CustomEvent("shadowwrite-tracking-state", {
        detail: { tracking: false, hasConversation: false },
      }));

      // Re-initialise for new conversation (allow DOM to settle)
      setTimeout(() => this._init(), 500);
    }

    /* -------------------------------------------------------------- */
    /*  Event Listeners                                                 */
    /* -------------------------------------------------------------- */

    _setupEventListeners() {
      // Settings update from popup / background
      chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
        switch (message.type) {
          case "settingsUpdated":
          case "settingsChanged":
            if (message.settings) {
              Object.assign(this.settings, message.settings);
              console.log("[ShadowWrite] Settings updated:", this.settings);
            }
            break;

          // Popup queries tracking state for the "当前对话追踪" toggle
          case "getTrackingState":
            sendResponse({
              isTracking: this.isTracking,
              hasConversation: !!this.currentConversationId,
              conversationId: this.currentConversationId,
            });
            break;

          // Popup sets tracking on/off (same as clicking the dot)
          case "setTracking":
            if (message.enabled) {
              this.enableTracking();
            } else {
              this.disableTracking();
            }
            sendResponse({ ok: true });
            break;

          // Context mode changed from popup
          case "setContextMode": {
            const newMode = message.mode || "off";
            this._contextMode = newMode;
            if (newMode === "inject" && this.isTracking) {
              this._setupHiddenInject();
            } else {
              this._teardownHiddenInject();
            }
            sendResponse({ ok: true });
            break;
          }

          // Popup requests visible context injection
          case "injectContextVisible":
            (async () => {
              const ok = await this.injectContextVisible();
              sendResponse({ ok });
            })();
            return true; // async response

          // Popup requests context mode info
          case "getContextMode":
            sendResponse({
              mode: this._contextMode,
              hasConversation: !!this.currentConversationId,
            });
            break;
        }
      });
    }

    /* -------------------------------------------------------------- */
    /*  Context Injection (Scheme 2 & 4)                                */
    /* -------------------------------------------------------------- */

    // ── Context-update marker regexes (same as CLI) ────────────

    static _CONTEXT_BLOCK_RE =
      /<!--\s*context-update-start\s*-->\s*?\n([\s\S]*?)\n\s*?<!--\s*context-update-end\s*-->/g;
    static _CONTEXT_INLINE_RE =
      /<!--\s*context-update:\s*([\s\S]*?)\s*-->/g;

    /**
     * Build the context prefix to prepend to user messages.
     * Mirrors CLI's build_system_prompt().
     */
    _buildContextPrefix() {
      const parts = [];
      if (this._contextContent) {
        parts.push(
          `=== PROJECT CONTEXT ===\n${this._contextContent}\n=== END PROJECT CONTEXT ===`
        );
      }
      parts.push(
        "IMPORTANT: A persistent context file is attached to this session.\n" +
        "This file is the LONG-TERM MEMORY that survives context window truncation.\n" +
        "When the conversation grows long, early messages will be lost, but this\n" +
        "file's content will always be re-injected.\n\n" +
        "Two marker formats are available:\n\n" +
        "1) STRUCTURED BLOCK (preferred for rich content):\n" +
        "<!-- context-update-start -->\n" +
        "## Section Heading\n" +
        "- Specific detail 1\n" +
        "<!-- context-update-end -->\n\n" +
        "2) INLINE note:\n" +
        "<!-- context-update: concise description -->\n\n" +
        "Only emit markers for genuinely important information that should be preserved long-term."
      );
      return parts.join("\n\n");
    }

    /**
     * Fetch the latest context content from server.
     */
    async _fetchContext() {
      if (!this.currentConversationId) return "";
      try {
        const resp = await chrome.runtime.sendMessage({
          type: "getContext",
          conversationId: this.currentConversationId,
        });
        if (resp?.success && resp.data) {
          this._contextContent = resp.data.content || "";
          return this._contextContent;
        }
      } catch (err) {
        if (this._handleContextInvalidated(err)) return "";
        console.warn("[ShadowWrite] Failed to fetch context:", err.message);
      }
      return "";
    }

    /**
     * Extract context-update markers from AI response content
     * and post incremental updates to the server.
     */
    async _extractAndSaveContextUpdates(messages) {
      if (this._contextMode === "off") return;
      if (!this.currentConversationId) return;

      const blocks = [];
      const inlines = [];

      for (const msg of messages) {
        if (msg.sender !== "AI") continue;
        const text = msg.content || "";

        // Block markers
        let match;
        const blockRe = new RegExp(BaseShadowWriteAdapter._CONTEXT_BLOCK_RE.source, "g");
        while ((match = blockRe.exec(text)) !== null) {
          blocks.push(match[1]);
        }

        // Inline markers (exclude those inside blocks)
        const textWithoutBlocks = text.replace(BaseShadowWriteAdapter._CONTEXT_BLOCK_RE, "");
        const inlineRe = new RegExp(BaseShadowWriteAdapter._CONTEXT_INLINE_RE.source, "g");
        while ((match = inlineRe.exec(textWithoutBlocks)) !== null) {
          inlines.push(match[1]);
        }
      }

      if (blocks.length === 0 && inlines.length === 0) return;

      console.log(`[ShadowWrite] Extracted ${blocks.length} blocks, ${inlines.length} inline context updates`);

      try {
        await chrome.runtime.sendMessage({
          type: "postContext",
          payload: {
            conversationId: this.currentConversationId,
            blocks,
            inlines,
          },
        });
      } catch (err) {
        if (this._handleContextInvalidated(err)) return;
        console.warn("[ShadowWrite] Failed to save context updates:", err.message);
      }
    }

    /* ── Hidden inject: hook the submit button/Enter key ─────── */

    /**
     * Override in subclasses to provide platform-specific input element.
     * Return the chat input element (textarea / contenteditable div), or null.
     */
    getInputElement() {
      return null;
    }

    /**
     * Override in subclasses to provide platform-specific submit trigger.
     * Return the send button element, or null.
     */
    getSubmitButton() {
      return null;
    }

    /**
     * Set text in the platform input box (handles both textarea and contenteditable).
     */
    _setInputText(el, text) {
      if (!el) return;
      if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") {
        // For React-controlled textareas, we need to use native input setter
        const nativeSetter = Object.getOwnPropertyDescriptor(
          window.HTMLTextAreaElement.prototype, "value"
        )?.set || Object.getOwnPropertyDescriptor(
          window.HTMLInputElement.prototype, "value"
        )?.set;
        if (nativeSetter) {
          nativeSetter.call(el, text);
        } else {
          el.value = text;
        }
        el.dispatchEvent(new Event("input", { bubbles: true }));
      } else {
        // contenteditable div (ChatGPT uses ProseMirror)
        el.textContent = text;
        el.dispatchEvent(new Event("input", { bubbles: true }));
      }
    }

    /**
     * Get current text from the platform input box.
     */
    _getInputText(el) {
      if (!el) return "";
      if (el.tagName === "TEXTAREA" || el.tagName === "INPUT") {
        return el.value;
      }
      return el.textContent || el.innerText || "";
    }

    /**
     * Setup hidden inject: intercept Enter key to prepend context before sending.
     */
    _setupHiddenInject() {
      if (this._submitHooked) return;
      this._submitHooked = true;

      this._origSubmitHandler = async (e) => {
        // Only intercept Enter without Shift (most platforms send on Enter)
        if (e.key !== "Enter" || e.shiftKey) return;

        const inputEl = this.getInputElement();
        if (!inputEl || inputEl !== e.target) return;
        if (this._contextMode !== "inject") return;

        const userText = this._getInputText(inputEl).trim();
        if (!userText) return;

        // Fetch latest context
        await this._fetchContext();
        if (!this._contextContent) return; // nothing to inject

        // Prevent the original send
        e.preventDefault();
        e.stopImmediatePropagation();

        const prefix = this._buildContextPrefix();
        const fullText = `${prefix}\n\n---\n\n${userText}`;

        // Set combined text and trigger send
        this._setInputText(inputEl, fullText);

        // Small delay to let the framework pick up the change, then click send
        await new Promise((r) => setTimeout(r, 50));
        const sendBtn = this.getSubmitButton();
        if (sendBtn) {
          sendBtn.click();
        } else {
          // Fallback: dispatch Enter event without our handler intercepting it
          this._submitHooked = false; // temporarily unhook
          inputEl.dispatchEvent(new KeyboardEvent("keydown", {
            key: "Enter", code: "Enter", keyCode: 13, bubbles: true
          }));
          this._submitHooked = true;
        }
      };

      // Use capture phase to intercept before platform handlers
      document.addEventListener("keydown", this._origSubmitHandler, true);
      console.log("[ShadowWrite] Hidden context inject hooked");
    }

    _teardownHiddenInject() {
      if (!this._submitHooked || !this._origSubmitHandler) return;
      document.removeEventListener("keydown", this._origSubmitHandler, true);
      this._submitHooked = false;
      this._origSubmitHandler = null;
    }

    /**
     * Visible inject: fill input box with context prefix + cursor placeholder.
     * Called from popup "注入上下文" button.
     */
    async injectContextVisible() {
      await this._fetchContext();
      const prefix = this._buildContextPrefix();
      const inputEl = this.getInputElement();
      if (!inputEl) {
        console.warn("[ShadowWrite] Cannot find input element for context injection");
        return false;
      }
      const current = this._getInputText(inputEl).trim();
      const text = current
        ? `${prefix}\n\n---\n\n${current}`
        : `${prefix}\n\n---\n\n`;
      this._setInputText(inputEl, text);
      inputEl.focus();
      return true;
    }

    /* -------------------------------------------------------------- */
    /*  HTTP Delivery                                                   */
    /* -------------------------------------------------------------- */

    async _sendToService(messages) {
      if (!messages || messages.length === 0) return; // final guard

      const payload = {
        platform: this.platform,
        url: this.pageUrl,
        conversationId: this.currentConversationId,
        title: this.extractTitle(),
        messages,
      };

      // ChatGPT project conversations: include project field
      const projectInfo = this.extractProject?.();
      if (projectInfo) {
        payload.project = projectInfo.project;
        payload.title   = projectInfo.title;
      }

      // Include per-conversation output directory if set
      try {
        const data = await chrome.storage.local.get({ convOutputDirs: {} });
        const customDir = data.convOutputDirs[this.currentConversationId];
        if (customDir) {
          payload.outputDir = customDir;
        }
      } catch { /* ignore */ }

      try {
        // Relay through background service worker to bypass page CSP.
        const resp = await chrome.runtime.sendMessage({
          type: "sendToServer",
          host: this.settings.host,
          port: this.settings.port,
          payload,
        });

        if (resp && resp.ok) {
          console.log(
            `[ShadowWrite] Saved ${messages.length} messages → ${resp.body}`
          );
          window.dispatchEvent(
            new CustomEvent("shadowwrite-save-success", {
              detail: { count: messages.length },
            })
          );
          // Extract context-update markers from AI responses
          this._extractAndSaveContextUpdates(messages);
        } else {
          const errMsg = resp?.error || `Server responded ${resp?.status}: ${resp?.body?.substring(0, 120)}`;
          console.warn(`[ShadowWrite] ${errMsg}`);
          window.dispatchEvent(
            new CustomEvent("shadowwrite-save-error", {
              detail: { error: errMsg },
            })
          );
        }
      } catch (err) {
        if (this._handleContextInvalidated(err)) return;
        console.warn("[ShadowWrite] Cannot reach background:", err.message);
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
     * Extract content from an element and convert HTML to Markdown,
     * preserving headings, bold/italic, lists, code blocks, tables, etc.
     */
    extractFormattedContent(element) {
      if (!element) return "";
      // Clone to avoid side effects
      const clone = element.cloneNode(true);
      // Remove hidden/script elements
      clone
        .querySelectorAll("script, style, .sr-only")
        .forEach((el) => el.remove());
      return this._htmlToMarkdown(clone);
    }

    /**
     * Lightweight HTML-to-Markdown converter.
     * Handles: headings, bold, italic, strikethrough, inline code,
     * fenced code blocks, links, images, lists (nested), paragraphs,
     * blockquotes, horizontal rules, and tables.
     */
    _htmlToMarkdown(root) {
      const self = this;

      function processNode(node) {
        if (node.nodeType === Node.TEXT_NODE) {
          return node.textContent;
        }
        if (node.nodeType !== Node.ELEMENT_NODE) return "";

        const tag = node.tagName.toLowerCase();
        const children = () =>
          Array.from(node.childNodes).map(processNode).join("");

        switch (tag) {
          // --- Headings ---
          case "h1":
          case "h2":
          case "h3":
          case "h4":
          case "h5":
          case "h6": {
            const level = parseInt(tag[1]);
            return (
              "\n\n" + "#".repeat(level) + " " + children().trim() + "\n\n"
            );
          }
          // --- Inline formatting ---
          case "strong":
          case "b":
            return "**" + children() + "**";
          case "em":
          case "i":
            return "*" + children() + "*";
          case "del":
          case "s":
            return "~~" + children() + "~~";
          case "mark":
            return "==" + children() + "==";
          case "sub":
            return "~" + children() + "~";
          case "sup":
            return "^" + children() + "^";
          // --- Code ---
          case "code": {
            // If inside <pre>, handled by 'pre' case
            if (
              node.parentElement &&
              node.parentElement.tagName.toLowerCase() === "pre"
            ) {
              return node.textContent;
            }
            return "`" + node.textContent + "`";
          }
          case "pre": {
            const codeEl = node.querySelector("code");
            const lang = codeEl
              ? (codeEl.className.match(/(?:language|lang|highlight)-(\S+)/i)?.[1] || "")
              : "";
            const code = codeEl ? codeEl.textContent : node.textContent;
            return (
              "\n\n```" + lang + "\n" + code.replace(/\n$/, "") + "\n```\n\n"
            );
          }
          // --- Links & images ---
          case "a": {
            const href = node.getAttribute("href") || "";
            const text = children();
            return href ? "[" + text + "](" + href + ")" : text;
          }
          case "img": {
            const alt = node.getAttribute("alt") || "";
            const src = node.getAttribute("src") || "";
            return "![" + alt + "](" + src + ")";
          }
          // --- Lists ---
          case "ul":
          case "ol":
            return "\n" + self._processListItems(node, "") + "\n";
          // --- Block elements ---
          case "p":
            return "\n\n" + children().trim() + "\n\n";
          case "br":
            return "\n";
          case "hr":
            return "\n\n---\n\n";
          case "blockquote": {
            const content = children().trim();
            return (
              "\n\n" +
              content
                .split("\n")
                .map((l) => "> " + l)
                .join("\n") +
              "\n\n"
            );
          }
          // --- Tables ---
          case "table":
            return "\n\n" + self._processTable(node) + "\n\n";
          // --- Math (KaTeX / MathJax) ---
          case "math": {
            const tex =
              node.getAttribute("alttext") ||
              node.getAttribute("data-latex") ||
              node.textContent;
            return "$" + tex + "$";
          }
          // --- Pass-through containers ---
          default:
            return children();
        }
      }

      return processNode(root).replace(/\n{3,}/g, "\n\n").trim();
    }

    /**
     * Process <ul>/<ol> items with proper nesting and indentation.
     */
    _processListItems(listNode, indent) {
      const isOrdered = listNode.tagName.toLowerCase() === "ol";
      let result = "";
      let idx = 0;

      for (const child of listNode.children) {
        if (child.tagName.toLowerCase() !== "li") continue;
        idx++;
        const prefix = isOrdered ? `${idx}. ` : "- ";
        const continuation = " ".repeat(prefix.length);

        let inlineContent = "";
        let nestedBlocks = "";

        for (const liChild of child.childNodes) {
          if (liChild.nodeType === Node.ELEMENT_NODE) {
            const liTag = liChild.tagName.toLowerCase();
            if (liTag === "ul" || liTag === "ol") {
              nestedBlocks += this._processListItems(
                liChild,
                indent + continuation
              );
              continue;
            }
          }
          inlineContent += this._htmlToMarkdown(liChild);
        }

        // First line gets the bullet, subsequent lines get continuation indent
        const trimmed = inlineContent.trim();
        if (trimmed) {
          const lines = trimmed.split("\n");
          result += indent + prefix + lines[0];
          for (let i = 1; i < lines.length; i++) {
            if (lines[i].trim()) {
              result += "\n" + indent + continuation + lines[i];
            }
          }
          result += "\n";
        }

        if (nestedBlocks) {
          result += nestedBlocks;
        }
      }

      return result;
    }

    /**
     * Process <table> into Markdown pipe table.
     */
    _processTable(tableNode) {
      const rows = tableNode.querySelectorAll("tr");
      if (rows.length === 0) return "";

      let result = "";
      let isFirst = true;

      for (const row of rows) {
        const cells = Array.from(row.querySelectorAll("th, td"));
        const cellTexts = cells.map((c) =>
          this._htmlToMarkdown(c).trim().replace(/\|/g, "\\|").replace(/\n/g, " ")
        );
        result += "| " + cellTexts.join(" | ") + " |\n";

        if (isFirst) {
          result += "| " + cellTexts.map(() => "---").join(" | ") + " |\n";
          isFirst = false;
        }
      }

      return result;
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
      if (this.throttleTimer) clearTimeout(this.throttleTimer);
      if (this.urlCheckInterval) clearInterval(this.urlCheckInterval);
    }
  }

  // Export to window for adapter scripts
  window.BaseShadowWriteAdapter = BaseShadowWriteAdapter;
})();
