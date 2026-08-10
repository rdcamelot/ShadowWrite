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
  const SEND_RETRY_DELAY = 5000;          // retry failed local-server sends
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
      this.lastMessagesSignature = "";     // compact snapshot signature for diff
      this.lastKnownUrl = this.getUrlKey(location.href);
      this.isTracking = false;             // per-conversation tracking toggle
      this._epoch = 0;                     // incremented on URL change; stale timers check this

      // Observers / timers
      this.contentObserver = null;
      this.observerRoot = null;
      this.observerRetryTimer = null;
      this.debounceTimer = null;
      this.throttleTimer = null;           // max-wait during streaming
      this.urlCheckInterval = null;
      this.initialCaptureHandle = null;
      this.initialCaptureUsesIdle = false;
      this.captureIdleHandle = null;
      this.captureIdleUsesIdle = false;
      this.sendInFlight = false;
      this.queuedSnapshot = null;
      this.failedSnapshot = null;
      this.sendRetryTimer = null;
      this._formatCache = new WeakMap();

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
      this._origSubmitClickHandler = null;
      this._contextSubmitInFlight = false;
      this._bypassContextInject = false;
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

    /**
     * Return the smallest stable DOM subtree that contains chat messages.
     * Subclasses can narrow this further (for example Gemini's #chat-history).
     */
    getObserverRoot() {
      return document.querySelector("main") || document.body;
    }

    /**
     * Return the URL identity used by the SPA watcher. Most platforms keep
     * conversation identity in the path and can ignore query parameters.
     */
    getUrlKey(url) {
      return String(url).split("?")[0];
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
        this.settings.contextMode = "off";
        this._contextMode = "off";
        if (result.contextMode !== "off") {
          await chrome.storage.sync.set({ contextMode: "off" });
        }
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
        const trackedConversations = this._normalizeTrackedConversations(
          data.trackedConversations
        );
        const entry = trackedConversations[this.currentConversationId];

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
        this._scheduleInitialCapture();
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
        const trackedConversations = this._normalizeTrackedConversations(
          data.trackedConversations
        );
        trackedConversations[this.currentConversationId] = {
          title: this.extractTitle(),
          platform: this.platform,
          url: this.pageUrl,
          enabledAt: new Date().toISOString(),
        };
        await chrome.storage.local.set({ trackedConversations });
      } catch (err) {
        if (this._handleContextInvalidated(err)) return;
        console.warn("[ShadowWrite] Failed to save tracking state:", err);
      }

      this.isTracking = true;
      // Clear snapshot cache so _captureAndSend always sends a fresh snapshot
      this.lastMessagesSignature = "";
      window.dispatchEvent(new CustomEvent("shadowwrite-tracking-state", {
        detail: { tracking: true, hasConversation: true, conversationId: this.currentConversationId },
      }));

      // Start capturing
      this._scheduleInitialCapture();
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
        const trackedConversations = this._normalizeTrackedConversations(
          data.trackedConversations
        );
        // Store disabled marker so autoCapture won't re-enable this conversation
        trackedConversations[this.currentConversationId] = { disabled: true };
        await chrome.storage.local.set({ trackedConversations });
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
      this.observerRoot = null;
      this.queuedSnapshot = null;
      this._clearSendRetry();
      if (this.observerRetryTimer) {
        clearTimeout(this.observerRetryTimer);
        this.observerRetryTimer = null;
      }
      this._cancelInitialCapture();
      this._cancelPendingChangeCheck();
      this._teardownHiddenInject();
    }

    /* -------------------------------------------------------------- */
    /*  MutationObserver                                                */
    /* -------------------------------------------------------------- */

    _normalizeTrackedConversations(value) {
      return value && typeof value === "object" && !Array.isArray(value)
        ? value
        : {};
    }

    _setupMutationObserver() {
      if (this.contentObserver) {
        this.contentObserver.disconnect();
      }
      if (this.observerRetryTimer) {
        clearTimeout(this.observerRetryTimer);
        this.observerRetryTimer = null;
      }

      const root = this.getObserverRoot();
      if (!root) {
        this._scheduleObserverRetry();
        return;
      }
      this.observerRoot = root;

      this.contentObserver = new MutationObserver((mutations) => {
        // autoCapture controls whether new conversations are enabled
        // automatically; it must not suspend an already tracked conversation.
        if (!this.isTracking) return;

        let hasRelevant = false;
        for (const mutation of mutations) {
          this._invalidateFormatCacheForMutation(mutation);

          if (mutation.type === "childList") {
            for (const node of mutation.addedNodes) {
              const candidate = node.nodeType === Node.ELEMENT_NODE
                ? node
                : mutation.target;
              if (candidate && this.isMessageElement(candidate)) {
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
          } else if (
            mutation.type === "attributes" &&
            this.isMessageElement(mutation.target)
          ) {
            hasRelevant = true;
          }
          if (hasRelevant) break;
        }

        if (hasRelevant) {
          this._debouncedCapture();
        }
      });

      this.contentObserver.observe(root, {
        childList: true,
        subtree: true,
        characterData: true,
        attributes: true,
        attributeFilter: ["href", "src", "alt"],
      });
    }

    _invalidateFormatCacheForMutation(mutation) {
      let element = mutation.target?.nodeType === Node.ELEMENT_NODE
        ? mutation.target
        : mutation.target?.parentElement;

      while (element) {
        this._formatCache.delete(element);
        if (element === this.observerRoot) break;
        element = element.parentElement;
      }
    }

    _scheduleObserverRetry() {
      if (!this.isTracking || this.observerRetryTimer) return;
      const epoch = this._epoch;
      this.observerRetryTimer = setTimeout(() => {
        this.observerRetryTimer = null;
        if (this.isTracking && epoch === this._epoch) {
          this._setupMutationObserver();
        }
      }, 1000);
    }

    _scheduleInitialCapture() {
      this._cancelInitialCapture();
      const epoch = this._epoch;
      const run = () => {
        this.initialCaptureHandle = null;
        if (this.isTracking && epoch === this._epoch) {
          this._captureAndSend(0, epoch);
        }
      };

      if (typeof window.requestIdleCallback === "function") {
        this.initialCaptureUsesIdle = true;
        this.initialCaptureHandle = window.requestIdleCallback(run, { timeout: 2500 });
      } else {
        this.initialCaptureUsesIdle = false;
        this.initialCaptureHandle = setTimeout(run, 500);
      }
    }

    _cancelInitialCapture() {
      if (this.initialCaptureHandle === null) return;
      if (
        this.initialCaptureUsesIdle &&
        typeof window.cancelIdleCallback === "function"
      ) {
        window.cancelIdleCallback(this.initialCaptureHandle);
      } else {
        clearTimeout(this.initialCaptureHandle);
      }
      this.initialCaptureHandle = null;
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
      this._scheduleChangeCheck(epoch);
    }

    _scheduleChangeCheck(epoch) {
      if (this.captureIdleHandle !== null) return;

      const run = () => {
        this.captureIdleHandle = null;
        if (!this.isTracking) return;
        if (epoch !== undefined && epoch !== this._epoch) return;
        this._checkForChanges();
      };

      if (typeof window.requestIdleCallback === "function") {
        this.captureIdleUsesIdle = true;
        this.captureIdleHandle = window.requestIdleCallback(run, { timeout: 1200 });
      } else {
        this.captureIdleUsesIdle = false;
        this.captureIdleHandle = setTimeout(run, 0);
      }
    }

    _cancelPendingChangeCheck() {
      if (this.captureIdleHandle === null) return;
      if (
        this.captureIdleUsesIdle &&
        typeof window.cancelIdleCallback === "function"
      ) {
        window.cancelIdleCallback(this.captureIdleHandle);
      } else {
        clearTimeout(this.captureIdleHandle);
      }
      this.captureIdleHandle = null;
    }

    /**
     * Second-pass verification: re-extract messages, compare JSON snapshot.
     */
    _checkForChanges() {
      if (!this.currentConversationId) return;

      // Skip only when an existing message is being edited. The composer often
      // stays focused while the model streams, and must not block syncing.
      if (this.isEditingMessage()) return;

      const messages = this.extractMessages();
      if (messages.length === 0) return; // nothing to send

      const signature = this._buildSnapshotSignature(messages);

      if (signature !== this.lastMessagesSignature) {
        this.lastMessagesSignature = signature;
        this._cancelInitialCapture();
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
      if (!this.isTracking || epoch !== this._epoch) return;

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
      this.lastMessagesSignature = this._buildSnapshotSignature(messages);
      this._sendToService(messages);
    }

    /* -------------------------------------------------------------- */
    /*  URL Watcher                                                    */
    /* -------------------------------------------------------------- */

    _startUrlWatcher() {
      if (this.urlCheckInterval) clearInterval(this.urlCheckInterval);

      this.urlCheckInterval = setInterval(() => {
        const current = this.getUrlKey(location.href);
        if (current !== this.lastKnownUrl) {
          const oldUrl = this.lastKnownUrl;
          this.lastKnownUrl = current;
          console.log(`[ShadowWrite] URL changed: ${oldUrl} → ${current}`);
          this._handleUrlChange();
        } else {
          this._refreshObserverRootIfNeeded();
        }
      }, URL_POLL_INTERVAL);
    }

    _refreshObserverRootIfNeeded() {
      if (!this.isTracking || !this.contentObserver) return;

      const preferredRoot = this.getObserverRoot();
      if (!preferredRoot) {
        if (!this.observerRoot || !this.observerRoot.isConnected) {
          this._setupMutationObserver();
        }
        return;
      }

      if (preferredRoot !== this.observerRoot || !this.observerRoot?.isConnected) {
        this._setupMutationObserver();
      }
    }

    _handleUrlChange() {
      // Bump epoch — all pending timers from the old conversation become stale
      this._epoch++;

      // Reset state
      this.lastMessagesSignature = "";
      this._formatCache = new WeakMap();
      this._contextContent = "";
      this.isTracking = false;
      if (this.contentObserver) {
        this.contentObserver.disconnect();
        this.contentObserver = null;
      }
      this.observerRoot = null;
      // Clear pending capture timers
      if (this.debounceTimer)  { clearTimeout(this.debounceTimer);  this.debounceTimer  = null; }
      if (this.throttleTimer) { clearTimeout(this.throttleTimer); this.throttleTimer = null; }
      if (this.observerRetryTimer) { clearTimeout(this.observerRetryTimer); this.observerRetryTimer = null; }
      this._cancelInitialCapture();
      this._cancelPendingChangeCheck();
      this.queuedSnapshot = null;
      this._clearSendRetry();

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
              this.settings.contextMode = "off";
              this._contextMode = "off";
              this._teardownHiddenInject();
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
            this.settings.contextMode = "off";
            this._contextMode = "off";
            this._teardownHiddenInject();
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
      const conversationId = this.currentConversationId;
      if (!conversationId) {
        this._contextContent = "";
        return "";
      }
      try {
        const resp = await chrome.runtime.sendMessage({
          type: "getContext",
          conversationId,
        });
        if (conversationId !== this.currentConversationId) return "";
        if (resp?.success && resp.data) {
          this._contextContent = resp.data.content || "";
          return this._contextContent;
        }
      } catch (err) {
        if (this._handleContextInvalidated(err)) return "";
        console.warn("[ShadowWrite] Failed to fetch context:", err.message);
      }
      this._contextContent = "";
      return "";
    }

    /**
     * Extract context-update markers from AI response content
     * and post incremental updates to the server.
     */
    async _extractAndSaveContextUpdates(messages, conversationId = this.currentConversationId) {
      if (this._contextMode === "off") return;
      if (!conversationId) return;

      const updates = [];

      for (const msg of messages) {
        if (msg.sender !== "AI") continue;
        const text = msg.content || "";

        // Block markers
        let match;
        let markerIndex = 0;
        const blockRe = new RegExp(BaseShadowWriteAdapter._CONTEXT_BLOCK_RE.source, "g");
        while ((match = blockRe.exec(text)) !== null) {
          const value = match[1].trim();
          updates.push({
            kind: "block",
            value,
            key: this._contextUpdateKey(msg.messageId, "block", markerIndex++, value),
          });
        }

        // Inline markers (exclude those inside blocks)
        const textWithoutBlocks = text.replace(BaseShadowWriteAdapter._CONTEXT_BLOCK_RE, "");
        markerIndex = 0;
        const inlineRe = new RegExp(BaseShadowWriteAdapter._CONTEXT_INLINE_RE.source, "g");
        while ((match = inlineRe.exec(textWithoutBlocks)) !== null) {
          const value = match[1].trim();
          updates.push({
            kind: "inline",
            value,
            key: this._contextUpdateKey(msg.messageId, "inline", markerIndex++, value),
          });
        }
      }

      if (updates.length === 0) return;

      const storageKey = `contextUpdateHistory:${conversationId}`;
      let seen = new Set();
      try {
        const stored = await chrome.storage.local.get({ [storageKey]: [] });
        seen = new Set(Array.isArray(stored[storageKey]) ? stored[storageKey] : []);
      } catch (err) {
        if (this._handleContextInvalidated(err)) return;
        console.warn("[ShadowWrite] Failed to load context update history:", err.message);
      }

      const pending = updates.filter((update) => !seen.has(update.key));
      if (pending.length === 0) return;

      const blocks = pending
        .filter((update) => update.kind === "block")
        .map((update) => update.value);
      const inlines = pending
        .filter((update) => update.kind === "inline")
        .map((update) => update.value);

      console.log(`[ShadowWrite] Extracted ${blocks.length} blocks, ${inlines.length} inline context updates`);

      try {
        const resp = await chrome.runtime.sendMessage({
          type: "postContext",
          payload: {
            conversationId,
            blocks,
            inlines,
          },
        });
        if (!resp?.success) {
          console.warn("[ShadowWrite] Context update was not saved:", resp?.error || resp?.data?.error);
          return;
        }

        pending.forEach((update) => seen.add(update.key));
        await chrome.storage.local.set({
          [storageKey]: Array.from(seen).slice(-1000),
        });
      } catch (err) {
        if (this._handleContextInvalidated(err)) return;
        console.warn("[ShadowWrite] Failed to save context updates:", err.message);
      }
    }

    _contextUpdateKey(messageId, kind, index, value) {
      let hash = 0x811c9dc5;
      const text = String(value || "");
      for (let i = 0; i < text.length; i++) {
        hash = Math.imul(hash ^ text.charCodeAt(i), 0x01000193);
      }
      return `${messageId || "unknown"}:${kind}:${index}:${text.length}:${hash >>> 0}`;
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

      this._origSubmitHandler = (e) => {
        // Only intercept Enter without Shift (most platforms send on Enter)
        if (e.key !== "Enter" || e.shiftKey || e.isComposing) return;
        if (this._bypassContextInject) return;

        const inputEl = this.getInputElement();
        if (!inputEl || inputEl !== e.target) return;
        this._interceptContextSubmit(e, inputEl);
      };

      this._origSubmitClickHandler = (e) => {
        if (this._bypassContextInject || e.button !== 0) return;
        const sendBtn = this.getSubmitButton();
        if (!sendBtn || (e.target !== sendBtn && !sendBtn.contains(e.target))) return;
        const inputEl = this.getInputElement();
        if (!inputEl) return;
        this._interceptContextSubmit(e, inputEl);
      };

      document.addEventListener("keydown", this._origSubmitHandler, true);
      document.addEventListener("click", this._origSubmitClickHandler, true);
      console.log("[ShadowWrite] Hidden context inject hooked");
    }

    _interceptContextSubmit(event, inputEl) {
      if (this._contextMode !== "inject") return false;

      const userText = this._getInputText(inputEl).trim();
      if (!userText) return false;

      // Event dispatch does not await async listeners, so the page submission
      // must be stopped before fetching context.
      event.preventDefault();
      event.stopImmediatePropagation();
      if (this._contextSubmitInFlight) return true;

      this._submitWithContext(inputEl, userText);
      return true;
    }

    async _submitWithContext(inputEl, userText) {
      this._contextSubmitInFlight = true;
      try {
        await this._fetchContext();
        if (!this._hasContextPrefix(userText)) {
          const prefix = this._buildContextPrefix();
          const fullText = `${prefix}\n\n---\n\n${userText}`;
          this._setInputText(inputEl, fullText);
        }

        await new Promise((r) => setTimeout(r, 50));
        this._bypassContextInject = true;
        const sendBtn = this.getSubmitButton();
        if (sendBtn && !sendBtn.disabled) {
          sendBtn.click();
        } else {
          inputEl.dispatchEvent(new KeyboardEvent("keydown", {
            key: "Enter",
            code: "Enter",
            keyCode: 13,
            bubbles: true,
            cancelable: true,
          }));
        }
      } finally {
        this._bypassContextInject = false;
        this._contextSubmitInFlight = false;
      }
    }

    _hasContextPrefix(text) {
      const value = String(text || "");
      return value.includes("=== PROJECT CONTEXT ===")
        || value.includes("IMPORTANT: A persistent context file is attached to this session.");
    }

    _stripInjectedContextPrefix(content) {
      const text = String(content || "").trim();
      if (!this._hasContextPrefix(text)) return text;

      const separator = /\n\s*\n---\n\s*\n/g;
      let boundary = -1;
      let match;
      while ((match = separator.exec(text)) !== null) {
        boundary = match.index + match[0].length;
      }
      return boundary >= 0 ? text.slice(boundary).trim() : text;
    }

    _teardownHiddenInject() {
      if (!this._submitHooked || !this._origSubmitHandler) return;
      document.removeEventListener("keydown", this._origSubmitHandler, true);
      if (this._origSubmitClickHandler) {
        document.removeEventListener("click", this._origSubmitClickHandler, true);
      }
      this._submitHooked = false;
      this._origSubmitHandler = null;
      this._origSubmitClickHandler = null;
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

      return this._sendPayloadToService(payload);
    }

    async _sendPayloadToService(payload) {
      if (!payload || !payload.messages || payload.messages.length === 0) return false;

      if (this.sendInFlight) {
        // During streaming, keep only the newest full snapshot. Older queued
        // snapshots would be overwritten by newer content anyway.
        this.queuedSnapshot = payload;
        return false;
      }

      this.sendInFlight = true;
      let allDelivered = true;
      let nextPayload = payload;
      try {
        while (nextPayload) {
          this.queuedSnapshot = null;
          const delivered = await this._deliverSnapshot(nextPayload);
          if (!delivered) {
            allDelivered = false;
            const retryPayload = this.queuedSnapshot || nextPayload;
            this.queuedSnapshot = null;
            this._scheduleSendRetry(retryPayload);
            break;
          }
          nextPayload = this.queuedSnapshot;
        }
      } finally {
        this.sendInFlight = false;
      }
      return allDelivered;
    }

    async _deliverSnapshot(payload) {
      // Include per-conversation output directory if set
      try {
        const data = await chrome.storage.local.get({ convOutputDirs: {} });
        const customDir = data.convOutputDirs[payload.conversationId];
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
            `[ShadowWrite] Saved ${payload.messages.length} messages → ${resp.body}`
          );
          window.dispatchEvent(
            new CustomEvent("shadowwrite-save-success", {
              detail: { count: payload.messages.length },
            })
          );
          // Extract context-update markers from AI responses
          await this._extractAndSaveContextUpdates(payload.messages, payload.conversationId);
          if (payload.conversationId === this.failedSnapshot?.conversationId) {
            this._clearSendRetry();
          }
          return true;
        } else {
          const errMsg = resp?.error || `Server responded ${resp?.status}: ${resp?.body?.substring(0, 120)}`;
          console.warn(`[ShadowWrite] ${errMsg}`);
          window.dispatchEvent(
            new CustomEvent("shadowwrite-save-error", {
              detail: { error: errMsg },
            })
          );
          return false;
        }
      } catch (err) {
        if (this._handleContextInvalidated(err)) return;
        console.warn("[ShadowWrite] Cannot reach background:", err.message);
        window.dispatchEvent(
          new CustomEvent("shadowwrite-save-error", {
            detail: { error: err.message },
          })
        );
        return false;
      }
    }

    _scheduleSendRetry(payload) {
      if (!payload || !payload.conversationId || this._contextInvalidated) return;
      if (!this.isTracking || payload.conversationId !== this.currentConversationId) return;

      this.failedSnapshot = payload;
      if (this.sendRetryTimer) return;

      this.sendRetryTimer = setTimeout(() => {
        this.sendRetryTimer = null;
        const retryPayload = this.failedSnapshot;
        if (!retryPayload) return;
        if (!this.isTracking || retryPayload.conversationId !== this.currentConversationId) {
          this._clearSendRetry();
          return;
        }
        this._sendPayloadToService(retryPayload);
      }, SEND_RETRY_DELAY);
    }

    _clearSendRetry() {
      if (this.sendRetryTimer) {
        clearTimeout(this.sendRetryTimer);
        this.sendRetryTimer = null;
      }
      this.failedSnapshot = null;
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
     * Build a compact change signature without allocating a huge JSON string.
     * This still scans message text, but avoids the extra full-snapshot copy
     * that becomes expensive in long conversations.
     */
    _buildSnapshotSignature(messages) {
      let hashA = 0x811c9dc5;
      let hashB = 0x45d9f3b;
      let charCount = 0;

      const update = (value) => {
        const text = value == null ? "" : String(value);
        charCount += text.length;
        for (let i = 0; i < text.length; i++) {
          const code = text.charCodeAt(i);
          hashA = Math.imul(hashA ^ code, 0x01000193);
          hashB = Math.imul(hashB + code, 0x27d4eb2d) ^ (hashB >>> 15);
        }
        hashA = Math.imul(hashA ^ 0x1f, 0x01000193);
        hashB = Math.imul(hashB + 0x9e3779b9, 0x27d4eb2d) ^ (hashB >>> 15);
      };

      update(messages.length);
      for (const msg of messages) {
        update(msg.messageId);
        update(msg.sender);
        update(msg.position);
        update(msg.content);
        update(msg.thinking);
      }

      return `${messages.length}:${charCount}:${hashA >>> 0}:${hashB >>> 0}`;
    }

    /**
     * Extract content from an element and convert HTML to Markdown,
     * preserving headings, bold/italic, lists, code blocks, tables, etc.
     */
    extractFormattedContent(element) {
      if (!element) return "";
      const cached = this._formatCache.get(element);
      if (cached !== undefined) return cached;

      // Clone to avoid side effects
      const clone = element.cloneNode(true);
      // Remove hidden/script elements
      clone
        .querySelectorAll("script, style, .sr-only")
        .forEach((el) => el.remove());
      const markdown = this._htmlToMarkdown(clone);
      this._formatCache.set(element, markdown);
      return markdown;
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

    isEditingMessage() {
      const active = document.activeElement;
      if (!active || active === document.body) return false;
      const editable = active.matches?.("textarea, input, [contenteditable]")
        ? active
        : active.closest?.("textarea, input, [contenteditable]");
      if (!editable) return false;
      return this.isMessageElement(editable);
    }

    /**
     * Cleanup on unload.
     */
    destroy() {
      if (this.contentObserver) this.contentObserver.disconnect();
      if (this.debounceTimer) clearTimeout(this.debounceTimer);
      if (this.throttleTimer) clearTimeout(this.throttleTimer);
      if (this.urlCheckInterval) clearInterval(this.urlCheckInterval);
      if (this.observerRetryTimer) clearTimeout(this.observerRetryTimer);
      this._cancelInitialCapture();
      this._cancelPendingChangeCheck();
      this.queuedSnapshot = null;
      this._clearSendRetry();
    }
  }

  // Export to window for adapter scripts
  window.BaseShadowWriteAdapter = BaseShadowWriteAdapter;
})();
