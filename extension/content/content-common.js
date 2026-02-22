/**
 * ShadowWrite — Content Common
 *
 * Shared utilities injected on all supported AI platforms.
 * Provides a minimal status indicator (floating dot) and
 * global event handling.
 */

"use strict";

(() => {
  if (window.__shadowWriteCommon) return;
  window.__shadowWriteCommon = true;

  /* ------------------------------------------------------------------ */
  /*  Status Indicator (floating dot)                                    */
  /* ------------------------------------------------------------------ */

  const STATUS_ID = "shadowwrite-status";

  function createStatusIndicator() {
    if (document.getElementById(STATUS_ID)) return;

    const dot = document.createElement("div");
    dot.id = STATUS_ID;
    dot.title = "ShadowWrite: click to toggle tracking";

    // Click to toggle tracking on/off for current conversation
    dot.addEventListener("click", () => {
      const adapter = window.__shadowWriteAdapter;
      if (!adapter || !adapter.currentConversationId) {
        console.log("[ShadowWrite] No active conversation to track.");
        return;
      }
      if (adapter.isTracking) {
        adapter.disableTracking();
      } else {
        adapter.enableTracking();
      }
    });

    document.body.appendChild(dot);
  }

  function setStatus(state, detail) {
    const dot = document.getElementById(STATUS_ID);
    if (!dot) return;

    // Remove transient state classes (keep sw-tracking!)
    dot.classList.remove(
      "sw-status-saving",
      "sw-status-ok",
      "sw-status-error"
    );

    switch (state) {
      case "idle":
        // Just remove transient classes; tracking class stays
        dot.title = dot.classList.contains("sw-tracking")
          ? "ShadowWrite: tracking ON (click to stop)"
          : "ShadowWrite: tracking OFF (click to start)";
        break;
      case "saving":
        dot.classList.add("sw-status-saving");
        dot.title = "ShadowWrite: sending…";
        break;
      case "ok":
        dot.classList.add("sw-status-ok");
        dot.title = `ShadowWrite: saved (${detail || ""})`;
        setTimeout(() => setStatus("idle"), 3000);
        break;
      case "error":
        dot.classList.add("sw-status-error");
        dot.title = `ShadowWrite: error — ${detail || "unknown"}`;
        setTimeout(() => setStatus("idle"), 8000);
        break;
    }
  }

  /* ------------------------------------------------------------------ */
  /*  Event Listeners                                                    */
  /* ------------------------------------------------------------------ */

  window.addEventListener("shadowwrite-save-success", (e) => {
    setStatus("ok", `${e.detail?.count || 0} messages`);
  });

  window.addEventListener("shadowwrite-save-error", (e) => {
    setStatus("error", e.detail?.error);
  });

  // Tracking state toggle — update dot appearance
  window.addEventListener("shadowwrite-tracking-state", (e) => {
    const dot = document.getElementById(STATUS_ID);
    if (!dot) return;
    const { tracking, hasConversation } = e.detail || {};
    if (!hasConversation) {
      // Not on a valid conversation page
      dot.classList.remove("sw-tracking");
      dot.title = "ShadowWrite: not a conversation page";
      return;
    }
    if (tracking) {
      dot.classList.add("sw-tracking");
      dot.title = "ShadowWrite: tracking ON (click to stop)";
    } else {
      dot.classList.remove("sw-tracking");
      dot.title = "ShadowWrite: tracking OFF (click to start)";
    }
  });

  /* ------------------------------------------------------------------ */
  /*  Init                                                               */
  /* ------------------------------------------------------------------ */

  // Wait for DOM ready
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", createStatusIndicator);
  } else {
    createStatusIndicator();
  }

  // Export
  window.shadowWriteCommon = { setStatus };
})();
