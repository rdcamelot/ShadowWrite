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

    // Keep host-page handlers from treating the indicator as page UI.
    for (const eventName of ["pointerdown", "pointerup", "mousedown", "mouseup"]) {
      dot.addEventListener(eventName, (event) => event.stopPropagation(), true);
      dot.addEventListener(eventName, (event) => event.stopPropagation());
    }

    // Click to toggle tracking on/off for current conversation. Use capture
    // phase so the handler still runs while the event is isolated from hosts.
    dot.addEventListener("click", (event) => {
      event.preventDefault();
      event.stopPropagation();
      const adapter = window.__shadowWriteAdapter;
      if (!adapter) return;
      if (adapter._contextInvalidated) {
        // Extension was reloaded — refreshing is the only fix
        if (confirm("ShadowWrite 扩展已更新，需要刷新页面才能继续使用。\n\n立即刷新？")) {
          location.reload();
        }
        return;
      }
      if (!adapter.currentConversationId) {
        console.log("[ShadowWrite] No active conversation to track.");
        return;
      }
      if (adapter.isTracking) {
        adapter.disableTracking();
      } else {
        adapter.enableTracking();
      }
    }, true);

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

  // Extension context invalidated — show red dot with refresh prompt
  window.addEventListener("shadowwrite-context-invalidated", () => {
    const dot = document.getElementById(STATUS_ID);
    if (!dot) return;
    dot.classList.remove("sw-tracking", "sw-status-saving", "sw-status-ok");
    dot.classList.add("sw-status-error");
    dot.title = "ShadowWrite: 扩展已更新，点击刷新页面";
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
