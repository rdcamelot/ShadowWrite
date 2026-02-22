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
    dot.title = "ShadowWrite: idle";
    document.body.appendChild(dot);
  }

  function setStatus(state, detail) {
    const dot = document.getElementById(STATUS_ID);
    if (!dot) return;

    dot.classList.remove(
      "sw-status-idle",
      "sw-status-saving",
      "sw-status-ok",
      "sw-status-error"
    );

    switch (state) {
      case "idle":
        dot.classList.add("sw-status-idle");
        dot.title = "ShadowWrite: idle";
        break;
      case "saving":
        dot.classList.add("sw-status-saving");
        dot.title = "ShadowWrite: sending…";
        break;
      case "ok":
        dot.classList.add("sw-status-ok");
        dot.title = `ShadowWrite: saved (${detail || ""})`;
        // Revert to idle after 3s
        setTimeout(() => setStatus("idle"), 3000);
        break;
      case "error":
        dot.classList.add("sw-status-error");
        dot.title = `ShadowWrite: error — ${detail || "unknown"}`;
        // Revert to idle after 8s
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
