/**
 * ShadowWrite — Clip Watcher (SPA navigation detection)
 *
 * Injected via chrome.scripting.executeScript when clip tracking
 * is active on a tab. Monitors URL changes for SPA-style navigation
 * (history.pushState, popstate) and notifies the background.
 *
 * For full page loads, chrome.tabs.onUpdated handles re-clipping;
 * this script covers the SPA case where the page doesn't fully reload.
 */
(function () {
  "use strict";

  // Guard: avoid duplicate watchers
  if (window._shadowWriteClipWatcher) return;
  window._shadowWriteClipWatcher = true;

  let lastUrl = location.href;

  function onUrlChange() {
    const current = location.href;
    if (current === lastUrl) return;
    lastUrl = current;
    chrome.runtime.sendMessage({
      type: "clipPageChanged",
      url: current,
    }).catch(() => {
      // Extension context invalidated — stop watching
      cleanup();
    });
  }

  // Intercept history.pushState / replaceState
  const origPush = history.pushState;
  const origReplace = history.replaceState;

  history.pushState = function (...args) {
    origPush.apply(this, args);
    setTimeout(onUrlChange, 0);
  };
  history.replaceState = function (...args) {
    origReplace.apply(this, args);
    setTimeout(onUrlChange, 0);
  };

  // popstate (back/forward)
  window.addEventListener("popstate", () => setTimeout(onUrlChange, 0));

  // hashchange
  window.addEventListener("hashchange", () => setTimeout(onUrlChange, 0));

  // Fallback: periodic check (some sites mutate URL without standard APIs)
  const intervalId = setInterval(onUrlChange, 2000);

  function cleanup() {
    clearInterval(intervalId);
    history.pushState = origPush;
    history.replaceState = origReplace;
    window._shadowWriteClipWatcher = false;
  }
})();
