/**
 * ShadowWrite — Background Service Worker
 *
 * Responsibilities:
 *  - Manage extension settings (chrome.storage.sync)
 *  - Forward configuration to content scripts
 *  - Handle badge/icon updates
 */

"use strict";

// ── Default settings ──────────────────────────────────────────────
const DEFAULT_SETTINGS = {
  host: "127.0.0.1",
  port: 24601,
  autoCapture: true,
  enabled: true,
};

// ── Initialisation ────────────────────────────────────────────────
chrome.runtime.onInstalled.addListener(async (details) => {
  if (details.reason === "install") {
    await chrome.storage.sync.set(DEFAULT_SETTINGS);
    console.log("[ShadowWrite] Extension installed, defaults saved.");
  }
});

// ── Badge helpers ─────────────────────────────────────────────────
function setBadge(tabId, text, color) {
  chrome.action.setBadgeText({ text, tabId });
  chrome.action.setBadgeBackgroundColor({ color, tabId });
}

// ── Message handler (content scripts ↔ background) ───────────────
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  const tabId = sender.tab?.id;

  switch (message.type) {
    // Content script requests current settings
    case "getSettings":
      chrome.storage.sync.get(DEFAULT_SETTINGS).then(sendResponse);
      return true; // async

    // Content script reports a successful save
    case "saveSuccess":
      if (tabId) setBadge(tabId, "✓", "#22c55e");
      // Clear after 3 seconds
      setTimeout(() => {
        if (tabId) setBadge(tabId, "", "#22c55e");
      }, 3000);
      break;

    // Content script reports a save error
    case "saveError":
      if (tabId) setBadge(tabId, "✗", "#ef4444");
      // Clear after 8 seconds
      setTimeout(() => {
        if (tabId) setBadge(tabId, "", "#ef4444");
      }, 8000);
      break;

    // Popup toggles enabled state
    case "setEnabled":
      chrome.storage.sync.set({ enabled: message.enabled });
      break;

    // Popup updates settings
    case "updateSettings":
      chrome.storage.sync.set(message.settings);
      break;

    default:
      console.warn("[ShadowWrite] Unknown message type:", message.type);
  }
});

// ── Settings change → notify active tabs ─────────────────────────
chrome.storage.onChanged.addListener((changes, area) => {
  if (area !== "sync") return;

  const updated = {};
  for (const [key, { newValue }] of Object.entries(changes)) {
    updated[key] = newValue;
  }

  // Broadcast to all tabs that might have content scripts
  chrome.tabs.query({}, (tabs) => {
    for (const tab of tabs) {
      if (tab.id) {
        chrome.tabs.sendMessage(tab.id, {
          type: "settingsChanged",
          settings: updated,
        }).catch(() => {
          // Tab might not have a content script — ignore
        });
      }
    }
  });
});
