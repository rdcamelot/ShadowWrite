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

    // Popup settings — popup writes directly to chrome.storage.sync;
    // the onChanged listener below broadcasts to content scripts.
    case "setEnabled":
    case "updateSettings":
    case "settingsUpdated":
      break;

    // Content script asks background to relay HTTP to local server
    // (background fetch is NOT subject to page CSP)
    case "sendToServer":
      (async () => {
        try {
          let host = message.host || "http://127.0.0.1";
          if (!/^https?:\/\//i.test(host)) {
            host = `http://${host}`;
          }
          const url = `${host}:${message.port || 24601}/api/messages`;
          const resp = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(message.payload),
          });
          const body = await resp.text();
          sendResponse({ ok: resp.ok, status: resp.status, body });
        } catch (err) {
          sendResponse({ ok: false, error: err.message });
        }
      })();
      return true; // keep message channel open for async sendResponse

    // Popup asks background to GET server config
    case "getServerConfig":
      (async () => {
        try {
          const s = await chrome.storage.sync.get(DEFAULT_SETTINGS);
          const url = `http://${s.host}:${s.port}/api/config`;
          const resp = await fetch(url, { signal: AbortSignal.timeout(3000) });
          const data = await resp.json();
          sendResponse({ success: true, data });
        } catch (err) {
          sendResponse({ success: false, error: err.message });
        }
      })();
      return true;

    // Popup asks background to POST updated config to server
    case "setServerConfig":
      (async () => {
        try {
          const s = await chrome.storage.sync.get(DEFAULT_SETTINGS);
          const url = `http://${s.host}:${s.port}/api/config`;
          const resp = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(message.config),
          });
          const data = await resp.json();
          sendResponse({ success: resp.ok, data });
        } catch (err) {
          sendResponse({ success: false, error: err.message });
        }
      })();
      return true;

    // Popup asks server to open a native directory picker dialog
    case "browseDirectory":
      (async () => {
        try {
          const s = await chrome.storage.sync.get(DEFAULT_SETTINGS);
          const url = `http://${s.host}:${s.port}/api/browse-directory`;
          const body = message.initialDir ? JSON.stringify({ initialDir: message.initialDir }) : "{}";
          const resp = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body,
            signal: AbortSignal.timeout(120000), // user may take time to pick
          });
          const data = await resp.json();
          sendResponse({ success: resp.ok, data });
        } catch (err) {
          sendResponse({ success: false, error: err.message });
        }
      })();
      return true;

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
