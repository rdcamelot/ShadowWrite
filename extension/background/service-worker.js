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
const EXTENSION_UPDATE_ALARM = "shadowwrite-extension-update";
const PENDING_RELOAD_TABS_KEY = "_shadowwritePendingReloadTabs";
const EXTENSION_UPDATE_ATTEMPT_KEY = "_shadowwriteExtensionUpdateAttempt";
const SUPPORTED_AI_PAGE = /^https:\/\/(?:chatgpt\.com|chat\.openai\.com|chat\.deepseek\.com|gemini\.google\.com|claude\.ai|grok\.com|kimi\.moonshot\.cn|kimi\.com|[^/]+\.kimi\.com|www\.doubao\.com|yuanbao\.tencent\.com)\//;

let extensionUpdateCheckInFlight = null;

function buildServerUrl(host, port, path, searchParams) {
  let rawHost = String(host || DEFAULT_SETTINGS.host).trim();
  if (!/^https?:\/\//i.test(rawHost)) {
    rawHost = `http://${rawHost}`;
  }

  const url = new URL(rawHost);
  if (port !== undefined && port !== null && port !== "") {
    url.port = String(port || DEFAULT_SETTINGS.port);
  } else if (!url.port) {
    url.port = String(DEFAULT_SETTINGS.port);
  }
  url.pathname = path;
  url.search = "";

  if (searchParams) {
    for (const [key, value] of Object.entries(searchParams)) {
      if (value !== undefined && value !== null) {
        url.searchParams.set(key, value);
      }
    }
  }
  return url.toString();
}

function getLoadedExtensionVersion() {
  const manifest = chrome.runtime.getManifest();
  return manifest.version_name || manifest.version;
}

function ensureExtensionUpdateAlarm() {
  chrome.alarms.create(EXTENSION_UPDATE_ALARM, { periodInMinutes: 1 });
}

async function finishPendingExtensionReload() {
  const stored = await chrome.storage.local.get({ [PENDING_RELOAD_TABS_KEY]: [] });
  const tabIds = stored[PENDING_RELOAD_TABS_KEY];
  if (!Array.isArray(tabIds) || tabIds.length === 0) return;

  await chrome.storage.local.remove(PENDING_RELOAD_TABS_KEY);
  for (const tabId of tabIds) {
    try {
      await chrome.tabs.reload(tabId);
    } catch {
      // The tab may have been closed while the extension was reloading.
    }
  }
}

function checkForExtensionUpdate() {
  if (extensionUpdateCheckInFlight) return extensionUpdateCheckInFlight;

  extensionUpdateCheckInFlight = (async () => {
    try {
      const settings = await chrome.storage.sync.get(DEFAULT_SETTINGS);
      const response = await fetch(buildServerUrl(
        settings.host,
        settings.port,
        "/api/health",
      ), { signal: AbortSignal.timeout(3000) });
      if (!response.ok) return;

      const health = await response.json();
      const availableVersion = health.extensionVersionName || health.extensionVersion;
      const loadedVersion = getLoadedExtensionVersion();
      if (!availableVersion || availableVersion === "unknown" || availableVersion === loadedVersion) {
        if (availableVersion === loadedVersion) {
          await chrome.storage.local.remove(EXTENSION_UPDATE_ATTEMPT_KEY);
        }
        return;
      }

      const stored = await chrome.storage.local.get({ [EXTENSION_UPDATE_ATTEMPT_KEY]: null });
      const previousAttempt = stored[EXTENSION_UPDATE_ATTEMPT_KEY];
      if (
        previousAttempt?.loadedVersion === loadedVersion &&
        previousAttempt?.availableVersion === availableVersion
      ) {
        return;
      }

      const tabs = await chrome.tabs.query({});
      const reloadTabIds = tabs
        .filter((tab) => tab.id && SUPPORTED_AI_PAGE.test(tab.url || ""))
        .map((tab) => tab.id);
      await chrome.storage.local.set({
        [PENDING_RELOAD_TABS_KEY]: reloadTabIds,
        [EXTENSION_UPDATE_ATTEMPT_KEY]: { loadedVersion, availableVersion },
      });
      console.log(`[ShadowWrite] Reloading extension ${loadedVersion} -> ${availableVersion}`);
      chrome.runtime.reload();
    } catch {
      // The local server may be offline; retry on the next alarm.
    }
  })().finally(() => {
    extensionUpdateCheckInFlight = null;
  });

  return extensionUpdateCheckInFlight;
}

// ── Clip tracking persistence (chrome.storage.session) ───────────
// Survives MV3 service worker restarts within the browser session.
// Structure: { _clipTabs: { "tabId": { lastUrl, group } } }

async function getClipTabs() {
  const { _clipTabs } = await chrome.storage.session.get({ _clipTabs: {} });
  return _clipTabs;
}

async function getClipTab(tabId) {
  const tabs = await getClipTabs();
  return tabs[String(tabId)] || null;
}

async function setClipTab(tabId, data) {
  const tabs = await getClipTabs();
  tabs[String(tabId)] = data;
  await chrome.storage.session.set({ _clipTabs: tabs });
}

async function removeClipTab(tabId) {
  const tabs = await getClipTabs();
  delete tabs[String(tabId)];
  await chrome.storage.session.set({ _clipTabs: tabs });
}

// ── Initialisation ────────────────────────────────────────────────
chrome.runtime.onInstalled.addListener(async (details) => {
  if (details.reason === "install") {
    await chrome.storage.sync.set(DEFAULT_SETTINGS);
    console.log("[ShadowWrite] Extension installed, defaults saved.");
  }

  await chrome.contextMenus.removeAll();
  // Context menu for web clipping (hidden feature — no UI mention)
  chrome.contextMenus.create({
    id: "shadowwrite-clip",
    title: "剪藏到 ShadowWrite",
    contexts: ["page", "selection"],
  });
  ensureExtensionUpdateAlarm();
  checkForExtensionUpdate();
});

chrome.runtime.onStartup.addListener(() => {
  ensureExtensionUpdateAlarm();
  finishPendingExtensionReload();
  checkForExtensionUpdate();
});

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === EXTENSION_UPDATE_ALARM) {
    checkForExtensionUpdate();
  }
});

ensureExtensionUpdateAlarm();
finishPendingExtensionReload();
checkForExtensionUpdate();

// ── Clip helper: inject clipper.js → POST /api/clip ──────────────
async function doClipTab(tabId, overrideGroup) {
  try {
    const results = await chrome.scripting.executeScript({
      target: { tabId },
      files: ["content/clipper.js"],
    });
    const data = results?.[0]?.result;
    if (!data || data.error) {
      console.error("[ShadowWrite] Clip extraction failed:", data?.error);
      setBadge(tabId, "✗", "#ef4444");
      setTimeout(() => setBadge(tabId, "", "#ef4444"), 5000);
      return { ok: false, error: data?.error || "Extraction failed" };
    }
    const group = overrideGroup || data.group || "";
    const s = await chrome.storage.sync.get(DEFAULT_SETTINGS);
    const url = buildServerUrl(s.host, s.port, "/api/clip");
    const resp = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        title: data.title,
        url: data.url,
        domain: data.domain,
        content: data.content,
        category: "clips",
        group: group,
      }),
    });
    if (resp.ok) {
      const body = await resp.json();
      console.log("[ShadowWrite] Clipped:", body.title, "→", body.path);
      setBadge(tabId, "📎", "#3b82f6");
      setTimeout(() => setBadge(tabId, "", "#3b82f6"), 3000);
      return { ok: true, group, ...body };
    } else {
      const errText = await resp.text();
      console.error("[ShadowWrite] Clip server error:", resp.status, errText);
      setBadge(tabId, "✗", "#ef4444");
      setTimeout(() => setBadge(tabId, "", "#ef4444"), 5000);
      return { ok: false, error: `Server ${resp.status}: ${errText}` };
    }
  } catch (err) {
    console.error("[ShadowWrite] Clip failed:", err);
    setBadge(tabId, "✗", "#ef4444");
    setTimeout(() => setBadge(tabId, "", "#ef4444"), 5000);
    return { ok: false, error: err.message };
  }
}

// ── Context menu handler ──────────────────────────────────────────
chrome.contextMenus.onClicked.addListener(async (info, tab) => {
  if (info.menuItemId !== "shadowwrite-clip") return;
  if (!tab?.id) return;
  await doClipTab(tab.id);
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
          const url = buildServerUrl(message.host, message.port, "/api/messages");
          const timeoutMs = Number(message.timeoutMs) || 15000;
          const resp = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(message.payload),
            signal: AbortSignal.timeout(timeoutMs),
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
          const url = buildServerUrl(s.host, s.port, "/api/config", {
            conversationId: message.conversationId,
          });
          const resp = await fetch(url, { signal: AbortSignal.timeout(3000) });
          const data = await resp.json();
          sendResponse({ success: resp.ok, data });
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
          const url = buildServerUrl(s.host, s.port, "/api/config");
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
          const url = buildServerUrl(s.host, s.port, "/api/browse-directory");
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

    // ── Context file API relay ────────────────────────────────
    case "getContext":
      (async () => {
        try {
          const s = await chrome.storage.sync.get(DEFAULT_SETTINGS);
          const url = buildServerUrl(s.host, s.port, "/api/context", {
            conversationId: message.conversationId,
          });
          const resp = await fetch(url, { signal: AbortSignal.timeout(5000) });
          const data = await resp.json();
          sendResponse({ success: resp.ok, data });
        } catch (err) {
          sendResponse({ success: false, error: err.message });
        }
      })();
      return true;

    case "postContext":
      (async () => {
        try {
          const s = await chrome.storage.sync.get(DEFAULT_SETTINGS);
          const url = buildServerUrl(s.host, s.port, "/api/context");
          const resp = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(message.payload),
          });
          const data = await resp.json();
          sendResponse({ success: resp.ok, data });
        } catch (err) {
          sendResponse({ success: false, error: err.message });
        }
      })();
      return true;

    case "summarizeContext":
      (async () => {
        try {
          const s = await chrome.storage.sync.get(DEFAULT_SETTINGS);
          const url = buildServerUrl(s.host, s.port, "/api/context/summarize");
          const resp = await fetch(url, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ conversationId: message.conversationId }),
            signal: AbortSignal.timeout(120000),
          });
          const data = await resp.json();
          sendResponse({ success: resp.ok, data });
        } catch (err) {
          sendResponse({ success: false, error: err.message });
        }
      })();
      return true;

    // ── Clip tracking (popup-based, hidden feature) ────────────
    case "getClipTrackingState": {
      const qTabId = message.tabId || tabId;
      (async () => {
        const clipTab = await getClipTab(qTabId);
        sendResponse({ isTracking: !!clipTab });
      })();
      return true;
    }

    case "setClipTracking": {
      const tTabId = message.tabId || tabId;
      if (!tTabId) {
        sendResponse({ ok: false, error: "No tab ID" });
        return true;
      }
      (async () => {
        try {
          if (message.enabled) {
            // Clip immediately
            const result = await doClipTab(tTabId);
            if (!result?.ok) {
              sendResponse(result || { ok: false, error: "Initial clip failed" });
              return;
            }
            // Store tracking state with group from first clip
            const curTab = await chrome.tabs.get(tTabId);
            await setClipTab(tTabId, {
              lastUrl: curTab?.url || "",
              group: result?.group || "",
            });
            // Inject SPA watcher for auto-tracking
            const { autoCapture } = await chrome.storage.sync.get({ autoCapture: true });
            if (autoCapture) {
              try {
                await chrome.scripting.executeScript({
                  target: { tabId: tTabId },
                  files: ["content/clip-watcher.js"],
                });
              } catch (e) {
                console.warn("[ShadowWrite] Watcher inject failed:", e);
              }
            }
            sendResponse({ ok: true, ...(result || {}) });
          } else {
            await removeClipTab(tTabId);
            setBadge(tTabId, "", "#3b82f6");
            sendResponse({ ok: true });
          }
        } catch (err) {
          sendResponse({ ok: false, error: err.message });
        }
      })();
      return true;
    }

    // clip-watcher.js detected SPA URL change
    case "clipPageChanged": {
      if (tabId) {
        (async () => {
          const clipTab = await getClipTab(tabId);
          if (!clipTab) return;
          const { autoCapture } = await chrome.storage.sync.get({ autoCapture: true });
          if (!autoCapture) return;
          if (message.url && message.url !== clipTab.lastUrl) {
            await setClipTab(tabId, { ...clipTab, lastUrl: message.url });
            await doClipTab(tabId, clipTab.group);
          }
        })();
      }
      break;
    }

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

// ── Clip tracking: auto-clip on full-page navigation ─────────────
chrome.tabs.onUpdated.addListener(async (tabId, changeInfo, tab) => {
  if (changeInfo.status !== "complete") return;

  const clipTab = await getClipTab(tabId);
  if (!clipTab) return;

  const { autoCapture } = await chrome.storage.sync.get({ autoCapture: true });
  if (!autoCapture) return;

  if (tab.url === clipTab.lastUrl) return; // same URL — skip
  await setClipTab(tabId, { ...clipTab, lastUrl: tab.url });

  await doClipTab(tabId, clipTab.group);

  // Re-inject watcher (destroyed on full navigation)
  try {
    await chrome.scripting.executeScript({
      target: { tabId },
      files: ["content/clip-watcher.js"],
    });
  } catch (e) { /* restricted page — ignore */ }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  removeClipTab(tabId);
});
