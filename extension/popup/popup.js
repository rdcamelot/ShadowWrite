/**
 * ShadowWrite — Popup Script
 *
 * - "当前对话追踪" toggle: queries and controls the active tab's
 *   tracking state (synced with the floating dot indicator).
 * - "自动追踪新对话" toggle: stored in chrome.storage.sync, controls
 *   whether new conversations auto-start tracking.
 * - Host/Port: stored in chrome.storage.sync (extension-local).
 * - outputDir / chatHtml: loaded from & saved to the running server
 *   via background relay (GET/POST /api/config).
 */

"use strict";

const $ = (id) => document.getElementById(id);

// ── Helpers ───────────────────────────────────────────────────────

function setStatus(text, type) {
  const el = $("status");
  el.textContent = text;
  el.className = "st" + (type ? ` ${type}` : "");
}

/** Send a message to background and return the response. */
function bg(msg) {
  return new Promise((resolve) => chrome.runtime.sendMessage(msg, resolve));
}

// ── Load ──────────────────────────────────────────────────────────

async function loadSettings() {
  // 1. Extension-local settings
  const local = await chrome.storage.sync.get({
    host: "127.0.0.1",
    port: 24601,
    autoCapture: true,
  });
  $("autoCapture").checked = local.autoCapture;
  $("host").value = local.host;
  $("port").value = local.port;

  // 2. Query active tab for tracking state (synced with dot)
  loadTrackingState();

  // 3. Server config (outputDir, chatHtml)
  await loadServerConfig();
}

/**
 * Query the active tab's content script for the current
 * tracking state and update the popup toggle accordingly.
 */
async function loadTrackingState() {
  const toggle = $("trackingToggle");
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    if (!tab?.id) {
      toggle.checked = false;
      toggle.disabled = true;
      return;
    }
    chrome.tabs.sendMessage(tab.id, { type: "getTrackingState" }, (resp) => {
      if (chrome.runtime.lastError || !resp) {
        // No content script on this page (not a supported AI platform)
        toggle.checked = false;
        toggle.disabled = true;
        return;
      }
      toggle.disabled = !resp.hasConversation;
      toggle.checked = !!resp.isTracking;
    });
  } catch {
    toggle.checked = false;
    toggle.disabled = true;
  }
}

async function loadServerConfig() {
  setStatus("正在连接…", "");
  try {
    const res = await bg({ type: "getServerConfig" });
    if (res && res.success) {
      const cfg = res.data; // { outputDir, chatHtml }
      $("outputDir").value = cfg.outputDir ?? "";
      $("chatHtml").checked = !!cfg.chatHtml;
      setStatus("✓ 本地服务已连接", "ok");
    } else {
      setStatus("✗ 无法读取服务器配置", "err");
    }
  } catch {
    setStatus("✗ 无法连接本地服务", "err");
  }
}

// ── Save ──────────────────────────────────────────────────────────

function attachListeners() {
  // ── Tracking toggle → send to active tab's content script ───
  $("trackingToggle").addEventListener("change", async () => {
    const enabled = $("trackingToggle").checked;
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (!tab?.id) return;
      chrome.tabs.sendMessage(tab.id, { type: "setTracking", enabled });
    } catch {
      // ignore
    }
  });

  // ── autoCapture → extension storage ─────────────────────────
  $("autoCapture").addEventListener("change", () => {
    chrome.storage.sync.set({ autoCapture: $("autoCapture").checked });
    broadcastLocal();
  });

  // ── Host / Port (extension-local, debounced) ────────────────
  let connTimer;
  const debounceConn = () => {
    clearTimeout(connTimer);
    connTimer = setTimeout(() => {
      const host = $("host").value.trim() || "127.0.0.1";
      const port = parseInt($("port").value, 10) || 24601;
      chrome.storage.sync.set({ host, port });
      broadcastLocal();
      loadServerConfig(); // re-fetch config from new host:port
    }, 600);
  };
  $("host").addEventListener("input", debounceConn);
  $("port").addEventListener("input", debounceConn);

  // ── Server config: outputDir (debounced) ────────────────────
  let dirTimer;
  $("outputDir").addEventListener("input", () => {
    clearTimeout(dirTimer);
    dirTimer = setTimeout(() => pushServerConfig(), 800);
  });

  // ── Server config: chatHtml (immediate) ─────────────────────
  $("chatHtml").addEventListener("change", () => pushServerConfig());

  // ── Browse directory button ─────────────────────────────────
  $("browseDir").addEventListener("click", async () => {
    const btn = $("browseDir");
    btn.disabled = true;
    setStatus("正在打开目录选择…", "");
    try {
      const currentDir = $("outputDir").value.trim() || undefined;
      const res = await bg({ type: "browseDirectory", initialDir: currentDir });
      if (res?.success && res.data?.selected) {
        $("outputDir").value = res.data.selected;
        pushServerConfig();
      } else if (res?.success && res.data?.cancelled) {
        setStatus("✓ 本地服务已连接", "ok");
      } else {
        setStatus("✗ " + (res?.data?.error || res?.error || "打开失败"), "err");
      }
    } catch {
      setStatus("✗ 无法连接本地服务", "err");
    } finally {
      btn.disabled = false;
    }
  });
}

/** Push outputDir + chatHtml to the running server. */
async function pushServerConfig() {
  const config = {
    outputDir: $("outputDir").value.trim() || "./outputs",
    chatHtml: $("chatHtml").checked,
  };
  try {
    const res = await bg({ type: "setServerConfig", config });
    if (res && res.success) {
      setStatus("✓ 已同步到服务器", "ok");
    } else {
      setStatus("✗ 保存失败", "err");
    }
  } catch {
    setStatus("✗ 无法连接本地服务", "err");
  }
}

/** Broadcast extension-local settings to content scripts. */
function broadcastLocal() {
  chrome.runtime.sendMessage({
    type: "settingsUpdated",
    settings: {
      host: $("host").value.trim() || "127.0.0.1",
      port: parseInt($("port").value, 10) || 24601,
      autoCapture: $("autoCapture").checked,
    },
  });
}

// ── Init ──────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadSettings();
  attachListeners();
});
