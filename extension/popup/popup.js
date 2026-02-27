/**
 * ShadowWrite — Popup Script
 *
 * - "当前对话追踪" toggle: queries and controls the active tab's
 *   tracking state (synced with the floating dot indicator).
 * - "自动追踪新对话" toggle: stored in chrome.storage.sync.
 * - Host/Port: stored in chrome.storage.sync (extension-local).
 * - outputDir: per-conversation (chrome.storage.local + server),
 *   falls back to server global default.
 * - chatHtml: global server config.
 */

"use strict";

const $ = (id) => document.getElementById(id);

let activeConversationId = null;
let globalOutputDir = "";
let isClipMode = false;   // true when on a non-AI page (hidden clip feature)
let currentTabId = null;

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

function showDirHint(isCustom) {
  const hint = $("outputDirHint");
  const resetBtn = $("resetDir");
  if (isCustom) {
    hint.textContent = `全局默认: ${globalOutputDir}`;
    hint.style.display = "";
    resetBtn.style.display = "";
  } else {
    hint.style.display = "none";
    resetBtn.style.display = "none";
  }
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

  // 2. Query active tab for tracking state + conversationId
  await loadTrackingState();

  // 3. Server config (outputDir, chatHtml)
  await loadServerConfig();
}

/**
 * Query the active tab's content script for the current
 * tracking state and extract conversationId.
 * On non-AI pages (no content script), falls back to clip tracking mode.
 */
function loadTrackingState() {
  const toggle = $("trackingToggle");
  return new Promise((resolve) => {
    chrome.tabs.query({ active: true, currentWindow: true }).then(([tab]) => {
      if (!tab?.id) {
        toggle.checked = false;
        toggle.disabled = true;
        resolve();
        return;
      }
      currentTabId = tab.id;
      chrome.tabs.sendMessage(tab.id, { type: "getTrackingState" }, async (resp) => {
        if (chrome.runtime.lastError || !resp) {
          // No AI-platform content script → clip tracking mode
          isClipMode = true;
          toggle.disabled = false;
          try {
            const clipResp = await bg({ type: "getClipTrackingState", tabId: tab.id });
            toggle.checked = !!clipResp?.isTracking;
          } catch {
            toggle.checked = false;
          }
          resolve();
          return;
        }
        isClipMode = false;
        toggle.disabled = !resp.hasConversation;
        toggle.checked = !!resp.isTracking;
        activeConversationId = resp.conversationId || null;
        resolve();
      });
    }).catch(() => {
      toggle.checked = false;
      toggle.disabled = true;
      resolve();
    });
  });
}

async function loadServerConfig() {
  setStatus("正在连接…", "");
  try {
    const msg = { type: "getServerConfig" };
    if (activeConversationId) msg.conversationId = activeConversationId;

    const res = await bg(msg);
    if (res && res.success) {
      const cfg = res.data;
      globalOutputDir = cfg.globalDir || cfg.outputDir || "";
      $("chatHtml").checked = !!cfg.chatHtml;

      if (activeConversationId) {
        $("outputDirLabel").textContent = "输出目录 (当前对话)";

        // Check server-side custom dir first, then extension storage
        let customDir = null;
        if (cfg.isCustom) {
          customDir = cfg.outputDir;
        } else {
          const data = await chrome.storage.local.get({ convOutputDirs: {} });
          customDir = data.convOutputDirs[activeConversationId] || null;
        }

        if (customDir && customDir !== globalOutputDir) {
          $("outputDir").value = customDir;
          showDirHint(true);
        } else {
          $("outputDir").value = globalOutputDir;
          showDirHint(false);
        }
      } else {
        $("outputDirLabel").textContent = "输出目录 (全局)";
        $("outputDir").value = cfg.outputDir ?? "";
        showDirHint(false);
      }

      setStatus("✓ 本地服务已连接", "ok");
    } else {
      setStatus("✗ 无法读取服务器配置", "err");
    }
  } catch {
    setStatus("✗ 无法连接本地服务", "err");
  }
}

// ── Save ──────────────────────────────────────────────────────────

/** Push outputDir change — per-conversation or global depending on context. */
async function pushDirConfig() {
  const dir = $("outputDir").value.trim();

  if (activeConversationId) {
    // Per-conversation
    const isCustom = !!(dir && dir !== globalOutputDir);

    // Persist to extension storage
    const data = await chrome.storage.local.get({ convOutputDirs: {} });
    if (isCustom) {
      data.convOutputDirs[activeConversationId] = dir;
    } else {
      delete data.convOutputDirs[activeConversationId];
    }
    await chrome.storage.local.set(data);

    // Notify server (may trigger file move)
    try {
      const config = {
        conversationId: activeConversationId,
        outputDir: isCustom ? dir : null,
      };
      const res = await bg({ type: "setServerConfig", config });
      setStatus(res?.success ? "✓ 已保存" : "✗ 保存失败", res?.success ? "ok" : "err");
    } catch {
      setStatus("✗ 无法连接本地服务", "err");
    }

    showDirHint(isCustom);
    if (!isCustom) $("outputDir").value = globalOutputDir;
  } else {
    // Global config
    try {
      const config = { outputDir: dir || "./outputs" };
      const res = await bg({ type: "setServerConfig", config });
      if (res?.success) {
        globalOutputDir = config.outputDir;
        setStatus("✓ 已同步到服务器", "ok");
      } else {
        setStatus("✗ 保存失败", "err");
      }
    } catch {
      setStatus("✗ 无法连接本地服务", "err");
    }
  }
}

/** Push chatHtml change (always global). */
async function pushChatHtmlConfig() {
  try {
    const config = { chatHtml: $("chatHtml").checked };
    const res = await bg({ type: "setServerConfig", config });
    setStatus(res?.success ? "✓ 已同步到服务器" : "✗ 保存失败", res?.success ? "ok" : "err");
  } catch {
    setStatus("✗ 无法连接本地服务", "err");
  }
}

function attachListeners() {
  // ── Tracking toggle ─────────────────────────────────────────────
  $("trackingToggle").addEventListener("change", async () => {
    const enabled = $("trackingToggle").checked;

    if (isClipMode) {
      // Non-AI page: clip tracking via background
      try {
        const res = await bg({ type: "setClipTracking", tabId: currentTabId, enabled });
        if (enabled && res?.ok) {
          setStatus("📎 已剪藏", "ok");
        } else if (!enabled) {
          setStatus("✓ 已停止追踪", "ok");
        } else {
          setStatus("✗ 剪藏失败: " + (res?.error || ""), "err");
        }
      } catch {
        setStatus("✗ 无法连接本地服务", "err");
      }
      return;
    }

    // AI platform page: normal tracking toggle
    try {
      const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
      if (tab?.id) chrome.tabs.sendMessage(tab.id, { type: "setTracking", enabled });
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

  // ── Output dir (debounced) ──────────────────────────────────
  let dirTimer;
  $("outputDir").addEventListener("input", () => {
    clearTimeout(dirTimer);
    dirTimer = setTimeout(() => pushDirConfig(), 800);
  });

  // ── Chat HTML (immediate, always global) ────────────────────
  $("chatHtml").addEventListener("change", () => pushChatHtmlConfig());

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
        await pushDirConfig();
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

  // ── Reset dir to global default ─────────────────────────────
  $("resetDir").addEventListener("click", async () => {
    $("outputDir").value = globalOutputDir;
    await pushDirConfig();
  });
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
