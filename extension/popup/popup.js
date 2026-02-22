/**
 * ShadowWrite — Popup Script
 *
 * Reads/writes settings from chrome.storage.sync and provides
 * a quick connection test to the local server.
 */

"use strict";

const $ = (id) => document.getElementById(id);

const FIELDS = ["enabled", "autoCapture", "host", "port"];

// ── Load settings ─────────────────────────────────────────────────
async function loadSettings() {
  const settings = await chrome.storage.sync.get({
    host: "127.0.0.1",
    port: 24601,
    autoCapture: true,
    enabled: true,
  });

  $("enabled").checked = settings.enabled;
  $("autoCapture").checked = settings.autoCapture;
  $("host").value = settings.host;
  $("port").value = settings.port;

  // Test connection on load
  testConnection(settings.host, settings.port);
}

// ── Save on change ────────────────────────────────────────────────
function attachListeners() {
  $("enabled").addEventListener("change", () => {
    chrome.storage.sync.set({ enabled: $("enabled").checked });
    chrome.runtime.sendMessage({
      type: "setEnabled",
      enabled: $("enabled").checked,
    });
  });

  $("autoCapture").addEventListener("change", () => {
    chrome.storage.sync.set({ autoCapture: $("autoCapture").checked });
  });

  // Debounce text inputs
  let saveTimer;
  const debounceSave = () => {
    clearTimeout(saveTimer);
    saveTimer = setTimeout(() => {
      const host = $("host").value.trim() || "127.0.0.1";
      const port = parseInt($("port").value, 10) || 24601;
      chrome.storage.sync.set({ host, port });
      chrome.runtime.sendMessage({
        type: "updateSettings",
        settings: { host, port },
      });
      testConnection(host, port);
    }, 600);
  };

  $("host").addEventListener("input", debounceSave);
  $("port").addEventListener("input", debounceSave);
}

// ── Connection test ───────────────────────────────────────────────
async function testConnection(host, port) {
  const statusEl = $("status");
  statusEl.className = "status";
  statusEl.textContent = "正在连接...";

  try {
    const resp = await fetch(`http://${host}:${port}/api/health`, {
      method: "GET",
      signal: AbortSignal.timeout(3000),
    });
    if (resp.ok) {
      statusEl.className = "status ok";
      statusEl.textContent = "✓ 本地服务已连接";
    } else {
      statusEl.className = "status err";
      statusEl.textContent = `✗ 服务返回 ${resp.status}`;
    }
  } catch {
    statusEl.className = "status err";
    statusEl.textContent = "✗ 无法连接本地服务";
  }
}

// ── Init ──────────────────────────────────────────────────────────
document.addEventListener("DOMContentLoaded", () => {
  loadSettings();
  attachListeners();
});
