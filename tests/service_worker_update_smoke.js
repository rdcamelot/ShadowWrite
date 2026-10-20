"use strict";

const fs = require("node:fs");
const path = require("node:path");
const vm = require("node:vm");

const ROOT = path.resolve(__dirname, "..");
const SERVICE_WORKER = fs.readFileSync(
  path.join(ROOT, "extension", "background", "service-worker.js"),
  "utf8",
);
const CURRENT_VERSION = "0.1.0+local.current";

function eventHook() {
  return { addListener() {} };
}

function createChromeMock({ loadedVersion, localState, reloadCalls, tabReloads, clipData, sessionState }) {
  return {
    runtime: {
      getManifest: () => ({ version: "0.1.0", version_name: loadedVersion }),
      reload: () => reloadCalls.push(loadedVersion),
      onInstalled: eventHook(),
      onStartup: eventHook(),
      onMessage: eventHook(),
      sendMessage: async () => ({ ok: true }),
    },
    storage: {
      sync: {
        get: async (defaults) => ({ ...(defaults || {}) }),
        set: async () => {},
      },
      local: {
        get: async (defaults) => ({ ...(defaults || {}), ...localState }),
        set: async (values) => Object.assign(localState, values),
        remove: async (key) => { delete localState[key]; },
      },
      session: {
        get: async (defaults) => ({ ...(defaults || {}), ...sessionState }),
        set: async (values) => Object.assign(sessionState, values),
      },
      onChanged: eventHook(),
    },
    alarms: {
      create() {},
      onAlarm: eventHook(),
    },
    tabs: {
      query: async () => [
        { id: 11, url: "https://gemini.google.com/app/test" },
        { id: 12, url: "https://example.com/" },
      ],
      reload: async (tabId) => tabReloads.push(tabId),
      sendMessage: async () => {},
      onUpdated: eventHook(),
      onRemoved: eventHook(),
    },
    contextMenus: {
      create() {},
      removeAll: async () => {},
      onClicked: eventHook(),
    },
    action: {
      setBadgeText() {},
      setBadgeBackgroundColor() {},
    },
    scripting: { executeScript: async () => [{ result: clipData }] },
  };
}

async function runWorker(loadedVersion, shared) {
  const chrome = createChromeMock({ loadedVersion, ...shared });
  const context = {
    chrome,
    URL,
    AbortSignal,
    setTimeout,
    clearTimeout,
    fetch: async (url, options) => {
      if (new URL(url).pathname === "/api/clip") {
        shared.clipRequests.push(JSON.parse(options.body));
        return { ok: true, json: async () => ({ title: "Chapter One", path: "clip.md" }) };
      }
      return {
        ok: true,
        json: async () => ({
          service: "ShadowWrite",
          extensionVersion: "0.1.0",
          extensionVersionName: CURRENT_VERSION,
        }),
      };
    },
    console,
  };
  vm.runInNewContext(SERVICE_WORKER, context, { filename: "service-worker.js" });
  await new Promise((resolve) => setTimeout(resolve, 30));
  return context;
}

async function main() {
  const shared = {
    localState: {},
    sessionState: { _clipTabs: { "11": { lastUrl: "https://example.com/book/1", group: "Old group" } } },
    reloadCalls: [],
    tabReloads: [],
    clipRequests: [],
    clipData: {
      title: "Chapter One", url: "https://example.com/book/1",
      domain: "example.com", content: "Chapter text", group: "Example Book", groupIsBook: true,
    },
  };

  await runWorker("0.1.0+local.old", shared);
  if (shared.reloadCalls.length !== 1) {
    throw new Error(`expected one extension reload, got ${shared.reloadCalls.length}`);
  }
  const pending = shared.localState._shadowwritePendingReloadTabs;
  if (JSON.stringify(pending) !== JSON.stringify([11])) {
    throw new Error(`expected only the Gemini tab to be pending, got ${JSON.stringify(pending)}`);
  }

  // If Chrome reloads from a different unpacked directory, the same old build
  // must not enter an immediate self-reload loop.
  await runWorker("0.1.0+local.old", shared);
  if (shared.reloadCalls.length !== 1) {
    throw new Error("the same failed version pair should not reload twice");
  }

  const currentWorker = await runWorker(CURRENT_VERSION, shared);
  if (shared.reloadCalls.length !== 1) {
    throw new Error("current extension version should not reload again");
  }
  if (JSON.stringify(shared.tabReloads) !== JSON.stringify([11])) {
    throw new Error(`expected pending Gemini tab refresh, got ${JSON.stringify(shared.tabReloads)}`);
  }
  if (shared.localState._shadowwritePendingReloadTabs !== undefined) {
    throw new Error("pending reload tabs should be cleared after extension reload");
  }
  if (shared.localState._shadowwriteExtensionUpdateAttempt !== undefined) {
    throw new Error("update attempt should be cleared after the new version loads");
  }

  console.log("OK service-worker-extension-update");

  await currentWorker.doClipTab(11, "Old chapter subtitle group");
  if (shared.clipRequests[0]?.group !== "Example Book") {
    throw new Error("current book name should override the old tracking group");
  }
  if (shared.sessionState._clipTabs["11"].group !== "Example Book") {
    throw new Error("the identified book name should persist for subsequent chapter pages");
  }
  shared.clipData.group = "";
  shared.clipData.groupIsBook = false;
  await currentWorker.doClipTab(11, "Previous series");
  if (shared.clipRequests[1]?.group !== "Previous series") {
    throw new Error("tracking group should remain a fallback when no book is identified");
  }
  shared.clipData.group = "Chapter subtitle";
  await currentWorker.doClipTab(11, "Previous series");
  if (shared.clipRequests[2]?.group !== "Previous series") {
    throw new Error("a guessed chapter subtitle must not replace the tracking group");
  }
  console.log("OK service-worker-clip-book-group");
}

main().catch((error) => {
  console.error(error);
  process.exitCode = 1;
});
