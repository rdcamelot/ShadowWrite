"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { spawnSync } = require("node:child_process");

const ROOT = path.resolve(__dirname, "..");
const TMP_DIR = path.join(ROOT, ".tmp_adapter_dom_smoke");

function findChrome() {
  const candidates = [
    process.env.CHROME_PATH,
    path.join(process.env.LOCALAPPDATA || "", "Google", "Chrome", "Application", "chrome.exe"),
    path.join(process.env.PROGRAMFILES || "", "Google", "Chrome", "Application", "chrome.exe"),
    path.join(process.env["PROGRAMFILES(X86)"] || "", "Google", "Chrome", "Application", "chrome.exe"),
    path.join(process.env.PROGRAMFILES || "", "Microsoft", "Edge", "Application", "msedge.exe"),
  ].filter(Boolean);

  for (const candidate of candidates) {
    if (fs.existsSync(candidate)) return candidate;
  }
  throw new Error("Chrome/Edge executable not found. Set CHROME_PATH to run this smoke test.");
}

function inlineScript(filePath) {
  return fs.readFileSync(filePath, "utf8").replace(/<\/script/gi, "<\\/script");
}

function buildHtml(adapterName, body, afterExtract = "") {
  const base = inlineScript(path.join(ROOT, "extension", "content", "base-adapter.js"));
  const adapter = inlineScript(path.join(ROOT, "extension", "content", "adapters", `${adapterName}.js`));
  return `<!doctype html>
<html>
<head><meta charset="utf-8"><title>Adapter Smoke</title></head>
<body>
${body}
<script>
window.chrome = {
  storage: {
    sync: { get: async (defaults) => defaults || {} },
    local: { get: async (defaults) => defaults || {}, set: async () => {} }
  },
  runtime: {
    onMessage: { addListener: () => {} },
    sendMessage: async () => ({ ok: true, body: "{}" })
  }
};
</script>
<script>${base}</script>
<script>${adapter}</script>
<script>
${afterExtract}
const messages = window.__shadowWriteAdapter.extractMessages();
document.body.setAttribute("data-result", encodeURIComponent(JSON.stringify(messages)));
</script>
</body>
</html>`;
}

function runCase(chrome, testCase) {
  fs.mkdirSync(TMP_DIR, { recursive: true });
  const htmlPath = path.join(TMP_DIR, `${testCase.name}.html`);
  fs.writeFileSync(htmlPath, testCase.html, "utf8");

  const result = spawnSync(chrome, [
    "--headless=new",
    "--disable-gpu",
    "--disable-gpu-compositing",
    "--disable-gpu-sandbox",
    "--disable-software-rasterizer",
    "--disable-dev-shm-usage",
    "--disable-features=UseSkiaRenderer,VizDisplayCompositor",
    "--no-sandbox",
    "--single-process",
    "--no-first-run",
    "--no-default-browser-check",
    ...(testCase.virtualTimeBudget ? [`--virtual-time-budget=${testCase.virtualTimeBudget}`] : []),
    "--dump-dom",
    `file:///${htmlPath.replace(/\\/g, "/")}`,
  ], {
    encoding: "utf8",
    timeout: 30000,
    windowsHide: true,
  });

  if (result.error) throw result.error;
  if (result.status !== 0) {
    throw new Error(`${testCase.name}: Chrome exited ${result.status}\n${result.stderr}`);
  }

  const match = result.stdout.match(/data-result="([^"]*)"/);
  if (!match) {
    throw new Error(`${testCase.name}: data-result not found in dumped DOM`);
  }

  const messages = JSON.parse(decodeURIComponent(match[1]));
  testCase.assert(messages);
  console.log(`OK ${testCase.name}: ${messages.length} message(s)`);
}

function buildRetryHtml() {
  const base = inlineScript(path.join(ROOT, "extension", "content", "base-adapter.js"));
  const chatgpt = inlineScript(path.join(ROOT, "extension", "content", "adapters", "chatgpt.js"));
  return `<!doctype html>
<html>
<head><meta charset="utf-8"><title>Retry Smoke</title></head>
<body>
<main>
  <article>
    <div data-message-author-role="user">retry user</div>
    <div data-message-author-role="assistant"><div class="markdown prose">retry assistant</div></div>
  </article>
</main>
<script>
let sendCount = 0;
window.chrome = {
  storage: {
    sync: { get: async (defaults) => defaults || {} },
    local: { get: async (defaults) => defaults || {}, set: async () => {} }
  },
  runtime: {
    onMessage: { addListener: () => {} },
    sendMessage: async () => {
      sendCount += 1;
      return sendCount === 1
        ? { ok: false, error: "server unavailable" }
        : { ok: true, body: "{}" };
    }
  }
};
</script>
<script>${base}</script>
<script>${chatgpt}</script>
<script>
window.__shadowWriteAdapter.currentConversationId = "retry_conv";
window.__shadowWriteAdapter.pageUrl = "https://chatgpt.com/c/retry_conv";
window.__shadowWriteAdapter.isTracking = true;
window.__shadowWriteAdapter._sendToService([
  { messageId: "m1", sender: "user", content: "retry user", thinking: "", position: 0 },
  { messageId: "m2", sender: "AI", content: "retry assistant", thinking: "", position: 1 }
]);
setTimeout(() => {
  document.body.setAttribute("data-result", encodeURIComponent(JSON.stringify([
    { messageId: "sendCount", sender: "test", content: String(sendCount), thinking: "", position: 0 }
  ])));
}, 5600);
</script>
</body>
</html>`;
}

function buildIdleMutationHtml() {
  const base = inlineScript(path.join(ROOT, "extension", "content", "base-adapter.js"));
  const chatgpt = inlineScript(path.join(ROOT, "extension", "content", "adapters", "chatgpt.js"));
  return `<!doctype html>
<html>
<head><meta charset="utf-8"><title>Idle Mutation Smoke</title></head>
<body>
<main>
  <article>
    <div data-message-author-role="user">idle user</div>
    <div data-message-author-role="assistant"><div class="markdown prose"><span id="reply">idle assistant</span></div></div>
  </article>
</main>
<script>
let sendCount = 0;
let lastPayloadCount = 0;
let lastAssistantContent = "";
window.chrome = {
  storage: {
    sync: { get: async (defaults) => defaults || {} },
    local: { get: async (defaults) => defaults || {}, set: async () => {} }
  },
  runtime: {
    onMessage: { addListener: () => {} },
    sendMessage: async (message) => {
      if (message?.type === "sendToServer") {
        sendCount += 1;
        lastPayloadCount = message.payload?.messages?.length || 0;
        lastAssistantContent = message.payload?.messages?.find((item) => item.sender === "AI")?.content || "";
      }
      return { ok: true, body: "{}" };
    }
  }
};
</script>
<script>${base}</script>
<script>${chatgpt}</script>
<script>
const idleAdapter = window.__shadowWriteAdapter;
idleAdapter.currentConversationId = "idle_conv";
idleAdapter.pageUrl = "https://chatgpt.com/c/idle_conv";
idleAdapter.isTracking = true;
idleAdapter.lastMessagesSignature = idleAdapter._buildSnapshotSignature(idleAdapter.extractMessages());
idleAdapter._setupMutationObserver();
document.getElementById("reply").firstChild.nodeValue = "idle assistant updated";
setTimeout(() => {
  document.body.setAttribute("data-result", encodeURIComponent(JSON.stringify([
    {
      messageId: "idleMutation",
      sender: "test",
      content: JSON.stringify({ sendCount, lastPayloadCount, lastAssistantContent }),
      thinking: "",
      position: 0
    }
  ])));
}, 2600);
</script>
</body>
</html>`;
}

function buildAttributeMutationHtml() {
  const base = inlineScript(path.join(ROOT, "extension", "content", "base-adapter.js"));
  const chatgpt = inlineScript(path.join(ROOT, "extension", "content", "adapters", "chatgpt.js"));
  return `<!doctype html>
<html>
<head><meta charset="utf-8"><title>Attribute Mutation Smoke</title></head>
<body>
<main>
  <article>
    <div data-message-author-role="user">image user</div>
    <div data-message-author-role="assistant">
      <div class="markdown prose">image assistant <img id="reply-image" alt="pending image"></div>
    </div>
  </article>
</main>
<script>
let sendCount = 0;
let lastAssistantContent = "";
window.chrome = {
  storage: {
    sync: { get: async (defaults) => defaults || {} },
    local: { get: async (defaults) => defaults || {}, set: async () => {} }
  },
  runtime: {
    onMessage: { addListener: () => {} },
    sendMessage: async (message) => {
      if (message?.type === "sendToServer") {
        sendCount += 1;
        lastAssistantContent = message.payload?.messages?.find((item) => item.sender === "AI")?.content || "";
      }
      return { ok: true, body: "{}" };
    }
  }
};
</script>
<script>${base}</script>
<script>${chatgpt}</script>
<script>
const attributeAdapter = window.__shadowWriteAdapter;
attributeAdapter.currentConversationId = "attribute_conv";
attributeAdapter.pageUrl = "https://chatgpt.com/c/attribute_conv";
attributeAdapter.isTracking = true;
attributeAdapter.lastMessagesSignature = attributeAdapter._buildSnapshotSignature(attributeAdapter.extractMessages());
attributeAdapter._setupMutationObserver();
const replyImage = document.getElementById("reply-image");
replyImage.setAttribute("alt", "final image");
replyImage.setAttribute("src", "https://example.com/final.png");
setTimeout(() => {
  document.body.setAttribute("data-result", encodeURIComponent(JSON.stringify([
    {
      messageId: "attributeMutation",
      sender: "test",
      content: JSON.stringify({ sendCount, lastAssistantContent }),
      thinking: "",
      position: 0
    }
  ])));
}, 2600);
</script>
</body>
</html>`;
}

function buildContextInjectionHtml(
  trigger,
  withButton = true,
  contextContent = "## Memory\n- Persistent fact"
) {
  const base = inlineScript(path.join(ROOT, "extension", "content", "base-adapter.js"));
  const chatgpt = inlineScript(path.join(ROOT, "extension", "content", "adapters", "chatgpt.js"));
  const button = withButton ? '<button data-testid="send-button">Send</button>' : "";
  const enterScript = `input.dispatchEvent(new KeyboardEvent("keydown", {
        key: "Enter", code: "Enter", keyCode: 13, bubbles: true, cancelable: true
      }));`;
  const triggerScript = trigger === "click"
    ? 'document.querySelector(\'[data-testid="send-button"]\').click();'
    : trigger === "visible-enter"
      ? `adapter.injectContextVisible().then(() => { ${enterScript} });`
      : enterScript;

  return `<!doctype html>
<html>
<head><meta charset="utf-8"><title>Context Injection Smoke</title></head>
<body>
<main>
  <div id="prompt-textarea" contenteditable="true">Original question</div>
  ${button}
</main>
<script>
let contextFetchCount = 0;
window.chrome = {
  storage: {
    sync: { get: async (defaults) => defaults || {} },
    local: { get: async (defaults) => defaults || {}, set: async () => {} }
  },
  runtime: {
    onMessage: { addListener: () => {} },
    sendMessage: async (message) => {
      if (message?.type === "getContext") {
        contextFetchCount += 1;
        return { success: true, data: { content: ${JSON.stringify(contextContent)} } };
      }
      return { ok: true, body: "{}" };
    }
  }
};
</script>
<script>${base}</script>
<script>${chatgpt}</script>
<script>
setTimeout(() => {
  const adapter = window.__shadowWriteAdapter;
  const input = document.getElementById("prompt-textarea");
  const sendButton = document.querySelector('[data-testid="send-button"]');
  let sendCount = 0;
  let keyboardSendCount = 0;
  let sentText = "";

  input.addEventListener("keydown", (event) => {
    if (event.key === "Enter") {
      keyboardSendCount += 1;
      sendCount += 1;
      sentText = input.textContent || "";
    }
  });
  sendButton?.addEventListener("click", () => {
    sendCount += 1;
    sentText = input.textContent || "";
  });

  adapter.currentConversationId = "context-conversation";
  adapter.isTracking = true;
  adapter._contextMode = "inject";
  adapter._setupHiddenInject();
  ${triggerScript}

  setTimeout(() => {
    const strippedText = adapter._stripInjectedContextPrefix(sentText);
    document.body.setAttribute("data-result", encodeURIComponent(JSON.stringify([
      {
        messageId: "contextInjection",
        sender: "test",
        content: JSON.stringify({ sendCount, keyboardSendCount, contextFetchCount, sentText, strippedText }),
        thinking: "",
        position: 0
      }
    ])));
  }, 350);
}, 0);
</script>
</body>
</html>`;
}

function buildContextUpdateDedupeHtml() {
  const base = inlineScript(path.join(ROOT, "extension", "content", "base-adapter.js"));
  const chatgpt = inlineScript(path.join(ROOT, "extension", "content", "adapters", "chatgpt.js"));
  return `<!doctype html>
<html>
<head><meta charset="utf-8"><title>Context Update Dedupe Smoke</title></head>
<body>
<main></main>
<script>
const localState = {};
let postCount = 0;
let postedBlocks = 0;
let postedInlines = 0;
window.chrome = {
  storage: {
    sync: { get: async (defaults) => defaults || {} },
    local: {
      get: async (defaults) => ({ ...(defaults || {}), ...localState }),
      set: async (values) => Object.assign(localState, values)
    }
  },
  runtime: {
    onMessage: { addListener: () => {} },
    sendMessage: async (message) => {
      if (message?.type === "postContext") {
        postCount += 1;
        postedBlocks += message.payload?.blocks?.length || 0;
        postedInlines += message.payload?.inlines?.length || 0;
        return { success: true, data: { count: postedBlocks + postedInlines } };
      }
      if (message?.type === "getContext") return { success: false, error: "missing" };
      return { ok: true, body: "{}" };
    }
  }
};
</script>
<script>${base}</script>
<script>${chatgpt}</script>
<script>
setTimeout(async () => {
  const adapter = window.__shadowWriteAdapter;
  adapter.currentConversationId = "context-dedupe";
  adapter._contextMode = "auto-summary";
  const messages = [{
    messageId: "assistant-1",
    sender: "AI",
    content: "<!-- context-update-start -->\\n## Canon\\n- Persistent fact\\n<!-- context-update-end -->\\n<!-- context-update: inline fact -->",
    thinking: "",
    position: 0
  }];

  await adapter._extractAndSaveContextUpdates(messages);
  await adapter._extractAndSaveContextUpdates(messages);
  adapter._contextContent = "stale conversation memory";
  adapter.currentConversationId = "new-conversation";
  await adapter._fetchContext();
  const historyCount = Object.values(localState)[0]?.length || 0;

  document.body.setAttribute("data-result", encodeURIComponent(JSON.stringify([
    {
      messageId: "contextDedupe",
      sender: "test",
      content: JSON.stringify({
        postCount,
        postedBlocks,
        postedInlines,
        historyCount,
        staleContent: adapter._contextContent
      }),
      thinking: "",
      position: 0
    }
  ])));
}, 0);
</script>
</body>
</html>`;
}

function buildCommonIsolationHtml() {
  const common = inlineScript(path.join(ROOT, "extension", "content", "content-common.js"));
  return `<!doctype html>
<html>
<head><meta charset="utf-8"><title>Common Isolation Smoke</title></head>
<body>
<script>
let hostClicks = 0;
let enableCount = 0;
document.addEventListener("click", () => {
  hostClicks += 1;
}, true);
window.__shadowWriteAdapter = {
  currentConversationId: "isolation_conv",
  isTracking: false,
  _contextInvalidated: false,
  enableTracking() {
    enableCount += 1;
    this.isTracking = true;
  },
  disableTracking() {
    this.isTracking = false;
  }
};
</script>
<script>${common}</script>
<script>
setTimeout(() => {
  const frame = document.getElementById("shadowwrite-status-frame");
  const dot = frame?.contentDocument?.getElementById("shadowwrite-status")
    || document.getElementById("shadowwrite-status");
  if (dot) {
    const eventWindow = frame?.contentWindow || window;
    dot.dispatchEvent(new eventWindow.MouseEvent("click", {
      bubbles: true,
      cancelable: true,
      view: eventWindow,
    }));
  }
  setTimeout(() => {
    document.body.setAttribute("data-result", encodeURIComponent(JSON.stringify([
      {
        messageId: "isolation",
        sender: "test",
        content: JSON.stringify({
          hasFrame: !!frame,
          hasDot: !!dot,
          dotConnected: !!dot?.isConnected,
          hostClicks,
          enableCount,
        }),
        thinking: "",
        position: 0
      }
    ])));
  }, 0);
}, 0);
</script>
</body>
</html>`;
}

function buildDisabledContextMigrationHtml() {
  const base = inlineScript(path.join(ROOT, "extension", "content", "base-adapter.js"));
  const chatgpt = inlineScript(path.join(ROOT, "extension", "content", "adapters", "chatgpt.js"));
  return `<!doctype html>
<html>
<head><meta charset="utf-8"><title>Disabled Context Migration Smoke</title></head>
<body>
<main>
  <div id="prompt-textarea" contenteditable="true"></div>
  <button data-testid="send-button">Send</button>
</main>
<script>
const storedSettings = { contextMode: "inject" };
let runtimeListener = null;
window.chrome = {
  storage: {
    sync: {
      get: async (defaults) => ({ ...(defaults || {}), ...storedSettings }),
      set: async (values) => Object.assign(storedSettings, values)
    },
    local: { get: async (defaults) => defaults || {}, set: async () => {} }
  },
  runtime: {
    onMessage: { addListener: (listener) => { runtimeListener = listener; } },
    sendMessage: async () => ({ ok: true, body: "{}" })
  }
};
</script>
<script>${base}</script>
<script>${chatgpt}</script>
<script>
setTimeout(() => {
  const adapter = window.__shadowWriteAdapter;
  adapter.currentConversationId = "context-disabled";
  adapter.isTracking = true;

  // Simulate an already-installed hook, then verify legacy mode messages
  // can only disable it in the current extension release.
  adapter._contextMode = "inject";
  adapter._setupHiddenInject();
  runtimeListener?.({ type: "setContextMode", mode: "inject" }, {}, () => {});

  document.body.setAttribute("data-result", encodeURIComponent(JSON.stringify([{
    messageId: "contextDisabled",
    sender: "test",
    content: JSON.stringify({
      mode: adapter._contextMode,
      settingsMode: adapter.settings.contextMode,
      storedMode: storedSettings.contextMode,
      submitHooked: adapter._submitHooked
    }),
    thinking: "",
    position: 0
  }])));
}, 50);
</script>
</body>
</html>`;
}

function assertIncludes(value, expected, label) {
  if (!String(value).includes(expected)) {
    throw new Error(`${label}: expected ${JSON.stringify(value)} to include ${JSON.stringify(expected)}`);
  }
}

function assertExcludes(value, unexpected, label) {
  if (String(value).includes(unexpected)) {
    throw new Error(`${label}: expected ${JSON.stringify(value)} to exclude ${JSON.stringify(unexpected)}`);
  }
}

function main() {
  const chrome = findChrome();
  const cases = [
    {
      name: "gemini-conversation-container",
      html: buildHtml("gemini", `
        <div id="chat-history">
          <div class="conversation-container">
            <user-query><div class="query-text">你说 hello gemini</div></user-query>
            <model-response><div class="model-response-text"><p>hi <strong>there</strong></p></div></model-response>
          </div>
        </div>
      `),
      assert(messages) {
        if (messages.length !== 2) throw new Error(`expected 2 messages, got ${messages.length}`);
        assertIncludes(messages[0].content, "hello gemini", "Gemini user content");
        assertIncludes(messages[1].content, "**there**", "Gemini AI markdown");
      },
    },
    {
      name: "gemini-turn-fallback",
      html: buildHtml("gemini", `
        <div id="chat-history">
          <user-query><div class="query-text">Question without wrapper</div></user-query>
          <model-response><div class="model-response-text">Answer without wrapper</div></model-response>
        </div>
      `),
      assert(messages) {
        if (messages.length !== 2) throw new Error(`expected 2 messages, got ${messages.length}`);
        assertIncludes(messages[0].content, "Question without wrapper", "Gemini fallback user content");
        assertIncludes(messages[1].content, "Answer without wrapper", "Gemini fallback AI content");
      },
    },
    {
      name: "gemini-main-root-fallback",
      html: buildHtml("gemini", `
        <main>
          <user-query><div data-test-id="user-query-content">Gemini main user</div></user-query>
          <model-response><div data-test-id="response-content">Gemini main assistant</div></model-response>
        </main>
      `),
      assert(messages) {
        if (messages.length !== 2) throw new Error(`expected 2 messages, got ${messages.length}`);
        assertIncludes(messages[0].content, "Gemini main user", "Gemini main fallback user");
        assertIncludes(messages[1].content, "Gemini main assistant", "Gemini main fallback AI");
      },
    },
    {
      name: "gemini-current-sidebar-title",
      html: buildHtml("gemini", `
        <nav>
          <a href="/app/other"><span data-test-id="conversation-title">Wrong conversation</span></a>
          <a href="/app/e38cfb0d8fe5271c" aria-current="page">
            <span class="conversation-title">异世界融合</span>
          </a>
        </nav>
        <main>
          <user-query><div class="query-text">参考小说或者 RPG 式的形式，构想一个很长的新设定</div></user-query>
          <model-response><div data-test-id="response-content">placeholder</div></model-response>
        </main>
      `, `
        window.__shadowWriteAdapter.pageUrl = "https://gemini.google.com/app/e38cfb0d8fe5271c";
        document.querySelector('[data-test-id="response-content"]').textContent =
          window.__shadowWriteAdapter.extractTitle() || "__NO_TITLE__";
      `),
      assert(messages) {
        if (messages.length !== 2) throw new Error(`expected 2 messages, got ${messages.length}`);
        if (messages[1].content !== "异世界融合") {
          throw new Error(`unexpected current Gemini title: ${messages[1].content}`);
        }
      },
    },
    {
      name: "gemini-ignores-temporary-prompt-title",
      html: buildHtml("gemini", `
        <nav>
          <a href="/app/e38cfb0d8fe5271c" aria-current="page">
            <span data-test-id="conversation-title">参考小说或者RPG式的形式，构想一个新的设定，我是一个穿越者，正如常见的设定一样，我有着极强的天赋，预设我的能力是类似于领域系空间系的能力</span>
          </a>
        </nav>
        <main>
          <user-query><div class="query-text">参考小说或者RPG式的形式，构想一个新的设定，我是一个穿越者，正如常见的设定一样，我有着极强的天赋，预设我的能力是类似于领域系空间系的能力</div></user-query>
          <model-response><div data-test-id="response-content">placeholder</div></model-response>
        </main>
      `, `
        window.__shadowWriteAdapter.pageUrl = "https://gemini.google.com/app/e38cfb0d8fe5271c";
        document.title = "Google Gemini";
        document.querySelector('[data-test-id="response-content"]').textContent =
          window.__shadowWriteAdapter.extractTitle() || "__NO_TITLE__";
      `),
      assert(messages) {
        if (messages.length !== 2) throw new Error(`expected 2 messages, got ${messages.length}`);
        if (messages[1].content !== "__NO_TITLE__") {
          throw new Error(`temporary prompt leaked into Gemini title: ${messages[1].content}`);
        }
      },
    },
    {
      name: "gemini-page-title-strips-brand",
      html: buildHtml("gemini", `
        <main>
          <user-query><div class="query-text">short question</div></user-query>
          <model-response><div data-test-id="response-content">placeholder</div></model-response>
        </main>
      `, `
        window.__shadowWriteAdapter.pageUrl = "https://gemini.google.com/app/e38cfb0d8fe5271c";
        document.title = "异世界融合：常陆的开局 - Google Gemini";
        document.querySelector('[data-test-id="response-content"]').textContent =
          window.__shadowWriteAdapter.extractTitle() || "__NO_TITLE__";
      `),
      assert(messages) {
        if (messages.length !== 2) throw new Error(`expected 2 messages, got ${messages.length}`);
        if (messages[1].content !== "异世界融合：常陆的开局") {
          throw new Error(`unexpected Gemini page-title fallback: ${messages[1].content}`);
        }
      },
    },
    {
      name: "gemini-nonhex-conversation-id",
      html: buildHtml("gemini", `
        <main><user-query>placeholder</user-query></main>
      `, `
        document.querySelector("user-query").textContent =
          window.__shadowWriteAdapter.extractConversationInfo(
            "https://gemini.google.com/app/ABC_12-xy"
          ).conversationId;
      `),
      assert(messages) {
        if (messages.length !== 1) throw new Error(`expected 1 message, got ${messages.length}`);
        if (messages[0].content !== "gemini_ABC_12-xy") {
          throw new Error(`unexpected Gemini conversation ID: ${messages[0].content}`);
        }
      },
    },
    {
      name: "gemini-strips-injected-context-from-export",
      html: buildHtml("gemini", `
        <main>
          <user-query><div class="query-text">
            IMPORTANT: A persistent context file is attached to this session.
            <hr>
            Gemini clean question
          </div></user-query>
        </main>
      `),
      assert(messages) {
        if (messages.length !== 1) throw new Error(`expected 1 message, got ${messages.length}`);
        if (messages[0].content !== "Gemini clean question") {
          throw new Error(`unexpected Gemini exported user content: ${messages[0].content}`);
        }
      },
    },
    {
      name: "chatgpt-focused-composer-does-not-block",
      html: buildHtml("chatgpt", `
        <main>
          <article>
            <div data-message-author-role="user">Plain user fallback</div>
            <section data-message-author-role="assistant"><div class="markdown prose"><p>Assistant <em>reply</em></p></div></section>
          </article>
          <textarea id="composer">still focused</textarea>
        </main>
      `, `document.getElementById("composer").focus();`),
      assert(messages) {
        if (messages.length !== 2) throw new Error(`expected 2 messages, got ${messages.length}`);
        assertIncludes(messages[0].content, "Plain user fallback", "ChatGPT user fallback");
        assertIncludes(messages[1].content, "*reply*", "ChatGPT assistant markdown");
      },
    },
    {
      name: "chatgpt-role-elements-without-article",
      html: buildHtml("chatgpt", `
        <main>
          <div data-message-author-role="user"><div class="whitespace-pre-wrap">ChatGPT direct user</div></div>
          <section data-message-author-role="assistant">
            <div class="markdown prose">ChatGPT <strong>direct assistant</strong></div>
          </section>
        </main>
      `),
      assert(messages) {
        if (messages.length !== 2) throw new Error(`expected 2 messages, got ${messages.length}`);
        assertIncludes(messages[0].content, "ChatGPT direct user", "ChatGPT direct user content");
        assertIncludes(messages[1].content, "**direct assistant**", "ChatGPT direct AI content");
      },
    },
    {
      name: "chatgpt-strips-injected-context-from-export",
      html: buildHtml("chatgpt", `
        <main>
          <div data-message-author-role="user" class="whitespace-pre-wrap">
            IMPORTANT: A persistent context file is attached to this session.
            <hr>
            ChatGPT clean question
          </div>
        </main>
      `),
      assert(messages) {
        if (messages.length !== 1) throw new Error(`expected 1 message, got ${messages.length}`);
        if (messages[0].content !== "ChatGPT clean question") {
          throw new Error(`unexpected ChatGPT exported user content: ${messages[0].content}`);
        }
      },
    },
    {
      name: "claude-assistant-testid",
      html: buildHtml("claude", `
        <main>
          <div data-testid="user-message">Claude user</div>
          <div data-testid="assistant-message"><p>Claude <strong>assistant</strong></p></div>
        </main>
      `),
      assert(messages) {
        if (messages.length !== 2) throw new Error(`expected 2 messages, got ${messages.length}`);
        assertIncludes(messages[1].content, "**assistant**", "Claude assistant fallback");
      },
    },
    {
      name: "deepseek-basic-turns",
      html: buildHtml("deepseek", `
        <main>
          <div class="_9663006"><div class="fbb737a4">DeepSeek user</div></div>
          <div class="_4f9bf79 _43c05b5">
            <div class="ds-message"><div class="ds-markdown">DeepSeek <strong>assistant</strong></div></div>
          </div>
          <textarea id="composer">focused composer</textarea>
        </main>
      `, `document.getElementById("composer").focus();`),
      assert(messages) {
        if (messages.length !== 2) throw new Error(`expected 2 messages, got ${messages.length}`);
        assertIncludes(messages[0].content, "DeepSeek user", "DeepSeek user content");
        assertIncludes(messages[1].content, "**assistant**", "DeepSeek assistant markdown");
      },
    },
    {
      name: "deepseek-semantic-role-fallback",
      html: buildHtml("deepseek", `
        <main>
          <div data-message-author-role="user"><div data-testid="message-content">DeepSeek semantic user</div></div>
          <div data-message-author-role="assistant">
            <div data-testid="thinking-content">DeepSeek private thought</div>
            <div data-testid="message-content">DeepSeek semantic assistant</div>
            <div role="toolbar"><button>Copy response</button></div>
          </div>
        </main>
      `),
      assert(messages) {
        if (messages.length !== 2) throw new Error(`expected 2 messages, got ${messages.length}`);
        assertIncludes(messages[1].thinking, "private thought", "DeepSeek fallback thinking");
        assertIncludes(messages[1].content, "semantic assistant", "DeepSeek fallback AI content");
        assertExcludes(messages[1].content, "Copy response", "DeepSeek fallback toolbar");
        assertExcludes(messages[1].content, "private thought", "DeepSeek fallback thinking duplication");
      },
    },
    {
      name: "doubao-basic-turns",
      html: buildHtml("doubao", `
        <main>
          <div data-testid="union_message">
            <div data-testid="send_message"><div data-testid="message_text_content">Doubao user</div></div>
          </div>
          <div data-testid="union_message">
            <div data-testid="receive_message"><div data-testid="message_text_content">Doubao assistant</div></div>
          </div>
          <textarea id="composer">focused composer</textarea>
        </main>
      `, `document.getElementById("composer").focus();`),
      assert(messages) {
        if (messages.length !== 2) throw new Error(`expected 2 messages, got ${messages.length}`);
        assertIncludes(messages[0].content, "Doubao user", "Doubao user content");
        assertIncludes(messages[1].content, "Doubao assistant", "Doubao assistant content");
      },
    },
    {
      name: "doubao-direct-role-markers",
      html: buildHtml("doubao", `
        <main>
          <div data-testid="send_message"><div data-testid="message_text_content">Doubao direct user</div></div>
          <div data-testid="receive_message"><div data-testid="message_text_content">Doubao direct assistant</div></div>
        </main>
      `),
      assert(messages) {
        if (messages.length !== 2) throw new Error(`expected 2 messages, got ${messages.length}`);
        assertIncludes(messages[0].content, "Doubao direct user", "Doubao direct user content");
        assertIncludes(messages[1].content, "Doubao direct assistant", "Doubao direct AI content");
      },
    },
    {
      name: "kimi-basic-turns",
      html: buildHtml("kimi", `
        <main>
          <div class="chat-content-item chat-content-item-user"><div class="user-content">Kimi user</div></div>
          <div class="chat-content-item chat-content-item-assistant">
            <div class="markdown-container">Kimi <em>assistant</em></div>
          </div>
          <textarea id="composer">focused composer</textarea>
        </main>
      `, `document.getElementById("composer").focus();`),
      assert(messages) {
        if (messages.length !== 2) throw new Error(`expected 2 messages, got ${messages.length}`);
        assertIncludes(messages[0].content, "Kimi user", "Kimi user content");
        assertIncludes(messages[1].content, "*assistant*", "Kimi assistant markdown");
      },
    },
    {
      name: "kimi-semantic-role-fallback",
      html: buildHtml("kimi", `
        <main>
          <div data-role="user"><div data-testid="message-content">Kimi semantic user</div></div>
          <div data-role="assistant">
            <div data-testid="thinking-content">Kimi hidden thought</div>
            <div data-testid="message-content">Kimi semantic assistant</div>
            <button>Regenerate</button>
          </div>
        </main>
      `),
      assert(messages) {
        if (messages.length !== 2) throw new Error(`expected 2 messages, got ${messages.length}`);
        assertIncludes(messages[1].content, "Kimi semantic assistant", "Kimi fallback AI content");
        assertExcludes(messages[1].content, "hidden thought", "Kimi fallback thinking content");
        assertExcludes(messages[1].content, "Regenerate", "Kimi fallback button");
      },
    },
    {
      name: "yuanbao-basic-turns",
      html: buildHtml("yuanbao", `
        <main>
          <div class="agent-chat__list__item--human"><div class="hyc-content-text">Yuanbao user</div></div>
          <div class="agent-chat__list__item--ai"><div class="hyc-component-reasoner__text">Yuanbao assistant</div></div>
          <textarea id="composer">focused composer</textarea>
        </main>
      `, `document.getElementById("composer").focus();`),
      assert(messages) {
        if (messages.length !== 2) throw new Error(`expected 2 messages, got ${messages.length}`);
        assertIncludes(messages[0].content, "Yuanbao user", "Yuanbao user content");
        assertIncludes(messages[1].content, "Yuanbao assistant", "Yuanbao assistant content");
      },
    },
    {
      name: "yuanbao-semantic-role-fallback",
      html: buildHtml("yuanbao", `
        <main>
          <div data-testid="user-message"><div data-testid="message-content">Yuanbao semantic user</div></div>
          <div data-testid="assistant-message">
            <div data-testid="thinking-content">Yuanbao private thought</div>
            <div data-testid="message-content">Yuanbao semantic assistant</div>
            <div role="toolbar">Share answer</div>
          </div>
        </main>
      `),
      assert(messages) {
        if (messages.length !== 2) throw new Error(`expected 2 messages, got ${messages.length}`);
        assertIncludes(messages[1].thinking, "private thought", "Yuanbao fallback thinking");
        assertIncludes(messages[1].content, "semantic assistant", "Yuanbao fallback AI content");
        assertExcludes(messages[1].content, "private thought", "Yuanbao fallback thinking duplication");
        assertExcludes(messages[1].content, "Share answer", "Yuanbao fallback toolbar");
      },
    },
    {
      name: "grok-response-branch",
      html: buildHtml("grok", `<main></main>`, `
        window.__shadowWriteAdapter._cachedMessages =
          window.__shadowWriteAdapter._buildMessagesFromResponses({
            responses: [
              { responseId: "u1", sender: "human", message: "Grok user", createTime: "2026-01-01T00:00:00Z" },
              { responseId: "a1", sender: "assistant", parentResponseId: "u1", message: "Grok assistant", createTime: "2026-01-01T00:00:01Z" }
            ]
          });
      `),
      assert(messages) {
        if (messages.length !== 2) throw new Error(`expected 2 messages, got ${messages.length}`);
        assertIncludes(messages[0].content, "Grok user", "Grok user content");
        assertIncludes(messages[1].content, "Grok assistant", "Grok assistant content");
      },
    },
    {
      name: "grok-strips-injected-context-from-api-export",
      html: buildHtml("grok", `<main></main>`, `
        window.__shadowWriteAdapter._cachedMessages =
          window.__shadowWriteAdapter._buildMessagesFromResponses({
            responses: [{
              responseId: "u1",
              sender: "human",
              message: "IMPORTANT: A persistent context file is attached to this session.\\n\\n---\\n\\nGrok clean question",
              createTime: "2026-01-01T00:00:00Z"
            }]
          });
      `),
      assert(messages) {
        if (messages.length !== 1) throw new Error(`expected 1 message, got ${messages.length}`);
        if (messages[0].content !== "Grok clean question") {
          throw new Error(`unexpected Grok exported user content: ${messages[0].content}`);
        }
      },
    },
    {
      name: "grok-semantic-dom-fallback",
      html: buildHtml("grok", `
        <main>
          <div data-testid="human-message"><div class="prose">Grok DOM user</div></div>
          <div data-testid="model-message">
            <div data-testid="thinking-content">Grok private thought</div>
            <div class="prose">Grok DOM assistant</div>
            <div role="toolbar"><button>Copy Grok response</button></div>
          </div>
        </main>
      `, `window.__shadowWriteAdapter._cachedMessages = window.__shadowWriteAdapter._extractMessagesFromDom();`),
      assert(messages) {
        if (messages.length !== 2) throw new Error(`expected 2 messages, got ${messages.length}`);
        assertIncludes(messages[1].thinking, "private thought", "Grok DOM thinking");
        assertIncludes(messages[1].content, "Grok DOM assistant", "Grok DOM AI content");
        assertExcludes(messages[1].content, "private thought", "Grok DOM thinking duplication");
        assertExcludes(messages[1].content, "Copy Grok response", "Grok DOM toolbar");
      },
    },
    {
      name: "grok-rid-url-key",
      html: buildHtml("grok", `<main></main>`, `
        const key = window.__shadowWriteAdapter.getUrlKey(
          "https://grok.com/c/conversation-id?foo=ignored&rid=response%2F2"
        );
        window.__shadowWriteAdapter._cachedMessages = [{
          messageId: "url-key",
          sender: "test",
          content: key,
          thinking: "",
          position: 0
        }];
      `),
      assert(messages) {
        if (messages.length !== 1) throw new Error(`expected 1 message, got ${messages.length}`);
        if (messages[0].content !== "https://grok.com/c/conversation-id?rid=response%2F2") {
          throw new Error(`unexpected Grok URL key: ${messages[0].content}`);
        }
      },
    },
    {
      name: "context-inject-enter-once",
      html: buildContextInjectionHtml("enter", true),
      virtualTimeBudget: 1200,
      assert(messages) {
        if (messages.length !== 1) throw new Error(`expected 1 result message, got ${messages.length}`);
        const result = JSON.parse(messages[0].content);
        if (
          result.sendCount !== 1 ||
          result.keyboardSendCount !== 0 ||
          result.contextFetchCount !== 1 ||
          !result.sentText.includes("Persistent fact") ||
          !result.sentText.includes("Original question") ||
          result.strippedText !== "Original question"
        ) {
          throw new Error(`expected one context-injected Enter submit, got ${JSON.stringify(result)}`);
        }
      },
    },
    {
      name: "context-inject-click-once",
      html: buildContextInjectionHtml("click", true),
      virtualTimeBudget: 1200,
      assert(messages) {
        if (messages.length !== 1) throw new Error(`expected 1 result message, got ${messages.length}`);
        const result = JSON.parse(messages[0].content);
        if (
          result.sendCount !== 1 ||
          result.keyboardSendCount !== 0 ||
          result.contextFetchCount !== 1 ||
          !result.sentText.includes("Persistent fact")
        ) {
          throw new Error(`expected one context-injected click submit, got ${JSON.stringify(result)}`);
        }
      },
    },
    {
      name: "context-visible-preview-does-not-double-inject",
      html: buildContextInjectionHtml("visible-enter", true),
      virtualTimeBudget: 1200,
      assert(messages) {
        if (messages.length !== 1) throw new Error(`expected 1 result message, got ${messages.length}`);
        const result = JSON.parse(messages[0].content);
        const prefixCount = (result.sentText.match(/=== PROJECT CONTEXT ===/g) || []).length;
        if (
          result.sendCount !== 1 ||
          result.contextFetchCount !== 2 ||
          prefixCount !== 1
        ) {
          throw new Error(`expected one visible context prefix, got ${JSON.stringify(result)}`);
        }
      },
    },
    {
      name: "context-inject-keyboard-fallback-once",
      html: buildContextInjectionHtml("enter", false),
      virtualTimeBudget: 1200,
      assert(messages) {
        if (messages.length !== 1) throw new Error(`expected 1 result message, got ${messages.length}`);
        const result = JSON.parse(messages[0].content);
        if (
          result.sendCount !== 1 ||
          result.keyboardSendCount !== 1 ||
          result.contextFetchCount !== 1 ||
          !result.sentText.includes("Persistent fact")
        ) {
          throw new Error(`expected one keyboard fallback submit, got ${JSON.stringify(result)}`);
        }
      },
    },
    {
      name: "context-inject-empty-memory-bootstrap",
      html: buildContextInjectionHtml("enter", true, ""),
      virtualTimeBudget: 1200,
      assert(messages) {
        if (messages.length !== 1) throw new Error(`expected 1 result message, got ${messages.length}`);
        const result = JSON.parse(messages[0].content);
        if (
          result.sendCount !== 1 ||
          !result.sentText.includes("context-update-start") ||
          !result.sentText.includes("Original question")
        ) {
          throw new Error(`expected empty context bootstrap instructions, got ${JSON.stringify(result)}`);
        }
      },
    },
    {
      name: "context-update-dedupes-and-clears-stale-cache",
      html: buildContextUpdateDedupeHtml(),
      virtualTimeBudget: 1200,
      assert(messages) {
        if (messages.length !== 1) throw new Error(`expected 1 result message, got ${messages.length}`);
        const result = JSON.parse(messages[0].content);
        if (
          result.postCount !== 1 ||
          result.postedBlocks !== 1 ||
          result.postedInlines !== 1 ||
          result.historyCount !== 2 ||
          result.staleContent !== ""
        ) {
          throw new Error(`expected deduped context updates and cleared cache, got ${JSON.stringify(result)}`);
        }
      },
    },
    {
      name: "context-mode-migrates-to-disabled",
      html: buildDisabledContextMigrationHtml(),
      virtualTimeBudget: 1000,
      assert(messages) {
        if (messages.length !== 1) throw new Error(`expected 1 result message, got ${messages.length}`);
        const result = JSON.parse(messages[0].content);
        if (
          result.mode !== "off" ||
          result.settingsMode !== "off" ||
          result.storedMode !== "off" ||
          result.submitHooked
        ) {
          throw new Error(`expected context mode to remain disabled, got ${JSON.stringify(result)}`);
        }
      },
    },
    {
      name: "send-retry-after-local-server-failure",
      html: buildRetryHtml(),
      virtualTimeBudget: 6500,
      assert(messages) {
        if (messages.length !== 1) throw new Error(`expected 1 result message, got ${messages.length}`);
        if (messages[0].content !== "2") {
          throw new Error(`expected failed send to be retried once, got sendCount=${messages[0].content}`);
        }
      },
    },
    {
      name: "idle-mutation-capture-still-sends",
      html: buildIdleMutationHtml(),
      virtualTimeBudget: 3500,
      assert(messages) {
        if (messages.length !== 1) throw new Error(`expected 1 result message, got ${messages.length}`);
        const result = JSON.parse(messages[0].content);
        if (
          result.sendCount !== 1 ||
          result.lastPayloadCount !== 2 ||
          !result.lastAssistantContent?.includes("updated")
        ) {
          throw new Error(`expected one idle mutation send with 2 messages, got ${JSON.stringify(result)}`);
        }
      },
    },
    {
      name: "message-attribute-mutation-still-sends",
      html: buildAttributeMutationHtml(),
      virtualTimeBudget: 3500,
      assert(messages) {
        if (messages.length !== 1) throw new Error(`expected 1 result message, got ${messages.length}`);
        const result = JSON.parse(messages[0].content);
        if (
          result.sendCount !== 1 ||
          !result.lastAssistantContent?.includes("![final image](https://example.com/final.png)")
        ) {
          throw new Error(`expected one attribute mutation send with final image, got ${JSON.stringify(result)}`);
        }
      },
    },
    {
      name: "status-indicator-plain-dot-toggle",
      html: buildCommonIsolationHtml(),
      virtualTimeBudget: 1000,
      assert(messages) {
        if (messages.length !== 1) throw new Error(`expected 1 result message, got ${messages.length}`);
        const result = JSON.parse(messages[0].content);
        if (result.hasFrame) throw new Error("expected status indicator to be a plain page dot");
        if (!result.hasDot || !result.dotConnected) {
          throw new Error(`expected status dot to be connected, got ${JSON.stringify(result)}`);
        }
        if (result.enableCount !== 1) {
          throw new Error(`expected status click to toggle adapter once, got ${JSON.stringify(result)}`);
        }
      },
    },
  ];

  try {
    for (const testCase of cases) {
      runCase(chrome, testCase);
    }
  } finally {
    fs.rmSync(TMP_DIR, { recursive: true, force: true });
  }
}

main();
