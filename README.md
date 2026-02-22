核心需求：**需要一个不仅能保存聊天，还能“实时、静默追加”到本地文档（如 Markdown）的方案，以便在不打断对话的前提下，随时在文档上对小说或推演内容进行润色和修改。同时，还要避免长对话导致的上下文截断问题。**

为了满足这个需求，下面是这三个技术方向的原理、优缺点以及实践建议：

---

## 1. 传统浏览器 DOM 插件（“事后快照”流）

这类工具适合**轻度提取**，你可以先下载体验一下，看看它们的源码，了解浏览器是如何与网页内容交互的。

* **工作原理：** 插件通过执行 JavaScript 脚本，直接读取你当前网页的 HTML DOM 树（也就是网页的骨架）。它会寻找特定的 CSS 类名（比如 `<div data-message-author-role="assistant">`），把里面的文字抠出来，转换成 Markdown 格式，然后打包成一个文件让你下载。
* **代表工具及源码参考：**
* **ChatGPT Exporter:** 在 GitHub 上开源，你可以去搜索它的源码，重点看它是如何解析 DOM 节点的（通常在 `content.js` 文件中）。
* **MarkDownload:** 一个通用的网页转 Markdown 工具，源码同样在 GitHub 可查。你可以学习它内部使用的 HTML 转 Markdown 的核心逻辑（如 `Turndown.js` 库）。


* **为何无法满足你的终极需求：**
* **非实时：** 只能手动点击触发，每次都会生成一个全新的文件，无法与你正在编辑的文档双向同步。
* **虚拟滚动（懒加载）陷阱：** 如果你的小说对话拉得很长，网页为了节省内存会把上面的对话从 DOM 里删掉。插件抓不到不在屏幕上的内容，导致你的长篇大论直接丢失。

---

## 2. 官方 API + 纯本地脚本（“硬核全场景”流）

这是最稳妥、最符合你“沉浸式写作与润色”需求的方案。跳过现成的网页界面，自己造一个纯本地的对话环境。

* **工作原理：**
你用 Python 或 Node.js 写一个脚本，直接通过 API（如 Gemini API、OpenAI API）发送你的提示词。脚本收到 AI 的回复后，做两件事：
1. 在屏幕（终端）上打印出来给你看。
2. 调用系统的文件操作命令（如 Node.js 的 `fs.appendFileSync`），把这段回复静默追加到你指定的 `novel_draft.md` 文件末尾。


* **完美契合你的工作流：**
你可以直接在 VS Code 里操作——左侧开一个内置终端（Terminal）专门用来和 AI 对话、输入剧情指令；右侧打开那个 `novel_draft.md` 文件。随着你在左边聊天，右边的文档会自动往下写，你可以随时在右侧修改设定、润色文笔，两边互不干扰。
* **优缺点：**
* **优点：** 100% 实时镜像，绝对不会因为网页崩溃或懒加载丢失任何历史记录；本地处理，隐私最安全。
* **缺点：** 失去了官方网页端花哨的 UI 界面；调用 API 可能会消耗额度（Token 计费）。



---

## 3. 油猴脚本 + 本地 Node.js 监听服务（“极客缝合怪”流）

如果你既想白嫖网页端免费且直观的对话界面，又想实现“实时同步到本地文档”的体验，这就是最终的折中方案。

* **工作原理：**
这是一个“里应外合”的架构，分为两部分：
1. **前端（监听者）：** 在浏览器里安装 Tampermonkey（油猴插件），写一段简单的脚本注入到聊天网页中。这段脚本利用 `MutationObserver` API，死死盯住网页的对话框区域。只要 AI 吐出了一段新话，脚本就立刻把这段话抓取下来。
2. **后端（接收者）：** 在你的电脑上跑一个非常极简的 Node.js 本地服务器（比如运行在 `localhost:3000`）。
油猴脚本抓到新内容后，通过 HTTP POST 请求，悄悄发给你的本地 Node.js 服务。Node.js 收到后，立刻将其追加（Append）到你的本地 Markdown 文件里。


* **优缺点：**
* **优点：** 巧妙绕过了浏览器的本地文件读写限制；即使网页开启了虚拟滚动，因为你是“增量实时抓取”，旧记录被清理了也无所谓，反正已经存到本地了。
* **缺点：** 维护成本稍高。如果 AI 平台某天更新了网页的前端代码（比如改了 CSS 类名），你的油猴脚本可能就会失效，需要重新检查网页元素并更新脚本。


### 现有产品的分析
Chat Memo 的核心原理：增量监听 + 浏览器本地存储

传统的插件是“拍快照”，而 Chat Memo 的核心逻辑更像是一个**“浏览器内置的录音机”**。

* **实时监听（MutationObserver）：** 它也是在网页里注入了脚本，利用 `MutationObserver` 实时盯着聊天框。AI 只要蹦出一个新字，它就立刻抓取下来。
* **沙盒内存储（IndexedDB / chrome.storage）：** **这是它和我们“Node.js 方案”最大的区别。** 抓到新文字后，它并没有穿透浏览器写到你电脑的 D 盘或桌面上，而是把数据塞进了**浏览器内部的本地数据库**（比如 IndexedDB）。
* **按需导出：** 当你想看的时候，你可以打开插件的独立面板，里面有你所有的聊天记录（从浏览器数据库里读出来的）。此时你可以点击“导出”，它会调用浏览器的下载功能，给你生成一个 `.md` 或 `.zip` 文件。

**好消息是，它完美解决了“长对话截断（懒加载）”的问题。**
因为它是“边聊边记”，所以无论网页怎么为了省内存而清理上面的对话节点，Chat Memo 早已把数据存进它自己的内部数据库里了。你的长篇小说推演不会丢。

**坏消息是，它依然打破不了“实时沉浸式双屏写作”的壁垒。**
由于浏览器的安全沙盒机制，Chat Memo 依然**无法直接操作你电脑硬盘上的文件系统**。
这意味着：

1. 你**不能**在 VS Code 里打开一个 Markdown 文件，然后一边在网页里和 AI 聊天，一边看着这个文件自动往下写。
2. 你的数据被“困”在了浏览器的内部存储里。如果你想在外部编辑器里润色，你依然需要经历：**打开插件 -> 点击导出 -> 下载新文件 -> 在本地打开修改** 这样的割裂流程。如果你在本地文件修改了前半部分，接下来导出的新文件依然无法和你本地的修改自动合并。

| 功能维度 | 传统 DOM 插件 (如 ChatGPT Exporter) | Chat Memo | 你的终极理想型 (网页监听 + Node.js 服务) |
| --- | --- | --- | --- |
| **工作模式** | 静态快照（事后算账） | 动态录音机（边聊边记） | 实时管道（边聊边穿透写入硬盘） |
| **是否防长对话丢失** | ❌ 否（网页滑不上去的就丢了） | ✅ 是（记录在浏览器内部数据库） | ✅ 是（直接写进本地 Markdown） |
| **存储位置** | 无，直接生成下载文件 | 浏览器的内部存储沙盒 | **你电脑上的任意文件夹** |
| **是否支持 VS Code 实时镜像润色** | ❌ 否 | ❌ 否（需要手动导出） | ✅ **是**（文件就在本地，随时可改，随时追加） |
| **隐私安全性** | 高（纯本地提取） | 高（不上传第三方服务器） | 极高（纯本地网络循环） |

Chat Memo 是一个非常优秀的折中产品。它把浏览器插件能做到的极限（增量保存 + 内部数据库）做到了最好。如果你觉得“边聊边在另一个窗口实时改代码/改小说”并不是绝对的刚需，只是想确保**“聊得再长也不丢数据，最后能完整导出”**，那么直接安装使用 Chat Memo 是性价比最高的选择，你完全不需要自己写代码了。

但如果你追求的是那种极致的极客体验——**左半屏是网页聊天，右半屏是 VS Code 里的 `.md` 自动打字，并且你随时可以停下来在右边直接改大纲和设定**——那么浏览器沙盒这堵墙你必须得翻过去，也就是必须得跑一个本地的 Node.js 接收服务。

---

## 适用环境

ShadowWrite 的核心理念是**将 AI 对话实时持久化到本地文件**。这个理念不仅适用于自建 CLI，
也适用于各类 AI 编程助手 / 对话插件的使用场景：

| 环境 | 说明 |
|------|------|
| **ShadowWrite CLI** | 本项目的主力方案，终端对话 → 本地 .md/.html |
| **VS Code + Copilot / Codex** | Copilot Chat 的对话历史不持久化到文件，可借助 ShadowWrite 的 Context File 机制维护项目上下文 |
| **VS Code + Claude (Cline / Continue)** | 同理，会话窗口关闭后上下文丢失，Context File 可作为跨会话记忆补充 |
| **VS Code + 其他 AI 插件** | 任何支持自定义 system prompt 的工具都可以手动加载 context.md 内容 |
| **浏览器 + Chrome 扩展（方案 3）** | 直接在网页端对话，通过扩展穿透到本地文件 |

> **关键洞察**：即使不使用 ShadowWrite CLI，仅使用 `context.md`（上下文记忆文件）
> 也能为任何 AI 工具提供跨会话的项目记忆。你可以手动将 context.md 的内容
> 粘贴到 Copilot / Claude 的对话开头，或者通过 `.github/copilot-instructions.md`
> 等机制自动注入。

---

## 4. 当前项目进展（2026-02-21）

### 已完成

1. 已克隆并建立参考代码库：
   - `external/chatgpt-exporter`
   - `external/markdownload`
   - `external/turndown`
2. 已获取并分析 Chat Memo 扩展源码（`test/` 目录）：
   - 多平台适配器架构（`BasePlatformAdapter` + 子类：ChatGPT/DeepSeek/Gemini/Claude/Kimi 等）
   - `MutationObserver` + debounce 增量监听机制
   - 锚点匹配算法处理懒加载场景
3. 已实现"方案 2"的本地可运行 MVP（无第三方依赖）：
   - `shadowwrite_cli.py`
   - `.env.example`
   - `LOCAL_API_WORKFLOW.md`
4. 当前 CLI 能力：
   - 终端连续对话
   - OpenAI-compatible 与 Gemini 的流式响应（边显示边写入）
   - 用户输入默认折叠保存（`<details>`），AI 回复作为主体正文
   - 支持 Typora 兼容回退（`SHADOWWRITE_USER_INPUT_MODE=blockquote`）
   - 时间默认可隐藏，仍保留可索引元数据（`<!-- sw: turn="N" ... -->`）
   - 每轮对话带 turn ID 锚点（`<a id="sw-turn-N"></a>`），支持文档内跳转
   - 轮次之间以 `---` 分隔线分割，便于视觉区分
   - 每轮对话自动追加到本地 Markdown（默认 `novel_draft.md`）
   - 可选同步输出聊天风格 HTML 视图（默认 `novel_draft.chat.html`）
   - HTML 视图支持助手 Markdown 渲染（CDN 可用时）
   - 支持手动结构化命令：`/section`、`/note`
   - 支持快照命令：`/snapshot`
   - 上下文记忆文件：`--context-file`，AI 自动维护 + 用户可审阅（见 §6）
   - 记录与上下文独立开关：`--no-record` 可仅聊天不保存文件，`--context-file` 可独立启用
   - 适配 `openai_compat` 与 `gemini` 两种接口层

### 方案 2 后续功能优先级（更新）

#### P0（已完成）

- **流式输出 + 流式写入 `.md` / `.chat.html`**
- **Gemini delta 算法修复**（段落重置不再重复累加）
- **网络重试**（指数退避 3 次，覆盖超时 / 429 / 5xx）
- **turn ID 锚点 + 元数据注释**（`<!-- sw: turn="N" ... -->`）
- **轮次分隔线**（Markdown `---`、HTML `<hr>`）
- **快照尊重 user_input_mode**（`details` / `blockquote` 均正确）
- **`detect_next_turn_id` 尾部 8 KB 快速扫描**
- **上下文记忆文件**（`--context-file`，自动注入 system prompt + AI 自动更新标记）
- **记录独立开关**（`--record / --no-record`，记录与上下文解耦）

#### P1（下一步重点）

- **上下文窗口管理**
  - history 增长到阈值后自动截断 / 摘要，防止 token 溢出
  - 与 Context File 联动：截断前将关键决策写入 context 文件

#### P2（可选增强）

- 自动分章节 / 标签写入（剧情 / 设定 / 灵感）
- 按日归档、会话导出索引
- 断点恢复（CLI 重启后自动恢复对话上下文）

### 当前结论

先走"方案 2"是正确路线。CLI 已具备跨模型流式写作、断点续写（turn ID）、
网络容错等核心能力。Context File 已实现双标记格式（结构化块 + 行内日志），下一阶段重点是上下文窗口管理。

方案 3（Chrome 扩展 + 本地 HTTP 服务）M0 骨架已完成，见下方。

### 路线 3 实现（M0 骨架）

- 详见：[ROUTE3_BASELINE.md](ROUTE3_BASELINE.md)
- 架构方向：**Chrome 扩展（Manifest V3）+ 本地 Python HTTP 服务**

#### 已实现（M0）

**Chrome 扩展**（`extension/` 目录）：
- `manifest.json` — MV3 配置，7 平台 content_scripts，`chrome.storage` + `tabs` 权限
- `content/base-adapter.js` — `BaseShadowWriteAdapter` 基类
  - 4 个抽象方法：`isValidConversationUrl`、`extractConversationInfo`、`extractMessages`、`isMessageElement`
  - `MutationObserver`（childList + subtree + characterData）+ 1 秒 debounce
  - JSON 快照差异比较，仅发送增量消息
  - URL 轮询（1 秒）自动切换会话
  - `fetch(POST)` 直接发送到本地 HTTP 服务
- `content/content-common.js` — 状态指示器小圆点（idle/saving/ok/error）
- `content/adapters/` — 7 个平台适配器：
  - `chatgpt.js` — ChatGPT/OpenAI
  - `deepseek.js` — DeepSeek（含 thinking 提取）
  - `gemini.js` — Google Gemini
  - `claude.js` — Anthropic Claude（含 thinking 过滤）
  - `kimi.js` — Kimi/Moonshot
  - `doubao.js` — 豆包（含 thinking 提取）
  - `yuanbao.js` — 腾讯元宝
- `background/service-worker.js` — 设置管理、badge 更新、设置变更广播
- `popup/popup.html` + `popup.js` — 配置 UI（host/port/开关/连接测试）
- `css/content.css` — 状态指示器样式

**本地 HTTP 服务**（`shadowwrite_server.py`）：
- 纯 stdlib，无第三方依赖
- `POST /api/messages` — 接收扩展发送的增量消息，写入 `.md` + `.chat.html`
- `GET /api/health` — 连接测试
- `GET /api/conversations` — 查看当前活跃会话
- 线程安全的会话状态管理（`ConversationState`）
- `messageId` 幂等去重
- CORS 支持
- 默认端口 `24601`，输出到 `./outputs/`

#### 使用方法

**第一步：启动本地 HTTP 服务**

```bash
cd d:\code\ShadowWrite
python shadowwrite_server.py
```

看到以下输出说明成功：

```
╔══════════════════════════════════════════════════════════════╗
║  ShadowWrite Local Server                                    ║
║  Listening:   http://127.0.0.1:24601                        ║
║  ...                                                         ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
```

验证服务正常（PowerShell 中 `curl` 是 `Invoke-WebRequest` 的别名，建议用 `curl.exe`）：

```powershell
curl.exe http://127.0.0.1:24601/api/health
# 预期返回：{"status": "ok", "service": "ShadowWrite", "version": "0.1.0"}
```

**第二步：加载 Chrome 扩展**

1. 打开 `chrome://extensions`
2. 右上角开启**开发者模式**
3. 点击**"加载已解压的扩展程序"**
4. 选择 `extension/` 目录

> 图标文件已预先生成（`extension/icons/icon{16,48,128}.png`），加载时不会报错。

**第三步：打开 AI 平台对话**

打开 ChatGPT / Claude / Gemini / DeepSeek / Kimi / 豆包 / 元宝任意支持的平台，进入对话。

扩展会在页面**右下角**显示一个小圆点状态指示器：

| 颜色 | 含义 |
|------|------|
| 灰色（暗） | 待机中 |
| 橙色 | 正在发送到本地服务 |
| 绿色（3 秒后消失） | 保存成功 |
| 红色（持续） | 本地服务未启动或连接失败 |

**第四步：查看输出**

每条对话保存到 `outputs/{platform}_{conversationId}/` 目录下：

```
outputs/
└── chatgpt_abc123/
    ├── chatgpt_abc123.md           ← Markdown 主文档（可在 VS Code 直接编辑）
    └── chatgpt_abc123.chat.html    ← 聊天视图（浏览器打开查看）
```

#### 快速冒烟测试（不需要打开浏览器）

```powershell
# 模拟一条来自扩展的对话数据
curl.exe -X POST http://127.0.0.1:24601/api/messages `
  -H "Content-Type: application/json" `
  -d '{\"platform\":\"chatgpt\",\"conversationId\":\"test_001\",\"title\":\"测试对话\",\"url\":\"https://chatgpt.com/c/test_001\",\"messages\":[{\"messageId\":\"msg_001\",\"sender\":\"user\",\"content\":\"你好\",\"thinking\":\"\",\"position\":0},{\"messageId\":\"msg_002\",\"sender\":\"AI\",\"content\":\"你好！有什么可以帮你的？\",\"thinking\":\"\",\"position\":1}]}'

# 预期：{"status": "ok", "written": 2, "skipped": 0, "conversationId": "test_001"}

# 检查生成的文件
Get-ChildItem outputs\chatgpt_test_001\
```

#### M1（下一步）

- 多轮去重优化（基于内容 hash 而非仅 position）
- 扩展 popup 显示当前会话列表
- 消息格式与 CLI 对齐（统一 turn 元数据格式）
- CSS 选择器热更新机制（应对平台前端变更）

## 5. CLI 多行输入（`/ml`）

当你要一次性输入多段提示词（比如包含空行、列表、代码块）时，使用多行模式更合适。

### 用法

1. 在 `You>` 提示符输入：`/ml`（或 `/multi`）
2. 进入多行输入后，逐行输入内容
3. 输入 `/end` 提交给模型
4. 输入 `/cancel` 取消本次多行输入

### 终端示例

```text
You> /ml
Multi-line mode: type /end on a new line to submit, /cancel to abort.
... 你是我的小说协作助手，请根据以下设定写一段开场。
... 
... 设定：
... - 时间：雨夜
... - 地点：废弃车站
... - 人物：失忆侦探
... /end

AI> （模型开始正常回复，并流式写入 .md/.html）
```

### 写入效果说明

- 这不是本地假输入，`/end` 后会按正常流程调用 API。
- 在 `details` 模式下，用户多行输入会以 `<br>` 保留换行写入 Markdown，兼容 Typora 的折叠限制。
- 在 `blockquote` 模式下，会按 Markdown 引用块格式写入。

## 6. 上下文记忆文件（Context File）

### 问题

LLM 对话的上下文窗口是有限的。无论是长篇小说协作、论文实验设计、还是项目开发，
当会话积累到几万 token 后，早期的关键信息会被滑窗截断丢失：
- 小说：角色表、世界观、剧情决策
- 论文：motivation、story line、实验设计主线、已有结果
- 项目：出发点、架构决策、已完成/未完成任务、关键变更日志

这导致 AI “忘记”之前约定的内容，仅从近期对话中找信息会产生偏差。

### 解决思路

引入一个**人工可审阅、AI 可更新**的结构化文件（如 context.md），
作为对话之外的“持久记忆层”——相当于一个实时、准确的项目说明文档：

`
project_root/
├── novel_draft.md          ← 主输出文件
├── novel_draft.chat.html   ← 聊天视图
└── context.md              ← 上下文记忆文件
`

### 设计要点

1. **自动注入**：CLI 启动时读取 context 文件，将其内容作为 system prompt 的一部分发送给模型。
   模型从第一轮就"知道"所有关键上下文。
2. **AI 可更新（双标记格式）**：system prompt 中附带指令，AI 可在回复中使用两种标记：
   - **结构化块标记**（推荐，用于输出完整的设定、角色表、决策细节等）：
     ```
     <!-- context-update-start -->
     ## 角色设定
     - 陆明：25岁记者，好奇心旺盛，战斗力弱
     - 从者：职阶待定，真名保密，初始态度冷静审视
     <!-- context-update-end -->
     ```
     CLI 解析后将块内容**原样追加**到 context 文件，保留 Markdown 结构。
   - **行内标记**（用于简短的状态记录）：
     `<!-- context-update: 第一幕完成，主角进入地下排水系统 -->`
     CLI 解析后追加为带时间戳的日志行。
3. **用户可审阅**：context 文件是普通 Markdown 文件，随时可在 VS Code 中查看、编辑、
   删减，保持上下文精简准确。也方便项目交接和进度同步。
4. **与截断联动**（可选）：当 history 消息数超过阈值触发截断时，截断前自动要求模型
   生成一段摘要写入 context 文件，确保关键信息不丢失。

> **关于文件结构**：context.md 的结构完全由用户自定义。AI 不会自动选择"小说模式"
> 或"论文模式"——你创建什么结构，AI 就在那个结构上追加增量更新。
> 下面的示例只是模板参考，实际内容会随项目发展变得更丰富。
> 首次使用时也可以通过 `/context update` 让 AI 根据对话内容生成初始版本。

### 记录与上下文的独立开关

记录写入（`.md` / `.html`）和上下文记忆文件是**完全解耦**的两个功能：

```bash
# 小说协作：记录 + 上下文（auto 按输出文件名自动生成）
python shadowwrite_cli.py --context-file auto

# 项目开发：只要上下文，不需要记录每次对话
python shadowwrite_cli.py --no-record --context-file auto

# 零散提问：只保存记录，不需要上下文（默认行为）
python shadowwrite_cli.py

# 纯聊天：什么都不保存
python shadowwrite_cli.py --no-record
```

| 场景 | `--record` | `--context-file` | 说明 |
|------|-----------|-----------------|------|
| 小说 / 长文协作 | ✅ on | ✅ 指定 | 完整记录 + 持久上下文 |
| 项目开发迭代 | ❌ off | ✅ 指定 | 只维护项目上下文，不记录零碎对话 |
| 随手提问 / 学习 | ✅ on | ❌ 不指定 | 保留对话记录以备查阅 |
| 一次性聊天 | ❌ off | ❌ 不指定 | 纯终端对话 |

### 预期用法

`bash
# 自动生成 context 文件（按输出文件名派生，如 test_03_context.md）
python shadowwrite_cli.py --context-file auto

# 指定 context 文件名
python shadowwrite_cli.py --context-file my_context.md

# 也可在 .env 中设置：SHADOWWRITE_CONTEXT_FILE=auto

# 交互中手动触发 context 更新
You> /context           # 查看当前 context 文件内容
You> /context update    # 要求 AI 生成上下文摘要并写入
`

### Context 文件示例结构

根据不同场景，文件结构可以灵活调整：

**小说协作**：
`markdown
# 项目上下文

## 核心设定
- 时代：近未来 2089 年
- 地点：新东京地下城
- 主题：记忆交易与身份认同

## 角色表
| 角色 | 特征 | 当前状态 |
|------|------|---------|
| 林夜 | 失忆侦探 | 正在调查“空白人”案件 |

## 关键决策记录
- [Turn 5] 决定采用第一人称叙事
`

**论文实验**：
`markdown
# 项目上下文

## Motivation & Story
- 核心问题：...的暴露偏差问题
- 主线：通过...方法降低偏差同时保持性能

## 实验设计
| 实验 | 目的 | 状态 |
|------|------|------|
| Table 1 | 主实验对比 | ✅ 完成 |
| Table 2 | Ablation | ⚠️ 进行中 |

## 已有结果摘要
- Baseline: 78.3%
- Ours (v1): 82.1%
`

**项目开发**（如本项目）：
`markdown
# 项目上下文

## 出发点
- 实时、静默追加 AI 对话到本地 Markdown

## 架构决策
- 方案 2 (API + CLI) 为主，同步探索方案 3 (Chrome 扩展)
- 纯 stdlib，无第三方依赖

## 任务跟踪
- [x] 流式输出 + 双文件写入
- [x] Gemini delta 修复
- [x] turn ID 锚点
- [ ] Context File 实现
- [ ] 上下文窗口管理

## 变更日志
- 2026-02-21: 完成 turn 分隔线、网络重试、尾部扫描优化
`

