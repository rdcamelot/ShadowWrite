# ShadowWrite

**将 AI 对话实时、静默持久化到本地 Markdown 文件。**

在网页端和 AI 对话的同时，VS Code 右侧的 `.md` 文件自动往下写——随时可以停下来润色、改大纲、调设定，两边互不干扰。

## 特性

- **双路线并行**：CLI 终端对话（API 直连）+ Chrome 扩展（7 大 AI 平台网页端）
- **实时流式写入**：对话即文件，每句话实时追加到本地 `.md` + `.chat.html`
- **零依赖**：纯 Python stdlib，无需 `pip install`
- **上下文记忆**：`context.md` 持久化关键设定，跨会话不丢失
- **隐私优先**：所有数据纯本地处理，不经过任何第三方服务器

## 两种使用方式

| | CLI（API 直连） | Chrome 扩展（网页端） |
|---|---|---|
| **入口** | 终端 `python shadowwrite_cli.py` | 浏览器打开 AI 平台 |
| **模型** | OpenAI-compatible / Gemini API | 网页端免费模型 |
| **费用** | API Token 计费 | 免费 |
| **输出** | `.md` + `.chat.html` | `.md` + `.chat.html` |
| **上下文记忆** | ✅ Context File | — |
| **适用场景** | 深度写作、项目开发 | 日常对话备份 |

---

## 快速开始

### 环境要求

- Python 3.10+（标准库即可，无需 pip install）
- Windows / macOS / Linux
- Chrome 浏览器（扩展功能）

### 方式一：CLI 终端对话

```bash
# 1. 复制配置文件
cp .env.example .env

# 2. 编辑 .env，填入 API Key
#    SHADOWWRITE_API_KEY=your-api-key-here
#    SHADOWWRITE_BASE_URL=https://api.openai.com/v1  (或其他兼容端点)

# 3. 启动对话
python shadowwrite_cli.py

# 4. 输出文件自动生成在 outputs/ 目录
```

VS Code 推荐布局：左侧终端对话，右侧打开 `outputs/xxx/xxx.md` 实时查看。

### 方式二：Chrome 扩展（网页端对话）

**第一步：启动本地 HTTP 服务**

```bash
python shadowwrite_server.py
# 看到 "ShadowWrite Local Server" 横幅即成功
```

**第二步：加载 Chrome 扩展**

1. 打开 `chrome://extensions`，开启**开发者模式**
2. 点击**"加载已解压的扩展程序"**，选择 `extension/` 目录

**第三步：开始追踪**

打开 ChatGPT 或 Gemini，进入对话。也可打开 Claude / DeepSeek / Kimi / 豆包 / 元宝，但这些平台目前仅有基础支持，适配完善程度不如前两者。

页面右下角出现一个小圆点——**点击它开启/关闭追踪**，也可以通过右上角扩展图标的弹窗来控制：

| 状态 | 外观 | 操作 |
|------|------|------|
| 未追踪（默认） | 灰色暗淡 | 点击 → 开启 |
| 追踪中 | **白色发光** | 点击 → 关闭 |
| 正在同步 | 橙色 | 自动 |
| 同步成功 | 绿色 | 自动恢复 |
| 连接失败 | 红色 | 检查服务是否运行 |

> **自动追踪**：弹窗中"自动追踪新对话"默认开启，对新开一个对话会自动开启追踪。进入一个历史对话，默认不追踪。切换到扩展已经记录过状态的对话时，以该对话上次保存的状态为准，不会强制重新开启。手动关闭过的对话会记住"已关闭"状态，不会被自动重新追踪。弹窗中的"当前对话追踪"开关与页面右下角的圆点完全同步。

> 网页端对话重命名后，服务器会在下一次同步时自动将本地文件及目录一并重命名。

> **按对话指定输出目录**：在 AI 会话页面打开弹窗，"输出目录"一栏变为"**输出目录 (当前对话)**"，可单独为这个对话指定保存路径（例如直接保存到相关项目目录中）。修改后底部会显示全局默认路径，点击 ↩ 按钮可随时重置回全局。修改目录时，服务器会自动将该对话已有的文件整体迁移到新路径。

**第四步：查看输出**

```
outputs/
└── gemini/
    └── 炼金术与魔法学院/
        ├── 炼金术与魔法学院.md           ← Markdown（可直接编辑）
        └── 炼金术与魔法学院.chat.html    ← 聊天视图（浏览器打开）
```

> 冒烟测试（不需要浏览器）：
> ```powershell
> curl.exe http://127.0.0.1:24601/api/health
> ```

---

## CLI 功能详解

### 基本对话

```bash
python shadowwrite_cli.py                        # 默认：记录到 outputs/
python shadowwrite_cli.py --no-record             # 纯聊天，不保存文件
python shadowwrite_cli.py -o my_novel.md          # 指定输出文件名
```

### 交互命令

| 命令 | 功能 |
|------|------|
| `/ml` 或 `/multi` | 进入多行输入模式（`/end` 提交，`/cancel` 取消） |
| `/section 标题` | 插入章节分隔标题 |
| `/note 内容` | 插入批注（不发给 AI） |
| `/snapshot` | 生成当前对话的完整快照 |
| `/context` | 查看当前 context 文件内容 |
| `/context update` | 要求 AI 生成上下文摘要并写入 |
| `/quit` 或 `/exit` | 退出 |

### 多行输入

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

AI> （流式回复，实时写入 .md）
```

### 上下文记忆文件（Context File）

LLM 上下文窗口有限，长对话后早期信息会被截断丢失。Context File 解决这个问题——作为对话之外的**持久记忆层**：

```bash
# 自动生成 context 文件（按输出文件名派生）
python shadowwrite_cli.py --context-file auto

# 指定 context 文件名
python shadowwrite_cli.py --context-file my_context.md

# 只要上下文，不记录对话
python shadowwrite_cli.py --no-record --context-file auto
```

**工作原理：**
1. CLI 启动时读取 context 文件，作为 system prompt 注入
2. AI 可在回复中使用标记更新 context：
   - 结构化块：`<!-- context-update-start -->` ... `<!-- context-update-end -->`
   - 行内标记：`<!-- context-update: 状态描述 -->`
3. 用户随时可在 VS Code 中直接编辑 context 文件

**场景矩阵：**

| 场景 | `--record` | `--context-file` | 效果 |
|------|-----------|-----------------|------|
| 小说 / 长文协作 | ✅ | ✅ | 完整记录 + 持久上下文 |
| 项目开发迭代 | ❌ | ✅ | 只维护上下文 |
| 随手提问 | ✅ | ❌ | 保留对话记录 |
| 一次性聊天 | ❌ | ❌ | 纯终端对话 |

> **跨工具适用**：即使不用 ShadowWrite CLI，`context.md` 也可手动粘贴到 Copilot / Claude / 其他 AI 工具的对话开头，提供跨会话记忆。

---

## 架构

```
┌─────────────────────────────────────────────────┐
│               ShadowWrite                        │
│                                                  │
│  ┌──────────┐    API     ┌─────────────────┐    │
│  │ CLI      │ ────────→  │ OpenAI / Gemini │    │
│  │ 终端对话  │ ←────────  │ API Server      │    │
│  └────┬─────┘  stream    └─────────────────┘    │
│       │                                          │
│       │ append                                   │
│       ▼                                          │
│  ┌─────────┐                                     │
│  │ outputs/ │  ← .md + .chat.html                │
│  └────▲─────┘                                    │
│       │                                          │
│       │ POST /api/messages                       │
│       │                                          │
│  ┌────┴──────────┐    relay    ┌──────────────┐  │
│  │ Local Server   │ ←────────  │ Chrome Ext   │  │
│  │ :24601         │            │ 7 Adapters   │  │
│  └───────────────┘             └──────────────┘  │
│                                                  │
└─────────────────────────────────────────────────┘
```

### CLI (`shadowwrite_cli.py`)

- 单文件 ~1800 行，纯 Python stdlib
- 支持 OpenAI-compatible 和 Gemini 两种 API 接口
- 流式响应，边显示边写入
- 网络重试（指数退避 3 次）
- Turn ID 锚点 + 元数据注释（`<!-- sw: turn="N" ... -->`）
- 用户输入默认折叠保存（`<details>`），可选 blockquote 模式

### Chrome 扩展 (`extension/`)

- Manifest V3，7 平台 content scripts
- `BaseShadowWriteAdapter` 基类 + 平台子类
- `MutationObserver` + 1s debounce / 3s throttle 实时监听（串流期间最多 3s 延迟）
- JSON 快照差异比较 + 流式 upsert（内容变化时截断文件重写，不产生重复）
- HTTP 通过 background service worker 中继（绕过 CSP）
- 按对话粒度追踪开关，状态持久化到 `chrome.storage.local`
- 自动追踪新对话（autoCapture），手动关闭的对话记住"已禁用"状态
- Epoch 计数器防止跨对话的过期定时器触发
- Popup 弹窗：与运行中服务器双向同步配置（输出目录、Chat HTML 开关）；支持通过服务器弹出原生目录选择对话框；可为每个对话单独设置输出目录，修改时自动迁移已有文件

**支持平台：**

| 平台 | 适配器 | 支持程度 | 特殊能力 |
|------|--------|---------|----------|
| ChatGPT | `chatgpt.js` | ✅ 完善 | 项目对话子目录 |
| Gemini | `gemini.js` | ✅ 完善 | 智能标题提取 |
| Claude | `claude.js` | ⚠️ 基础 | thinking 过滤 |
| DeepSeek | `deepseek.js` | ⚠️ 基础 | thinking 提取 |
| Kimi | `kimi.js` | ⚠️ 基础 | — |
| 豆包 | `doubao.js` | ⚠️ 基础 | thinking 提取 |
| 元宝 | `yuanbao.js` | ⚠️ 基础 | — |

> ⚠️ 基础支持：适配器可正常运行，但 DOM 选择器未经过充分测试，平台前端更新后可能需要调整。

### 本地 HTTP 服务 (`shadowwrite_server.py`)

- 纯 stdlib，端口 `24601`
- `POST /api/messages` — 接收增量消息，写入 `.md` + `.chat.html`；流式 upsert：同一消息内容变化时截断重写
- `GET  /api/health` — 连接测试
- `GET  /api/conversations` — 活跃会话列表
- `GET  /api/config` — 读取运行时配置；附 `?conversationId=` 可获取该对话的实际输出目录及是否使用自定义路径
- `POST /api/config` — 热更新配置：不传 `conversationId` 时修改全局目录/Chat HTML；传入 `conversationId` 时设置对话级自定义目录，服务器自动迁移已有文件（`outputDir: null` 重置回全局）
- `POST /api/browse-directory` — 在服务器端弹出原生目录选择对话框，返回所选路径
- 线程安全，`messageId` 幂等去重，CORS 支持
- 对话标题变更时自动重命名本地文件及目录
- 输出结构：`{output_dir}/{platform}/{标题}/`

---

## 配置

复制 `.env.example` 为 `.env` 并编辑：

```bash
# --- CLI (shadowwrite_cli.py) ---
SHADOWWRITE_API_KEY=your-api-key
SHADOWWRITE_BASE_URL=https://api.openai.com/v1
SHADOWWRITE_MODEL=gpt-4o
SHADOWWRITE_OUTPUT=outputs/novel_draft.md
SHADOWWRITE_CONTEXT_FILE=auto

# --- Server (shadowwrite_server.py) ---
SHADOWWRITE_SERVER_HOST=127.0.0.1
SHADOWWRITE_SERVER_PORT=24601
SHADOWWRITE_OUTPUT_DIR=./outputs
SHADOWWRITE_SERVER_CHAT_HTML=true   # false 则不生成 .chat.html
```

---

## 项目结构

```
ShadowWrite/
├── shadowwrite_cli.py          ← CLI 主程序
├── shadowwrite_server.py       ← 本地 HTTP 服务
├── .env.example                ← 配置模板
├── extension/                  ← Chrome 扩展
│   ├── manifest.json
│   ├── background/
│   │   └── service-worker.js
│   ├── content/
│   │   ├── base-adapter.js
│   │   ├── content-common.js
│   │   └── adapters/           ← 平台适配器
│   ├── popup/
│   ├── css/
│   └── icons/
└──outputs/                    ← 输出目录
```

---

## 已知限制

- **懒加载**：部分平台（如 Gemini）对长对话使用虚拟滚动，扩展只能抓取当前 DOM 中已加载的消息。建议开启追踪前先手动滚动到对话顶部。
- **Gemini Deep Research**：Deep Research 模式生成的报告使用独立渲染组件，当前不支持抓取。
- **服务须手动启动**：需在终端手动运行 `python shadowwrite_server.py`。
- **CSS 选择器耦合**：平台前端更新可能导致适配器失效，需更新选择器。

## Roadmap

**CLI：**

- [ ] 上下文窗口管理（自动截断 / 摘要，防 token 溢出）
- [ ] 断点恢复（CLI 重启后自动恢复对话上下文）
- [ ] 自动分章节 / 标签写入

**Chrome 扩展：**

- [x] 自动追踪新对话（autoCapture）
- [x] Popup 与服务器配置双向同步
- [x] 对话重命名时自动同步本地文件
- [x] 每个对话可单独指定输出目录（含自动文件迁移）
- [ ] 自动启停本地服务（Chrome Native Messaging + 系统注册）
- [ ] 首次追踪时自动滚动加载完整对话历史
- [ ] CSS 选择器热更新机制

---

## License

MIT
