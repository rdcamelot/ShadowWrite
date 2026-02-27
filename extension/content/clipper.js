/**
 * ShadowWrite — Web Clipper (injected on demand)
 *
 * Self-contained IIFE injected via chrome.scripting.executeScript.
 * Returns { title, url, domain, content } where content is Markdown.
 */
(function () {
  "use strict";

  // ── Known site selectors ────────────────────────────────────────
  // domain-substring → CSS selector for the main content container
  const SITE_SELECTORS = {
    "alicesw.com": ".read-content",
    "qidian.com": ".read-content",
    "zongheng.com": ".content",
    "17k.com": "#chapterContent",
    "jjwxc.net": ".noveltext",
    "ciweimao.com": "#J_BookRead",
    "faloo.com": ".noveContent",
    "69shu.pro": ".txtnav",
    "biquge": "#content",
    "xbiquge": "#content",
    "ptwxz.com": "#content",
    "zhihu.com": ".Post-RichText, .RichContent-inner",
    "mp.weixin.qq.com": "#js_content",
    "jianshu.com": "article",
    "cnblogs.com": "#cnblogs_post_body",
    "csdn.net": "#content_views",
    "juejin.cn": ".article-content",
    "medium.com": "article",
    "substack.com": ".body",
    "wordpress.com": ".entry-content",
  };

  // ── Find the main content element ──────────────────────────────
  function findContentElement() {
    const hostname = location.hostname;

    // 1. Known site selectors
    for (const [domain, selector] of Object.entries(SITE_SELECTORS)) {
      if (hostname.includes(domain)) {
        const el = document.querySelector(selector);
        if (el && el.textContent.trim().length > 50) return el;
      }
    }

    // 2. User text selection
    const selection = window.getSelection();
    if (selection && selection.rangeCount > 0 && !selection.isCollapsed) {
      const range = selection.getRangeAt(0);
      const text = selection.toString().trim();
      if (text.length > 50) {
        const container = document.createElement("div");
        container.appendChild(range.cloneContents());
        return container;
      }
    }

    // 3. Generic semantic selectors
    const genericSelectors = [
      "article",
      '[role="article"]',
      '[role="main"]',
      "main",
      ".post-content",
      ".article-content",
      ".entry-content",
      ".content-body",
      ".story-body",
      ".markdown-body",
    ];
    for (const sel of genericSelectors) {
      const el = document.querySelector(sel);
      if (el && el.textContent.trim().length > 200) return el;
    }

    // 4. Scoring: find container with max paragraph text
    let best = null;
    let bestScore = 0;
    const candidates = document.querySelectorAll("div, section, article");
    for (const el of candidates) {
      const paragraphs = el.querySelectorAll("p");
      let score = 0;
      for (const p of paragraphs) {
        const len = p.textContent.trim().length;
        if (len > 25) score += len;
      }
      // Penalise very large containers (whole page body)
      if (el === document.body || el.children.length > 100) {
        score *= 0.3;
      }
      if (score > bestScore) {
        bestScore = score;
        best = el;
      }
    }
    if (best && bestScore > 200) return best;

    // 5. Last resort
    return document.body;
  }

  // ── Lightweight HTML → Markdown converter ──────────────────────
  function htmlToMarkdown(root) {
    function processNode(node) {
      if (node.nodeType === Node.TEXT_NODE) {
        return node.textContent;
      }
      if (node.nodeType !== Node.ELEMENT_NODE) return "";

      const tag = node.tagName.toLowerCase();
      const children = () =>
        Array.from(node.childNodes).map(processNode).join("");

      switch (tag) {
        // Headings
        case "h1":
        case "h2":
        case "h3":
        case "h4":
        case "h5":
        case "h6": {
          const level = parseInt(tag[1]);
          return "\n\n" + "#".repeat(level) + " " + children().trim() + "\n\n";
        }
        // Inline formatting
        case "strong":
        case "b":
          return "**" + children() + "**";
        case "em":
        case "i":
          return "*" + children() + "*";
        case "del":
        case "s":
          return "~~" + children() + "~~";
        case "mark":
          return "==" + children() + "==";
        case "sub":
          return "~" + children() + "~";
        case "sup":
          return "^" + children() + "^";
        // Code
        case "code": {
          if (
            node.parentElement &&
            node.parentElement.tagName.toLowerCase() === "pre"
          ) {
            return node.textContent;
          }
          return "`" + node.textContent + "`";
        }
        case "pre": {
          const codeEl = node.querySelector("code");
          const lang = codeEl
            ? codeEl.className.match(
                /(?:language|lang|highlight)-(\S+)/i
              )?.[1] || ""
            : "";
          const code = codeEl ? codeEl.textContent : node.textContent;
          return (
            "\n\n```" + lang + "\n" + code.replace(/\n$/, "") + "\n```\n\n"
          );
        }
        // Links & images
        case "a": {
          const href = node.getAttribute("href") || "";
          const text = children();
          return href ? "[" + text + "](" + href + ")" : text;
        }
        case "img": {
          const alt = node.getAttribute("alt") || "";
          const src = node.getAttribute("src") || "";
          return "![" + alt + "](" + src + ")";
        }
        // Lists
        case "ul":
        case "ol":
          return "\n" + processListItems(node, "") + "\n";
        // Block elements
        case "p":
          return "\n\n" + children().trim() + "\n\n";
        case "div": {
          // treat divs that act as paragraphs
          const inner = children().trim();
          return inner ? "\n\n" + inner + "\n\n" : "";
        }
        case "br":
          return "\n";
        case "hr":
          return "\n\n---\n\n";
        case "blockquote": {
          const content = children().trim();
          return (
            "\n\n" +
            content
              .split("\n")
              .map((l) => "> " + l)
              .join("\n") +
            "\n\n"
          );
        }
        // Tables
        case "table":
          return "\n\n" + processTable(node) + "\n\n";
        // Math
        case "math": {
          const tex =
            node.getAttribute("alttext") ||
            node.getAttribute("data-latex") ||
            node.textContent;
          return "$" + tex + "$";
        }
        // Skip noise
        case "script":
        case "style":
        case "noscript":
        case "iframe":
        case "svg":
          return "";
        // Pass-through
        default:
          return children();
      }
    }

    function processListItems(listNode, indent) {
      const isOrdered = listNode.tagName.toLowerCase() === "ol";
      let result = "";
      let idx = 0;
      for (const child of listNode.children) {
        if (child.tagName.toLowerCase() !== "li") continue;
        idx++;
        const prefix = isOrdered ? `${idx}. ` : "- ";
        const continuation = " ".repeat(prefix.length);
        let inlineContent = "";
        let nestedBlocks = "";
        for (const liChild of child.childNodes) {
          if (liChild.nodeType === Node.ELEMENT_NODE) {
            const liTag = liChild.tagName.toLowerCase();
            if (liTag === "ul" || liTag === "ol") {
              nestedBlocks += processListItems(liChild, indent + continuation);
              continue;
            }
          }
          inlineContent += processNode(liChild);
        }
        const trimmed = inlineContent.trim();
        if (trimmed) {
          const lines = trimmed.split("\n");
          result += indent + prefix + lines[0];
          for (let i = 1; i < lines.length; i++) {
            if (lines[i].trim()) {
              result += "\n" + indent + continuation + lines[i];
            }
          }
          result += "\n";
        }
        if (nestedBlocks) result += nestedBlocks;
      }
      return result;
    }

    function processTable(tableNode) {
      const rows = tableNode.querySelectorAll("tr");
      if (rows.length === 0) return "";
      let result = "";
      let isFirst = true;
      for (const row of rows) {
        const cells = Array.from(row.querySelectorAll("th, td"));
        const cellTexts = cells.map((c) =>
          processNode(c).trim().replace(/\|/g, "\\|").replace(/\n/g, " ")
        );
        result += "| " + cellTexts.join(" | ") + " |\n";
        if (isFirst) {
          result += "| " + cellTexts.map(() => "---").join(" | ") + " |\n";
          isFirst = false;
        }
      }
      return result;
    }

    return processNode(root).replace(/\n{3,}/g, "\n\n").trim();
  }

  // ── Group/series extraction ─────────────────────────────────────
  /**
   * Try to extract a series/group name from the page title by stripping
   * chapter/episode markers. e.g.:
   *   "第一卷 第1章 _楠楠的暴露系列" → "楠楠的暴露系列"
   *   "深度学习基础入门（二）" → "深度学习基础入门"
   */
  function extractGroup(title) {
    let work = title;

    // Leading chapter/volume markers
    const leadingPatterns = [
      /^第[一二三四五六七八九十百千万零〇\d]+[卷章节话回篇集部幕场]\s*/,
      /^[Cc]hapter\s+\d+[\s.:：]*/,
      /^[Pp]art\s+\d+[\s.:：]*/,
      /^[Ss]ection\s+\d+[\s.:：]*/,
      /^[Ee]pisode\s+\d+[\s.:：]*/,
      /^\d+[\s.、:：]+/,
    ];

    let changed = true;
    while (changed) {
      changed = false;
      for (const pat of leadingPatterns) {
        const before = work;
        work = work.replace(pat, "");
        if (work !== before) changed = true;
      }
      // Strip separators between markers
      const s = work.replace(/^[\s_\-:：·|、]+/, "");
      if (s !== work) {
        work = s;
        changed = true;
      }
    }

    // Trailing chapter markers: （一）, (2), (上), etc.
    work = work
      .replace(
        /\s*[（(]\s*[一二三四五六七八九十百千万零〇\d]+\s*[)）]\s*$/,
        ""
      )
      .replace(/\s*[（(]\s*(?:上|中|下|续|完|终)\s*[)）]\s*$/, "")
      .trim();

    if (work && work !== title.trim() && work.length >= 2) {
      return work;
    }
    return null;
  }

  // ── Extract & return ───────────────────────────────────────────
  try {
    const contentEl = findContentElement();
    const clone = contentEl.cloneNode(true);

    // Remove noise elements
    clone
      .querySelectorAll(
        "script, style, noscript, iframe, .sr-only, nav, footer, header," +
          ".ads, .advertisement, .ad-container, .social-share, .comments," +
          ".sidebar, .related-posts, .recommend, [aria-hidden='true']"
      )
      .forEach((el) => el.remove());

    const markdown = htmlToMarkdown(clone);

    // Derive a clean title
    let title = document.title || "";
    // Strip common suffixes: "xxx - Site Name", "xxx | Site Name"
    title = title.replace(/\s*[|\-–—_]\s*[^|\-–—_]+$/, "").trim();
    if (!title) title = "Untitled";

    return {
      title: title,
      url: location.href,
      domain: location.hostname,
      content: markdown,
      group: extractGroup(title) || "",
    };
  } catch (err) {
    return { error: err.message };
  }
})();
