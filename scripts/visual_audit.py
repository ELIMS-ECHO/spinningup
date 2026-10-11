#!/usr/bin/env python3
"""Browser audit of GitHub's actual rendered public Markdown pages."""
import json
import os
from pathlib import Path
from urllib.parse import quote

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "dist" / "github-visual-audit"
OUTPUT.mkdir(parents=True, exist_ok=True)
REPO = "tianyueni78-cyber/spinningup"
SHA = os.environ.get("GITHUB_SHA", "learning-edition-zh")
CASES = [
    ("algorithms/vpg_zh.md", "原始策略梯度", "math"),
    ("algorithms/ddpg_zh.md", "深度确定性策略梯度", "math"),
    ("spinningup/rl_intro3_zh.md", "第三部分", "math"),
    ("spinningup/rl_intro2_zh.md", "第二部分", "image"),
    ("user/running_zh.md", "命令", "code"),
    ("spinningup/keypapers_zh.md", "深度强化学习关键论文", "bibliography"),
]
# Audit every generated chapter's actual GitHub page, not just six examples.
covered = {name for name, _, _ in CASES}
for path in sorted((ROOT / "online" / "教材").rglob("*_zh.md")):
    name = path.relative_to(ROOT / "online" / "教材").as_posix()
    if name in covered:
        continue
    heading = next((line[2:].strip() for line in path.read_text(encoding="utf-8").splitlines()
                    if line.startswith("# ")), path.stem)
    CASES.append((name, heading, "presence"))
if len(CASES) != 24:
    raise RuntimeError("Expected 24 real GitHub pages, found " + str(len(CASES)))

result = {"repository": REPO, "sha": SHA, "pages": [], "errors": []}

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 1440, "height": 1000}, device_scale_factor=1)
    for name, required, kind in CASES:
        url = "https://github.com/" + REPO + "/blob/" + SHA + "/online/" + quote("教材") + "/" + name
        item = {"chapter": name, "url": url, "type": kind}
        try:
            response = None
            last_error = None
            for attempt in range(3):
                try:
                    response = page.goto(url, wait_until="domcontentloaded", timeout=60000)
                    page.locator("article.markdown-body, .markdown-body").first.wait_for(timeout=20000)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
            if last_error is not None:
                raise last_error
            page.wait_for_timeout(900 if kind == "presence" else 2500)
            item["http_status"] = response.status if response else None
            item["has_expected_text"] = required in page.locator("article.markdown-body, .markdown-body").first.inner_text()
            item["math_renderers"] = page.locator("math-renderer, .katex, mjx-container, .js-render-math").count()
            item["content_images"] = page.locator("article.markdown-body img, .markdown-body img").count()
            item["broken_content_images"] = page.locator("article.markdown-body img, .markdown-body img").evaluate_all(
                "(imgs) => imgs.filter(i => !i.complete || i.naturalWidth === 0).length")
            item["code_blocks"] = page.locator("article.markdown-body pre, .markdown-body pre").count()
            item["raw_parsed_literal"] = 'class="parsed-literal"' in page.locator("article.markdown-body, .markdown-body").first.inner_text()
            item["visible_body_characters"] = len(page.locator("article.markdown-body, .markdown-body").first.inner_text())
            # Guard previously observed real GitHub rendering regressions.
            body_text = page.locator("article.markdown-body, .markdown-body").first.inner_text()
            if name == "algorithms/vpg_zh.md" and "$J(" in body_text:
                result["errors"].append({"chapter": name, "reason": "unrendered inline J math"})
            if name == "user/running_zh.md" and "\\\\[algo name\\\\]" in body_text:
                result["errors"].append({"chapter": name, "reason": "escaped shell placeholder"})
            if name == "spinningup/spinningup_zh.md" and "文献 47" not in body_text:
                result["errors"].append({"chapter": name, "reason": "numbered references missing"})
            item["screenshot"] = name.replace("/", "-") + ".png"
            page.screenshot(path=str(OUTPUT / item["screenshot"]), full_page=False)
            if item["http_status"] != 200 or not item["has_expected_text"] or item["raw_parsed_literal"]:
                result["errors"].append({"chapter": name, "reason": "page/text/raw-html"})
            if kind == "math" and item["math_renderers"] == 0:
                result["errors"].append({"chapter": name, "reason": "no rendered math signal"})
            if kind == "image" and (item["content_images"] == 0 or item["broken_content_images"]):
                result["errors"].append({"chapter": name, "reason": "missing/broken images"})
            if kind == "code" and item["code_blocks"] == 0:
                result["errors"].append({"chapter": name, "reason": "no rendered code blocks"})
            if kind == "bibliography" and "Playing Atari" not in page.locator("article.markdown-body, .markdown-body").first.inner_text():
                result["errors"].append({"chapter": name, "reason": "bibliography missing"})
        except Exception as e:
            result["errors"].append({"chapter": name, "reason": str(e)[:300]})
        print(json.dumps({"page_audit": item}, ensure_ascii=False))
        result["pages"].append(item)
    browser.close()

(OUTPUT / "report.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps({"pages": len(result["pages"]), "errors": result["errors"]}, ensure_ascii=False))
if result["errors"]:
    raise SystemExit(1)
