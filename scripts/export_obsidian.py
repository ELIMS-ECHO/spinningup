#!/usr/bin/env python3
"""Export the 24 reviewed Chinese Spinning Up RST chapters to an Obsidian vault."""
import ast
import html
import json
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
VAULT = ROOT / "dist" / "SpinningUp-Obsidian-Vault"
MATERIAL = VAULT / "教材"
BASE_URL = "https://github.com/tianyueni78-cyber/spinningup/blob/learning-edition-zh/"
UPSTREAM = "https://github.com/openai/spinningup/blob/master/docs/"
SECTION = {"User Documentation": "用户文档", "Introduction to RL": "强化学习基础",
           "Resources": "研究资源", "Algorithms Docs": "算法原理",
           "Utilities Docs": "工具文档", "Etc.": "附录"}
API_PATTERN = re.compile(r"(?m)^\.\. (autoclass|automodule|automethod|autofunction)::\s*([\w.]+)(?:\n[ \t]+:[^\n]*)*")
LINK_PATTERN = re.compile(r"(\]\()([^)]+)(\))")
DIV_PATTERN = re.compile(r'(?s)<div class="(contents|admonition|note|warning|tip|important|hint|caution)"[^>]*>\s*(.*?)\s*</div>')
ROLE_PATTERN = re.compile(r":(?:ref|doc|class|func|meth):`([^`]+)`")
FIGURE_RE = re.compile(r'(?s)<figure>\s*<img\s+src="([^"]+)"[^>]*\/>\s*<figcaption>(.*?)</figcaption>\s*</figure>')


def pandoc(text):
    proc = subprocess.run(["pandoc", "-f", "rst", "-t", "gfm+tex_math_dollars", "--wrap=none"],
                          input=text, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode:
        raise RuntimeError(proc.stderr)
    return proc.stdout


def api_text(kind, dotted):
    chunks = dotted.split(".")
    mod = dotted if kind == "automodule" else ".".join(chunks[:-1])
    src = ROOT / (mod.replace(".", "/") + ".py")
    output = ["### API：" + dotted, "", "源代码：" + BASE_URL + src.relative_to(ROOT).as_posix(), ""]
    if not src.is_file():
        return "\n".join(output + ["源代码未找到；请查阅原始 RST 页面。"])
    tree = ast.parse(src.read_text(encoding="utf-8"))

    def show(node, parent=""):
        label = parent + node.name
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            label += "(" + ast.unparse(node.args) + ")"
        output.extend(["#### " + label, ""])
        doc = ast.get_docstring(node, clean=True)
        if doc:
            try:
                output.extend([pandoc(doc).strip(), ""])
            except Exception:
                output.extend([doc, ""])
        if kind == "autoclass" and isinstance(node, ast.ClassDef):
            for child in node.body:
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) and (
                    not child.name.startswith("_") or child.name == "__init__"
                ):
                    show(child, node.name + ".")

    nodes = tree.body
    if kind != "automodule":
        names = [chunks[-1]]
        nodes = [n for n in tree.body if getattr(n, "name", "") in names]
        if not nodes and kind == "automethod":
            nodes = [n for cls in tree.body if isinstance(cls, ast.ClassDef)
                     for n in cls.body if getattr(n, "name", "") == names[0]]
    else:
        nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef))
                 and not n.name.startswith("_")]
    if not nodes:
        output.append("该对象可能来自继承关系，请查阅源码。")
    for node in nodes:
        show(node)
    return "\n".join(output)


def preserve_console_blocks(text):
    """Protect REPL prompts and their text output from Pandoc paragraph joining."""
    lines = text.splitlines(keepends=True)
    result = []
    i = 0
    while i < len(lines):
        if lines[i].startswith(">>> "):
            end = i
            while end < len(lines) and lines[end].strip():
                end += 1
            result.extend([".. code-block:: text\n", "\n"])
            for line in lines[i:end]:
                result.append("    " + line)
            result.append("\n")
            i = end
        else:
            result.append(lines[i])
            i += 1
    return "".join(result)


def postprocess(text):
    def div_replace(m):
        kind, body = m.groups()
        if kind == "contents":
            return ""
        lines = body.strip().splitlines()
        if kind == "admonition" and lines:
            label = lines.pop(0).strip()
        else:
            label = kind
        while lines and not lines[0].strip():
            lines.pop(0)
        output = ["> [!note] " + label]
        output += ["> " + line if line.strip() else ">" for line in lines]
        return "\n".join(output) + "\n"
    text = DIV_PATTERN.sub(div_replace, text)
    def literal_replace(m):
        body = html.unescape(m.group(1))
        body = "\n".join(line for line in body.splitlines() if line.strip() != chr(96) * 3).strip()
        return "\n```text\n" + body + "\n```\n"
    text = re.sub(r'(?s)<div class="parsed-literal"[^>]*>\s*(.*?)\s*</div>', literal_replace, text)
    def figure_replace(m):
        src = html.unescape(m.group(1))
        raw_caption = html.unescape(m.group(2).strip())
        plain = re.sub(r"<[^>]*>", "", raw_caption)
        rich = re.sub(r"</?strong>", "**", raw_caption)
        rich = re.sub(r"<[^>]*>", "", rich)
        return "![" + plain[:80].replace("]", "") + "](" + src + ")\n\n*" + rich + "*\n"
    return FIGURE_RE.sub(figure_replace, text)



def github_math_and_pseudocode(text):
    """Translate unsupported Sphinx/algorithmic LaTeX for GitHub Markdown."""
    def balanced(source, start):
        if start >= len(source) or source[start] != "{":
            raise ValueError("Expected braced LaTeX argument")
        depth = 0
        for i in range(start, len(source)):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    return source[start + 1:i], i + 1
        raise ValueError("Unclosed LaTeX argument")

    token = r"\underE"
    while token in text:
        start = text.index(token)
        pos = start + len(token)
        if pos >= len(text) or text[pos] != "{":
            raise ValueError("Unexpected underE syntax")
        subscript, pos = balanced(text, pos)
        expectation, end = balanced(text, pos)
        text = text[:start] + r"\mathbb{E}_{" + subscript + r"}\left[" + expectation + r"\right]" + text[end:]

    pattern = re.compile(r'(?:<span nowrap="">)?\$\$\s*(?:\\begin\{aligned\}\s*)?\\begin\{algorithm\}(?:\[H\])?(.*?)\\end\{algorithm\}\s*(?:\\end\{aligned\})?\s*\$\$(?:</span>)?', re.S)
    def algorithm(m):
        block = m.group(1)
        title = re.search(r"\\caption\{([^{}]*)\}", block)
        lines = ["", "**算法伪代码" + ("：" + title.group(1) if title else "") + "**", ""]
        block = re.sub(r"\\(?:begin|end)\{algorithmic\}(?:\[\d+\])?", "", block)
        block = re.sub(r"\\(?:caption|label)\{[^{}]*\}", "", block)
        block = block.replace(r"\begin{equation*}", "\n" + "$" + "$" + "\n").replace(r"\end{equation*}", "\n" + "$" + "$" + "\n")
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            if line == "$":
                line = "$" + "$"
            if line == r"\begin{align*}":
                line = "$" + "$" + "\n" + r"\begin{aligned}"
            if line == r"\end{align*}":
                line = r"\end{aligned}" + "\n" + "$" + "$"
            line = re.sub(r"^\\REPEAT$", "- **重复：**", line)
            line = re.sub(r"^\\UNTIL\{(.*)\}$", r"- **直到：** \1", line)
            line = re.sub(r"^\\IF\{(.*)\}$", r"- **如果：** \1", line)
            line = re.sub(r"^\\ELSE$", "- **否则：**", line)
            line = re.sub(r"^\\ENDIF$", "- **结束条件**", line)
            line = re.sub(r"^\\STATE\s*", "- ", line)
            line = re.sub(r"^\\FOR\{(.*)\}$", r"- **循环：** \1", line)
            line = re.sub(r"^\\ENDFOR\s*$", "- **结束循环**", line)
            lines.append(line)
        return "\n".join(lines) + "\n"
    text = pattern.sub(algorithm, text)
    if r"\begin{algorithm}" in text or r"\underE" in text:
        raise AssertionError("Unsupported algorithm/math macro remains")
    return text


def remap_links(text, folder, translated):
    def change(m):
        raw = m.group(2)
        target, sep, anchor = raw.partition("#")
        if "://" in target or not target.endswith(".html"):
            return m.group(0)
        base = (folder / target).resolve().with_suffix("")
        zh_source = base.with_name(base.name + "_zh.rst")
        if zh_source in translated:
            href = target[:-5] + "_zh.md"
            # Sphinx English anchors do not match GitHub slugs of Chinese headings.
            # Link to the valid translated chapter rather than a nonexistent section.
            anchor = ""
            sep = ""
        else:
            rst = base.with_suffix(".rst")
            if not rst.is_file() or not rst.is_relative_to(DOCS):
                return m.group(0)
            href = UPSTREAM + rst.relative_to(DOCS).as_posix()
        return m.group(1) + href + (sep + anchor if sep else "") + m.group(3)
    return LINK_PATTERN.sub(change, text)


def index_chapters():
    label = None
    ordered = []
    for line in (DOCS / "index.rst").read_text(encoding="utf-8").splitlines():
        m = re.search(r":caption:\s*(.+)$", line)
        if m:
            label = m.group(1)
        m = re.fullmatch(r"\s{3}(user|spinningup|algorithms|utils|etc)/(\w+)", line)
        if m and label in SECTION:
            path = DOCS / m.group(1) / (m.group(2) + "_zh.rst")
            if path.is_file():
                ordered.append((label, path))
    if len(ordered) != 24:
        raise AssertionError("Expected 24 translated main chapters; found " + str(len(ordered)))
    return ordered


def main():
    chapters = index_chapters()
    if VAULT.exists():
        shutil.rmtree(VAULT)
    MATERIAL.mkdir(parents=True)
    originals = VAULT / "98_原始RST备份"
    originals.mkdir(parents=True)
    translated = {p.resolve() for _, p in chapters}
    listing, manifest = [], []

    for num, (section, rst) in enumerate(chapters, 1):
        rel = rst.relative_to(DOCS)
        backup = originals / rel
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rst, backup)
        text = rst.read_text(encoding="utf-8")
        placeholders = {}

        def replace_api(m):
            token = "APIEXPORTTOKEN" + str(len(placeholders)) + "END"
            placeholders[token] = api_text(m.group(1), m.group(2))
            return token

        pre = API_PATTERN.sub(replace_api, text)
        # Keep auto-numbered bibliography entries that Pandoc otherwise drops.
        pre = re.sub(r"(?m)^\.\. \[#\] (.*)$", r"- \1", pre)
        # Treat Sphinx parsed-literal content as literal code before Pandoc conversion.
        pre = re.sub(r"(?m)^(\\s*)\\.\\. parsed-literal::", r"\\1.. code-block:: text", pre)
        pre = preserve_console_blocks(pre)
        pre = ROLE_PATTERN.sub(lambda m: m.group(1), pre)
        md = postprocess(pandoc(pre))
        for token, actual in placeholders.items():
            if token not in md:
                raise AssertionError("Lost API content in " + str(rel))
            md = md.replace(token, actual)
        md = github_math_and_pseudocode(md)
        md = remap_links(md, rst.parent.resolve(), translated)
        title_match = re.search(r"(?m)^# (.+)$", md)
        title = title_match.group(1).strip() if title_match else rst.stem
        destination = MATERIAL / rel.with_suffix(".md")
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = [
            "---",
            "tags: [SpinningUp, 深度强化学习, 中文教材]",
            "source_rst: " + (rst.relative_to(ROOT).as_posix()),
            "source_github: " + BASE_URL + rst.relative_to(ROOT).as_posix(),
            "---", "",
        ]
        destination.write_text("\n".join(metadata) + md.rstrip() + "\n", encoding="utf-8")
        listing.append((num, section, rel.with_suffix(".md"), title))
        manifest.append({"number": num, "original": str(rel), "markdown": str(rel.with_suffix(".md")),
                         "api_extracts": len(placeholders), "original_characters": len(text)})

    # Keep original image paths relative to the corresponding chapter.
    if (DOCS / "images").is_dir():
        shutil.copytree(DOCS / "images", MATERIAL / "images",
                        ignore=shutil.ignore_patterns("*.mp4", "*.xml"))
    for _, _, rel, _ in listing:
        page = MATERIAL / rel
        text = page.read_text(encoding="utf-8")
        for m in re.finditer(r"!\[[^\]]*\]\(([^)\s]+)", text):
            url = m.group(1).split("#", 1)[0]
            if "://" not in url and not (page.parent / url).resolve().is_file():
                raise AssertionError("Missing image: " + str(rel) + " -> " + url)

    # Audit locally referenced Markdown pages and media on every export.
    from urllib.parse import unquote
    missing_links = []
    checked_links = 0
    for page in MATERIAL.rglob("*_zh.md"):
        source = page.read_text(encoding="utf-8")
        for match in re.finditer(r"!?\[[^\]]*\]\(([^)\n]+)\)", source):
            url = match.group(1).strip().split(" ", 1)[0]
            target = unquote(url.split("#", 1)[0].split("?", 1)[0])
            if not target or target.startswith(("/", "//", "data:", "mailto:")) or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
                continue
            checked_links += 1
            if not (page.parent / target).resolve().is_file():
                missing_links.append(f"{page.relative_to(MATERIAL)} -> {target}")
    if missing_links:
        raise AssertionError("Broken local Markdown links:\n" + "\n".join(missing_links[:30]))
    print(json.dumps({"local_links_checked": checked_links, "broken_links": len(missing_links)}))

    contents = ["# Spinning Up 深度强化学习 · 中文教材", "",
                "> 已翻译的 24 篇主目录文章。第25篇《作者》尚未完成，不在本笔记库中。",
                "", "按原站点目录顺序阅读；本公开教材包不包含个人学习笔记。", ""]
    last = ""
    for num, sec, rel, title in listing:
        if sec != last:
            last = sec
            contents += ["## " + SECTION[sec], ""]
        contents.append(str(num).zfill(2) + ". [[" + "教材/" + rel.with_suffix("").as_posix() + "|" + title + "]]")
    contents += ["", "[[导入说明]]"]
    (VAULT / "00_课程导航.md").write_text("\n".join(contents) + "\n", encoding="utf-8")
    (VAULT / "导入说明.md").write_text(
        "# 导入和使用说明\n\n"
        "1. 将下载的压缩包解压到电脑。\n"
        "2. 在 Obsidian 选择 Open folder as vault（将文件夹打开为仓库）。\n"
        "3. 选择文件夹 SpinningUp-Obsidian-Vault，并打开 [[00_课程导航]]。\n\n"
        "教材是 RST 转换的 Markdown 阅读副本，原始已审计的 RST 文件均在 98_原始RST备份。"
        "复杂公式、Sphinx 引用与自动生成 API 可能存在显示差异，应以 GitHub 原件为准。"
        "API 章节从真实 Python 源码静态提取文档字符串，未运行模型训练。"
        "未包含未翻译的补充附录或第25篇；未复制大型 MP4 视频文件。\n\n"
        "仓库：" + BASE_URL + "\n", encoding="utf-8"
    )
    (VAULT / "转换清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    count = len(list(MATERIAL.rglob("*_zh.md")))
    assert count == 24, count
    print(json.dumps({"result": "ok", "chapters": count,
                      "api_objects": sum(m["api_extracts"] for m in manifest),
                      "image_files": len(list((MATERIAL / "images").rglob("*.*")))},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
, '', body).strip()
        return "\n```text\n" + body + "\n```\n"
    text = re.sub(r'(?s)<div class="parsed-literal"[^>]*>\s*(.*?)\s*</div>', literal_replace, text)
    def figure_replace(m):
        src = html.unescape(m.group(1))
        raw_caption = html.unescape(m.group(2).strip())
        plain = re.sub(r"<[^>]*>", "", raw_caption)
        rich = re.sub(r"</?strong>", "**", raw_caption)
        rich = re.sub(r"<[^>]*>", "", rich)
        return "![" + plain[:80].replace("]", "") + "](" + src + ")\n\n*" + rich + "*\n"
    return FIGURE_RE.sub(figure_replace, text)



def github_math_and_pseudocode(text):
    """Translate unsupported Sphinx/algorithmic LaTeX for GitHub Markdown."""
    def balanced(source, start):
        if start >= len(source) or source[start] != "{":
            raise ValueError("Expected braced LaTeX argument")
        depth = 0
        for i in range(start, len(source)):
            if source[i] == "{":
                depth += 1
            elif source[i] == "}":
                depth -= 1
                if depth == 0:
                    return source[start + 1:i], i + 1
        raise ValueError("Unclosed LaTeX argument")

    token = r"\underE"
    while token in text:
        start = text.index(token)
        pos = start + len(token)
        if pos >= len(text) or text[pos] != "{":
            raise ValueError("Unexpected underE syntax")
        subscript, pos = balanced(text, pos)
        expectation, end = balanced(text, pos)
        text = text[:start] + r"\mathbb{E}_{" + subscript + r"}\left[" + expectation + r"\right]" + text[end:]

    pattern = re.compile(r'(?:<span nowrap="">)?\$\$\s*(?:\\begin\{aligned\}\s*)?\\begin\{algorithm\}(?:\[H\])?(.*?)\\end\{algorithm\}\s*(?:\\end\{aligned\})?\s*\$\$(?:</span>)?', re.S)
    def algorithm(m):
        block = m.group(1)
        title = re.search(r"\\caption\{([^{}]*)\}", block)
        lines = ["", "**算法伪代码" + ("：" + title.group(1) if title else "") + "**", ""]
        block = re.sub(r"\\(?:begin|end)\{algorithmic\}(?:\[\d+\])?", "", block)
        block = re.sub(r"\\(?:caption|label)\{[^{}]*\}", "", block)
        block = block.replace(r"\begin{equation*}", "\n" + "$" + "$" + "\n").replace(r"\end{equation*}", "\n" + "$" + "$" + "\n")
        for line in block.splitlines():
            line = line.strip()
            if not line:
                continue
            if line == "$":
                line = "$" + "$"
            if line == r"\begin{align*}":
                line = "$" + "$" + "\n" + r"\begin{aligned}"
            if line == r"\end{align*}":
                line = r"\end{aligned}" + "\n" + "$" + "$"
            line = re.sub(r"^\\REPEAT$", "- **重复：**", line)
            line = re.sub(r"^\\UNTIL\{(.*)\}$", r"- **直到：** \1", line)
            line = re.sub(r"^\\IF\{(.*)\}$", r"- **如果：** \1", line)
            line = re.sub(r"^\\ELSE$", "- **否则：**", line)
            line = re.sub(r"^\\ENDIF$", "- **结束条件**", line)
            line = re.sub(r"^\\STATE\s*", "- ", line)
            line = re.sub(r"^\\FOR\{(.*)\}$", r"- **循环：** \1", line)
            line = re.sub(r"^\\ENDFOR\s*$", "- **结束循环**", line)
            lines.append(line)
        return "\n".join(lines) + "\n"
    text = pattern.sub(algorithm, text)
    if r"\begin{algorithm}" in text or r"\underE" in text:
        raise AssertionError("Unsupported algorithm/math macro remains")
    return text


def remap_links(text, folder, translated):
    def change(m):
        raw = m.group(2)
        target, sep, anchor = raw.partition("#")
        if "://" in target or not target.endswith(".html"):
            return m.group(0)
        base = (folder / target).resolve().with_suffix("")
        zh_source = base.with_name(base.name + "_zh.rst")
        if zh_source in translated:
            href = target[:-5] + "_zh.md"
            # Sphinx English anchors do not match GitHub slugs of Chinese headings.
            # Link to the valid translated chapter rather than a nonexistent section.
            anchor = ""
            sep = ""
        else:
            rst = base.with_suffix(".rst")
            if not rst.is_file() or not rst.is_relative_to(DOCS):
                return m.group(0)
            href = UPSTREAM + rst.relative_to(DOCS).as_posix()
        return m.group(1) + href + (sep + anchor if sep else "") + m.group(3)
    return LINK_PATTERN.sub(change, text)


def index_chapters():
    label = None
    ordered = []
    for line in (DOCS / "index.rst").read_text(encoding="utf-8").splitlines():
        m = re.search(r":caption:\s*(.+)$", line)
        if m:
            label = m.group(1)
        m = re.fullmatch(r"\s{3}(user|spinningup|algorithms|utils|etc)/(\w+)", line)
        if m and label in SECTION:
            path = DOCS / m.group(1) / (m.group(2) + "_zh.rst")
            if path.is_file():
                ordered.append((label, path))
    if len(ordered) != 24:
        raise AssertionError("Expected 24 translated main chapters; found " + str(len(ordered)))
    return ordered


def main():
    chapters = index_chapters()
    if VAULT.exists():
        shutil.rmtree(VAULT)
    MATERIAL.mkdir(parents=True)
    originals = VAULT / "98_原始RST备份"
    originals.mkdir(parents=True)
    translated = {p.resolve() for _, p in chapters}
    listing, manifest = [], []

    for num, (section, rst) in enumerate(chapters, 1):
        rel = rst.relative_to(DOCS)
        backup = originals / rel
        backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(rst, backup)
        text = rst.read_text(encoding="utf-8")
        placeholders = {}

        def replace_api(m):
            token = "APIEXPORTTOKEN" + str(len(placeholders)) + "END"
            placeholders[token] = api_text(m.group(1), m.group(2))
            return token

        pre = API_PATTERN.sub(replace_api, text)
        # Keep auto-numbered bibliography entries that Pandoc otherwise drops.
        pre = re.sub(r"(?m)^\.\. \[#\] (.*)$", r"- \1", pre)
        # Treat Sphinx parsed-literal content as literal code before Pandoc conversion.
        pre = re.sub(r"(?m)^(\\s*)\\.\\. parsed-literal::", r"\\1.. code-block:: text", pre)
        pre = preserve_console_blocks(pre)
        pre = ROLE_PATTERN.sub(lambda m: m.group(1), pre)
        md = postprocess(pandoc(pre))
        for token, actual in placeholders.items():
            if token not in md:
                raise AssertionError("Lost API content in " + str(rel))
            md = md.replace(token, actual)
        md = github_math_and_pseudocode(md)
        md = remap_links(md, rst.parent.resolve(), translated)
        title_match = re.search(r"(?m)^# (.+)$", md)
        title = title_match.group(1).strip() if title_match else rst.stem
        destination = MATERIAL / rel.with_suffix(".md")
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = [
            "---",
            "tags: [SpinningUp, 深度强化学习, 中文教材]",
            "source_rst: " + (rst.relative_to(ROOT).as_posix()),
            "source_github: " + BASE_URL + rst.relative_to(ROOT).as_posix(),
            "---", "",
        ]
        destination.write_text("\n".join(metadata) + md.rstrip() + "\n", encoding="utf-8")
        listing.append((num, section, rel.with_suffix(".md"), title))
        manifest.append({"number": num, "original": str(rel), "markdown": str(rel.with_suffix(".md")),
                         "api_extracts": len(placeholders), "original_characters": len(text)})

    # Keep original image paths relative to the corresponding chapter.
    if (DOCS / "images").is_dir():
        shutil.copytree(DOCS / "images", MATERIAL / "images",
                        ignore=shutil.ignore_patterns("*.mp4", "*.xml"))
    for _, _, rel, _ in listing:
        page = MATERIAL / rel
        text = page.read_text(encoding="utf-8")
        for m in re.finditer(r"!\[[^\]]*\]\(([^)\s]+)", text):
            url = m.group(1).split("#", 1)[0]
            if "://" not in url and not (page.parent / url).resolve().is_file():
                raise AssertionError("Missing image: " + str(rel) + " -> " + url)

    # Audit locally referenced Markdown pages and media on every export.
    from urllib.parse import unquote
    missing_links = []
    checked_links = 0
    for page in MATERIAL.rglob("*_zh.md"):
        source = page.read_text(encoding="utf-8")
        for match in re.finditer(r"!?\[[^\]]*\]\(([^)\n]+)\)", source):
            url = match.group(1).strip().split(" ", 1)[0]
            target = unquote(url.split("#", 1)[0].split("?", 1)[0])
            if not target or target.startswith(("/", "//", "data:", "mailto:")) or re.match(r"^[A-Za-z][A-Za-z0-9+.-]*:", target):
                continue
            checked_links += 1
            if not (page.parent / target).resolve().is_file():
                missing_links.append(f"{page.relative_to(MATERIAL)} -> {target}")
    if missing_links:
        raise AssertionError("Broken local Markdown links:\n" + "\n".join(missing_links[:30]))
    print(json.dumps({"local_links_checked": checked_links, "broken_links": len(missing_links)}))

    contents = ["# Spinning Up 深度强化学习 · 中文教材", "",
                "> 已翻译的 24 篇主目录文章。第25篇《作者》尚未完成，不在本笔记库中。",
                "", "按原站点目录顺序阅读；本公开教材包不包含个人学习笔记。", ""]
    last = ""
    for num, sec, rel, title in listing:
        if sec != last:
            last = sec
            contents += ["## " + SECTION[sec], ""]
        contents.append(str(num).zfill(2) + ". [[" + "教材/" + rel.with_suffix("").as_posix() + "|" + title + "]]")
    contents += ["", "[[导入说明]]"]
    (VAULT / "00_课程导航.md").write_text("\n".join(contents) + "\n", encoding="utf-8")
    (VAULT / "导入说明.md").write_text(
        "# 导入和使用说明\n\n"
        "1. 将下载的压缩包解压到电脑。\n"
        "2. 在 Obsidian 选择 Open folder as vault（将文件夹打开为仓库）。\n"
        "3. 选择文件夹 SpinningUp-Obsidian-Vault，并打开 [[00_课程导航]]。\n\n"
        "教材是 RST 转换的 Markdown 阅读副本，原始已审计的 RST 文件均在 98_原始RST备份。"
        "复杂公式、Sphinx 引用与自动生成 API 可能存在显示差异，应以 GitHub 原件为准。"
        "API 章节从真实 Python 源码静态提取文档字符串，未运行模型训练。"
        "未包含未翻译的补充附录或第25篇；未复制大型 MP4 视频文件。\n\n"
        "仓库：" + BASE_URL + "\n", encoding="utf-8"
    )
    (VAULT / "转换清单.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    count = len(list(MATERIAL.rglob("*_zh.md")))
    assert count == 24, count
    print(json.dumps({"result": "ok", "chapters": count,
                      "api_objects": sum(m["api_extracts"] for m in manifest),
                      "image_files": len(list((MATERIAL / "images").rglob("*.*")))},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
