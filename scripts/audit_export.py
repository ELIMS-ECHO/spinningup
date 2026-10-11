#!/usr/bin/env python3
"""Independent, conservative RST -> Markdown publication audit.

No LLM grading and no rewriting of sources. Uncertain equivalence is marked
for manual review rather than silently accepted.
"""
import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
ONLINE = ROOT / "online" / "教材"

def sha(value):
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]

def headings_rst(text):
    ls = text.splitlines()
    return [ls[i - 1].strip() for i in range(1, len(ls))
            if re.fullmatch(r"[=~^\-]{3,}", ls[i].strip())
            and ls[i-1].strip() and not ls[i-1].lstrip().startswith((">>>", "|", "    "))
            and len(ls[i].strip()) >= len(ls[i-1].strip())]

def headings_md(text):
    return [x.strip() for x in re.findall(r"(?m)^#{1,6} (.+)$", text)]

def footnotes(text):
    return re.findall(r"(?m)^\.\. \[#\] (.+)$", text)

def code_directives(text):
    return re.findall(r"(?m)^\s*\.\. (?:code-block|code|parsed-literal|literalinclude)::", text)

def math_units(text):
    # Deliberately only count explicit math directives and algorithm environments.
    return len(re.findall(r"(?m)^\s*\.\. math::", text)), len(re.findall(r"\\begin\{algorithm\}", text))

def paragraph_samples(text):
    chunks = re.split(r"\n\s*\n", text)
    return [c.strip() for c in chunks if len(c.strip()) > 100 and
            not c.lstrip().startswith((".. ", "#", "~~~", "```"))]


def normalize_prose(value):
    """Normalize surface syntax; never pretend this proves semantic identity."""
    value = re.sub(r"\`([^<\`]+)\s*<https?://[^>]+>\`_", r"\1", value)
    value = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", value)
    value = re.sub(r"https?://\S+", "", value)
    value = re.sub(r"(?m)^\s*\.\. [^:\n]+::.*$", "", value)
    value = re.sub(r"[\s\`*\\_#|\[\]()<>{}~]", "", value)
    return value

def prose_alignment(source, target):
    """Find high-signal windows from every substantive source paragraph."""
    normalized_target = normalize_prose(target)
    entries = []
    for idx, para in enumerate(paragraph_samples(source), 1):
        clean = normalize_prose(para)
        if len(re.findall(r"[\u4e00-\u9fff]", clean)) < 35:
            continue
        width = min(28, len(clean))
        anchors = [clean[:width], clean[max(0, len(clean)//2 - width//2):][:width], clean[-width:]]
        hits = [bool(x and x in normalized_target) for x in anchors]
        entries.append({"source_paragraph": idx, "size": len(clean),
                        "windows_found": sum(hits), "windows_total": len(hits),
                        "fingerprint": sha(clean),
                        "excerpt": clean[:70] if not all(hits) else None})
    suspicious = [x for x in entries if x["windows_found"] == 0]
    partial = [x for x in entries if 0 < x["windows_found"] < x["windows_total"]]
    return {"checked": len(entries), "zero_window_matches": len(suspicious),
            "partial_matches": len(partial),
            "suspected_loss": suspicious[:30], "partial_examples": partial[:15]}

def math_alignment(source, target):
    """Track explicit RST math expressions without asserting LaTeX equivalence."""
    blocks = re.findall(r"(?ms)^\s*\.\. math::\s*\n((?:\s{3,}.+\n|\s*\n)+)", source)
    blocks += re.findall(r"(?s)\\begin\{equation\*?\}(.*?)\\end\{equation\*?\}", source)
    target_compact = re.sub(r"\s+", "", target)
    matched, uncertain = 0, []
    for idx, expression in enumerate(blocks, 1):
        compact = re.sub(r"\s+", "", expression)
        compact = compact.strip()
        if not compact:
            continue
        if compact in target_compact:
            matched += 1
        else:
            uncertain.append({"expression": idx, "fingerprint": sha(compact),
                              "excerpt": compact[:110]})
    return {"source_expression_units": len(blocks), "literal_preserved": matched,
            "requires_review": len(uncertain), "examples": uncertain[:20]}

def audit_pair(name):
    src = DOCS / (name + "_zh.rst")
    dst = ONLINE / (name + "_zh.md")
    out = {"chapter": name, "source": str(src.relative_to(ROOT)),
           "target": str(dst.relative_to(ROOT)), "issues": [], "review": []}
    if not src.is_file() or not dst.is_file():
        out["issues"].append({"kind": "missing_file"})
        return out
    a, b = src.read_text(encoding="utf-8"), dst.read_text(encoding="utf-8")
    out["sha256_short"] = {"rst": sha(a), "md": sha(b)}
    ah, bh = headings_rst(a), headings_md(b)
    missing_heads = [h for h in ah if h not in bh]
    if missing_heads:
        out["issues"].append({"kind": "missing_headings", "examples": missing_heads})
    references = footnotes(a)
    # RST auto-numbered references may become inline Markdown footnotes or
    # bibliography bullet entries. Flag each missing *link target*, not text.
    ref_urls = [re.findall(r"https?://[^\s<>]+", item) for item in references]
    missing_urls = [u for urls in ref_urls for u in urls
                    if u.rstrip("_,.;") not in b]
    if missing_urls:
        out["issues"].append({"kind": "missing_footnote_urls",
                              "count": len(missing_urls),
                              "examples": missing_urls[:10]})
    code_src = len(code_directives(a))
    code_md = len(re.findall(r"(?m)^```", b)) // 2
    if code_md < code_src:
        out["review"].append({"kind": "code_block_count_lower",
                              "source": code_src, "target": code_md})
    math_src, alg_src = math_units(a)
    if math_src and "$" not in b:
        out["issues"].append({"kind": "math_missing"})
    if alg_src and "算法伪代码" not in b:
        out["issues"].append({"kind": "pseudocode_missing"})
    if 'class="parsed-literal"' in b:
        out["issues"].append({"kind": "raw_parsed_literal"})
    if any(tok in b for tok in ("\\underE", "\\begin{algorithm}", "\\STATE")):
        out["issues"].append({"kind": "unsupported_latex"})
    if len(re.findall(r"(?m)^```", b)) % 2:
        out["issues"].append({"kind": "unclosed_code_fence"})
    if b.count("$$") % 2:
        out["issues"].append({"kind": "unclosed_display_math"})
    out["prose_alignment"] = prose_alignment(a, b)
    out["math_alignment"] = math_alignment(a, b)
    if out["prose_alignment"]["zero_window_matches"]:
        out["review"].append({"kind": "suspected_paragraph_loss",
            "count": out["prose_alignment"]["zero_window_matches"]})
    if out["math_alignment"]["requires_review"]:
        out["review"].append({"kind": "math_equivalence_unverified",
            "count": out["math_alignment"]["requires_review"]})
    # Non-identical prose is not necessarily missing: Pandoc reformats it.
    # Preserve samples for deterministic/manual review, do not claim semantic
    # paragraph equivalence from a heading or character-count match.
    out["review"].append({"kind": "prose_semantics_not_proven",
                          "source_paragraph_candidates": len(paragraph_samples(a)),
                          "target_paragraph_candidates": len(paragraph_samples(b))})
    out["counts"] = {"rst_headings": len(ah), "md_headings": len(bh),
                     "rst_auto_footnotes": len(references),
                     "rst_code_directives": code_src, "md_code_fences": code_md,
                     "rst_math_directives": math_src,
                     "rst_algorithm_environments": alg_src}
    return out

def chapter_list():
    index = (DOCS / "index.rst").read_text(encoding="utf-8")
    names = re.findall(r"(?m)^   ((?:user|spinningup|algorithms|utils|etc)/\w+)$", index)
    return [n for n in names if (DOCS / (n + "_zh.rst")).is_file()]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="dist/audit_export.json")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    names = chapter_list()
    rows = [audit_pair(n) for n in names]
    report = {"schema_version": 1, "audit_scope": "committed 24 Chinese RST/Markdown pairs",
              "commit": subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                   capture_output=True, text=True).stdout.strip(),
              "chapters": len(rows),
              "blocking_issues": sum(len(x["issues"]) for x in rows),
              "manual_review_required": True,
              "limitations": ["Does not prove prose/LaTeX semantic equivalence",
                              "Does not capture actual GitHub browser rendering",
                              "Counts are signals, not a substitute for paragraph-level alignment"],
              "results": rows}
    if len(names) != 24:
        report["blocking_issues"] += 1
    target = ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"chapters": len(rows), "blocking_issues": report["blocking_issues"],
                      "manual_review_required": True, "report": str(target)}, ensure_ascii=False))
    print(json.dumps({"prose_windows_checked": sum(x.get("prose_alignment", {}).get("checked", 0) for x in rows),
        "prose_zero_windows": sum(x.get("prose_alignment", {}).get("zero_window_matches", 0) for x in rows),
        "prose_partial": sum(x.get("prose_alignment", {}).get("partial_matches", 0) for x in rows),
        "math_source_units": sum(x.get("math_alignment", {}).get("source_expression_units", 0) for x in rows),
        "math_needs_review": sum(x.get("math_alignment", {}).get("requires_review", 0) for x in rows)}, ensure_ascii=False))
    for entry in rows:
        for issue in entry["issues"]:
            print(json.dumps({"chapter": entry["chapter"], "issue": issue}, ensure_ascii=False))
    if args.strict and report["blocking_issues"]:
        raise SystemExit(1)

if __name__ == "__main__":
    main()
