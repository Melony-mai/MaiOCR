"""Code-aware OCR post-processing.

Pipeline:
    raw OCR text -> clean -> strip editor artifacts (gutters/prompts)
    -> looks_like_code? -> detect language/format -> analyze structure
    -> markdown block

Code mode is authoritative: the user explicitly asked for code, so the
output is always delivered as a fenced code block unless the content is
blatantly natural prose.
"""

import re
from dataclasses import dataclass, field

from maiocr.utils.logger import get_logger

logger = get_logger()

# --------------------------------------------------------------------------
# Cleaning
# --------------------------------------------------------------------------

_FULLWIDTH_MAP = {
    "（": "(", "）": ")", "：": ":", "；": ";", "，": ",", "．": ".",
    "。": ".", "！": "!", "？": "?", "【": "[", "】": "]", "｛": "{",
    "｝": "}", "＜": "<", "＞": ">", "％": "%", "＃": "#", "＆": "&",
    "＄": "$", "＠": "@", "＂": '"', "＇": "'", "｜": "|", "～": "~",
    "＊": "*", "＝": "=", "＋": "+", "－": "-", "／": "/", "＼": "\\",
    "＿": "_", "“": '"', "”": '"', "‘": "'", "’": "'", "«": '"',
    "»": '"', "\u3000": " ", "\u200b": "", "\ufeff": "",
}

_FULLWIDTH_TABLE = str.maketrans(_FULLWIDTH_MAP)

# Multi-char math/logic glyphs OCR renders as unicode symbols instead of
# their ASCII spellings inside code.
_SYMBOL_FIXES = [
    (re.compile(r"→|➜|➔"), "->"),
    (re.compile(r"←"), "<-"),
    (re.compile(r"⇒|⟹"), "=>"),
    (re.compile(r"…|⋯"), "..."),
    (re.compile(r"≥|≧"), ">="),
    (re.compile(r"≤|≦"), "<="),
    (re.compile(r"≠|̸="), "!="),
    (re.compile("\u2212"), "-"),   # minus sign
    (re.compile("\u2013|\u2014|\u2500"), "-"),
]

_BLANK_RUN = re.compile(r"\n{3,}")


def clean_code_text(text: str) -> str:
    """Normalize characters that OCR misreads from monospaced screenshots."""
    cleaned = text.translate(_FULLWIDTH_TABLE)
    for rx, replacement in _SYMBOL_FIXES:
        cleaned = rx.sub(replacement, cleaned)
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines())
    cleaned = _BLANK_RUN.sub("\n\n", cleaned)
    return cleaned.strip()


def clean_plain_text(text: str) -> str:
    """
    Whitespace-only cleanup, preserving CJK punctuation.
    Used when OCR content turns out NOT to be code.
    """
    cleaned = text.replace("\u200b", "").replace("\ufeff", "")
    cleaned = "\n".join(line.rstrip() for line in cleaned.splitlines())
    cleaned = _BLANK_RUN.sub("\n\n", cleaned)
    return cleaned.strip()


# --------------------------------------------------------------------------
# Code vs natural text
# --------------------------------------------------------------------------

_CODE_KEYWORDS = re.compile(
    r"\b(def|function|func|fn|import|include|export|class|struct|impl|enum|"
    r"interface|namespace|package|public|private|protected|static|void|return|"
    r"const|let|var|val|final|mut|async|await|yield|lambda|new|delete|"
    r"if|elif|else|switch|case|match|for|foreach|while|do|break|continue|"
    r"try|catch|except|finally|raise|throw|throws|with|using|require|"
    r"true|false|null|nil|none|undefined|self|this)\b"
)

_IDENTIFIER = re.compile(r"\b(?:_?[a-z]+(?:_[a-z0-9]+)+|[a-z]+(?:[A-Z][a-z0-9]*)+)\b")
_CONSTANT = re.compile(r"\b[A-Z][A-Z0-9_]{2,}\b")
_CALL = re.compile(r"\b[a-zA-Z_]\w*\([^)]*\)")
_SYMBOLS = set("{}[]()<>=+-*/%!&|;:#@$~^\\")
_CJK_SENTENCE = re.compile(r"[。！？，、；：“”‘’（）]")
_NL_TAIL = re.compile(r"[。！？]$")
_COMMON_WORD = re.compile(
    r"\b(the|and|is|are|was|were|of|to|in|that|it|for|on|with|as|at|by|"
    r"这个|那个|我们|你们|他们|因为|所以|但是|如果)\b",
    re.IGNORECASE,
)


def code_score(text: str) -> float:
    """Heuristic score; > 0 means code-like, magnitude is confidence."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if not lines:
        return 0.0

    score = 0.0
    indented = 0
    symbol_chars = 0
    total_chars = 0

    for ln in lines:
        stripped = ln.strip()
        total_chars += len(stripped)
        symbol_chars += sum(1 for ch in stripped if ch in _SYMBOLS)

        if ln[:1] in (" ", "\t") and len(ln) - len(ln.lstrip()) >= 2:
            indented += 1

        if stripped[-1] in ";{}():":
            score += 0.5
        if re.search(r":=|=>|<-|\w+\s*=\s*", stripped):
            score += 0.5
        if _CALL.search(stripped):
            score += 0.5
        if _IDENTIFIER.search(stripped):
            score += 0.25
        if _CONSTANT.search(stripped):
            score += 0.25

    keyword_hits = len(_CODE_KEYWORDS.findall(text))
    score += min(keyword_hits * 0.75, 6.0)

    indent_ratio = indented / len(lines)
    if indent_ratio >= 0.25:
        score += 1.5

    symbol_density = symbol_chars / max(total_chars, 1)
    if symbol_density >= 0.12:
        score += 2.0
    elif symbol_density <= 0.03:
        score -= 1.5

    cjk_punct = len(_CJK_SENTENCE.findall(text))
    score -= min(cjk_punct * 0.5, 3.0)
    score -= min(len(_COMMON_WORD.findall(text)) * 0.25, 3.0)

    if _NL_TAIL.search(text.strip()):
        score -= 1.0

    return score


CODE_SCORE_THRESHOLD = 2.5


def looks_like_code(text: str) -> bool:
    return code_score(text) >= CODE_SCORE_THRESHOLD


# --------------------------------------------------------------------------
# Screenshot artifacts: editor gutters and shell/REPL prompts
# --------------------------------------------------------------------------

_GUTTER_PREFIX = re.compile(r"^(\d{1,4})[.:|\)]?\s+(?=\S)")
_SHELL_PROMPT = re.compile(r"^\$\s+|^>\s+|^❯\s+")
_PS_PROMPT = re.compile(r"^PS [A-Za-z]:\\[^>]*>\s*")
_REPL_PROMPTS = [
    re.compile(r"^>>> ?"),
    re.compile(r"^\.\.\. ?"),
    re.compile(r"^In \[\d+\]: ?"),
    re.compile(r"^Out\[\d+\]: ?"),
    _SHELL_PROMPT,
    _PS_PROMPT,
]


def strip_editor_artifacts(text: str) -> tuple[str, dict]:
    """
    Remove artifacts that come from screenshotting editors and terminals.

    - Line-number gutters (``12  def foo():``): stripped only when most
      lines carry one and the numbers form an increasing column, so data
      lines that merely start with digits survive.
    - Shell / REPL / PowerShell prompts: stripped when present on at least
      a third of the lines.

    Returns (cleaned text, info dict with what was removed).
    """
    lines = text.splitlines()
    info: dict = {"gutter_stripped": False, "prompts_stripped": 0}

    # -- line numbers ------------------------------------------------------
    matches = [(i, m) for i, ln in enumerate(lines) if (m := _GUTTER_PREFIX.match(ln))]
    if len(matches) >= 3:
        frac = len(matches) / max(len(lines), 1)
        nums = [int(m.group(1)) for _, m in matches]
        increasing = sum(1 for a, b in zip(nums, nums[1:]) if b > a)
        if frac >= 0.6 and (nums[0] <= 3 or increasing >= max(len(nums) - 1, 1) * 0.5):
            kept: list[str] = []
            for ln in lines:
                m = _GUTTER_PREFIX.match(ln)
                if m:
                    kept.append(ln[len(m.group(0)) :])
                elif ln.strip().isdigit() and len(ln.strip()) <= 4:
                    continue  # a bare line-number row
                else:
                    kept.append(ln)
            lines = kept
            info["gutter_stripped"] = True
    elif len(matches) >= 2:
        # Partial gutter: OCR glued most numbers into the code, only some
        # lines start with one. Strip them when the numbers increase and at
        # least one remainder looks like code (not prose sentences).
        nums = [int(m.group(1)) for _, m in matches]
        strictly_increasing = all(b > a for a, b in zip(nums, nums[1:]))
        remainders = [lines[i][len(m.group(0)) :] for i, m in matches]
        code_ish = any(
            r.rstrip()[-1:] in ";{}():=" or "(" in r for r in remainders
        )
        if strictly_increasing and code_ish:
            kill = {i for i, _ in matches}
            kept = []
            for i, ln in enumerate(lines):
                m = _GUTTER_PREFIX.match(ln)
                if i in kill:
                    kept.append(ln[len(m.group(0)) :])
                elif ln.strip().isdigit() and len(ln.strip()) <= 4:
                    continue  # orphaned gutter row
                else:
                    kept.append(ln)
            lines = kept
            info["gutter_stripped"] = True

    # -- prompts -----------------------------------------------------------
    prompt_hits = sum(
        1
        for ln in lines
        if any(rx.match(ln) for rx in _REPL_PROMPTS)
    )
    if prompt_hits >= max(2, len(lines) / 3):
        stripped_lines = []
        for ln in lines:
            for rx in _REPL_PROMPTS:
                ln = rx.sub("", ln, count=1)
            stripped_lines.append(ln)
        info["prompts_stripped"] = prompt_hits
        lines = stripped_lines

    return "\n".join(lines).strip("\n"), info


# --------------------------------------------------------------------------
# Code structure recognition
# --------------------------------------------------------------------------

_DEF_RX = re.compile(
    r"(?m)^\s*(?:pub(?:\(\w+\))?\s+)?(?:async\s+)?(?:export\s+)?"
    r"(?:default\s+)?(?:def|fn|func|function|fun)\s+[A-Za-z_]"
)
_CLASS_RX = re.compile(
    r"(?m)^\s*(?:export\s+|public\s+|private\s+|protected\s+|abstract\s+|"
    r"final\s+|data\s+|sealed\s+)*(?:class|struct|interface|enum|impl|trait|"
    r"object)\s+[A-Za-z_]"
)
_IMPORT_RX = re.compile(
    r"(?m)^\s*(?:import\b|from\s+\S+\s+import\b|use\b|require\b|#include\b|"
    r"using\b|@import\b|package\b)"
)
_COMMENT_LINE_RX = re.compile(r"(?m)^\s*(#(?!!)|//|--)")


@dataclass
class CodeStructure:
    """Coarse structural summary of recognized code."""

    functions: int = 0
    classes: int = 0
    imports: int = 0
    comment_lines: int = 0
    balanced: bool | None = None   # None when no brackets at all

    def as_dict(self) -> dict:
        return {
            "functions": self.functions,
            "classes": self.classes,
            "imports": self.imports,
            "comment_lines": self.comment_lines,
            "balanced": self.balanced,
        }


_PAIRS = {"(": ")", "[": "]", "{": "}"}


def _brackets_balanced(text: str) -> bool | None:
    """
    True/False when the text contains bracket pairs, None when it has none.
    String literals and comments are skipped.
    """
    stack: list[str] = []
    i, n = 0, len(text)
    in_str: str | None = None
    comment: str | None = None
    saw_bracket = False

    while i < n:
        ch = text[i]
        two = text[i : i + 2]
        three = text[i : i + 3]

        if comment:
            if comment == "/*":
                if two == "*/":
                    comment = None
                    i += 1
            elif ch == "\n":
                comment = None
        elif in_str:
            if ch == "\\":
                i += 1
            elif three in ('"""', "'''") and in_str in ('"', "'"):
                in_str = None
                i += 2
            elif ch == in_str:
                in_str = None
        elif three in ('"""', "'''"):
            in_str = ch
            i += 2
        elif ch in "\"'":
            in_str = ch
        elif ch == "#":
            comment = "#"
        elif two in ("//", "--", "/*"):
            comment = two
        elif ch in _PAIRS:
            stack.append(_PAIRS[ch])
            saw_bracket = True
        elif ch in ")]}":
            saw_bracket = True
            if not stack or stack[-1] != ch:
                return False
            stack.pop()
        i += 1

    return (not stack) if saw_bracket else None


def analyze_structure(text: str) -> CodeStructure:
    return CodeStructure(
        functions=len(_DEF_RX.findall(text)),
        classes=len(_CLASS_RX.findall(text)),
        imports=len(_IMPORT_RX.findall(text)),
        comment_lines=len(_COMMENT_LINE_RX.findall(text)),
        balanced=_brackets_balanced(text),
    )


# --------------------------------------------------------------------------
# Language detection
# --------------------------------------------------------------------------

_LANG_RULES: list[tuple[str, list[tuple[re.Pattern, float]]]] = [
    (
        "python",
        [
            # Signature with optional return annotation:
            #   def f(...) -> list[int]:
            (re.compile(r"^\s*(?:async\s+)?def\s+\w+\s*\(.*\)\s*(?:->\s*[^:]+)?:", re.MULTILINE), 3),
            (re.compile(r"^\s*import\s+\w+", re.MULTILINE), 2),
            (re.compile(r"^\s*from\s+\w+\s+import\b", re.MULTILINE), 3),
            (re.compile(r"\bself\."), 2),
            (re.compile(r"^\s*@[\w\.]+\s*$", re.MULTILINE), 2),
            (re.compile(r"\belif\b"), 2),
            (re.compile(r"f['\"]\{"), 1.5),
            (re.compile(r"\bprint\s*\("), 0.5),
            (re.compile(r"\b__name__\s*=="), 3),
        ],
    ),
    (
        "javascript",
        [
            (re.compile(r"\bconsole\.log\s*\("), 3),
            (re.compile(r"\bmodule\.exports\b"), 3),
            (re.compile(r"\brequire\s*\(\s*['\"]"), 2.5),
            (re.compile(r"=>"), 1),
            (re.compile(r"\bdocument\.\w+"), 2),
            (re.compile(r"\bfunction\s*\w*\s*\("), 1.5),
            (re.compile(r"\b(const|let)\s+\w+\s*="), 1),
            (re.compile(r"`[^`]*\$\{"), 1.5),
        ],
    ),
    (
        "typescript",
        [
            (re.compile(r":\s*(string|number|boolean|any|void)\b"), 3),
            (re.compile(r"\binterface\s+\w+\s*\{"), 3),
            (re.compile(r"\btype\s+\w+\s*="), 1.5),
            (re.compile(r"\bimplements\s+\w+"), 2),
            (re.compile(r"<[A-Z]\w*(?:,\s*\w+)?>\s*\("), 1.5),
            (re.compile(r"\benum\s+\w+"), 1.5),
        ],
    ),
    (
        "go",
        [
            (re.compile(r"^package\s+\w+", re.MULTILINE), 3),
            (re.compile(r"\bfmt\.\w+\("), 2.5),
            (re.compile(r":="), 2),
            (re.compile(r"^import\s*\(", re.MULTILINE), 2),
            (re.compile(r"\bfunc\s+(\(?\w+\s*\*?\w*\)?\s*)?\w+\("), 2),
            (re.compile(r"\bdefer\b|\bgo\s+\w+\("), 2),
        ],
    ),
    (
        "rust",
        [
            (re.compile(r"\bprintln!\s*\("), 3),
            (re.compile(r"\bfn\s+\w+"), 2.5),
            (re.compile(r"\blet\s+mut\b"), 3),
            (re.compile(r"\bimpl\s+\w+"), 2),
            (re.compile(r"::<"), 2),
            (re.compile(r"\bVec<"), 2),
            (re.compile(r"&str\b|\bString::from"), 2),
            (re.compile(r"\bmatch\s+\w+\s*\{"), 1.5),
        ],
    ),
    (
        "java",
        [
            (re.compile(r"\bSystem\.out\.print"), 3),
            (re.compile(r"\bpublic\s+(static\s+)?(final\s+)?(void|class|int|String|boolean)\b"), 2.5),
            (re.compile(r"@Override\b"), 2.5),
            (re.compile(r"^package\s+[\w\.]+;", re.MULTILINE), 2.5),
            (re.compile(r"\bprivate\s+\w+\s+\w+\s*;"), 1.5),
            (re.compile(r"\bnew\s+\w+\s*\("), 0.5),
        ],
    ),
    (
        "csharp",
        [
            (re.compile(r"\bConsole\.(WriteLine|Write)\s*\("), 3),
            (re.compile(r"^using\s+[\w\.]+;", re.MULTILINE), 2),
            (re.compile(r"\bnamespace\s+[\w\.]+"), 2),
            (re.compile(r"\bpublic\s+class\s+\w+"), 1.5),
            (re.compile(r"\bvar\s+\w+\s*=\s*new\b"), 1.5),
            (re.compile(r"\basync\s+Task\b"), 2),
        ],
    ),
    (
        "cpp",
        [
            (re.compile(r"#include\s*<(iostream|string|vector|map)>"), 3),
            (re.compile(r"\bstd::"), 3),
            (re.compile(r"\b(cout|cin)\s*(<<|>>)"), 3),
            (re.compile(r"\btemplate\s*<"), 2),
            (re.compile(r"\busing\s+namespace\s+std"), 2.5),
        ],
    ),
    (
        "c",
        [
            (re.compile(r"#include\s*<(stdio|stdlib|string|math)\.h>"), 3),
            (re.compile(r"\bprintf\s*\("), 2),
            (re.compile(r"\bscanf\s*\("), 2),
            (re.compile(r"\bmalloc\s*\(|\bfree\s*\("), 2),
            (re.compile(r"\bint\s+main\s*\("), 1.5),
        ],
    ),
    (
        "php",
        [
            (re.compile(r"<\?php"), 4),
            (re.compile(r"\$\w+\s*->\s*\w+"), 2.5),
            (re.compile(r"\becho\s+"), 1.5),
            (re.compile(r"\bfunction\s+__construct"), 2),
        ],
    ),
    (
        "ruby",
        [
            (re.compile(r"^\s*def\s+\w+", re.MULTILINE), 1.5),
            (re.compile(r"^\s*end$", re.MULTILINE), 2),
            (re.compile(r"\bputs\s+"), 2),
            (re.compile(r"\battr_(accessor|reader)\b"), 3),
            (re.compile(r"#\{[^}]*\}"), 1),
        ],
    ),
    (
        "kotlin",
        [
            (re.compile(r"\bfun\s+\w+\s*\("), 2.5),
            (re.compile(r"\bval\s+\w+\s*="), 2.5),
            (re.compile(r"\bprintln\s*\("), 1.5),
            (re.compile(r"\bdata class\b|\bcompanion object\b"), 3),
        ],
    ),
    (
        "swift",
        [
            (re.compile(r"\bfunc\s+\w+\s*\("), 1.5),
            (re.compile(r"\blet\s+\w+\s*(:|=\s*\[)"), 1),
            (re.compile(r"\\\(\w+\)"), 2.5),
            (re.compile(r"\bguard\s+let\b|\bif\s+let\b"), 3),
        ],
    ),
    (
        "sql",
        [
            (re.compile(r"\bSELECT\b.+\bFROM\b", re.IGNORECASE | re.DOTALL), 3),
            (re.compile(r"\bINSERT\s+INTO\b", re.IGNORECASE), 3),
            (re.compile(r"\bCREATE\s+TABLE\b", re.IGNORECASE), 3),
            (re.compile(r"\bWHERE\b", re.IGNORECASE), 1),
            (re.compile(r"\bGROUP\s+BY\b|\bORDER\s+BY\b", re.IGNORECASE), 2),
            (re.compile(r"\bJOIN\b", re.IGNORECASE), 1.5),
        ],
    ),
    (
        "html",
        [
            (re.compile(r"<!DOCTYPE\s+html", re.IGNORECASE), 4),
            (re.compile(r"<html[\s>]", re.IGNORECASE), 3),
            (re.compile(r"</(div|span|body|head|p|a)>", re.IGNORECASE), 2),
            (re.compile(r"<(div|span|p|a|h[1-6]|ul|li)\b[^>]*>", re.IGNORECASE), 1),
            (re.compile(r"<script|<link|<meta", re.IGNORECASE), 1.5),
        ],
    ),
    (
        "css",
        [
            (re.compile(r"[@#\.]?[\w\-]+\s*\{[^{}]*:[^{}]*;?\s*\}", re.DOTALL), 2),
            (re.compile(r":\s*(hover|focus|active|before|after)\b"), 2.5),
            (re.compile(r"@media\b"), 2.5),
            (re.compile(r"\b(margin|padding|color|font-size|display)\s*:"), 1.5),
        ],
    ),
    (
        "json",
        [
            (re.compile(r'^\s*[\{\[]', re.MULTILINE), 1),
            (re.compile(r'"[^"]+"\s*:\s*("[^"]*"|[\d\.]+|true|false|null|\{|\[)'), 2.5),
            (re.compile(r"^\s*\},?\s*$", re.MULTILINE), 1),
        ],
    ),
    (
        "yaml",
        [
            (re.compile(r"^[\w\.\-]+:\s*(\S.*)?$", re.MULTILINE), 2),
            (re.compile(r"^\s+-?\s*[\w\.\-]+:\s+(\S.*)?$", re.MULTILINE), 1),
            (re.compile(r"^\s*-\s+\S", re.MULTILINE), 0.5),
            (re.compile(r"^---\s*$", re.MULTILINE), 2),
            (re.compile(r":\s+(?:true|false|null|~)\s*$", re.MULTILINE | re.IGNORECASE), 2),
        ],
    ),
    (
        "bash",
        [
            (re.compile(r"^#!\s*/bin/(ba)?sh", re.MULTILINE), 4),
            (re.compile(r"\becho\s+"), 1),
            (re.compile(r"\$\{?\w+\}?"), 1),
            (re.compile(r"(^|\s)(fi|esac|then)\s*$", re.MULTILINE), 2),
            (re.compile(r"\bapt(-get)?\s+(install|update)"), 2),
            (re.compile(r"\|\s*\w+|&&"), 0.5),
        ],
    ),
    (
        "powershell",
        [
            (re.compile(r"\b(Get|Set|New|Remove|Write)-[A-Z]\w+-?[A-Za-z]*\b"), 3),
            (re.compile(r"\bWrite-Host\b"), 3),
            (re.compile(r"\$\w+\s*=\s*"), 1),
            (re.compile(r"\-\w+\s+\$?\w+"), 0.5),
        ],
    ),
]

# Disambiguation: TS-specific syntax required for typescript to beat javascript.
_TS_SPECIFIC = re.compile(
    r":\s*(string|number|boolean|any)\b|\binterface\s+\w+\s*\{"
)

_SCORE_THRESHOLD = 1.5


def detect_language(text: str) -> str:
    """
    Return a fenced-code language identifier ('python', 'cpp', ...).
    Empty string when nothing scores above threshold.
    """
    sample = text[:4000]
    scores: dict[str, float] = {}

    for lang, rules in _LANG_RULES:
        scores[lang] = sum(weight for rx, weight in rules if rx.search(sample))

    if not _TS_SPECIFIC.search(sample):
        scores["typescript"] *= 0.3

    best_lang, best_score = "", 0.0
    for lang, score in scores.items():
        if score > best_score:
            best_lang, best_score = lang, score

    return best_lang if best_score >= _SCORE_THRESHOLD else ""


# --------------------------------------------------------------------------
# Layout reconstruction: indentation + vertical whitespace from geometry
# --------------------------------------------------------------------------

_MIN_UNIT_PX = 8
_MAX_UNIT_PX = 240
_MAX_INDENT_LEVEL = 16


def render_layout(geom) -> str:
    """
    Rebuild code text from OCR line geometry.

    The recognizer emits each line *without* leading whitespace - the
    indentation only exists in the bounding-box x offsets. This function:

      1. estimates the indent unit (median gap between distinct left edges,
         falling back to a multiple of the median line height),
      2. converts each line's left offset into an indentation level,
      3. re-inserts blank lines where vertical gaps between consecutive
         rows exceed normal line spacing.

    ``geom`` is a sequence of objects with text/left/top/right/bottom
    (already merged and ordered top-down), as produced by the engine.
    """
    rows = [g for g in geom if g.text.strip()]
    if not rows:
        return ""

    heights = sorted(g.bottom - g.top for g in rows)
    med_h = max(heights[len(heights) // 2], 1)

    base = min(g.left for g in rows)
    lefts = sorted({g.left for g in rows})

    # The indent unit is the smallest consistent step between adjacent
    # distinct margins - every hierarchy level shows up as one such step in
    # real code. Glyph-bearing jitter produces tiny deltas; filter them with
    # a floor scaled to the detected text height.
    jitter_floor = max(_MIN_UNIT_PX, round(med_h * 0.5))
    deltas = [b - a for a, b in zip(lefts, lefts[1:]) if b - a >= jitter_floor]
    if deltas:
        unit = int(sorted(deltas)[len(deltas) // 2])
    else:
        unit = int(med_h * 1.8)
    unit = max(jitter_floor, min(_MAX_UNIT_PX, unit))

    # Line pitch (top-to-top advance) makes blank-line detection robust:
    # an empty editor line roughly doubles the gap between neighbours.
    # Only trusted with enough rows; tiny captures use height heuristics.
    rows_sorted = sorted(rows, key=lambda r: (r.top, r.left))
    top_diffs = [
        b.top - a.top
        for a, b in zip(rows_sorted, rows_sorted[1:])
        if b.top > a.top
    ]
    pitch = (
        int(sorted(top_diffs)[len(top_diffs) // 2])
        if len(top_diffs) >= 3
        else 0
    )

    out: list[str] = []
    prev = None
    for g in rows_sorted:
        level = round((g.left - base) / unit)
        level = max(0, min(_MAX_INDENT_LEVEL, level))

        if prev is not None:
            gap = g.top - prev.bottom
            if pitch:
                r = gap / pitch
                blanks = 2 if r > 1.95 else (1 if r > 0.95 else 0)
            else:
                blanks = 2 if gap > 3.2 * med_h else (1 if gap > 1.15 * med_h else 0)
            out.extend([""] * blanks)

        out.append("    " * level + g.text.rstrip())
        prev = g

    logger.debug(
        "Layout rebuilt: {} rows, base={}, unit={}px, levels={}",
        len(rows),
        base,
        unit,
        len({round((g.left - base) / unit) for g in rows}),
    )
    return "\n".join(out)


# --------------------------------------------------------------------------
# Triple-quote repair (Python docstrings)
# --------------------------------------------------------------------------

# Glyphs the recognizer produces when reading """ / ''' shapes.
# NOTE: capital-I runs (``III``) are one of the most common misreads.
_QUOTE_NOISE_CHARS = set("uvnliIlL|'\"`")
# Short tokens spelled entirely from those glyphs that are real code words.
_QUOTE_NOISE_EXCEPT = {
    "null", "nil", "fun", "in", "il", "li", "ill", "lli", "lin",
    "vin", "van", "nan", "vii",
}

# Rows matching this inside an open docstring terminate it (their indent
# betrayed them): they read as real code, not prose.
_CODE_LINE_RX = re.compile(
    r"^\s*(?:def\s|class\s|import\s|from\s|return\b|if\s|for\s|while\s|"
    r"with\s|try\b|elif\b|else\s*:|@|[A-Za-z_][\w\.]*\s*=[^=]|\}|#include)"
)


def _looks_like_quote_noise(stripped: str) -> bool:
    """
    True for lines that are almost certainly a mangled triple-quote:
    2-6 characters drawn only from quote-confusion glyphs
    (u v n l i I l L | ' "), ignoring spaces - e.g. ``u u n``, ``uun``,
    ``'''``, ``III``, ``"I``, ``" " "``.
    """
    compact = stripped.replace(" ", "")
    if not (2 <= len(compact) <= 6):
        return False
    if any(ch not in _QUOTE_NOISE_CHARS for ch in compact):
        return False
    return compact.lower() not in _QUOTE_NOISE_EXCEPT


def repair_triple_quotes(text: str) -> tuple[str, int]:
    r'''
    Restore triple-quote lines destroyed by OCR (e.g. ``u u n``).

    A small docstring state machine:

      - a noise line directly after a block opener (previous non-blank line
        ends with ``:``) becomes an opening quote
      - further noise lines while inside the docstring become the closing
        quote
      - body lines are kept, lifted to at least the opener's indentation;
        a shallower row that looks like real code closes the docstring
      - genuine triple-quote lines toggle the state untouched
      - an unclosed docstring at EOF gets a synthesized closer

    Returns (repaired text, number of repaired lines).
    '''
    out: list[str] = []
    repairs = 0
    in_doc = False
    doc_indent = 0
    prev_nonblank = ""

    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped:
            out.append(raw)
            continue

        indent = len(raw) - len(raw.lstrip(" "))

        if '"""' in stripped or "'''" in stripped:
            toggles = (stripped.count('"""') + stripped.count("'''")) % 2
            if toggles:
                in_doc = not in_doc
                doc_indent = indent if in_doc else 0
            out.append(raw)
            prev_nonblank = stripped
            continue

        if _looks_like_quote_noise(stripped):
            opens_block = prev_nonblank.rstrip().endswith(":")
            if not in_doc and opens_block:
                out.append(" " * indent + '"""')
                repairs += 1
                in_doc, doc_indent = True, indent
                prev_nonblank = '"""'
                continue
            if in_doc:
                out.append(" " * max(indent, doc_indent) + '"""')
                repairs += 1
                in_doc, doc_indent = False, 0
                prev_nonblank = '"""'
                continue
            # No context to justify a repair - leave the line untouched.
            out.append(raw)
            prev_nonblank = stripped
            continue

        if in_doc and indent < doc_indent and _CODE_LINE_RX.match(stripped):
            # Real code resumes at a shallower level: the docstring's
            # closer was lost to OCR - synthesize it before this line.
            out.append(" " * doc_indent + '"""')
            repairs += 1
            in_doc, doc_indent = False, 0
            out.append(raw)
        elif in_doc and indent < doc_indent:
            # Docstring body whose indentation OCR dropped: lift it back
            # under the quotes instead of leaking it out of the string.
            out.append(" " * doc_indent + stripped)
        else:
            out.append(raw)
        prev_nonblank = stripped

    if in_doc:
        # EOF with an unterminated docstring: synthesize the closer.
        out.append(" " * doc_indent + '"""')
        repairs += 1
    result = "\n".join(out)
    if text.endswith("\n") and not result.endswith("\n"):
        result += "\n"
    return result, repairs


# --------------------------------------------------------------------------
# Markdown output
# --------------------------------------------------------------------------

_FENCE_IN_CODE = re.compile(r"^`{3,}", re.MULTILINE)


def to_markdown(code: str, language: str = "") -> str:
    """Wrap code in a fenced Markdown block, safe against embedded fences."""
    inner = max((len(m.group(0)) for m in _FENCE_IN_CODE.finditer(code)), default=0)
    fence = "`" * max(3, inner + 1)
    info = language.strip().lower()
    return f"{fence}{info}\n{code.rstrip()}\n{fence}"


@dataclass
class CodeOcrResult:
    output: str          # what gets copied to the clipboard
    language: str        # '' when unknown / not code
    is_code: bool
    structure: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)


# Below this score, with no language detected, content is treated as prose
# even in explicit code mode (protects against accidentally capturing an
# article while pressing the code hotkey).
_PROSE_SCORE_FLOOR = 1.0


def process_code(raw_text: str, force_code: bool = True) -> CodeOcrResult:
    """
    Full code-mode post-processing.

    Code mode is authoritative: when ``force_code`` (the code hotkey), the
    result is always a fenced block with structure analysis - even if the
    heuristics are only lukewarm (config files, snippets, pseudo-code).
    Plain-text fallback remains solely for blatant natural-language prose.
    """
    cleaned = clean_code_text(raw_text)
    cleaned, artifacts = strip_editor_artifacts(cleaned)
    cleaned, quote_repairs = repair_triple_quotes(cleaned)
    if quote_repairs:
        logger.info("Repaired {} triple-quote line(s)", quote_repairs)

    score = code_score(cleaned)
    language = detect_language(cleaned)

    is_prose = (
        not looks_like_code(cleaned) and language == "" and score < _PROSE_SCORE_FLOOR
    )
    if not force_code or is_prose:
        # Natural-language content: keep CJK punctuation intact.
        return CodeOcrResult(
            output=clean_plain_text(raw_text),
            language="",
            is_code=False,
            artifacts=artifacts,
        )

    structure = analyze_structure(cleaned)
    return CodeOcrResult(
        output=to_markdown(cleaned, language),
        language=language,
        is_code=True,
        structure=structure.as_dict(),
        artifacts=artifacts,
    )
