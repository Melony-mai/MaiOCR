"""Unit tests for maiocr.ocr.code (no Qt / no models required)."""

import pytest

from maiocr.ocr.code import (
    analyze_structure,
    clean_code_text,
    code_score,
    detect_language,
    looks_like_code,
    process_code,
    render_layout,
    repair_triple_quotes,
    strip_editor_artifacts,
    to_markdown,
)
from maiocr.ocr.engine import GeomLine, drop_gutter_boxes, merge_fragment_lines


def _g(text, left, top=0, right=None, bottom=20):
    return GeomLine(
        text=text,
        left=left,
        top=top,
        right=right if right is not None else left + len(text) * 10,
        bottom=bottom,
    )

PYTHON_CODE = """\
import asyncio

async def fetch(url: str) -> dict:
    async with session.get(url) as resp:
        data = await resp.json()
    return {"url": url, "status": resp.status}

def main():
    results = asyncio.run(fetch("https://api.example.com/v1"))
    print(results["status"])
"""

JS_CODE = """\
const fetchData = async (url) => {
  const response = await fetch(url);
  const data = await response.json();
  console.log(data.items.length);
  return data;
};
"""

TS_CODE = """\
interface User {
  id: number;
  name: string;
}

export function getUser(id: number): User | null {
  return users.find((u) => u.id === id) ?? null;
}
"""

GO_CODE = """\
package main

import "fmt"

func main() {
    items := []string{"a", "b"}
    for i, v := range items {
        fmt.Println(i, v)
    }
}
"""

RUST_CODE = """\
fn main() {
    let mut count = 0u32;
    let names: Vec<&str> = vec!["ai", "bo"];
    for name in names {
        println!("hello {}", name);
        count += 1;
    }
}
"""

JAVA_CODE = """\
public class Main {
    public static void main(String[] args) {
        System.out.println("Hello, world!");
        int total = args.length;
    }
}
"""

CPP_CODE = """\
#include <iostream>
#include <vector>

int main() {
    std::vector<int> nums{1, 2, 3};
    std::cout << nums.size() << std::endl;
    return 0;
}
"""

SQL_CODE = """\
SELECT u.id, u.name, COUNT(o.id) AS order_count
FROM users u
LEFT JOIN orders o ON o.user_id = u.id
WHERE u.created_at > '2024-01-01'
GROUP BY u.id, u.name
ORDER BY order_count DESC;
"""

JSON_CODE = """\
{
  "name": "maiocr",
  "version": "0.1.0",
  "features": ["ocr", "code"],
  "gpu": true,
  "retries": 3
}
"""

HTML_CODE = """\
<!DOCTYPE html>
<html lang="zh">
<head>
  <meta charset="utf-8">
  <title>demo</title>
</head>
<body>
  <div id="app" class="container">hello</div>
</body>
</html>
"""

BASH_CODE = """\
#!/bin/bash
set -e

for dir in src tests; do
    echo "building $dir"
    cd "$dir"
done
"""

CHINESE_TEXT = (
    "MaiOCR 是一款完全本地离线运行的文字识别工具。"
    "它支持中英日多语言识别，以及混合文本场景。"
    "按下快捷键即可截图并自动识别。"
)


# ---------------------------------------------------------------- cleaning


class TestClean:
    def test_fullwidth_punctuation(self):
        assert clean_code_text("（a）：" ) == "(a):"
        assert clean_code_text("｛｝") == "{}"
        assert clean_code_text("＝＝") == "=="

    def test_smart_quotes(self):
        assert clean_code_text("\u201cx\u201d") == '"x"'
        assert clean_code_text("\u2018y\u2019") == "'y'"

    def test_ideographic_space(self):
        assert clean_code_text("if\u3000x:") == "if x:"

    def test_zero_width_removed(self):
        assert clean_code_text("a\u200bb\ufeffc") == "abc"

    def test_trailing_whitespace_and_blank_runs(self):
        cleaned = clean_code_text("a  \n\n\n\nb")
        assert cleaned == "a\n\nb"


# ------------------------------------------------------- code vs natural text


class TestLooksLikeCode:
    @pytest.mark.parametrize(
        "snippet",
        [PYTHON_CODE, JS_CODE, GO_CODE, RUST_CODE, JAVA_CODE, SQL_CODE],
        ids=["py", "js", "go", "rust", "java", "sql"],
    )
    def test_code_detected(self, snippet):
        assert looks_like_code(snippet), f"score={code_score(snippet)}"

    def test_natural_language_rejected(self):
        assert not looks_like_code(CHINESE_TEXT)

    def test_short_natural_sentence_rejected(self):
        assert not looks_like_code("this is a simple sentence about the weather")

    def test_empty_rejected(self):
        assert not looks_like_code("")
        assert code_score("") == 0.0


# ------------------------------------------------------------ language detection


class TestDetectLanguage:
    @pytest.mark.parametrize(
        ("snippet", "expected"),
        [
            (PYTHON_CODE, "python"),
            (JS_CODE, "javascript"),
            (TS_CODE, "typescript"),
            (GO_CODE, "go"),
            (RUST_CODE, "rust"),
            (JAVA_CODE, "java"),
            (CPP_CODE, "cpp"),
            (SQL_CODE, "sql"),
            (JSON_CODE, "json"),
            (HTML_CODE, "html"),
            (BASH_CODE, "bash"),
        ],
    )
    def test_known_languages(self, snippet, expected):
        assert detect_language(snippet) == expected, snippet

    def test_unknown_returns_empty(self):
        assert detect_language(CHINESE_TEXT) == ""

    def test_ts_requires_specific_syntax(self):
        plain_js = JS_CODE + "\nexport { fetchData };"
        # No TS-only constructs -> typescript score must be discounted.
        scores_ok = detect_language(plain_js) in ("javascript", "")
        assert scores_ok


# ------------------------------------------------------------------ markdown


class TestMarkdown:
    def test_basic_block(self):
        md = to_markdown("print(1)", "python")
        assert md.startswith("```python\n")
        assert md.endswith("\n```")

    def test_no_language(self):
        md = to_markdown("some stuff")
        assert md.startswith("```\n")

    def test_embedded_fence_upgrades(self):
        tricky = "text\n```\ncode\n```\nmore"
        md = to_markdown(tricky, "python")
        assert md.count("````python\n") == 1
        inner = md.split("\n", 1)[1]
        assert "```\n" in inner

    def test_trailing_newline_normalized(self):
        assert to_markdown("print(1)\n\n\n", "python").endswith("print(1)\n```")


# --------------------------------------------------------------- integration


class TestProcessCode:
    def test_code_path(self):
        res = process_code(PYTHON_CODE)
        assert res.is_code
        assert res.language == "python"
        assert res.output.startswith("```python\n")
        assert res.output.endswith("\n```")

    def test_non_code_falls_back_to_plain(self):
        res = process_code(CHINESE_TEXT)
        assert not res.is_code
        assert res.language == ""
        assert res.output == CHINESE_TEXT

    def test_fullwidth_cleanup_before_detection(self):
        dirty = PYTHON_CODE.replace(":", "：").replace("{", "｛").replace("}", "｝")
        res = process_code(dirty)
        assert res.is_code
        assert "{" in res.output and "}" in res.output

    def test_english_sentence_stays_plain_in_code_mode(self):
        res = process_code("This is a simple sentence about the weather.")
        assert not res.is_code


# --------------------------------------------------- code-mode authority
# Code mode must deliver code even when the heuristic is lukewarm.


class TestCodeModeAuthority:
    YAML_CONFIG = "server:\n  host: 0.0.0.0\n  port: 8080\nretries: 3\n"

    AMBIGUOUS = "width = right - left\nheight = bottom - top\n"

    def test_config_file_is_fenced_with_language(self):
        # Below looks_like_code threshold, but language detection fires.
        assert not looks_like_code(self.YAML_CONFIG)
        res = process_code(self.YAML_CONFIG)
        assert res.is_code
        assert res.language == "yaml"
        assert res.output.startswith("```yaml\n")

    def test_ambiguous_snippet_still_fenced(self):
        assert code_score(self.AMBIGUOUS) < 2.5  # old heuristic said "no"
        res = process_code(self.AMBIGUOUS)
        assert res.is_code
        assert res.output.startswith("```\n")
        assert "width = right - left" in res.output

    def test_force_code_false_keeps_legacy_behaviour(self):
        res = process_code(self.AMBIGUOUS, force_code=False)
        assert not res.is_code
        assert res.output.strip() == self.AMBIGUOUS.strip()


# ------------------------------------------------------ editor artifacts


class TestEditorArtifacts:
    def test_line_number_gutter_stripped(self):
        shot = (
            "12  def greet(name):\n"
            "13      print(f\"hi {name}\")\n"
            "14\n"
            "15  greet(\"world\")\n"
        )
        cleaned, info = strip_editor_artifacts(shot)
        assert info["gutter_stripped"]
        assert "12" not in cleaned.splitlines()[0]
        assert cleaned.startswith("def greet(name):")
        assert "greet(\"world\")" in cleaned
        assert "\n14" not in cleaned and "\n14\n" not in cleaned + "\n"

    def test_data_lines_starting_with_digits_survive(self):
        shot = "1 apples are red\n2 sky is blue\n"
        cleaned, info = strip_editor_artifacts(shot)
        assert not info["gutter_stripped"]
        assert cleaned.startswith("1 apples")

    def test_shell_prompts_stripped(self):
        shot = "$ pip install maiocr\n$ maiocr --help\n$ echo done\n"
        cleaned, info = strip_editor_artifacts(shot)
        assert info["prompts_stripped"] == 3
        assert cleaned.startswith("pip install")

    def test_repl_prompts_stripped(self):
        shot = ">>> def add(a, b):\n...     return a + b\n>>> add(1, 2)\n3\n"
        cleaned, info = strip_editor_artifacts(shot)
        assert info["prompts_stripped"] >= 3
        assert ">>>" not in cleaned
        assert "def add(a, b):" in cleaned

    def test_single_prompt_line_left_alone(self):
        shot = "total = a + b  # $ cheap math\ntotal += 1\n"
        _, info = strip_editor_artifacts(shot)
        assert info["prompts_stripped"] == 0


# ----------------------------------------------------- symbol normalization


class TestSymbolRepair:
    def test_unicode_operators_become_ascii(self):
        assert "->" in clean_code_text("a → b")
        assert ">=" in clean_code_text("a ≥ b")
        assert "!=" in clean_code_text("a ≠ b")
        assert "..." in clean_code_text("x…y")


# ------------------------------------------------------------ structure


class TestStructure:
    def test_python_structure_counts(self):
        s = analyze_structure(PYTHON_CODE)
        assert s.functions == 2
        assert s.imports == 1
        assert s.classes == 0
        assert s.balanced is True

    def test_class_detection(self):
        s = analyze_structure(JAVA_CODE)
        assert s.classes == 1
        assert s.imports == 0

    def test_unbalanced_brackets_flagged(self):
        broken = "def f():\n    d = {\"a\": 1\n"
        s = analyze_structure(broken)
        assert s.balanced is False

    def test_no_brackets_is_none(self):
        assert analyze_structure("import os\n").balanced is None

    def test_process_code_exposes_structure(self):
        res = process_code(PYTHON_CODE)
        assert res.structure["functions"] == 2
        assert res.structure["balanced"] is True


# ------------------------------------------------- layout reconstruction


class TestLayoutReconstruction:
    def test_indentation_levels_from_x_offsets(self):
        rows = [
            _g("def f():", left=100, top=0, bottom=20),
            _g("return 1", left=140, top=30, bottom=50),
            _g("if x:", left=180, top=60, bottom=80),
        ]
        out = render_layout(rows)
        lines = out.splitlines()
        assert lines[0] == "def f():"
        assert lines[1] == "    return 1"
        assert lines[2] == "        if x:"

    def test_blank_lines_from_vertical_gaps(self):
        rows = [
            _g("a = 1", left=100, top=0, bottom=20),
            _g("b = 2", left=100, top=70, bottom=90),   # gap 50 >> spacing
        ]
        out = render_layout(rows)
        assert out.splitlines() == ["a = 1", "", "b = 2"]

    def test_no_spurious_blanks_for_tight_lines(self):
        rows = [
            _g("one", left=100, top=0, bottom=20),
            _g("two", left=100, top=28, bottom=48),
            _g("three", left=100, top=56, bottom=76),
        ]
        assert render_layout(rows).splitlines() == ["one", "two", "three"]

    def test_flat_when_all_lines_share_left(self):
        rows = [_g(f"line{i}", left=100) for i in range(3)]
        assert all(not ln.startswith(" ") for ln in render_layout(rows).splitlines())

    def test_indentation_survives_markdown_fencing(self):
        rows = [
            _g("def f():", left=100, top=0, bottom=20),
            _g("return 1", left=140, top=30, bottom=50),
        ]
        rendered = render_layout(rows)
        res = process_code(rendered)
        assert res.is_code
        assert "    return 1" in res.output


class TestFragmentMerging:
    def test_same_row_boxes_concatenate(self):
        frags = [
            GeomLine(text="def f", left=100, top=0, right=160, bottom=20),
            GeomLine(text="():", left=163, top=2, right=190, bottom=20),
        ]
        merged = merge_fragment_lines(frags)
        assert len(merged) == 1
        assert merged[0].text.replace(" ", "") == "deff():"

    def test_wide_gap_becomes_single_space(self):
        frags = [
            GeomLine(text="total", left=0, top=0, right=50, bottom=20),
            GeomLine(text="=", left=90, top=0, right=105, bottom=20),
            GeomLine(text="42", left=115, top=0, right=135, bottom=20),
        ]
        merged = merge_fragment_lines(frags)
        assert merged[0].text == "total = 42"

    def test_different_rows_stay_separate(self):
        frags = [
            _g("alpha", left=100, top=0, bottom=20),
            _g("beta", left=300, top=40, bottom=60),
        ]
        assert len(merge_fragment_lines(frags)) == 2

    def test_rows_sorted_top_down(self):
        frags = [
            _g("second", left=100, top=40, bottom=60),
            _g("first", left=100, top=0, bottom=20),
        ]
        merged = merge_fragment_lines(frags)
        assert [m.text for m in merged] == ["first", "second"]


class TestGutterBoxRemoval:
    def test_gutter_column_dropped_before_merge(self):
        entries = []
        for i, (num, code) in enumerate(
            [(3, "using System;"), (4, "namespace X"), (5, "{"), (6, "class A")]
        ):
            top = i * 30
            # gutter number box, far left
            entries.append(GeomLine(str(num), 10, top, 28, top + 22))
            # the actual code box of the same row
            entries.append(GeomLine(code, 70, top, 70 + len(code) * 9, top + 22))

        remaining, dropped = drop_gutter_boxes(entries)
        assert dropped == 4
        assert all(e.text != "3" for e in remaining)
        merged = merge_fragment_lines(remaining)
        assert merged[1].left == 70  # true code margin, not gutter x
        assert "    " not in merged[0].text

    def test_meaningful_numbers_not_in_left_column_kept(self):
        entries = [
            _g("value", left=60),
            GeomLine("42", left=200, top=30, right=230, bottom=50),
            _g("other", left=60, top=60, bottom=80),
        ]
        remaining, dropped = drop_gutter_boxes(entries)
        assert dropped == 0
        assert any(e.text == "42" for e in remaining)

    def test_non_increasing_numbers_rejected(self):
        entries = [
            GeomLine("9", 10, 0, 25, 20),
            GeomLine("2", 10, 30, 25, 50),
            _g("code", left=70, top=0, bottom=20),
            _g("more", left=70, top=30, bottom=50),
        ]
        _, dropped = drop_gutter_boxes(entries)
        assert dropped == 0

    def test_fewer_than_two_numbers_noop(self):
        entries = [_g("7", left=10), _g("code", left=70)]
        _, dropped = drop_gutter_boxes(entries)
        assert dropped == 0


class TestPartialGutterFallback:
    """Numbers already glued into line text by OCR (no separate box)."""

    def test_partial_gutter_stripped(self):
        shot = (
            "using System;\n"
            "namespace Bitorrent\n"
            "{\n"
            "8 {\n"
            "public long Max { get; set; }\n"
            "10 public TimeSpan Window { get; set; }\n"
        )
        cleaned, info = strip_editor_artifacts(shot)
        assert info["gutter_stripped"]
        assert not any(ln.startswith(("8", "10")) for ln in cleaned.splitlines())
        assert "public long Max" in cleaned

    def test_numbered_prose_list_survives(self):
        shot = "1 apples are red\n2 sky is blue\n"
        cleaned, info = strip_editor_artifacts(shot)
        assert not info["gutter_stripped"]
        assert cleaned.startswith("1 apples")


class TestPipelineGeometryIntegration:
    """Code mode must consume engine geometry, not flat text."""

    def test_pipeline_uses_rendered_layout(self):
        from maiocr.core.pipeline import MaiOCRPipeline, Mode
        from maiocr.ocr.engine import OcrResult

        pipeline = MaiOCRPipeline.__new__(MaiOCRPipeline)
        pipeline._engine = type(
            "FakeEngine",
            (),
            {
                "recognize": lambda self, img: OcrResult(
                    lines=["def f():", "return 1"],
                    scores=[0.9, 0.9],
                    geom=[
                        _g("def f():", left=200, top=0, bottom=22),
                        _g("return 1", left=244, top=32, bottom=54),
                    ],
                ),
                "warmup": lambda self: None,
                "mode_label": "CPU",
            },
        )()
        result = pipeline.run(mode=Mode.CODE, image="unused")
        assert result.is_code
        assert "    return 1" in result.output


# --------------------------------------------------- triple-quote repair


class TestTripleQuoteRepair:
    USER_SAMPLE = (
        "def _build_warmup_images() -> list[Image.Image]:\n"
        "\n"
        "u u n\n"
        "\n"
        "Synthetic text screenshots covering det/cls/rec kernel shapes.\n"
    )

    def test_user_reported_case_repaired(self):
        cleaned, n = repair_triple_quotes(self.USER_SAMPLE)
        assert n >= 1
        assert '"""' in cleaned
        assert "u u n" not in cleaned

    def test_full_docstring_open_and_close(self):
        raw = (
            "def f():\n"
            "    u u n\n"
            "    Body line of the docstring.\n"
            "    u u n\n"
            "    return 1\n"
        )
        cleaned, n = repair_triple_quotes(raw)
        lines = cleaned.splitlines()
        assert lines[1].strip() == '"""'
        assert lines[3].strip() == '"""'
        assert n == 2
        assert lines[2] == "    Body line of the docstring."

    def test_body_lifted_to_quote_indentation(self):
        # OCR lost the body's leading spaces entirely.
        raw = "def f():\n    u u n\nBody text here.\nu u n\n"
        cleaned, _ = repair_triple_quotes(raw)
        assert "\n    Body text here." in cleaned

    def test_genuine_triple_quotes_untouched(self):
        raw = 'def f():\n    """Doc."""\n    return 0\n'
        cleaned, n = repair_triple_quotes(raw)
        assert n == 0
        assert cleaned == raw

    def test_single_glyph_mangle_repaired(self):
        # Another common rendering of """: a bare quote + capital I.
        # The missing closer is synthesized at EOF.
        raw = 'def f():\n"I\n\nBody text.\n'
        cleaned, n = repair_triple_quotes(raw)
        assert n == 2
        assert cleaned.count('"""') == 2
        assert "\nBody text." in cleaned

    def test_capital_i_run_variant_repaired(self):
        # PP-OCR frequently reads """ as a run of capital I's.
        raw = "def f():\nIII\n    Body text.\nIII\n"
        cleaned, n = repair_triple_quotes(raw)
        assert n == 2
        assert cleaned.count('"""') == 2
        assert "III" not in cleaned

    def test_real_word_null_not_touched(self):
        raw = "match v:\n    case:\nnull\n"
        _, n = repair_triple_quotes(raw)
        assert n == 0

    def test_process_code_repairs_and_fences(self):
        res = process_code(self.USER_SAMPLE)
        assert res.is_code
        assert res.language == "python"
        assert '"""' in res.output
        assert "u u n" not in res.output
