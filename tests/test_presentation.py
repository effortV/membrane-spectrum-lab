import json
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from xps_agent.config import Settings
from xps_agent.conversations import ConversationStore
from xps_agent.db import StateDB
from xps_agent.presentation import formula_reading, normalize_math_markdown


EXAMPLE = r"P = f_{\text{highE\_N}} \times W_{80\_O}"
REPORT = r"""### 4.1 谱熵比

- **公式**：
  $$R = \frac{S_N}{S_O + \epsilon}$$
  其中 $S_N$ 为 `N__spectral_entropy`，$\epsilon=10^{-12}$。
- **白话版**：比较两种谱形的熵。

### 4.2 异质性指数

- **公式**：
  $$P = f_{\text{highE,N}} \times W_{80,O}$$
  其中 $W_{80,O}$ 的单位为 eV。
- **状态**：`calculable`，74 个非空值。

### 4.3 含量乘熵

- **公式**：
  $$C = \text{N(%)} \times S_N$$
  其中 N(%) 为 `XPS参数 | N(%)`。

## 5. 证据定位

| 项目 | 证据 |
|---|---|
| 列 | `XPS参数 | N(%)` |

**暂定等级**：尚未独立验证。
"""


def test_actual_report_indented_formulas_percent_table_and_idempotence():
    result = normalize_math_markdown(REPORT)
    assert r"\text{N(\%)}" in result
    assert r"  $$" in result
    assert "  R = " in result and "  $$\n  其中" in result
    assert r"`XPS参数 \| N(%)`" in result
    assert "## 5. 证据定位\n\n" in result
    assert normalize_math_markdown(result) == result
    formulas, definitions = formula_reading(REPORT)
    assert formulas == [
        "R = (S_N) / (S_O + ε)",
        "ε=10^(-12)",
        "P = f_(highE,N) × W_(80,O)",
        "C = N(%) × S_N",
    ]
    assert len(definitions) == 4


def test_repeated_escaping_flattened_lists_and_bare_equation():
    source = r"""### 4.1 谱熵比（N\_O\_entropy\_ratio）
- \*\*公式\*\*：
R = \frac{S\_N}{S\_O + \epsilon}
其中 $S\_N$ 为 \`N\_\_spectral\_entropy\`。 - \*\*机制链\*\*： 1. 读取 XPS。 2. 检验假说。 - \*\*状态\*\*：待验证。 ## 5. 证据
"""
    result = normalize_math_markdown(source)
    assert r"\*\*" not in result and r"\`" not in result
    assert "`N__spectral_entropy`" in result
    assert "N_O_entropy_ratio" in result
    assert r"\frac{S_N}{S_O + \epsilon}" in result
    assert "\n\n- **机制链**" in result
    assert "\n2. 检验假说。" in result
    assert "\n\n## 5. 证据" in result
    assert normalize_math_markdown(result) == result


@pytest.mark.parametrize(
    "source",
    [
        r"$x = \unknown{a}$",
        r"$$x = \frac{a}{b$$",
        "$$x = 2\n## 后面的正文\n$$",
    ],
)
def test_unsafe_math_has_readable_fallback_instead_of_rendering_error(source):
    result = normalize_math_markdown(source)
    assert "原始公式已保留" in result
    assert "$$" not in result and r"\unknown" not in result


def test_report_component_latex_and_following_prose_without_paid_calls():
    app = AppTest.from_string(
        f"from xps_agent.presentation import render_model_output\nrender_model_output({REPORT!r})",
        default_timeout=20,
    ).run()
    assert not app.exception
    assert len(app.latex) == 3
    assert app.latex[2].value == "$$\n" + r"C = \text{N(\%)} \times S_N" + "\n$$"
    assert any("暂定等级" in value.value for value in app.markdown)
    assert any("## 5. 证据定位" in value.value for value in app.markdown)
    assert any("C = N(%) × S_N" == value.value for value in app.text)


def test_saved_relationship_report_reload_is_read_only_and_no_model_calls():
    settings = Settings.load()
    directory = settings.workspace_root / "runs" / "relationship_mock"
    directory.mkdir(parents=True)
    saved = directory / "agent_interpretation.md"
    saved.write_text(REPORT, encoding="utf-8")
    database = StateDB(settings.database_path)
    database.initialize()
    result = {
        "relationship": "XPS → 结构",
        "target": "Ra",
        "experiment_id": "mock",
        "cohort": {"rows_scored": 74, "groups": 12},
        "warnings": [],
        "out_of_fold": [],
        "run_dir": str(directory),
    }
    app = AppTest.from_string(
        "from xps_agent.config import Settings\n"
        "from xps_agent.db import StateDB\n"
        "from xps_agent.workbench_ui import _research_result\n"
        "settings = Settings.load()\n"
        f"_research_result({result!r}, settings, StateDB(settings.database_path), None, 'test')",
        default_timeout=20,
    ).run()
    assert not app.exception and len(app.latex) == 3
    assert any("已保存的 Agent 解读" in value.value for value in app.caption)
    assert saved.read_text(encoding="utf-8") == REPORT
    assert database.rows("SELECT COUNT(*) AS n FROM llm_request_events")[0]["n"] == 0


@pytest.mark.parametrize(
    "source",
    [
        "\\[\n" + EXAMPLE + "\n\\]",
        "[ " + EXAMPLE + " ]",
        r"[ P = f\_{\text{highE\_N}} \times W\_{80\_O} ]",
        "$$\n" + EXAMPLE + "\n$$",
    ],
)
def test_example_equation_renders_math_not_literal_tex(source):
    result = normalize_math_markdown(source)
    assert "$$\n" + EXAMPLE + "\n$$" in result
    assert r"\[" not in result and r"\]" not in result
    assert r"\_{" not in result
    assert r"\text{highE\_N}" in result
    assert normalize_math_markdown(result) == result


def test_inline_and_table_math_are_compatible():
    source = (
        r"比例 \(f_{\text{highE\_N}}\)，能量 \(E_{90}-E_{10}\)。"
        + "\n\n| 符号 | 含义 |\n|---|---|\n"
        + r"| \(W_{80\_O}\) | 能量宽度 |"
    )
    result = normalize_math_markdown(source)
    assert r"$f_{\text{highE\_N}}$" in result
    assert r"$E_{90}-E_{10}$" in result
    assert r"| $W_{80\_O}$ | 能量宽度 |" in result
    assert r"\(" not in result


@pytest.mark.parametrize(
    "source",
    [
        r"[evidence:one] [1] [待核验] [x = 2]",
        r"[ P = \alpha ](https://example.org/formula)",
        r"https://example.org/query?x=[P=\alpha]",
        r"`\[P = a \times b\]` 和 `raw_N__high_energy_fraction`",
        "```python\n" + r"value = '\[P = a \times b\]'" + "\n```",
        "~~~text\n" + r"\(P = a \times b\)" + "\n~~~",
        r"原始路径 C:\data\NF_N.xlsx，范围 [10,90]，费用 \$5 和 \$10",
    ],
)
def test_code_links_citations_paths_and_ordinary_brackets_are_preserved(source):
    assert normalize_math_markdown(source) == source
    assert formula_reading(source) == ([], [])


def test_math_outside_code_is_repaired_without_rewriting_code():
    source = r"`\(raw_formula\)` 与 \(x = a \times b\)"
    result = normalize_math_markdown(source)
    assert r"`\(raw_formula\)`" in result
    assert "$x = a \\times b$" in result


def test_plain_formula_known_glossary_and_no_invented_unknown_definitions():
    formulas, definitions = formula_reading("\\[" + EXAMPLE + "\\]")
    assert formulas == ["P = f_(highE_N) × W_(80_O)"]
    assert len(definitions) == 2
    assert any("不直接等于正电荷密度" in definition for definition in definitions)
    assert any("累积积分" in definition and "eV" in definition for definition in definitions)
    assert formula_reading(r"\[Z = a \times b\]") == (["Z = a × b"], [])


def test_complex_math_stays_typeset_not_misleading_flat_expression():
    source = r"\[f = \frac{\int_{E_c}^{E_m} I(E)\,dE}{\int_{E_0}^{E_m} I(E)\,dE}\]"
    result = normalize_math_markdown(source)
    assert "$$" in result and r"\frac{\int" in result
    assert formula_reading(source) == ([], [])


def test_normalization_keeps_entire_public_answer_and_order():
    source = (
        "中文摘要😀\n\n"
        + r"\[P = a \times b\]"
        + "\n\n引文 `evidence:one`，原始数据列 `O__central_80_width_eV`。"
    )
    result = normalize_math_markdown(source)
    assert result.startswith("中文摘要😀")
    assert result.endswith("引文 `evidence:one`，原始数据列 `O__central_80_width_eV`。")
    assert result.count("P = a") == 1


def test_markdown_export_repaired_json_and_database_raw_are_unchanged(tmp_path):
    database = StateDB(tmp_path / "state.sqlite3")
    database.initialize()
    store = ConversationStore(database)
    key = store.create("公式导出")
    raw = "\\[" + EXAMPLE + "\\]"
    turn = store.reserve(key, "解释指标", model="mock", task_id="mock-task")
    store.finish(turn, raw)
    assert "$$" in store.markdown(key)
    assert store.turns(key)[0]["answer"] == raw
    assert store.export(key)["turns"][0]["answer"] == raw
    assert json.loads(json.dumps(store.export(key)))["turns"][0]["answer"] == raw


def test_existing_answer_dashboard_math_and_chinese_explanation_no_model_calls():
    settings = Settings.load()
    database = StateDB(settings.database_path)
    database.initialize()
    store = ConversationStore(database)
    key = store.create("已有公式显示测试")
    turn = store.reserve(key, "怎么理解这个式子", model="mock-only", task_id="test-task")
    source = "\\[\n" + EXAMPLE + "\n\\]\n\n" + r"表格里的 \(W_{80\_O}\)。"
    store.finish(turn, source)
    st.cache_resource.clear()
    try:
        app = AppTest.from_file(
            str(Path(__file__).parents[1] / "src" / "xps_agent" / "dashboard.py"),
            default_timeout=30,
        )
        app.query_params["chat"] = key
        app.run()
        assert not app.exception
        assert any("$$\n" + EXAMPLE + "\n$$" == item.value for item in app.latex)
        assert any("P = f_(highE_N) × W_(80_O)" == item.value for item in app.text)
        assert any("这条式子的白话版" in item.value for item in app.markdown)
        assert store.turns(key)[0]["answer"] == source
        assert database.rows("SELECT COUNT(*) AS n FROM llm_request_events")[0]["n"] == 0
    finally:
        st.cache_resource.clear()
