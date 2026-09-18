"""Conservative display-only normalization of model-produced mathematics.

Original responses stay unchanged in SQLite/JSON. Do not rewrite code, URLs,
citations or formula meanings, and do not infer definitions of unknown symbols.
"""

from __future__ import annotations

import re
import textwrap


_PROTECTED = re.compile(
    r"(?P<fence>^(?:[ \t]*`{3,}[^\n]*\n[\s\S]*?^[ \t]*`{3,}[ \t]*$|"
    r"[ \t]*~{3,}[^\n]*\n[\s\S]*?^[ \t]*~{3,}[ \t]*$))"
    r"|(?P<code>`+[^`\n]+`+)"
    r"|(?P<link>!?\[[^\]\n]*\]\([^\n]*?\))"
    r"|(?P<url>https?://[^\s<>]+)",
    re.MULTILINE,
)
_MATH = re.compile(
    r"\\\[(?P<bracket>[\s\S]*?)\\\]"
    r"|\\\((?P<paren>[^\n]*?)\\\)"
    r"|(?<![\\$])\$\$(?P<display>[\s\S]*?)\$\$(?!\$)"
    r"|(?<![\\$])\$(?P<inline>[^$\n]+)\$(?!\$)"
    r"|(?<![\\\w])\[(?P<legacy>[^\[\]\n]+)\](?!\s*\()"
)
_TEX_MARKER = re.compile(
    r"\\(?:times|cdot|frac|text|mathrm|sqrt|int|sum|alpha|beta)\b|(?:_|\^|\\_)[{]"
)
_SIMPLE_COMMANDS = {
    "times": "×",
    "cdot": "·",
    "div": "÷",
    "pm": "±",
    "leq": "≤",
    "geq": "≥",
    "neq": "≠",
    "approx": "≈",
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "Delta": "Δ",
    "epsilon": "ε",
    "mu": "μ",
    "sigma": "σ",
}

# Lex math as a whole before looking for inline code inside it. Keep identifiers,
# citations, actual code, links, and Windows path separators out of prose repairs.
_LEXER = re.compile(
    _PROTECTED.pattern.split(r"|(?P<code>")[0]
    + r"|(?P<escaped_code>(?<!\\)\\`[^`\n]*?\\`)"
    + r"|(?P<code>(?<![\\`])`+[^`\n]+`+)"
    + r"|(?P<link>!?\[[^\]\n]*\]\([^\n]*?\))"
    + r"|(?P<url>https?://[^\s<>]+)"
    + r"|(?P<path>(?:[A-Za-z]:\\|\\\\[A-Za-z0-9_.-]+\\)[^\s`<>\"|]+)"
    + r"|(?P<double_bracket>\\\\\[[\s\S]*?\\\\\])"
    + _MATH.pattern.replace(r"\\\[(?P<bracket>", r"|\\\[(?P<bracket>", 1),
    re.MULTILINE,
)
_LITERAL_COMMANDS = {"text", "textrm", "textsf", "texttt", "mathrm", "operatorname", "mathtt"}
_SUPPORTED_COMMANDS = set(
    "frac dfrac tfrac cfrac sqrt text textrm textsf texttt textnormal textbf textit "
    "mathrm mathit mathbf mathsf mathtt mathbb mathcal mathscr mathrm operatorname "
    "times cdot div pm mp le leq ge geq ne neq approx sim simeq equiv propto "
    "alpha beta gamma delta epsilon varepsilon zeta eta theta vartheta iota kappa "
    "lambda mu nu xi omicron pi varpi rho varrho sigma varsigma tau upsilon phi "
    "varphi chi psi omega Gamma Delta Theta Lambda Xi Pi Sigma Upsilon Phi Psi Omega "
    "int iint iiint oint sum prod lim min max log ln exp sin cos tan sinh cosh "
    "arcsin arccos arctan inf sup det dim deg gcd lcm partial nabla infty "
    "left right big Big bigg Bigg bigl bigr Bigl Bigr biggl biggr Biggl Biggr "
    "begin end aligned array matrix pmatrix bmatrix vmatrix Vmatrix cases "
    "overline underline hat widehat bar vec dot ddot tilde widetilde "
    "overbrace underbrace overset underset stackrel binom dbinom tbinom "
    "underbracket overbracket boldsymbol bm phantom vphantom hphantom "
    "quad qquad hspace vspace displaystyle textstyle scriptstyle scriptscriptstyle "
    "ldots cdots vdots ddots dots lvert rvert vert lVert rVert Vert "
    "langle rangle lceil rceil lfloor rfloor "
    "to rightarrow leftarrow leftrightarrow Rightarrow Leftarrow Leftrightarrow "
    "longrightarrow longleftarrow mapsto in notin subset subseteq supset supseteq "
    "cup cap emptyset forall exists neg land lor not perp parallel degree "
    "mathop limits nolimits tag color textcolor cancel cancelto boxed fbox "
    "lbrack rbrack lbrace rbrace small textless textgreater".split()
)


def _tokens(text: str):
    cursor = 0
    for match in _LEXER.finditer(text):
        if match.start() > cursor:
            yield "prose", text[cursor : match.start()]
        kind = match.lastgroup
        if kind == "legacy" and ("=" not in match.group() or not _TEX_MARKER.search(match.group())):
            yield "prose", match.group()
        else:
            yield kind, match.group()
        cursor = match.end()
    if cursor < len(text):
        yield "prose", text[cursor:]


def _formula(kind: str, value: str) -> str:
    if kind == "double_bracket":
        return value[3:-3]
    if kind in {"bracket", "paren", "display"}:
        return value[2:-2]
    return value[1:-1]


def _repair_formula(formula: str) -> str:
    formula = textwrap.dedent(formula).strip()
    known = "|".join(sorted(_SUPPORTED_COMMANDS, key=len, reverse=True))
    formula = re.sub(r"(?<!\\)\\\\(?=(?:" + known + r")\b)", r"\\", formula)
    result, contexts = [], [False]
    pending_literal = False
    pending_label = False
    index = 0
    while index < len(formula):
        char = formula[index]
        if char == "\\" and index + 1 < len(formula):
            next_char = formula[index + 1]
            if next_char in "_^":
                operator = not contexts[-1] or (
                    next_char == "_" and formula[index + 2 : index + 3] == "{"
                )
                result.append(next_char if operator else "\\" + next_char)
                pending_label = operator
                index += 2
                continue
            command = re.match(r"\\([A-Za-z]+)", formula[index:])
            if command:
                result.append(command.group())
                pending_literal = command.group(1) in _LITERAL_COMMANDS
                index += len(command.group())
                continue
            result.append(formula[index : index + 2])
            index += 2
            continue
        if char == "{":
            contexts.append(contexts[-1] or pending_literal or pending_label)
            pending_literal = pending_label = False
        elif char == "}":
            if len(contexts) > 1:
                contexts.pop()
        elif char in "_^":
            pending_label = True
        elif char == "%":
            result.append(r"\%")
            index += 1
            continue
        elif not char.isspace():
            pending_literal = pending_label = False
        result.append(char)
        index += 1
    return "".join(result)


def _math_issue(formula: str) -> str | None:
    depth = 0
    for match in re.finditer(r"\\.|[{}]", formula):
        if match.group() == "{":
            depth += 1
        elif match.group() == "}":
            depth -= 1
            if depth < 0:
                return "公式括号不完整"
    if depth:
        return "公式括号不完整"
    if re.search(r"(?m)^\s*(?:#{1,6}\s|[-*+]\s+\*\*)", formula):
        return "公式中混入了正文"
    unknown = set(re.findall(r"\\([A-Za-z]+)", formula)) - _SUPPORTED_COMMANDS
    if unknown:
        return "公式含暂不支持的排版指令"
    return None


def _repair_prose(text: str) -> str:
    text = text.replace("\r\n", "\n")
    # Recover paired emphasis only, not multiplication/wildcards or arbitrary \*.
    text = re.sub(r"\\\*\\\*([^\n]*?)\\\*\\\*", r"**\1**", text)
    text = re.sub(r"\\_\\_([^\n]*?)\\_\\_", r"__\1__", text)
    text = re.sub(r"(?<=\w)\\_(?=\w)", "_", text)
    text = re.sub(
        r"(?m)^(\s*)(?:\\#){1,6}(?=\s)", lambda match: match.group().replace("\\", ""), text
    )
    text = re.sub(r"(?m)^(\s*)\\([-+*])(?=\s+)", r"\1\2", text)
    return text


def _repair_layout(text: str) -> str:
    text = re.sub(r"(?<=[^\n])[ \t]+(?=#{1,6}[ \t]+)", "\n\n", text)
    text = re.sub(r"(?<=[^\n])[ \t]+(?=[-+*][ \t]+\*\*[^*\n]{1,50}\*\*[：:])", "\n\n", text)
    text = re.sub(r"(?<=[。；：:])[ \t]+(?=\d{1,2}\.[ \t]+[A-Za-z\u3400-\u9fff])", "\n", text)
    text = re.sub(r"(?<=[。；])[ \t]+(?=-[ \t]+[A-Za-z\u3400-\u9fff])", "\n", text)
    lines, output = text.splitlines(keepends=True), []
    previous_list = False
    for index, line in enumerate(lines):
        heading = bool(re.match(r"^#{1,6}\s+", line))
        item = bool(re.match(r"^\s*(?:[-+*]|\d{1,2}\.)\s+", line))
        if output and output[-1].strip() and (heading or (item and not previous_list)):
            output.append("\n")
        output.append(line)
        if heading and index + 1 < len(lines) and lines[index + 1].strip():
            output.append("\n")
        previous_list = item
    return "".join(output)


def _protected_segments(text: str):
    cursor = 0
    for match in _PROTECTED.finditer(text):
        yield False, text[cursor : match.start()]
        yield True, match.group()
        cursor = match.end()
    yield False, text[cursor:]


def normalize_math_markdown(text: str) -> str:
    """Normalize model Markdown without altering stored text or scientific values."""
    prefix = "\ue000XPS_FORMAT_"
    while prefix in text:
        prefix += "X"
    protected: dict[str, tuple[str, str]] = {}

    def protect(kind, value):
        key = f"{prefix}{len(protected)}\ue001"
        protected[key] = (kind, value)
        return key

    pieces = []
    for kind, value in _tokens(text):
        if kind == "prose":
            pieces.append(_repair_prose(value))
        elif kind == "escaped_code":
            content = value[2:-2]
            if not re.match(r"[A-Za-z]:\\|\\\\", content):
                content = re.sub(r"\\([_*`])", r"\1", content)
            pieces.append(protect("code", "`" + content + "`"))
        else:
            pieces.append(protect(kind, value))
    layout = "".join(pieces)
    # Bare TeX equations are repaired only when an entire line is an equation.
    lines = []
    for line in layout.splitlines(keepends=True):
        body = line.strip()
        inspect = re.sub(r"\\(?:text|mathrm)\{[^{}]*\}", "", body)
        if (
            "=" in body
            and _TEX_MARKER.search(body)
            and not re.search(r"[\u3400-\u9fff]", inspect)
            and prefix not in body
        ):
            indentation = line[: len(line) - len(line.lstrip(" \t"))]
            lines.append(
                indentation
                + protect("display", "$$" + body.strip("$") + "$$")
                + ("\n" if line.endswith("\n") else "")
            )
        else:
            lines.append(line)
    layout = _repair_layout("".join(lines))
    marker = re.compile(re.escape(prefix) + r"\d+\ue001")

    def restore(match):
        kind, value = protected[match.group()]
        line_start = layout.rfind("\n", 0, match.start()) + 1
        before = layout[line_start : match.start()]
        line_end = layout.find("\n", match.end())
        after = layout[match.end() : line_end if line_end >= 0 else len(layout)]
        in_table = before.lstrip().startswith("|")
        if kind == "code":
            return re.sub(r"(?<!\\)\|", r"\|", value) if in_table else value
        if kind not in {"bracket", "double_bracket", "paren", "display", "inline", "legacy"}:
            return value
        formula = _repair_formula(_formula(kind, value))
        if in_table:
            formula = re.sub(r"(?<!\\)\|", r"\vert ", formula)
        issue = _math_issue(formula)
        if issue:
            if "正文" in issue:
                return f"（{issue}，原始公式已保留）\n\n" + formula
            plain = _plain_formula(formula)
            return ("`" + plain.replace("`", "") + "`") if plain else f"（{issue}，原始公式已保留）"
        if kind in {"inline", "paren"} or in_table:
            return f"${formula}$"
        indent = before if not before.strip() else ""
        content = "$$\n" + formula + "\n$$"
        if indent:
            content = content.replace("\n", "\n" + indent)
        return ("\n\n" if before.strip() else "") + content + ("\n\n" if after.strip() else "")

    return marker.sub(restore, layout)


def _group_at(text: str, start: int) -> tuple[str, int] | None:
    while start < len(text) and text[start].isspace():
        start += 1
    if start >= len(text) or text[start] != "{":
        return None
    depth, index = 1, start + 1
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if not depth:
                return text[start + 1 : index], index + 1
        index += 1
    return None


def _plain_formula(formula: str) -> str | None:
    text = _repair_formula(formula).replace(r"\_", "_").replace(r"\%", "%")
    # Fractions retain explicit grouping. Never flatten an integral or unknown command.
    while match := re.search(r"\\(?:frac|dfrac|tfrac)\b", text):
        numerator = _group_at(text, match.end())
        denominator = _group_at(text, numerator[1]) if numerator else None
        if numerator is None or denominator is None:
            return None
        top, bottom = _plain_formula(numerator[0]), _plain_formula(denominator[0])
        if top is None or bottom is None:
            return None
        text = text[: match.start()] + f"({top}) / ({bottom})" + text[denominator[1] :]
    text = re.sub(r"\\(?:text|mathrm|operatorname|mathbf)\{([^{}]*)\}", r"\1", text)
    for command, symbol in _SIMPLE_COMMANDS.items():
        text = re.sub(r"\\" + command + r"\b", lambda match, value=symbol: value, text)
    # Unknown commands (e.g. nested integrals/fractions) remain in typeset form.
    # Do not invent a flattened expression that could change their meaning.
    if "\\" in text:
        return None
    text = re.sub(r"([_^])\{([^{}]+)\}", r"\1(\2)", text)
    if "{" in text or "}" in text:
        return None
    return " ".join(text.split())


def formula_reading(text: str) -> tuple[list[str], list[str]]:
    formulas, definitions = [], []
    for kind, value in _tokens(normalize_math_markdown(text)):
        if kind in {"bracket", "double_bracket", "paren", "display", "inline", "legacy"}:
            plain = _plain_formula(_formula(kind, value))
            if plain is None:
                continue
            if "=" in plain and plain not in formulas:
                formulas.append(plain)
            if re.search(r"\bf_\(highE[_,]N\)", plain):
                definition = "f(highE_N)：N 1s 谱中，结合能高于谱质心部分的积分面积占比；无量纲，不直接等于正电荷密度。"
                if definition not in definitions:
                    definitions.append(definition)
            if re.search(r"\bW_\(80[_,]O\)", plain):
                definition = "W(80_O)：O 1s 谱的能量累积积分分布中，90% 与 10% 分位能量之差；单位 eV，不是拟合峰的半高宽。"
                if definition not in definitions:
                    definitions.append(definition)
            for symbol, label, column in (
                ("N", "氮", "N__spectral_entropy"),
                ("O", "氧", "O__spectral_entropy"),
            ):
                if re.search(r"\bS_(?:" + symbol + r"|\(" + symbol + r"\))(?=$|\W)", plain):
                    definition = f"S_{symbol}：{label}谱的谱形信息熵，对应列 {column}；描述强度分布，不直接确定化学物种数量。"
                    if definition not in definitions:
                        definitions.append(definition)
    return formulas, definitions


def render_model_output(text: str) -> None:
    import streamlit as st

    normalized = normalize_math_markdown(text)
    buffer = []
    continuation_indent = 0

    def flush():
        nonlocal continuation_indent
        prose = "".join(buffer).strip("\n")
        if continuation_indent:
            lines = prose.splitlines()
            for index, line in enumerate(lines):
                if not line.strip():
                    continue
                if line.startswith(" " * continuation_indent):
                    lines[index] = line[continuation_indent:]
                else:
                    break
            prose = "\n".join(lines)
        if prose.strip():
            st.markdown(prose)
        buffer.clear()
        continuation_indent = 0

    for kind, value in _tokens(normalized):
        if kind == "display":
            prefix = "".join(buffer).rsplit("\n", 1)[-1]
            indentation = len(prefix) if not prefix.strip() else 0
            flush()
            st.latex(_repair_formula(_formula(kind, value)))
            continuation_indent = indentation
        else:
            buffer.append(value)
    flush()
    formulas, definitions = formula_reading(text)
    if formulas:
        with st.expander("公式白话版与符号说明", expanded=True):
            for formula in formulas:
                st.text(formula)
            for definition in definitions:
                st.write(definition)
            if any(
                re.fullmatch(r"P\s*=\s*f_\(highE[_,]N\)\s*×\s*W_\(80[_,]O\)", formula)
                for formula in formulas
            ):
                st.write(
                    "这条式子的白话版：候选指标 P = N 1s 质心以上面积占比 × O 1s 中心 80% 能量宽度。"
                )
            st.caption(
                "白话符号中的 _(标签) 标明下标，不是函数调用。仅转换显示格式；未知符号以原回答的定义为准，公式本身不证明新机理。"
            )
