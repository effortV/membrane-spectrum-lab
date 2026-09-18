"""Pure UI entrypoint for Community Cloud. No local backend or database startup."""

from __future__ import annotations

import base64
from pathlib import Path
import re
import sys
import uuid

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from backend_client import BackendError, ConnectionSettings, SSHBackendClient, SubmissionUncertain

# The shared formatter uses only stdlib and lazily imports Streamlit. No backend.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from xps_agent.presentation import render_model_output  # noqa: E402

st.set_page_config(page_title="膜谱研究台", page_icon=":material/graphic_eq:", layout="wide")


@st.cache_resource
def client():
    try:
        settings = ConnectionSettings.from_mapping(st.secrets)
    except (FileNotFoundError, KeyError):
        raise BackendError("请先在 Streamlit Cloud Secrets 中配置服务器连接。") from None
    return SSHBackendClient(settings)


def call(method, path, *, body=None, query=None, binary=False):
    return client().request(
        method, path, st.session_state.owner, body=body, query=query, binary=binary
    )


def submit(kind, parameters=None, *, paid=False):
    request_id = uuid.uuid4().hex
    st.session_state.last_request_id = request_id
    result = call(
        "POST",
        "/v1/jobs",
        body={
            "request_id": request_id,
            "kind": kind,
            "parameters": parameters or {},
            "confirmed_paid": paid,
        },
    )
    st.success("任务已交给服务器。可切换页面，在侧栏查看进度。")
    return result


def table(rows):
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.info("暂无记录。")


@st.fragment(run_every=3)
def task_status():
    try:
        state = call("GET", "/v1/tasks")
        task = state.get("current")
        if task:
            st.caption(f"{task['label']} · {task['status']} · {task['elapsed_seconds']:.0f} 秒")
            progress = task.get("progress", {})
            st.caption(progress.get("stage", "等待更新"))
            for finished, total, unit in (
                ("documents_finished", "documents_total", "篇文献"),
                ("fits_finished", "fits_total", "次训练/验证"),
            ):
                if progress.get(total):
                    st.progress(
                        min(1.0, progress.get(finished, 0) / progress[total]),
                        text=f"{progress.get(finished, 0)}/{progress[total]} {unit}",
                    )
            if progress.get("document_title"):
                st.caption(progress["document_title"])
            if progress.get("page"):
                st.caption(f"正在读取 PDF 第 {progress['page']} 页")
            if (
                task.get("is_owner")
                and task["status"] == "running"
                and st.button("停止后续操作", key="cloud_cancel")
            ):
                call("POST", f"/v1/tasks/{task['task_id']}/cancel")
                st.rerun(scope="fragment")
            if task.get("error"):
                st.warning(task["error"])
        else:
            st.caption("服务器当前没有运行任务")
    except BackendError as error:
        st.warning(str(error))


def overview(summary):
    st.title("膜谱研究台")
    st.caption("制备—XPS—结构—性能 · 数据与机理融合研究")
    for column, (key, value) in zip(st.columns(4), list(summary["counts"].items())[:4]):
        column.metric(
            {
                "documents": "文献",
                "evidence": "证据",
                "hypotheses": "候选指标",
                "experiments": "研究结果",
            }.get(key, key),
            value,
        )
    st.subheader("NF / RO 数据状态")
    profile = summary["profile"]
    table(
        [
            {
                key: profile.get(key)
                for key in (
                    "model_table_rows",
                    "canonical_spectra_usable",
                    "paired_n_o_records",
                    "doi_group_rows_resolved",
                )
            }
        ]
    )
    st.caption("数据、文献、数据库、模型调用与计算均在服务器；云端不保存替代数据库。")


def data_page():
    st.title("数据与谱图")
    family = st.selectbox("膜体系", ["全部", "NF", "RO"])
    offset = st.number_input("起始行", 0, value=0, step=100)
    result = call(
        "GET",
        "/v1/data",
        query={"dataset": "" if family == "全部" else family, "offset": offset, "limit": 100},
    )
    st.caption(f"筛选后共 {result['total']} 行；当前分页读取最多 100 行。")
    default = [
        key
        for key in ("record_id", "dataset", "reference_number", "N__centroid_eV", "O__centroid_eV")
        if key in result["columns"]
    ]
    columns = st.multiselect("显示字段", result["columns"], default=default)
    values = pd.DataFrame(result["rows"])
    if not values.empty:
        st.dataframe(values[columns], hide_index=True, width="stretch")
        if "record_id" in values and any(name.endswith("centroid_eV") for name in columns):
            st.caption("当前展示谱形描述符，不将其解释为未经验证的化学态或膜性能。")
        spectral = values[
            values.get("N__spectrum_source_file", pd.Series(index=values.index, dtype=str)).notna()
            | values.get(
                "O__spectrum_source_file", pd.Series(index=values.index, dtype=str)
            ).notna()
        ]
        if not spectral.empty:
            record = st.selectbox("单样品原始 N/O 谱", spectral["record_id"].astype(str))
            spectra = call("GET", "/v1/spectrum", query={"record_id": record})
            figure = make_subplots(rows=1, cols=2, subplot_titles=["N 1s", "O 1s"])
            for position, element in enumerate(("N", "O"), 1):
                if element in spectra:
                    figure.add_trace(
                        go.Scatter(
                            x=spectra[element]["energy_eV"],
                            y=spectra[element]["normalized_intensity"],
                            mode="lines",
                            name=element + " 1s",
                        ),
                        row=1,
                        col=position,
                    )
                    figure.update_xaxes(
                        title_text="结合能 (eV)", autorange="reversed", row=1, col=position
                    )
            st.plotly_chart(figure, width="stretch")


def literature():
    st.title("文献中心")
    search_tab, library_tab, read_tab = st.tabs(["联网检索与下载", "全部文献", "正文与图页读取"])
    with search_tab:
        with st.form("cloud_search"):
            query = st.text_input("检索式", "XPS polyamide membrane interfacial polymerization")
            limit = st.number_input("最多检索", 1, 100, 30)
            if st.form_submit_button("检索并加入队列"):
                submit("search", {"query": query, "limit": int(limit)})
        batches = call("GET", "/v1/library/batches", query={"kind": "search"})
        if batches:
            selected = st.selectbox(
                "已保存的检索批次",
                batches,
                format_func=lambda item: (
                    f"{item['created_at']} · {item['query']} · {item['status']}"
                ),
            )
            receipt = call("GET", f"/v1/library/batches/{selected['batch_id']}")
            st.write(
                f"匹配 {receipt['summary']['matched']} 篇 · 新增 {receipt['summary']['added']} 篇 · 已有 {receipt['summary']['existing']} 篇"
            )
            items = receipt["items"]
            table(
                [
                    {
                        key: row.get(key)
                        for key in (
                            "title",
                            "doi",
                            "year",
                            "is_new",
                            "outcome",
                            "status",
                            "download_status",
                            "reason",
                        )
                    }
                    for row in items
                ]
            )
            pending = [row for row in items if not row.get("current_path")]
            choices = st.multiselect(
                "选择本批次待下载的文献", pending, format_func=lambda row: row["title"]
            )
            if st.button("下载所选可访问 PDF", disabled=not choices):
                submit(
                    "fetch",
                    {"doc_keys": [row["doc_key"] for row in choices], "limit": len(choices)},
                )
        else:
            st.info("检索完成后，这里会显示本次加入了哪些文献。可使用右上方刷新按钮。")
        st.caption("API 无法获取的全文由你补充上传；不会绕过期刊权限。")
    with library_tab:
        rows = call("GET", "/v1/library")
        text = st.text_input("按标题 / DOI 筛选")
        filtered = [
            row
            for row in rows
            if text.lower() in (str(row["title"]) + " " + str(row.get("doi", ""))).lower()
        ]
        table(filtered)
        pdfs = [row for row in filtered if row["has_pdf"]]
        if pdfs:
            selected = st.selectbox("读取已保存 PDF", pdfs, format_func=lambda row: row["title"])
            if st.button("准备下载这篇 PDF"):
                content = call(
                    "GET", "/v1/library/pdf", query={"doc_key": selected["doc_key"]}, binary=True
                )
                st.download_button(
                    "保存 PDF", content, file_name="reference.pdf", mime="application/pdf"
                )
        uploaded = st.file_uploader("补充无法通过 API 获取的 PDF", type=["pdf"])
        if uploaded and st.button("保存到服务器文献库"):
            call(
                "POST",
                "/v1/uploads",
                body={
                    "kind": "pdf",
                    "name": uploaded.name,
                    "content_base64": base64.b64encode(uploaded.getvalue()).decode(),
                },
            )
            st.success("已存入服务器，下一步读取正文。")
    with read_tab:
        st.write("读取 PDF 正文：提取本地文字；不调用视觉模型。")
        st.write("筛选 XPS 图页：根据正文线索生成视觉读取计划；不调用视觉模型。")
        a, b, c = st.columns(3)
        if a.button("索引本地文献"):
            submit("index")
        if b.button("读取待处理 PDF 正文"):
            submit("parse", {"limit": 500})
        if c.button("更新 XPS 图页计划"):
            submit("plan")
        count = st.number_input("本次最多读取文献数", 1, 20, 5)
        paid = st.checkbox("确认调用视觉模型读取图页（可能计费）")
        if st.button("开始视觉证据读取", disabled=not paid):
            submit("extraction", {"limit": int(count), "mode": "pending"}, paid=True)


@st.fragment(run_every=4)
def saved_turns(key):
    result = call("GET", f"/v1/conversations/{key}")
    for turn in result["turns"]:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            if turn["answer"]:
                render_model_output(turn["answer"])
            else:
                st.caption(f"{turn['status']} · {turn['error'] or '服务器正在处理'}")


def agent_page():
    st.title("证据与智能体")
    chat_tab, evidence_tab, hypotheses_tab = st.tabs(["研究对话", "文献证据", "新描述符候选"])
    with chat_tab:
        if st.button("新建研究会话"):
            key = call("POST", "/v1/conversations", body={"title": "新研究会话", "memory": ""})[
                "conversation_id"
            ]
            st.query_params["chat"] = key
            st.rerun()
        conversations = call("GET", "/v1/conversations")
        if conversations:
            keys = [row["conversation_id"] for row in conversations]
            active = st.query_params.get("chat", "")
            key = st.selectbox(
                "已保存会话",
                keys,
                index=keys.index(active) if active in keys else 0,
                format_func=lambda value: next(
                    row["title"] for row in conversations if row["conversation_id"] == value
                ),
            )
            st.query_params["chat"] = key
            record = call("GET", f"/v1/conversations/{key}")["conversation"]
            with st.expander("会话名称和长期研究备注"):
                with st.form("cloud_memory"):
                    title = st.text_input("名称", record["title"])
                    memory = st.text_area("长期备注", record["memory"])
                    if st.form_submit_button("保存备注"):
                        call(
                            "PATCH",
                            f"/v1/conversations/{key}",
                            body={"title": title, "memory": memory},
                        )
                st.download_button(
                    "下载已保存对话",
                    call("GET", f"/v1/conversations/{key}/markdown", binary=True),
                    file_name="research_chat.md",
                    mime="text/markdown",
                )
            saved_turns(key)
            with st.form("cloud_question", clear_on_submit=False):
                question = st.text_area("研究问题", max_chars=6000)
                paid = st.checkbox("确认调用主模型回答（可能计费）")
                if st.form_submit_button("提交问题") and paid and question.strip():
                    request_id = uuid.uuid4().hex
                    st.session_state.last_request_id = request_id
                    call(
                        "POST",
                        f"/v1/conversations/{key}/ask",
                        body={
                            "request_id": request_id,
                            "question": question,
                            "confirmed_paid": True,
                        },
                    )
                    st.success("已提交，回答保存在服务器，刷新不会丢失。")
        else:
            st.info("先新建一个会话。")
    with evidence_tab:
        text = st.text_input("检索证据")
        table(call("GET", "/v1/evidence", query={"query": text, "limit": 100}))
    with hypotheses_tab:
        count = st.number_input("最多提出候选数", 1, 20, 6)
        paid = st.checkbox("确认调用主模型提出新指标（可能计费）")
        if st.button("提出新描述符候选", disabled=not paid):
            submit("proposal", {"limit": int(count)}, paid=True)
        if st.button("安全计算已有候选"):
            submit("materialize")
        for item in call("GET", "/v1/hypotheses"):
            with st.expander(f"{item['status']} · {item['name']}"):
                spec = item["spec"]
                render_model_output(str(spec.get("equation", "")))
                for field_name in ("mechanism_chain", "falsification_tests"):
                    for text in spec.get(field_name, []):
                        render_model_output(str(text))
                st.json(spec)


def images_page():
    st.title("XPS 图分析")
    st.caption("只解释可见信息及有条件的结构/性能假说；不会把图片读数自动加入 ML 训练。")
    uploads = st.file_uploader(
        "同一样品的 1–4 张谱图",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
    )
    family = st.selectbox("膜类型", ["NF", "RO", "未知"])
    sample = st.text_input("样品说明")
    preparation = st.text_area("制备条件及已知实测值")
    paid = st.checkbox("确认调用视觉模型及主模型分析（可能计费）")
    if st.button("保存图片并交给服务器分析", disabled=not paid or not 1 <= len(uploads) <= 4):
        images = [
            call(
                "POST",
                "/v1/uploads",
                body={
                    "kind": "xps_image",
                    "name": image.name,
                    "content_base64": base64.b64encode(image.getvalue()).decode(),
                },
            )["artifact_id"]
            for image in uploads
        ]
        submit(
            "xps_image",
            {
                "artifact_ids": images,
                "context": {"membrane_type": family, "sample": sample, "preparation": preparation},
            },
            paid=True,
        )
    current = call("GET", "/v1/tasks").get("current")
    if current and current["kind"] == "xps_image" and current.get("result"):
        st.json(current["result"])
    history = call("GET", "/v1/analyses")
    if history:
        selected = st.selectbox(
            "已保存的图谱分析",
            history,
            format_func=lambda row: row["created_at"] + " · " + row["artifact_id"][:8],
        )
        report = call("GET", f"/v1/analyses/{selected['artifact_id']}")
        interpretation = report["interpretation"]
        render_model_output(interpretation["summary"])
        for name in (
            "findings",
            "conditional_structure",
            "conditional_performance",
            "alternatives",
            "missing_measurements",
            "suggested_experiments",
        ):
            for text in interpretation.get(name, []):
                render_model_output(str(text))
        with st.expander("完整报告与证据"):
            st.json(report)


def research_page():
    st.title("制备—XPS—结构—性能 · 关系研究")
    catalog = call("GET", "/v1/research/catalog")
    relation = st.selectbox(
        "研究关系",
        list(catalog["relationships"]),
        format_func=lambda key: catalog["relationships"][key]["label"],
    )
    spec = catalog["relationships"][relation]
    columns = catalog["columns"]
    targets = [
        row["column"]
        for row in columns
        if row["scientific_role"] == spec["target"] and row["numeric_count"] >= 3
    ]
    if targets:
        target = st.selectbox("独立实测目标（保持原表单位）", targets)
        groups = {
            role: st.multiselect(
                catalog["roles"][role] + "输入",
                [
                    row["column"]
                    for row in columns
                    if row["scientific_role"] == role and row["column"] != target
                ],
                key="cloud_group_" + role,
            )
            for role in spec["inputs"]
        }
        controls = st.multiselect(
            "共同控制的性能测试条件",
            [row["column"] for row in columns if row["scientific_role"] == "test_condition"],
        )
        families = st.multiselect(
            "膜体系",
            catalog["datasets"],
            default=["NF"] if "NF" in catalog["datasets"] else catalog["datasets"],
        )
        models = st.multiselect(
            "模型", ["ridge", "random_forest", "extra_trees"], default=["ridge"]
        )
        measured = st.checkbox("仅使用中间变量实测样本", value=True)
        if st.button(
            "服务器开始分组验证（不调用 LLM）",
            disabled=not families or not models or any(not values for values in groups.values()),
        ):
            submit(
                "relationship",
                {
                    "task": relation,
                    "target": target,
                    "groups": groups,
                    "controls": controls,
                    "datasets": families,
                    "models": models,
                    "measured_intermediates": measured,
                },
            )
    else:
        st.info("该关系没有可训练的独立目标；请核对表格字段分类或补充实测数据。")
    history = call("GET", "/v1/experiments")
    if history:
        selected = st.selectbox(
            "已保存研究结果",
            history,
            format_func=lambda row: row["created_at"] + " · " + row["experiment_id"][:8],
        )
        result = call("GET", f"/v1/experiments/{selected['experiment_id']}")
        table(result.get("out_of_fold", []))
        for warning in result.get("warnings", []):
            st.caption(warning)
        with st.expander("完整报告"):
            st.json(result)


def main():
    remembered = st.query_params.get("session", "")
    if "owner" not in st.session_state:
        st.session_state.owner = (
            remembered if re.fullmatch(r"[a-f0-9]{32}", remembered) else uuid.uuid4().hex
        )
    st.query_params["session"] = st.session_state.owner
    try:
        summary = call("GET", "/v1/overview")
    except BackendError as error:
        st.title("膜谱研究台")
        st.error(str(error))
        st.caption("连接未就绪，未启动模型、后台计算或云端数据库。")
        st.button("重新检查服务器连接")
        st.stop()
    st.sidebar.subheader("膜谱研究台", anchor=False)
    st.sidebar.caption("NF / RO 表面研究 · 云端界面")

    def leave_monitor():
        st.session_state.show_task_monitor = False

    page = st.sidebar.radio(
        "导航",
        [
            "总览",
            "数据与谱图",
            "文献中心",
            "证据与智能体",
            "XPS 图分析",
            "关系研究（ML）",
            "系统设置",
        ],
        on_change=leave_monitor,
    )
    st.sidebar.divider()
    settings = summary["settings"]
    st.sidebar.caption("主模型：" + settings["main_model"])
    st.sidebar.caption("视觉模型：" + settings["vision_model"])
    st.sidebar.caption(
        "SiliconFlow：" + ("已配置" if settings["siliconflow_key"] == "configured" else "未配置")
    )
    if st.sidebar.button("后台任务", width="stretch"):
        st.session_state.show_task_monitor = True
    with st.sidebar:
        task_status()
    st.button("刷新服务器结果", icon=":material/refresh:")
    try:
        if st.session_state.get("show_task_monitor"):
            st.title("现在正在做什么")
            state = call("GET", "/v1/tasks")
            if state["current"]:
                st.json(state["current"])
            table(state["history"])
        elif page == "总览":
            overview(summary)
        elif page == "数据与谱图":
            data_page()
        elif page == "文献中心":
            literature()
        elif page == "证据与智能体":
            agent_page()
        elif page == "XPS 图分析":
            images_page()
        elif page == "关系研究（ML）":
            research_page()
        else:
            st.title("服务器状态")
            st.json(settings)
            st.caption("模型及文献供应商密钥在服务器本机页面配置，不传给云端。")
            if st.button("重新审计 NF / RO 数据（本地计算）"):
                submit("data_audit")
        if st.session_state.get("last_request_id"):
            with st.expander("最近一次提交编号（用于核查，不会重发）"):
                request_id = st.session_state.last_request_id
                st.code(request_id)
                if st.button("查询这次提交"):
                    st.json(call("GET", "/v1/requests/" + request_id))
    except SubmissionUncertain as error:
        st.warning(str(error))
        st.code(st.session_state.get("last_request_id", ""))
    except BackendError as error:
        st.error(str(error))


if __name__ == "__main__":
    main()
