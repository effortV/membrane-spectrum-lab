from __future__ import annotations

import inspect
import json
import os
import threading
import uuid
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from xps_agent.agent import ResearchAgent
from xps_agent import __version__
from xps_agent.config import Settings, save_api_settings, save_model_settings
from xps_agent.data import DataAuditor
from xps_agent.db import StateDB
from xps_agent.diagnostics import health_report
from xps_agent.evidence import EvidenceExtractor
from xps_agent.hypotheses import HypothesisEngine
from xps_agent.literature import LiteratureService
from xps_agent.llm import SiliconFlowClient
from xps_agent.ml import NestedGroupEvaluator
from xps_agent.pdfs import PDFService
from xps_agent.features import SPECTRUM_SUMMARY_VERSION, is_known_descriptor, is_quality_column
from xps_agent.security import safe_error
from xps_agent.presentation import render_model_output
from xps_agent.tasks import BackgroundTasks, TaskContext
from xps_agent.server_runtime import get_runtime


st.set_page_config(
    page_title="膜谱研究台",
    page_icon=":material/graphic_eq:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def get_context(version: str = __version__) -> tuple[Settings, StateDB]:
    del version  # Upgrades must not reuse old Settings objects from a resource cache.
    return get_runtime().context()


@st.cache_resource
def get_paid_gate() -> tuple[threading.Lock, dict[str, str]]:
    runtime = get_runtime()
    return runtime.paid_lock, runtime.paid_state


@st.cache_resource
def get_task_manager() -> BackgroundTasks:
    return get_runtime().manager


@st.cache_resource
def get_backend_server():
    return get_runtime().start_api()


def task_owner() -> str:
    if "task_owner" not in st.session_state:
        st.session_state["task_owner"] = uuid.uuid4().hex
    return st.session_state["task_owner"]


def start_model_task(
    settings: Settings, db: StateDB, kind: str, value, *, mode: str = "pending"
) -> None:
    labels = {"extraction": "视觉证据抽取", "proposal": "候选生成"}
    if kind not in labels:
        raise ValueError("问答必须通过已保存会话提交。")

    def worker(context: TaskContext):
        with SiliconFlowClient(
            settings, db, progress=context.update, check_cancelled=context.check
        ) as llm:
            if kind == "extraction":
                return EvidenceExtractor(
                    settings, db, llm, progress=context.update, check_cancelled=context.check
                ).extract_registered(int(value), mode)
            if kind == "proposal":
                return HypothesisEngine(settings, db, llm).propose(int(value))

    get_task_manager().start(
        task_owner(),
        labels[kind],
        kind,
        worker,
        timeout=settings.action_timeout_seconds,
        journal_root=settings.workspace_root / "state" / "tasks",
    )


def start_literature_task(settings: Settings, db: StateDB, kind: str, value=None) -> None:
    labels = {
        "index": "索引本地文献",
        "parse": "读取 PDF 文字",
        "plan": "筛选 XPS 图页",
        "fetch": "获取可访问全文",
        "search": "检索 OpenAlex",
        "prepare": "索引并读取待处理文献",
    }

    def worker(context: TaskContext):
        context.update(stage=labels[kind])
        pdfs = PDFService(settings, db, progress=context.update, check_cancelled=context.check)
        service = LiteratureService(
            settings, db, progress=context.update, check_cancelled=context.check
        )
        try:
            roots = [
                settings.reference_root,
                settings.workspace_root / "library",
                settings.workspace_root / "inbox",
            ]
            if kind == "search":
                keys = service.search_openalex(value[0], int(value[1]))
                return {
                    "added_or_updated": len(keys),
                    **service.activity.summary(service.last_batch_id),
                }
            if kind == "fetch":
                if isinstance(value, dict):
                    return service.fetch_queued(int(value["limit"]), value.get("doc_keys"))
                return service.fetch_queued(int(value))
            if kind == "index":
                return service.index_local_pdfs(roots)
            if kind == "parse":
                return pdfs.parse_registered(int(value or 500))
            if kind == "plan":
                return pdfs.evidence_plan(5000)
            return {
                "index": service.index_local_pdfs(roots),
                "text": pdfs.parse_registered(500),
                "plan": pdfs.evidence_plan(5000),
            }
        finally:
            service.client.close()

    get_task_manager().start(
        task_owner(),
        labels[kind],
        kind,
        worker,
        timeout=1800,
        journal_root=settings.workspace_root / "state" / "tasks",
    )
    st.rerun()


@st.fragment(run_every=2)
def task_panel() -> None:
    manager = get_task_manager()
    owner = task_owner()
    task = manager.snapshot(owner)
    if not task:
        lock, state = get_paid_gate()
        if lock.locked():
            st.info(f"原有同步任务仍在运行：{state.get('label', '模型调用')}。请勿重复提交。")
        return
    progress = task["progress"]
    with st.container(border=True):
        st.subheader(f"任务状态 · {task['label']}")
        st.write(
            f"{task['status']} · 总耗时 {task['elapsed_seconds']:.0f} 秒 · {progress.get('stage', '')}"
        )
        if progress.get("document_title"):
            st.caption(
                f"文献 {progress.get('document_index', '?')}/{progress.get('documents_total', '?')}：{progress['document_title']}"
            )
        if progress.get("page"):
            st.caption(
                f"PDF 第 {progress['page']} 页 · 当前文献筛选页 {progress.get('page_index', '?')}/{progress.get('pages_total', '?')}"
            )
        if progress.get("stage") == "等待模型响应":
            st.info(
                f"本请求已等待 {task['phase_elapsed_seconds']:.0f} 秒；读取等待上限 {progress.get('wait_limit_seconds', '?')} 秒。网络超时不会自动重复提交。"
            )
        if progress.get("stage") in {"模型正在推理", "模型正在生成", "已连接，等待生成"}:
            st.caption(
                f"已收到 {progress.get('received_chunks', 0)} 段数据 · 回答 {progress.get('generated_chars', 0)} 字符 · 推理 {progress.get('reasoning_chars', 0)} 字符；完整返回并校验后才保存结果。"
            )
        total = progress.get("documents_total", 0)
        if total:
            st.progress(
                min(1.0, progress.get("documents_finished", 0) / total),
                text=f"已处理 {progress.get('documents_finished', 0)}/{total} 篇文献",
            )
        if progress.get("fits_total"):
            st.progress(
                min(1.0, progress.get("fits_finished", 0) / progress["fits_total"]),
                text=f"训练/验证 {progress.get('fits_finished', 0)}/{progress['fits_total']} · 外层 {progress.get('outer_fold', '?')} · {progress.get('feature_set', '')} · {progress.get('model', '')}",
            )
        if task["status"] == "running":
            st.caption(
                "后台运行中：刷新或切换页面不会重复启动。服务器重启会停止后台任务，但成功页检查点保留。"
            )
            if task["is_owner"] and st.button(
                "停止后续请求（保留已完成页）",
                key=f"cancel_{task['task_id']}",
                disabled=task["cancel_requested"],
            ):
                manager.cancel(task["task_id"], owner)
                st.rerun(scope="fragment")
            if task["cancel_requested"]:
                st.warning(
                    "已请求停止，等待当前请求返回或超时后停止；已发出的请求仍可能由供应商计费。"
                )
        else:
            if task["error"]:
                st.error(task["error"])
            elif task["status"] == "completed":
                if isinstance(task["result"], dict) and task["result"].get("failed"):
                    st.warning("批次结束，但有失败页/文献，详见结果；失败并不等于成功提取。")
                else:
                    st.success("任务已完成。")
            else:
                st.warning("任务已停止或达到批次上限，可以依据检查点分批继续。")
            if task["is_owner"] and st.session_state.get("consumed_task") != task["task_id"]:
                result_keys = {
                    "extraction": "last_extraction_result",
                    "proposal": "last_proposal_result",
                    "ask": "last_chat_result",
                    "relationship": "last_relationship_result",
                    "research_interpretation": "last_research_interpretation",
                    "membrane": "last_membrane_result",
                }
                if task["result"] is not None:
                    st.session_state[result_keys.get(task["kind"], "last_literature_result")] = (
                        task["result"]
                    )
                st.session_state["consumed_task"] = task["task_id"]
                st.rerun()
        with st.expander("任务详情（不含密钥和模型提示词）"):
            st.json({key: value for key, value in task.items() if key != "result"})


@contextmanager
def paid_operation(label: str):
    lock, state = get_paid_gate()
    if not lock.acquire(blocking=False):
        raise RuntimeError(
            f"另一个付费任务正在运行：{state.get('label', '模型调用')}。本次未重复提交。"
        )
    state["label"] = label
    try:
        yield
    finally:
        state.clear()
        lock.release()


@st.cache_data(show_spinner=False)
def read_csv_cached(path_text: str, modified: float) -> pd.DataFrame:
    del modified
    return pd.read_csv(path_text, low_memory=False)


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return read_csv_cached(str(path), path.stat().st_mtime)


def clear_data_cache() -> None:
    read_csv_cached.clear()


def one_count(db: StateDB, table: str) -> int:
    allowed = {"documents", "evidence", "hypotheses", "experiments"}
    if table not in allowed:
        raise ValueError("Unsupported table")
    rows = db.rows(f"SELECT COUNT(*) AS count FROM {table}")
    return int(rows[0]["count"]) if rows else 0


def hero(title: str, subtitle: str) -> None:
    st.title(title)
    st.caption(subtitle)


def show_error(exc: Exception) -> None:
    st.error(safe_error(exc))
    with st.expander("错误详情"):
        st.write(f"错误类型：{type(exc).__name__}。敏感凭据已过滤，不展示原始堆栈。")


def numeric_columns(frame: pd.DataFrame, minimum_fraction: float = 0.2) -> list[str]:
    result: list[str] = []
    for column in frame.columns:
        if pd.to_numeric(frame[column], errors="coerce").notna().mean() >= minimum_fraction:
            result.append(str(column))
    return result


def page_overview(settings: Settings, db: StateDB) -> None:
    hero("膜谱研究台", "NF / RO · 制备、表面化学、结构与性能")
    st.write(
        "从 NF / RO 数据和页级文献证据出发，研究制备—XPS—结构—性能关系，提出新描述符并设计验证实验。"
    )
    st.caption("相关性、预测能力与机理证据分别记录；候选可计算不等于新机理已成立。")
    profile_path = settings.workspace_root / "canonical" / "data_profile.json"
    profile = json.loads(profile_path.read_text(encoding="utf-8")) if profile_path.exists() else {}
    columns = st.columns(5)
    values = [
        ("样品记录", profile.get("model_table_rows", 0)),
        ("可用谱图", profile.get("canonical_spectra_usable", 0)),
        ("N/O 成对", profile.get("paired_n_o_records", 0)),
        ("文献目录", one_count(db, "documents")),
        ("候选假设", one_count(db, "hypotheses")),
    ]
    for column, (label, value) in zip(columns, values, strict=True):
        column.metric(label, value)

    st.subheader("研究流程")
    st.caption(f"统一科学数据目录：{settings.public_summary()['storage_root']}")
    stages = [
        (
            "1. 数据规范化",
            profile_path.exists(),
            "原始 NF/RO 表和 N/O 谱图哈希化、分组、保留不确定映射",
        ),
        (
            "2. 文献证据",
            one_count(db, "evidence") > 0,
            "本地文本优先，视觉模型只读取筛选后的图表页面",
        ),
        (
            "3. 机理假设",
            one_count(db, "hypotheses") > 0,
            "DeepSeek 只生成受限 DSL、可计算且可证伪的候选",
        ),
        (
            "4. 制备—XPS—结构—性能研究",
            one_count(db, "experiments") > 0,
            "按变量组比较制备→XPS、XPS→结构、结构→性能及 XPS→性能；相同 DOI 不跨折，负结果保留",
        ),
        ("5. 外部验证", False, "独立文献、时间批次或实验确认后才升级为发现"),
    ]
    for name, ready, description in stages:
        icon = "已就绪 ·" if ready else "待验证 ·"
        st.write(f"{icon} {name}")
        st.caption(description)

    left, right = st.columns([1.2, 1])
    with left:
        st.subheader("当前数据质量")
        if profile:
            st.json(
                {
                    "NF/RO 总行数": profile.get("model_table_rows"),
                    "可用 N/O 谱": profile.get("canonical_spectra_usable"),
                    "成对谱记录": profile.get("paired_n_o_records"),
                    "已解析 DOI 的样品行": profile.get("doi_group_rows_resolved"),
                    "谱清单错误/非标准工作表": profile.get("spectrum_errors"),
                }
            )
            if profile.get("spectrum_summary_version") != SPECTRUM_SUMMARY_VERSION:
                st.warning("谱图统计版本较旧，请在系统设置重新审计数据，再重新计算候选。")
        else:
            st.warning("尚未运行数据审计。")
    with right:
        st.subheader("模型与费用状态")
        summary = settings.public_summary()
        st.write(f"主推理：`{summary['main_model']}`")
        st.write(f"文献视觉：`{summary['vision_model']}`")
        st.write(f"SiliconFlow：**{summary['siliconflow_key']}**")
        st.caption(
            f"本次操作网络请求上限：{settings.llm_max_requests_per_action}（含重试，缓存命中不占上限）。"
        )
        usage = db.rows(
            "SELECT model, COUNT(*) calls, SUM(input_tokens) input_tokens, "
            "SUM(output_tokens) output_tokens FROM llm_request_events WHERE status='succeeded' GROUP BY model"
        )
        if usage:
            st.dataframe(pd.DataFrame(usage), hide_index=True, width="stretch")
        else:
            st.info("升级后尚无成功的模型网络调用。历史缓存条数不等于真实付费调用次数。")
        historical = db.rows("SELECT COUNT(*) count FROM llm_calls")[0]["count"]
        st.caption(f"历史唯一请求缓存：{historical} 条（保留旧账本，不反推费用）。")


def page_data(settings: Settings, db: StateDB) -> None:
    del db
    hero("数据与谱图", "查看新的规范化表、匹配状态和原始 XPS 曲线")
    table_path = settings.workspace_root / "canonical" / "model_table.csv"
    frame = read_csv(table_path)
    if frame.empty:
        st.warning("没有 model_table.csv，请先运行“系统设置”中的数据审计。")
        return

    filter_a, filter_b, filter_c = st.columns(3)
    dataset_options = sorted(frame["dataset"].dropna().astype(str).unique())
    selected_datasets = filter_a.multiselect("膜类型", dataset_options, default=dataset_options)
    spectrum_filter = filter_b.selectbox(
        "谱图关联", ["全部", "有 N 或 O", "N/O 都有", "没有关联谱图"]
    )
    query = filter_c.text_input("检索 record_id / 文献号", "")
    view = frame[frame["dataset"].astype(str).isin(selected_datasets)].copy()
    n_present = view.get("N__centroid_eV", pd.Series(index=view.index, dtype=float)).notna()
    o_present = view.get("O__centroid_eV", pd.Series(index=view.index, dtype=float)).notna()
    if spectrum_filter == "有 N 或 O":
        view = view[n_present | o_present]
    elif spectrum_filter == "N/O 都有":
        view = view[n_present & o_present]
    elif spectrum_filter == "没有关联谱图":
        view = view[~n_present & ~o_present]
    if query:
        mask = view["record_id"].astype(str).str.contains(query, case=False, na=False)
        mask |= view["reference_number"].astype(str).str.contains(query, case=False, na=False)
        view = view[mask]

    st.caption(f"当前显示 {len(view)} / {len(frame)} 行。未映射谱图不会被自动猜测配对。")
    default_columns = [
        column
        for column in (
            "record_id",
            "dataset",
            "reference_number",
            "validation_group",
            "N__centroid_eV",
            "N__spread_eV",
            "O__centroid_eV",
            "O__spread_eV",
        )
        if column in view.columns
    ]
    selected_columns = st.multiselect("表格字段", list(view.columns), default=default_columns)
    st.dataframe(view[selected_columns].head(1000), hide_index=True, width="stretch")
    st.download_button(
        "下载当前筛选 CSV",
        view.to_csv(index=False).encode("utf-8-sig"),
        file_name="xps_filtered_samples.csv",
        mime="text/csv",
    )

    st.subheader("变量分布")
    numeric = numeric_columns(view)
    if numeric:
        distribution_column = st.selectbox("数值变量", numeric)
        plot_frame = view[["dataset", distribution_column]].copy()
        plot_frame[distribution_column] = pd.to_numeric(
            plot_frame[distribution_column], errors="coerce"
        )
        figure = px.histogram(
            plot_frame.dropna(),
            x=distribution_column,
            color="dataset",
            marginal="box",
            barmode="overlay",
            opacity=0.65,
        )
        st.plotly_chart(figure, width="stretch")

    st.subheader("单样品 N/O 原始谱")
    spectral_records = view[
        view.get("N__spectrum_source_file", pd.Series(index=view.index, dtype=str)).notna()
        | view.get("O__spectrum_source_file", pd.Series(index=view.index, dtype=str)).notna()
    ]
    if spectral_records.empty:
        st.info("当前筛选范围没有已关联的谱文件。")
        return
    record_id = st.selectbox("样品", spectral_records["record_id"].astype(str).tolist())
    row = spectral_records[spectral_records["record_id"].astype(str) == record_id].iloc[0]
    figure = make_subplots(rows=1, cols=2, subplot_titles=("N 1s", "O 1s"))
    plotted = False
    for position, element in enumerate(("N", "O"), start=1):
        source = row.get(f"{element}__spectrum_source_file")
        if isinstance(source, str) and source and Path(source).exists():
            try:
                energy, intensity = DataAuditor._load_spectrum_xy(Path(source))
                figure.add_trace(
                    go.Scatter(x=energy, y=intensity, mode="lines", name=f"{element} 1s"),
                    row=1,
                    col=position,
                )
                figure.update_xaxes(
                    title_text="Binding energy (eV)", autorange="reversed", row=1, col=position
                )
                figure.update_yaxes(title_text="Normalized intensity", row=1, col=position)
                plotted = True
            except Exception as exc:
                st.warning(f"{element} 谱读取失败：{exc}")
    if plotted:
        figure.update_layout(height=430, legend_orientation="h")
        st.plotly_chart(figure, width="stretch")


def page_tasks(settings: Settings, db: StateDB) -> None:
    from xps_agent.task_monitor_ui import page_task_monitor

    page_task_monitor(settings, db, get_task_manager(), task_owner())


def _open_task_monitor() -> None:
    st.session_state["show_task_monitor"] = True


def _leave_task_monitor() -> None:
    st.session_state["show_task_monitor"] = False


@st.fragment(run_every=2)
def sidebar_task_status(settings: Settings) -> None:
    from xps_agent.task_monitor import progress_measure, read_task_history

    tasks = read_task_history(
        settings.workspace_root / "state" / "tasks", get_task_manager().snapshot(task_owner())
    )
    active = [task for task in tasks if task["status"] == "running"]
    if not active:
        st.caption("当前无运行任务")
        return
    for task in active:
        st.caption(f"运行中 · {task['label']}")
        st.caption(task["progress"].get("stage", "准备任务"))
        measure = progress_measure(task["progress"])
        if measure:
            st.progress(measure[0], text=measure[1])


def page_literature(settings: Settings, db: StateDB) -> None:
    from xps_agent.library_ui import page_library

    page_library(settings, db, start_literature_task, busy=get_paid_gate()[0].locked())


def page_evidence(settings: Settings, db: StateDB) -> None:
    hero("证据与智能体", "页级证据、受约束新描述符假设和可审计问答")
    evidence = pd.DataFrame(
        db.rows(
            """
            SELECT e.evidence_id,e.doc_key,e.page,e.kind,e.claim,e.locator,e.confidence,
                   e.model,d.title,d.doi
            FROM evidence e JOIN documents d USING(doc_key)
            ORDER BY e.created_at DESC
            """
        )
    )
    tab_extract, tab_evidence, tab_hypotheses, tab_ask = st.tabs(
        ["视觉抽取", "证据库", "候选假设", "智能体问答"],
        key="evidence_tab",
        on_change="rerun",
        default="智能体问答" if st.query_params.get("chat") else None,
    )
    with tab_extract:
        if "last_extraction_result" in st.session_state:
            st.json(st.session_state["last_extraction_result"])
        plan_path = settings.workspace_root / "state" / "evidence_plan.csv"
        plan = read_csv(plan_path)
        if not plan.empty:
            st.metric("计划中的筛选页面", int(plan["selected_page_count"].sum()))
            st.caption("默认每篇最多 4 页；相同 PDF 页面和模型响应按哈希缓存。")
        limit = st.slider("本次处理文献数", 1, 100, 5)
        mode_label = st.selectbox(
            "文献处理模式", ["仅处理待完成文献（推荐）", "仅重试失败文献", "待完成和失败文献"]
        )
        mode = {
            "仅处理待完成文献（推荐）": "pending",
            "仅重试失败文献": "failed",
            "待完成和失败文献": "all",
        }[mode_label]
        vision_wait = st.number_input(
            "视觉请求读取等待上限（秒）",
            30,
            1800,
            int(max(30, min(1800, settings.vision_timeout_seconds))),
            step=15,
        )
        st.caption(
            f"流式响应连续无数据等待上限 {vision_wait:g} 秒；超时重试 {settings.vision_timeout_retries} 次（额外请求可能重复计费）。429/5xx 等暂时性错误最多 {settings.vision_max_attempts} 次尝试。批次时限 {settings.action_timeout_seconds:g} 秒。"
        )
        confirmed = st.checkbox("确认调用付费视觉模型", key="confirm_vlm")
        if st.button("开始页级证据抽取", disabled=not confirmed):
            try:
                start_model_task(
                    replace(settings, vision_timeout_seconds=float(vision_wait)),
                    db,
                    "extraction",
                    limit,
                    mode=mode,
                )
                st.rerun()
            except Exception as exc:
                show_error(exc)
    with tab_evidence:
        if evidence.empty:
            st.info("证据库为空。请先小批量运行视觉抽取。")
        else:
            query = st.text_input("证据检索", key="evidence_query")
            view = evidence
            if query:
                view = view[
                    view["claim"].astype(str).str.contains(query, case=False, na=False)
                    | view["title"].astype(str).str.contains(query, case=False, na=False)
                ]
            st.dataframe(view.head(1000), hide_index=True, width="stretch")
    with tab_hypotheses:
        if "last_proposal_result" in st.session_state:
            with st.expander("最近一次候选生成（包含拒绝原因）", expanded=True):
                st.json(st.session_state["last_proposal_result"])
        hypotheses = db.rows(
            "SELECT hypothesis_id,name,status,spec_json,model,created_at FROM hypotheses ORDER BY created_at DESC"
        )
        count = st.slider("本次最多提出", 1, 20, 6)
        confirmed = st.checkbox("确认调用 DeepSeek-V4-Pro 生成候选", key="confirm_propose")
        if st.button("提出新的描述符候选", disabled=not confirmed):
            try:
                start_model_task(settings, db, "proposal", count)
                st.rerun()
            except Exception as exc:
                show_error(exc)
        if st.button("安全计算全部可用候选"):
            try:
                result = HypothesisEngine(settings, db).materialize()
                clear_data_cache()
                st.success(result)
            except Exception as exc:
                show_error(exc)
        if hypotheses:
            for item in hypotheses:
                with st.expander(
                    f"{item['status']} · {item['name']} · {item['hypothesis_id'][:8]}"
                ):
                    spec = json.loads(item["spec_json"])
                    equation = str(spec.get("equation", ""))
                    st.markdown("**公式**")
                    if "\\" in equation or "$" in equation:
                        render_model_output(equation if "$" in equation else "$$" + equation + "$$")
                    else:
                        st.text(equation)
                    st.caption(f"单位：{spec.get('units', '未注明')}")
                    for key, label in (
                        ("mechanism_chain", "机制链"),
                        ("falsification_tests", "否证实验"),
                    ):
                        st.markdown(f"**{label}**")
                        for value in spec.get(key, []):
                            render_model_output("- " + str(value))
                    with st.expander("原始候选 JSON（真实列名及审核字段）"):
                        st.json(spec)
        else:
            st.info("尚无候选假设。HABD/HCD/HAD/O2-O1 只会作为排除基线。")
    with tab_ask:
        from xps_agent.chat_ui import render_conversations

        render_conversations(settings, db, get_task_manager(), task_owner())


def page_ml(settings: Settings, db: StateDB) -> None:
    from xps_agent.workbench_ui import page_research

    page_research(settings, db, get_task_manager(), task_owner())


def page_images(settings: Settings, db: StateDB) -> None:
    from xps_agent.workbench_ui import page_images as render

    render(settings, db, get_task_manager(), task_owner())


def page_legacy_ml(settings: Settings, db: StateDB) -> None:
    hero("ML 验证", "相同 DOI 分组的嵌套交叉验证：基线 vs 基线 + 候选")
    enriched = settings.workspace_root / "canonical" / "model_table_enriched.csv"
    base = settings.workspace_root / "canonical" / "model_table.csv"
    table_path = base
    if enriched.exists():
        from xps_agent.utils import sha256_file

        manifest_path = enriched.with_suffix(".manifest.json")
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if (
                manifest.get("source_sha256") == sha256_file(base)
                and manifest.get("destination_sha256") == sha256_file(enriched)
                and manifest.get("spectrum_summary_version") == SPECTRUM_SUMMARY_VERSION
            ):
                table_path = enriched
            else:
                st.warning("候选增强表已过期，本页只使用基础表；请先重新安全计算候选。")
        except (ValueError, OSError):
            st.warning("候选增强表缺少有效清单，本页只使用基础表。")
    frame = read_csv(table_path)
    if frame.empty:
        st.warning("没有可用建模表。")
        return
    st.caption(f"建模表：{table_path} · {len(frame)} 行。测试折不会参与模型或描述符选择。")
    dataset_scope = None
    if "dataset" in frame:
        datasets = sorted(frame["dataset"].dropna().astype(str).unique())
        dataset_scope = st.multiselect("研究膜类型", datasets, default=datasets[:1])
        if len(dataset_scope) > 1:
            st.warning(
                "联合 NF/RO 建模前请确认目标单位一致，并控制单体体系、测试压力和材料类别差异。"
            )
    inventory = read_csv(settings.workspace_root / "canonical" / "column_inventory.csv")
    outcome_candidates = []
    condition_candidates = []
    if not inventory.empty:
        outcome_candidates = [
            column
            for column in inventory[inventory["role"] == "outcome"]["column"].astype(str)
            if column in frame
        ]
        condition_candidates = [
            column
            for column in inventory[inventory["role"] == "condition"]["column"].astype(str)
            if column in frame
        ]
    numeric = numeric_columns(frame, 0.1)
    outcomes = [column for column in outcome_candidates if column in numeric] or numeric
    group_options = [column for column in ("doi_group", "validation_group") if column in frame]
    if not outcomes or not group_options:
        st.warning("缺少可用数值目标或文献分组列，请检查规范化表。")
        return
    target = st.selectbox("目标变量", outcomes)
    group = st.selectbox("分组变量", group_options, index=0)
    baseline_options = [
        column for column in condition_candidates if column in numeric and column != target
    ]
    if not baseline_options:
        baseline_options = [
            column
            for column in numeric
            if column != target and not column.startswith(("N__", "O__"))
        ]
    baseline = st.multiselect("基线特征", baseline_options)
    hypothesis_symbols: list[str] = []
    for row in db.rows("SELECT spec_json FROM hypotheses WHERE status='calculable'"):
        try:
            symbol = json.loads(row["spec_json"])["symbol"]
            if symbol in frame:
                hypothesis_symbols.append(symbol)
        except (KeyError, TypeError, json.JSONDecodeError):
            continue
    spectrum_candidates = [
        column
        for column in numeric
        if column.startswith(("N__", "O__")) and not is_quality_column(column)
    ]
    candidate_options = [
        column
        for column in dict.fromkeys([*hypothesis_symbols, *spectrum_candidates])
        if column != target and column not in baseline and not is_known_descriptor(column)
    ]
    candidates = st.multiselect("候选描述符", candidate_options)
    policy_label = st.selectbox(
        "候选缺失处理", ["只使用实际测得候选的样本（推荐）", "训练折内填补缺失（探索性敏感性分析）"]
    )
    candidate_policy = "measured_only" if policy_label.startswith("只使用") else "impute"
    include_unresolved = st.checkbox("允许 DOI 未确认的备用分组（仅探索性，独立性不能保证）")
    st.warning(
        "按文献内行序关联的谱图/样品尚需人工核对；多次查看同一数据上的得分会造成适应性选择，不能替代独立验证。"
    )
    fold_a, fold_b = st.columns(2)
    outer = fold_a.slider("外层分组折", 3, 10, 5)
    inner = fold_b.slider("内层分组折", 2, 8, 4)
    confirmed = st.checkbox("确认开始计算；结果可能是否定性的，仍会完整保存")
    if st.button(
        "运行嵌套分组验证",
        disabled=not confirmed
        or not baseline
        or not candidates
        or not target
        or not group
        or dataset_scope == [],
    ):
        try:
            with st.spinner("正在执行相同外层折的基线/增强模型比较…"):
                result = NestedGroupEvaluator(settings, db).evaluate(
                    table_path,
                    target,
                    group,
                    baseline,
                    candidates,
                    outer_splits=outer,
                    inner_splits=inner,
                    candidate_policy=candidate_policy,
                    include_unresolved_groups=include_unresolved,
                    dataset_scope=dataset_scope,
                )
            st.success("验证完成并写入不可变运行目录。")
            st.json(result)
            st.rerun()
        except Exception as exc:
            show_error(exc)

    st.subheader("历史运行")
    experiments = db.rows(
        "SELECT experiment_id,hypothesis_id,run_dir,decision,created_at,metrics_json "
        "FROM experiments ORDER BY created_at DESC"
    )
    if not experiments:
        st.info("尚无 ML 运行。")
        return
    choices = {f"{row['created_at']} · {row['experiment_id']}": row for row in experiments}
    selected = choices[st.selectbox("运行", list(choices))]
    metrics = json.loads(selected["metrics_json"])
    metric_a, metric_b = st.columns(2)
    metric_a.metric("平均 ΔR²", f"{metrics.get('mean_delta_r2', float('nan')):.3f}")
    metric_b.metric(
        "探索性折符号翻转值（非发现显著性）",
        f"{metrics.get('one_sided_sign_permutation_p', float('nan')):.3f}",
    )
    if "cohort" in metrics:
        st.json(
            {
                "样本纳入报告": metrics["cohort"],
                "逐文献平均误差": metrics.get("group_balanced_out_of_fold"),
                "均值预测基线": metrics.get("training_mean_dummy_out_of_fold"),
            }
        )
    fold_path = Path(selected["run_dir"]) / "fold_metrics.csv"
    if fold_path.exists():
        folds = pd.read_csv(fold_path)
        figure = px.box(folds, x="feature_set", y="r2", points="all", color="feature_set")
        st.plotly_chart(figure, width="stretch")
        st.dataframe(folds, hide_index=True, width="stretch")


def page_settings(settings: Settings, db: StateDB) -> None:
    hero("系统设置", "数据目录、模型、API 配置与离线维护")
    notice = st.session_state.pop("api_settings_notice", None)
    if notice:
        st.success(notice)

    st.subheader("API 与模型访问")
    status = settings.public_summary()
    status_columns = st.columns(3)
    status_columns[0].metric("SiliconFlow", status["siliconflow_key"])
    status_columns[1].metric("OpenAlex", status["openalex_key"])
    status_columns[2].metric("Elsevier", status["elsevier_key"])
    st.caption("输入内容使用密码框，不会回显到页面、日志或模型提示词；留空表示保持原值。")
    with st.form("api_settings_form", clear_on_submit=True):
        siliconflow_key = st.text_input("SiliconFlow API key", type="password")
        openalex_key = st.text_input("OpenAlex API key（可选）", type="password")
        openalex_mailto = st.text_input("OpenAlex 联系邮箱（可选）")
        elsevier_key = st.text_input("Elsevier API key（可选）", type="password")
        elsevier_insttoken = st.text_input("Elsevier institutional token（可选）", type="password")
        with st.expander("清除已保存的配置"):
            clear_siliconflow = st.checkbox("清除 SiliconFlow key")
            clear_openalex = st.checkbox("清除 OpenAlex key 与联系邮箱")
            clear_elsevier = st.checkbox("清除 Elsevier key 与 institutional token")
        submitted = st.form_submit_button("保存 API 配置", type="primary")
    if submitted:
        candidates = {
            "SILICONFLOW_API_KEY": siliconflow_key,
            "OPENALEX_API_KEY": openalex_key,
            "OPENALEX_MAILTO": openalex_mailto,
            "ELSEVIER_API_KEY": elsevier_key,
            "ELSEVIER_INSTTOKEN": elsevier_insttoken,
        }
        updates = {name: value for name, value in candidates.items() if value.strip()}
        if clear_siliconflow:
            updates["SILICONFLOW_API_KEY"] = ""
        if clear_openalex:
            updates["OPENALEX_API_KEY"] = ""
            updates["OPENALEX_MAILTO"] = ""
        if clear_elsevier:
            updates["ELSEVIER_API_KEY"] = ""
            updates["ELSEVIER_INSTTOKEN"] = ""
        if not updates:
            st.info("没有需要保存的更改。")
        else:
            try:
                changed = save_api_settings(settings.project_root, updates)
                labels = {
                    "SILICONFLOW_API_KEY": "SiliconFlow",
                    "OPENALEX_API_KEY": "OpenAlex",
                    "OPENALEX_MAILTO": "OpenAlex 联系邮箱",
                    "ELSEVIER_API_KEY": "Elsevier",
                    "ELSEVIER_INSTTOKEN": "Elsevier institutional token",
                }
                names = "、".join(labels[name] for name in changed)
                st.session_state["api_settings_notice"] = f"已保存：{names}。新配置已立即生效。"
                get_context.clear()
                st.rerun()
            except Exception as exc:
                show_error(exc)

    st.subheader("模型等待与重试")
    st.caption(
        "流式接收可显示推理／生成是否仍在进行；无数据等待上限不是整个回答的生成时长。总时限在接收数据时检查，阻塞读取仍受无数据等待上限约束。超时后重发是新请求，可能重复计费，默认关闭。"
    )
    with st.form("model_request_settings"):
        stream_enabled = st.checkbox(
            "启用流式接收与生成进度（推荐）", value=settings.llm_stream_enabled
        )
        col_a, col_b = st.columns(2)
        main_wait = col_a.number_input(
            "主模型无数据等待上限（秒）", 30, 1800, int(settings.llm_timeout_seconds), step=30
        )
        vision_wait = col_b.number_input(
            "视觉模型无数据等待上限（秒）", 30, 1800, int(settings.vision_timeout_seconds), step=30
        )
        request_total = col_a.number_input(
            "单次请求总时限（秒）", 120, 7200, int(settings.request_timeout_seconds), step=60
        )
        batch_total = col_b.number_input(
            "模型批次总时限（秒）", 300, 14400, int(settings.action_timeout_seconds), step=300
        )
        retry_main = st.checkbox(
            "允许主模型超时／断流后额外重试 1 次（可能重复计费）",
            value=bool(settings.llm_timeout_retries),
        )
        retry_vision = st.checkbox(
            "允许视觉模型超时／断流后额外重试 1 次（可能重复计费）",
            value=bool(settings.vision_timeout_retries),
        )
        save_model = st.form_submit_button("保存模型等待设置")
    if save_model:
        try:
            save_model_settings(
                settings.project_root,
                {
                    "XPS_LLM_STREAM_ENABLED": str(stream_enabled).lower(),
                    "XPS_LLM_TIMEOUT_SECONDS": str(main_wait),
                    "XPS_VISION_TIMEOUT_SECONDS": str(vision_wait),
                    "XPS_REQUEST_TIMEOUT_SECONDS": str(request_total),
                    "XPS_ACTION_TIMEOUT_SECONDS": str(batch_total),
                    "XPS_LLM_TIMEOUT_RETRIES": "1" if retry_main else "0",
                    "XPS_VISION_TIMEOUT_RETRIES": "1" if retry_vision else "0",
                },
            )
            st.session_state["model_settings_notice"] = (
                "模型等待设置已保存，对下一次任务生效；正在运行的任务不重新提交。"
            )
            get_context.clear()
            st.rerun()
        except Exception as exc:
            show_error(exc)
    if st.session_state.get("model_settings_notice"):
        st.success(st.session_state["model_settings_notice"])
    with st.expander("当前路径与模型（不含密钥值）"):
        st.json(settings.public_summary())
    st.subheader("只读健康诊断")
    if st.button("检查数据、证据、候选与产物一致性（不调用 API）"):
        st.session_state["health_report"] = health_report(settings, db)
    if "health_report" in st.session_state:
        report = st.session_state["health_report"]
        st.json(report)
        st.download_button(
            "下载诊断 JSON（不含密钥）",
            json.dumps(report, ensure_ascii=False, indent=2).encode(),
            file_name="xps_health.json",
            mime="application/json",
        )
    st.subheader("离线维护")
    if st.button("重新审计数据并生成建模表"):
        try:
            with st.spinner("正在读取 NF.xlsx、RO.xlsx 与四组谱图…"):
                auditor = DataAuditor(settings, db)
                result = auditor.run()
                auditor.export_authoritative_tables()
            clear_data_cache()
            st.success(result)
        except Exception as exc:
            show_error(exc)
    if st.button("运行代码自检提示"):
        st.code(
            "Set-Location D:\\zzh\\XPS-agent\n"
            ".\\.venv\\Scripts\\ruff.exe check src tests\n"
            ".\\.venv\\Scripts\\python.exe -m pytest -q",
            language="powershell",
        )
    st.subheader("科学与安全约束")
    st.markdown(
        """
        - `final` 只提供谱图与来源，不导入旧模型、旧 R² 或旧切分。
        - HABD、HCD、HAD、交联度及 O2/O1 只作为已知基线和新颖性排除项。
        - 分别归一化的 N/O 高分辨谱不能直接计算跨元素化学计量。
        - 干态 XPS 不自动等同于湿态运行电荷，高结合能峰归属必须保留条件边界。
        - 付费视觉抽取、候选生成和智能体问答都需要界面内再次确认。
        """
    )


def main() -> None:
    settings, db = get_context(__version__)
    hold = settings.workspace_root / "state" / "migration_hold.json"
    if hold.exists():
        lock, _ = get_paid_gate()
        ready = {"pid": os.getpid(), "busy": lock.locked()}
        (hold.parent / "migration_ready.json").write_text(json.dumps(ready), encoding="utf-8")
        st.info("正在升级数据目录与研究工作流；已有任务结束后更新，不会重复提交模型请求。")
        return
    # Streamlit reruns scripts but may retain imported package modules after deployment.
    # Never submit work through a mixed old/new API before a deliberate server restart.
    if (
        "storage_root" not in Settings.__dataclass_fields__
        or "progress" not in inspect.signature(SiliconFlowClient.__init__).parameters
        or "history" not in inspect.signature(ResearchAgent.ask).parameters
    ):
        st.warning(
            "新版文件已部署，但当前进程仍加载旧模块。请在当前批次结束后重启 Streamlit；不会在此页面重复发送请求。"
        )
        st.code(
            "cd /d D:\\zzh\\XPS-agent\n.\\.venv\\Scripts\\python.exe -m streamlit run streamlit_app.py --server.address 127.0.0.1 --server.port 8501",
            language="bat",
        )
        return
    try:
        get_backend_server()
    except Exception as exc:
        st.warning(safe_error(exc))
    brand_icon, brand_title = st.sidebar.columns([1, 6], vertical_alignment="center", gap="small")
    brand_icon.markdown(
        '<svg width="32" height="32" viewBox="0 0 32 32" aria-label="谱线标志" role="img">'
        '<path d="M3 24H29M3 8H29" stroke="#A6B4C2" stroke-width="1.4"/>'
        '<path d="M3 21L8 21L11 11L14 21L18 21L22 5L26 21L29 21" '
        'fill="none" stroke="#345B76" stroke-width="2" stroke-linejoin="round"/>'
        "</svg>",
        unsafe_allow_html=True,
    )
    brand_title.subheader("膜谱研究台", anchor=False)
    st.sidebar.caption(f"NF / RO 表面研究 · v{__version__}")
    chat_key = st.query_params.get("chat", "")
    if "navigation_page" not in st.session_state and chat_key:
        rows = db.rows(
            "SELECT conversation_id,archived FROM chat_conversations WHERE conversation_id=?",
            (chat_key,),
        )
        if rows:
            st.session_state["navigation_page"] = "证据与智能体"
            st.session_state["active_conversation_id"] = chat_key
            st.session_state["chat_show_archived"] = bool(rows[0]["archived"])
    if st.session_state.get("navigation_page") == "后台任务":
        st.session_state["navigation_page"] = "总览"
        st.session_state["show_task_monitor"] = True
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
        key="navigation_page",
        on_change=_leave_task_monitor,
    )
    if page != "证据与智能体":
        st.query_params.pop("chat", None)
    st.sidebar.divider()
    st.sidebar.caption(f"主模型：{settings.main_model}")
    st.sidebar.caption(f"视觉模型：{settings.vision_model}")
    key_state = "已配置" if settings.siliconflow_api_key else "未配置"
    st.sidebar.caption(f"SiliconFlow：{key_state}")
    st.sidebar.button(
        "后台任务",
        icon=":material/tune:",
        width="stretch",
        on_click=_open_task_monitor,
        key="open_task_monitor",
    )
    with st.sidebar:
        sidebar_task_status(settings)
    task_panel()
    pages = {
        "总览": page_overview,
        "数据与谱图": page_data,
        "文献中心": page_literature,
        "证据与智能体": page_evidence,
        "XPS 图分析": page_images,
        "关系研究（ML）": page_ml,
        "系统设置": page_settings,
    }
    if st.session_state.get("show_task_monitor"):
        page_tasks(settings, db)
    else:
        pages[page](settings, db)


if __name__ == "__main__":
    main()
