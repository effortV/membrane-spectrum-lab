"""Streamlit task operations page, refreshed without blocking other pages."""

from datetime import datetime, timedelta, timezone
import json

import pandas as pd
import streamlit as st

from .task_monitor import RESULT_PAGES, STATUSES, progress_measure, read_task_history


def _time(value):
    try:
        return (
            datetime.fromisoformat(str(value).replace("Z", "+00:00"))
            .astimezone(timezone(timedelta(hours=8)))
            .strftime("%m-%d %H:%M:%S")
        )
    except (ValueError, TypeError):
        return "—"


def _open_page(page, conversation_id=None):
    st.session_state["show_task_monitor"] = False
    st.session_state["navigation_page"] = page
    if conversation_id:
        st.session_state["active_conversation_id"] = conversation_id
        st.session_state["chat_show_archived"] = False


def _details(task, manager, owner, *, can_cancel=False):
    progress = task["progress"]
    st.write(
        f"{STATUSES.get(task['status'], task['status'])} · {progress.get('stage', '暂无阶段信息')}"
    )
    st.caption(f"开始 {_time(task['created_at'])} · 总耗时 {task['elapsed_seconds']:.0f} 秒")
    if task["status"] == "running":
        st.caption(f"距最近进度更新 {task['update_age_seconds']:.0f} 秒；页面每 2 秒自动更新。")
    measure = progress_measure(progress)
    if measure:
        st.progress(measure[0], text=measure[1])
        if task["status"] == "completed" and measure[0] < 1:
            st.warning(
                "此批次已结束，但计数进度未全部完成。请查看失败／超时信息，未处理部分需依据检查点继续。"
            )
    elif task["status"] == "running":
        st.info("当前阶段暂无可计数进度；正在执行，不将等待时间当成完成百分比。")
    if progress.get("document_title"):
        st.text(f"当前文献：{progress['document_title']}")
    if progress.get("page"):
        st.caption(
            f"PDF 第 {progress['page']} 页 · 当前文献筛选页 {progress.get('page_index', '?')}/{progress.get('pages_total', '?')}"
        )
    if progress.get("outer_fold"):
        st.caption(
            f"外层折 {progress['outer_fold']}/{progress.get('outer_total', '?')} · 内层折 {progress.get('inner_fold', '?')} · 变量组 {progress.get('feature_set', '?')}"
        )
    if progress.get("model"):
        st.caption(f"当前模型：{progress['model']}")
    if "history_turns" in progress:
        st.caption(
            f"同一研究会话带入前文 {progress['history_turns']} 轮；更早未带入 {progress.get('omitted_turns', 0)} 轮。"
        )
    if "batch_evidence_saved" in progress:
        st.caption(f"本批次最近统计：已保存 {progress['batch_evidence_saved']} 条证据。")
    if progress.get("received_chunks"):
        st.caption(
            f"已收到 {progress['received_chunks']} 段数据 · 回答 {progress.get('generated_chars', 0)} 字符 · 推理 {progress.get('reasoning_chars', 0)} 字符。仅显示数量，不显示推理正文。"
        )
    if progress.get("request_trace_id"):
        st.caption(f"请求追踪 ID：{progress['request_trace_id']}")
    if progress.get("stage") == "等待模型响应" and task["status"] == "running":
        st.info(
            f"等待模型首批数据；连续无数据上限 {progress.get('wait_limit_seconds', '?')} 秒，单次请求总时限 {progress.get('request_total_limit_seconds', '?')} 秒。刷新不会重发请求。"
        )
    if progress.get("attempt"):
        st.caption(
            f"请求尝试 {progress['attempt']} · 本任务实际模型网络请求 {progress.get('network_requests', '—')} 次"
        )
    if task.get("error"):
        st.warning(task["error"])
    elif progress.get("last_error"):
        st.warning(f"最近步骤错误：{progress['last_error']}")
    if task.get("result_summary"):
        st.write("完成数量摘要")
        st.json(task["result_summary"])
        st.caption("added_or_updated 表示检索命中并新增或更新的数量，不等于新增篇数。")
    if task["status"] == "running" and can_cancel and task["is_owner"]:
        if st.button(
            "停止此任务后续步骤",
            key=f"monitor_cancel_{task['task_id']}",
            disabled=task["cancel_requested"],
        ):
            manager.cancel(task["task_id"], owner)
            st.rerun(scope="fragment")
        if task["cancel_requested"]:
            st.warning("已请求停止；等待当前请求返回后停止后续步骤，已发出的请求仍可能计费。")
    if task.get("events"):
        with st.expander("最近阶段记录"):
            st.dataframe(
                pd.DataFrame([{**event, "at": _time(event.get("at"))} for event in task["events"]]),
                hide_index=True,
                width="stretch",
            )


@st.fragment(run_every=2)
def task_monitor_body(settings, manager, owner):
    tasks = read_task_history(settings.workspace_root / "state" / "tasks", manager.snapshot(owner))
    active = [task for task in tasks if task["status"] == "running"]
    a, b, c = st.columns(3)
    a.metric("正在运行", len(active))
    b.metric("已完成（最近记录）", sum(task["status"] == "completed" for task in tasks))
    c.metric(
        "失败／停止／中断", sum(task["status"] not in {"running", "completed"} for task in tasks)
    )
    st.subheader("现在正在做什么")
    if not active:
        st.success("当前没有正在运行的后台任务。")
    for task in active:
        with st.container(border=True):
            st.subheader(task["label"])
            _details(task, manager, owner, can_cancel=True)
    st.subheader("任务历史与结果入口")
    if not tasks:
        st.info("还没有后台任务记录。启动检索、读取、下载、XPS 分析或关系研究后会自动记录。")
        return
    labels = list(STATUSES.values())
    statuses = st.multiselect("任务状态筛选", labels, default=labels, key="monitor_status_filter")
    view = [task for task in tasks if STATUSES.get(task["status"], task["status"]) in statuses]
    st.caption(
        "最多显示最近 200 条记录；时间为北京时间。已完成表示批次结束，不表示每页均成功，请查看失败摘要。重启保留历史，但不会自动恢复未完成模型请求。"
    )
    if not view:
        st.info("该筛选条件下没有任务。")
        return
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "开始时间": _time(task["created_at"]),
                    "任务": task["label"],
                    "状态": STATUSES.get(task["status"], task["status"]),
                    "阶段": task["progress"].get("stage", "—"),
                    "进度": (progress_measure(task["progress"]) or (None, "—"))[1],
                    "耗时（秒）": round(task["elapsed_seconds"]),
                    "当前／最后文献": task["progress"].get("document_title", "—"),
                    "任务 ID": task["task_id"],
                }
                for task in view
            ]
        ),
        hide_index=True,
        width="stretch",
    )
    lookup = {task["task_id"]: task for task in view}
    selected = st.selectbox(
        "查看任务详情",
        list(lookup),
        format_func=lambda key: (
            f"{_time(lookup[key]['created_at'])} · {lookup[key]['label']} · {STATUSES.get(lookup[key]['status'], lookup[key]['status'])}"
        ),
        key="monitor_selected_task",
    )
    task = lookup[selected]
    with st.container(border=True):
        _details(task, manager, owner)
        page = RESULT_PAGES.get(task["kind"])
        if page:
            if st.button(
                f"前往结果页面：{page}",
                key=f"monitor_result_{task['task_id']}",
                on_click=_open_page,
                args=(page, task["progress"].get("conversation_id")),
            ):
                st.rerun(scope="app")
            st.caption(
                "文献检索与下载的逐篇明细在“文献中心 → 检索与获取”；正文和图页计划在“本地读取”，实际抽取证据在证据库。问答、关系研究和 XPS 分析报告均有持久化历史。"
            )
    st.download_button(
        "下载任务记录 JSON（不含密钥、提示词和模型回答）",
        json.dumps(tasks, ensure_ascii=False, indent=2).encode(),
        file_name="xps_tasks.json",
        mime="application/json",
        key="monitor_download",
    )


def page_task_monitor(settings, db, manager, owner):
    del db
    st.title("后台任务 · 运行与进度")
    st.caption(
        "集中查看检索、下载、文献文字／图页读取、XPS 图分析、候选生成、智能体问答及 ML。查看任务不会启动任何 API 调用。"
    )
    task_monitor_body(settings, manager, owner)
