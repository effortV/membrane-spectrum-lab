"""Literature workbench: explicit receipts, full-text state, and local reading steps."""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import streamlit as st

from .library_activity import LibraryActivity
from .security import safe_error
from .utils import safe_filename

STATUS = {
    "discovered": "待获取全文",
    "queued": "待下载",
    "retry": "下载失败，可重试",
    "manual": "需要补传 PDF",
    "downloaded": "已下载，待读正文",
    "indexed": "本地 PDF，待读正文",
    "parsed": "正文已读取",
    "parse_failed": "正文读取失败",
    "evidence_failed": "图页读取未完成",
    "evidence_extracted": "图页证据已读取",
    "evidence_no_relevant_pages": "未筛到相关图页",
}


def _time(value):
    try:
        return (
            datetime.fromisoformat(value)
            .astimezone(timezone(timedelta(hours=8)))
            .strftime("%m-%d %H:%M")
        )
    except (TypeError, ValueError):
        return "时间未知"


def document_table(rows, *, receipts=False):
    records = []
    for row in rows:
        item = {
            "文献": row.get("title") or "未提供标题",
            "全文 / 读取状态": STATUS.get(row.get("status"), row.get("status", "")),
            "年份": row.get("year"),
            "DOI": row.get("doi") or "",
            "来源链接": row.get("landing_url") or "",
            "保存位置": row.get("current_path") or row.get("local_path") or "",
            "说明": safe_error(
                row.get("detail") or row.get("reason") or row.get("download_reason") or ""
            ),
        }
        if receipts:
            item = {
                "本次入库": "无法追溯"
                if row.get("is_new") is None
                else ("新增" if row["is_new"] else "已有"),
                "本次处理": {
                    "downloaded": "下载成功",
                    "manual": "需补传",
                    "retry": "待重试",
                    "pending": "尚未处理",
                    "new": "新增记录",
                    "existing": "已有记录",
                    "legacy": "升级前记录",
                }.get(row.get("outcome"), ""),
                **item,
            }
        if receipts:
            order = [
                "本次入库",
                "文献",
                "全文 / 读取状态",
                "年份",
                "DOI",
                "本次处理",
                "来源链接",
                "保存位置",
                "说明",
            ]
            item = {column: item[column] for column in order}
        records.append(item)
    return pd.DataFrame(records)


def _table(rows, *, receipts=False, filename="literature.csv", key):
    frame = document_table(rows, receipts=receipts)
    if frame.empty:
        st.caption("此处暂无文献记录。")
        return
    st.dataframe(
        frame,
        hide_index=True,
        width="stretch",
        column_config={
            "来源链接": st.column_config.LinkColumn("来源链接", display_text="打开来源"),
            "文献": st.column_config.TextColumn(width="medium"),
            "全文 / 读取状态": st.column_config.TextColumn(width="medium"),
        },
    )
    st.download_button(
        "导出当前清单 CSV",
        frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=filename,
        mime="text/csv",
        key=key,
    )
    lookup = {row["doc_key"]: row for row in rows}
    with st.expander("查看单篇详情：完整标题、状态与保存位置"):
        chosen = st.selectbox(
            "选择文献",
            list(lookup),
            key=key + "_detail",
            format_func=lambda value: lookup[value].get("title") or value,
        )
        row = lookup[chosen]
        st.write(row.get("title") or "未提供标题")
        st.caption(
            f"DOI：{row.get('doi') or '未提供'} · 状态：{STATUS.get(row.get('status'), row.get('status', '未注明'))}"
        )
        if path := row.get("current_path") or row.get("local_path"):
            st.code(str(path), language=None)
        else:
            st.caption("尚未关联本地 PDF；检索命中本身不会生成全文文件。")
        if reason := row.get("detail") or row.get("reason") or row.get("download_reason"):
            st.caption(safe_error(reason))


def page_library(settings, db, start_task, *, busy=False):
    activity = LibraryActivity(db)
    documents = activity.documents()

    def start(kind, value=None):
        try:
            start_task(settings, db, kind, value)
        except Exception as exc:
            st.error(safe_error(exc))

    st.title("文献中心")
    st.caption(
        "检索记录、全文获取与本地读取分别留痕。检索命中不等于全文已下载，图页计划也不等于模型已读图。"
    )
    if st.button("刷新文献记录"):
        st.rerun()
    a, b, c, d = st.columns(4)
    a.metric("文献记录", len(documents))
    b.metric("已关联本地 PDF", sum(bool(row["local_path"]) for row in documents))
    c.metric(
        "待读正文",
        sum(
            row["status"] in {"indexed", "downloaded", "parse_failed"} and bool(row["local_path"])
            for row in documents
        ),
    )
    d.metric("需要补传", sum(row["status"] == "manual" for row in documents))
    if busy:
        st.caption("有任务正在运行；此页可继续查看，待任务结束后再提交新任务。")

    search_tab, catalog_tab, reading_tab, upload_tab = st.tabs(
        ["检索与获取", "文献库", "本地读取", "补充 PDF"]
    )
    with search_tab:
        st.subheader("1. 检索文献")
        with st.form("literature_search"):
            left, right = st.columns([4, 1])
            query = left.text_input(
                "检索词（OpenAlex）", "XPS polyamide membrane interfacial polymerization"
            )
            limit = right.number_input("最多结果", 1, 100, 30)
            submitted = st.form_submit_button("检索文献", type="primary", disabled=busy)
        if submitted:
            start("search", (query, int(limit)))

        batches = activity.batches("search")
        lookup = {row["batch_id"]: row for row in batches}
        options = list(lookup) + ["legacy"]
        last = st.session_state.get("last_literature_result", {})
        if (
            isinstance(last, dict)
            and last.get("batch_id") in lookup
            and st.session_state.get("library_consumed_batch") != last["batch_id"]
        ):
            st.session_state["library_search_batch"] = last["batch_id"]
            st.session_state["library_consumed_batch"] = last["batch_id"]
        if st.session_state.get("library_search_batch") not in options:
            st.session_state["library_search_batch"] = options[0]
        selected = st.selectbox(
            "查看检索批次",
            options,
            key="library_search_batch",
            format_func=lambda key: (
                "升级前已检索文献（新增数量不可追溯）"
                if key == "legacy"
                else f"{_time(lookup[key]['created_at'])} · {lookup[key]['query']} · {lookup[key]['total']} 篇"
            ),
        )
        results = (
            activity.legacy_search_items() if selected == "legacy" else activity.items(selected)
        )
        if selected == "legacy":
            st.caption(
                "以下是现有数据库能确认的历史检索文献，不伪造当时的新增篇数。新检索会保留独立批次及逐篇入库记录，刷新后仍可查。"
            )
        else:
            batch = lookup[selected]
            summary = activity.summary(selected)
            st.write(
                f"命中 {summary['matched']} 篇 · 新增 {summary['added']} 篇 · 已有 {summary['existing']} 篇"
            )
            st.caption("已有文献也会列出；同一 DOI 不会重复建立文献记录。")
            if batch["status"] != "completed":
                label = {
                    "running": "尚未结束，实际进度见后台任务",
                    "failed": "检索失败",
                    "cancelled": "已停止",
                }.get(batch["status"], batch["status"])
                st.warning(
                    f"批次状态：{label}。{safe_error(batch.get('error') or '')} 已保存的逐篇记录仍可查看。"
                )
        _table(results, receipts=True, filename="search_results.csv", key="search_results_csv")

        st.subheader("2. 获取全文")
        scope = st.radio("下载范围", ["当前检索结果", "全部待下载文献"], horizontal=True)
        queued = [
            row
            for row in (results if scope == "当前检索结果" else documents)
            if row.get("download_status") in {"queued", "retry"}
        ]
        st.caption(
            f"所选范围有 {len(queued)} 篇待下载。可访问全文直接保存；无权限的文献进入“需要补传”，由你上传 PDF。"
        )
        left, right = st.columns([1, 3])
        fetch_limit = left.number_input("本次下载上限", 1, 500, 30)
        confirm = right.checkbox("确认获取所选范围的可访问全文")
        if st.button("下载可访问全文", disabled=busy or not confirm or not queued):
            start(
                "fetch",
                {
                    "limit": int(fetch_limit),
                    "doc_keys": [row["doc_key"] for row in queued]
                    if scope == "当前检索结果"
                    else None,
                },
            )
        downloads = activity.batches("fetch")
        if downloads:
            download_lookup = {row["batch_id"]: row for row in downloads}
            chosen = st.selectbox(
                "查看下载批次",
                list(download_lookup),
                format_func=lambda key: (
                    f"{_time(download_lookup[key]['created_at'])} · {download_lookup[key]['total']} 篇 · "
                    + {
                        "completed": "已结束",
                        "running": "尚未结束",
                        "failed": "失败",
                        "cancelled": "已停止",
                    }.get(download_lookup[key]["status"], download_lookup[key]["status"])
                ),
            )
            items = activity.items(chosen)
            st.write(
                f"下载成功 {sum(row['outcome'] == 'downloaded' for row in items)} 篇 · 需补传 {sum(row['outcome'] == 'manual' for row in items)} 篇 · 待重试 {sum(row['outcome'] == 'retry' for row in items)} 篇"
            )
            _table(
                items, filename="download_results.csv", receipts=True, key="download_results_csv"
            )

    with catalog_tab:
        st.subheader("文献库")
        query = st.text_input("查找标题或 DOI", key="library_catalog_query")
        category = st.radio(
            "显示范围", ["全部", "有本地 PDF", "待下载", "需要补传", "读取失败"], horizontal=True
        )
        view = documents
        if query:
            view = [
                row
                for row in view
                if query.casefold() in ((row["title"] or "") + " " + (row["doi"] or "")).casefold()
            ]
        if category == "有本地 PDF":
            view = [row for row in view if row["local_path"]]
        elif category == "待下载":
            view = [row for row in view if row["download_status"] in {"queued", "retry"}]
        elif category == "需要补传":
            view = [row for row in view if row["status"] == "manual"]
        elif category == "读取失败":
            view = [row for row in view if row["status"] in {"parse_failed", "evidence_failed"}]
        st.caption(f"显示 {len(view)} / {len(documents)} 篇；保存位置为服务器上的路径。")
        _table(view, key="library_catalog_csv")

    with reading_tab:
        st.subheader("3. 在本地读正文，再准备图页")
        st.write(
            "先把 PDF 正文提取成可检索的文字，再从页级关键词和图像信息中筛出 XPS 相关页。两步都在本地完成，不调用 LLM。"
        )
        if st.button("一键准备文献：索引 → 读正文 → 筛图页", type="primary", disabled=busy):
            start("prepare")
        left, right = st.columns(2)
        with left:
            st.markdown("#### 读取 PDF 正文")
            st.write(
                "处理已下载或已上传、尚未读完的 PDF，提取每页文字并缓存。扫描版可能没有可提取文字，需要后续读图。"
            )
            if st.button("读取待处理 PDF 正文", disabled=busy):
                start("parse", 500)
        with right:
            st.markdown("#### 筛选待读图页")
            st.write(
                "根据已提取文字和图像信息，生成待送视觉模型的页码清单。这一步只做计划，不理解图中曲线，也不收费。"
            )
            if st.button("更新 XPS 图页清单", disabled=busy):
                start("plan")
        plan = settings.workspace_root / "state" / "evidence_plan.csv"
        if plan.is_file():
            try:
                frame = pd.read_csv(plan)
                st.caption(
                    f"已有图页计划：{len(frame)} 篇文献 · {int(frame['selected_page_count'].sum())} 页。"
                )
                st.dataframe(
                    frame.rename(
                        columns={
                            "title": "文献",
                            "doi": "DOI",
                            "selected_pages": "待读页码",
                            "selected_page_count": "待读页数",
                            "pdf_pages": "PDF 总页数",
                        }
                    ),
                    hide_index=True,
                    width="stretch",
                )
            except (OSError, ValueError, KeyError):
                st.caption("图页清单为空或需更新。")
        st.info(
            "真正读图并提取页级证据：到“证据与智能体 → 视觉抽取”确认调用视觉模型后开始，可能产生费用。"
        )
        with st.expander("本地文件索引与最近操作"):
            if st.button("重新索引本地 PDF", disabled=busy):
                start("index")
            if last := st.session_state.get("last_literature_result"):
                st.json(last)

    with upload_tab:
        st.subheader("4. 补充无法下载的 PDF")
        manual = [row for row in documents if row["status"] == "manual"]
        _table(manual, filename="missing_literature.csv", key="missing_literature_csv")
        uploads = st.file_uploader("选择 PDF（可以多选）", type=["pdf"], accept_multiple_files=True)
        if st.button("保存并准备上传文献", disabled=busy or not uploads):
            inbox = settings.workspace_root / "inbox"
            inbox.mkdir(parents=True, exist_ok=True)
            saved = rejected = 0
            for upload in uploads or []:
                content = upload.getvalue()
                if not content.startswith(b"%PDF"):
                    rejected += 1
                    continue
                digest = hashlib.sha256(content).hexdigest()
                destination = inbox / f"{safe_filename(Path(upload.name).stem)}_{digest[:8]}.pdf"
                if not destination.exists():
                    destination.write_bytes(content)
                saved += 1
            st.session_state["last_literature_upload"] = {"saved": saved, "rejected": rejected}
            if saved:
                start("prepare")
        if last := st.session_state.get("last_literature_upload"):
            st.caption(
                f"最近上传：接受 {last['saved']} 个 PDF，拒绝 {last['rejected']} 个非 PDF 文件。"
            )
