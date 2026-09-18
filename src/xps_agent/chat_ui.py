"""Server-persisted conversation UI, including multi-browser/restart recovery."""

from __future__ import annotations

import json
import streamlit as st

from .conversations import ConversationStore, STATUS_LABELS, start_conversation_task
from .security import safe_error
from .presentation import render_model_output


@st.fragment(run_every=2)
def _history(db, key: str, journal_root) -> None:
    store = ConversationStore(db)
    store.recover(journal_root)
    turns = store.turns(key)
    if not turns:
        st.info("在下方提出研究问题。后续问题可以直接说“刚才那个指标”或“继续分析”。")
    for turn in turns:
        with st.chat_message("user"):
            st.markdown(turn["question"])
        with st.chat_message("assistant"):
            if turn["status"] == "completed":
                render_model_output(turn["answer"])
                st.caption(
                    f"{turn['model']} · 带入前文 {turn['history_turns']} 轮 · {turn['finished_at']}"
                )
            elif turn["status"] == "running":
                st.info("正在后台回答；可以切换页面或刷新，完整回答会自动保存到此会话。")
            else:
                st.warning(
                    f"{STATUS_LABELS[turn['status']]}：{turn['error'] or '未保存完整回答。'}"
                )
                st.caption("问题已保留，未自动重复请求。需要继续时，请明确重新提交。")


def render_conversations(settings, db, manager, owner: str) -> None:
    store = ConversationStore(db)
    journal_root = settings.workspace_root / "state" / "tasks"
    store.recover(journal_root)
    st.subheader("研究会话 · 自动保存")
    st.caption(
        "同一会话连续问答，切换会话不会混入其他研究的前文。刷新、关闭网页或重启后，从会话列表继续。"
    )
    show_archived = st.checkbox("查看已归档会话", key="chat_show_archived")

    def new_conversation():
        st.session_state["active_conversation_id"] = store.create()
        st.session_state["chat_show_archived"] = False

    st.button("新建会话", key="chat_new", on_click=new_conversation)
    conversations = store.list(archived=show_archived)
    if not conversations and not show_archived:
        store.create()
        conversations = store.list()
    if not conversations:
        st.info("没有已归档会话。归档只隐藏会话，不删除历史。")
        return
    options = {row["conversation_id"]: row for row in conversations}
    if st.session_state.get("active_conversation_id") not in options:
        st.session_state["active_conversation_id"] = next(iter(options))
    key = st.selectbox(
        "已保存的会话",
        list(options),
        key="active_conversation_id",
        format_func=lambda value: (
            f"{options[value]['title']} · {options[value]['turns']} 轮 · {value[:8]}"
        ),
    )
    if st.session_state.get("evidence_tab") == "智能体问答":
        st.query_params["chat"] = key
    conversation = store.get(key)
    if conversation["archived"]:

        def restore_conversation():
            store.archive(key, False)
            st.session_state["chat_show_archived"] = False

        st.button("恢复此会话", key=f"chat_restore_{key}", on_click=restore_conversation)
    with st.expander("会话设置、长期研究备注与导出"):
        with st.form(f"chat_settings_{key}"):
            title = st.text_input("会话名称", conversation["title"], max_chars=120)
            memory = st.text_area(
                "长期研究备注",
                conversation["memory"],
                max_chars=6000,
                help="填写要持续保留的研究目标、膜类型、条件、实验/证据 ID。每轮带入，不会自动生成或将推测当成事实。不要填写 API 密钥。",
            )
            if st.form_submit_button("保存会话设置"):
                try:
                    store.update(key, title=title, memory=memory)
                    st.rerun()
                except ValueError as exc:
                    st.error(safe_error(exc))
        left, right = st.columns(2)
        left.download_button(
            "导出完整对话 Markdown",
            store.markdown(key).encode(),
            file_name=f"conversation_{key[:8]}.md",
            mime="text/markdown",
            key=f"chat_export_md_{key}",
        )
        right.download_button(
            "导出完整对话 JSON",
            json.dumps(store.export(key), ensure_ascii=False, indent=2).encode(),
            file_name=f"conversation_{key[:8]}.json",
            mime="application/json",
            key=f"chat_export_json_{key}",
        )
        st.caption(f"服务器保存位置：{db.path}；导出包含问题和公开回答，不包含密钥或模型私有推理。")
        if not conversation["archived"] and st.button(
            "归档此会话（不删除）", key=f"chat_archive_{key}"
        ):
            try:
                store.archive(key)
                st.rerun()
            except ValueError as exc:
                st.error(safe_error(exc))
    _history(db, key, journal_root)
    if conversation["archived"]:
        return
    previous = store.context(key)
    st.caption(
        f"下一问将带入最近 {previous['history_turns']} 轮完整前文及长期研究备注（最多 20 轮 / 8 万字符）；所有历史均保留在服务器。"
    )
    if previous["omitted_turns"]:
        st.warning(
            f"更早的 {previous['omitted_turns']} 轮不会自动带入模型。需要持续保留的内容请放入长期研究备注；也可以从完整历史中复制具体问题、证据或实验 ID。"
        )
    question_key, confirm_key = f"chat_question_{key}", f"chat_confirm_{key}"
    st.text_area(
        "问题",
        key=question_key,
        max_chars=6000,
        placeholder="基于现有证据，当前最值得优先验证的新描述符是哪一个？",
    )
    confirmed = st.checkbox("确认本次智能体问答调用主模型", key=confirm_key)

    def submit():
        try:
            start_conversation_task(
                settings, db, manager, owner, key, st.session_state[question_key]
            )
            st.session_state[question_key] = ""
            st.session_state[confirm_key] = False
            st.session_state.pop("chat_submit_error", None)
        except Exception as exc:
            st.session_state["chat_submit_error"] = safe_error(exc)

    st.button(
        "询问智能体",
        on_click=submit,
        key=f"chat_send_{key}",
        disabled=not confirmed or not st.session_state[question_key].strip(),
    )
    if st.session_state.get("chat_submit_error"):
        st.error(st.session_state["chat_submit_error"])
