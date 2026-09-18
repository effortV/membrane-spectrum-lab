from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from .llm import SiliconFlowClient
from .agent import ResearchAgent
from .membrane import MembraneAnalyzer, save_xps_image
from .research import (
    TASKS,
    ROLES,
    ROLE_LABELS,
    RelationshipEvaluator,
    current_research_table,
    field_catalog,
    load_role_overrides,
    prepare_research_data,
    save_role_overrides,
)
from .security import safe_error
from .presentation import normalize_math_markdown, render_model_output


def _start(settings, manager, owner, label, kind, worker, timeout=None):
    try:
        manager.start(
            owner,
            label,
            kind,
            worker,
            timeout=timeout or settings.action_timeout_seconds,
            journal_root=settings.workspace_root / "state" / "tasks",
        )
        st.rerun()
    except Exception as exc:
        st.error(safe_error(exc))


def _research_result(result: dict, settings, db, manager, owner) -> None:
    st.subheader("关系研究结果 · 分组外推验证")
    st.write(result["relationship"])
    st.caption(
        f"目标：{result['target']} · 样本 {result['cohort']['rows_scored']} · 文献组 {result['cohort']['groups']}"
    )
    st.dataframe(pd.DataFrame(result["out_of_fold"]), hide_index=True, width="stretch")
    for warning in result["warnings"]:
        st.caption(warning)
    path = Path(result["run_dir"]) / "predictions.csv"
    if path.exists():
        predictions = pd.read_csv(path)
        families = predictions["feature_set"].astype(str) + " / " + predictions["model"].astype(str)
        selected = st.selectbox("查看同一验证集上的预测", sorted(families.unique()))
        view = predictions[families == selected]
        st.plotly_chart(
            px.scatter(
                view,
                x="observed",
                y="predicted",
                color="doi_group",
                hover_data=["record_id", "fold"],
                labels={"observed": "原表实测值", "predicted": "未见文献的预测值"},
            ),
            width="stretch",
        )
        st.download_button(
            "下载逐样本外推预测 CSV",
            path.read_bytes(),
            file_name="relationship_predictions.csv",
            mime="text/csv",
        )
    st.download_button(
        "下载关系研究报告 JSON",
        json.dumps(result, ensure_ascii=False, indent=2).encode(),
        file_name="relationship_summary.json",
        mime="application/json",
    )
    with st.expander("Agent × ML：结合文献解释关系、提出新指标与否证实验"):
        st.caption(
            "解释已有研究结果，不把最高 R² 当成机制；新指标先作为假说，再到“证据与智能体”生成、计算并在独立数据验证。"
        )
        confirmed = st.checkbox(
            "确认调用主模型解读该关系（可能计费）", key="confirm_relationship_agent"
        )
        if st.button(
            "结合文献解读这次研究", disabled=not confirmed or not settings.siliconflow_api_key
        ):
            question = f"请分析 experiment_id={result['experiment_id']} 的 {result['relationship']} 研究。先调用 get_experiment 精确读取该 ID，再调用 get_column_inventory、get_health_report 和 search_evidence，引用对应运行目录及 evidence_id/DOI/页码。区分已测量的制备、XPS、独立结构与性能，解释变量组差异、混杂、负结果和证据边界。提出不同于 HABD/HCD/HAD、交联度、O2/O1 的可计算新 XPS 物理描述符方向、机制链和否证实验，不凭外层最高分挑选或宣称新机制；本次结果已经被查看，后续候选必须独立验证。"

            def worker(context):
                with SiliconFlowClient(
                    settings, db, progress=context.update, check_cancelled=context.check
                ) as llm:
                    answer = ResearchAgent(settings, db, llm).ask(question)
                directory = Path(result["run_dir"])
                if not directory.resolve().is_relative_to(
                    (settings.workspace_root / "runs").resolve()
                ):
                    raise ValueError("研究报告路径不在当前数据目录。")
                (directory / "agent_interpretation.md").write_text(answer, encoding="utf-8")
                return {"experiment_id": result["experiment_id"], "answer": answer}

            _start(
                settings, manager, owner, "Agent 解读关系研究", "research_interpretation", worker
            )
        previous = st.session_state.get("last_research_interpretation")
        answer = None
        if previous and previous["experiment_id"] == result["experiment_id"]:
            answer = previous["answer"]
        else:
            saved = Path(result["run_dir"]) / "agent_interpretation.md"
            if (
                saved.resolve().is_relative_to((settings.workspace_root / "runs").resolve())
                and saved.is_file()
            ):
                answer = saved.read_text(encoding="utf-8")
        if answer:
            st.caption("已保存的 Agent 解读：刷新页面或查看历史结果无需再次调用模型。")
            render_model_output(answer)
            st.download_button(
                "下载排版后的 Agent 解读 Markdown",
                normalize_math_markdown(answer),
                file_name="agent_interpretation.md",
                mime="text/markdown",
            )
            with st.expander("原始回答（仅用于核查）"):
                st.code(answer, language="markdown")


def page_research(settings, db, manager, owner) -> None:
    st.title("制备—XPS—结构—性能 · 关系研究")
    st.caption(
        "从 NF / RO 原始表重新训练，不导入旧 ML 结果。比较变量组的解释和预测能力，不预设新指标一定有效。"
    )
    path = current_research_table(settings)
    if not path.exists():
        st.info("请先在系统设置中运行离线数据审计。")
        return
    frame = pd.read_csv(path, low_memory=False)
    symbols = [
        json.loads(row["spec_json"]).get("symbol")
        for row in db.rows("SELECT spec_json FROM hypotheses WHERE status='calculable'")
    ]
    overrides = load_role_overrides(settings)
    catalog = field_catalog(frame, overrides, symbols)
    with st.expander("字段分类与单位核对（不修改原 Excel）"):
        st.caption(
            "制备包括单体、基膜、浓度、IP 与后处理；结构指独立测得的厚度、粗糙度、接触角、孔径、电位等。HABD／交联度等是结构代理量，不作独立结构目标。不同单位列保留，不自动合并。"
        )
        st.caption(
            "孔径／MWCO 等的来源需查阅原文；若由同一截留率等性能反算，不能再当作该性能的独立预测变量。"
        )
        editor = st.data_editor(
            catalog,
            hide_index=True,
            width="stretch",
            disabled=[name for name in catalog.columns if name != "scientific_role"],
            column_config={
                "scientific_role": st.column_config.SelectboxColumn(
                    "字段类别", options=list(ROLES), required=True
                )
            },
            key="scientific_role_editor",
        )
        if st.button("保存字段分类"):
            save_role_overrides(
                settings, dict(zip(editor["column"], editor["scientific_role"], strict=True))
            )
            st.success("分类已保存；身份／来源／质量字段仍禁止作为物理变量。")
            st.rerun()
    task = st.selectbox("研究关系", list(TASKS), format_func=lambda key: TASKS[key]["label"])
    spec = TASKS[task]
    datasets = st.multiselect(
        "膜体系",
        sorted(frame["dataset"].dropna().unique()),
        default=["NF"],
        key="relationship_datasets",
    )
    selected_frame = frame[frame["dataset"].isin(datasets)]
    # Determine availability in the selected membrane family, not from the other family.
    available = (
        field_catalog(selected_frame, overrides, symbols) if len(selected_frame) else catalog
    )
    targets = (
        available[
            (available["scientific_role"] == spec["target"]) & (available["numeric_count"] >= 3)
        ]
        .sort_values("numeric_count", ascending=False)["column"]
        .tolist()
    )
    if not targets:
        st.info("该关系暂没有可训练的独立数值目标，请补充数据或核对字段分类。")
        return
    target = st.selectbox("实测目标（保持原表单位）", targets, key=f"relationship_target_{task}")
    groups = {}
    for role in spec["inputs"]:
        options = (
            available[
                (available["scientific_role"] == role)
                & (available["non_null"] >= 3)
                & (available["column"] != target)
            ]
            .sort_values("non_null", ascending=False)["column"]
            .tolist()
        )
        groups[role] = st.multiselect(
            f"{ROLE_LABELS[role]}输入",
            options,
            default=options[:2],
            key=f"relationship_features_{task}_{role}",
        )
    conditions = available[
        (available["scientific_role"] == "test_condition") & (available["non_null"] >= 3)
    ]["column"].tolist()
    controls = st.multiselect(
        "性能测试条件（在所有比较模型中共同控制）",
        conditions,
        default=conditions[:2] if spec["target"] == "performance" else [],
        key=f"relationship_controls_{task}",
    )
    measured = st.checkbox("只使用所选 XPS／结构输入有实测值的样本（推荐）", value=True)
    unresolved = st.checkbox("包含未关联 DOI 的样本（仅探索；不能证明独立文献外推）", value=False)
    cohort = None
    try:
        data, feature_sets, cohort = prepare_research_data(
            frame,
            catalog,
            task,
            target,
            groups,
            controls,
            datasets=datasets,
            allow_unresolved=unresolved,
            measured_intermediates=measured,
        )
        st.write(f"可研究样本：{len(data)} · 独立来源组：{cohort['groups']}")
        st.dataframe(
            pd.DataFrame(
                [
                    {"比较组": key, "变量": "；".join(names) if names else "训练文献目标均值"}
                    for key, names in feature_sets.items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        with st.expander("样本筛选／缺失值说明"):
            st.json(cohort)
            st.caption(
                "制备类别支持单体等文本变量；编码、数值填补及超参数选择仅使用当前训练文献。相同 DOI 不跨训练与测试。"
            )
    except ValueError as exc:
        st.info(str(exc))
    a, b, c = st.columns(3)
    outer = a.number_input("外层文献分组折数", 3, 10, 3)
    inner = b.number_input("内层调参折数", 2, 8, 2)
    models = c.multiselect(
        "模型", ["ridge", "random_forest", "extra_trees"], default=["ridge", "random_forest"]
    )
    enough = cohort and cohort["rows_scored"] >= 12 and cohort["groups"] >= 3 and bool(models)
    if st.button("开始关系研究（本地 ML，不调用 LLM）", type="primary", disabled=not enough):

        def worker(context):
            return RelationshipEvaluator(
                settings, db, progress=context.update, check_cancelled=context.check
            ).evaluate(
                path,
                task,
                target,
                groups,
                controls,
                datasets=datasets,
                models=models,
                outer_splits=int(outer),
                inner_splits=int(inner),
                allow_unresolved=unresolved,
                measured_intermediates=measured,
            )

        _start(settings, manager, owner, spec["label"], "relationship", worker, timeout=1800)
    if "last_relationship_result" in st.session_state:
        _research_result(st.session_state["last_relationship_result"], settings, db, manager, owner)
    history = db.rows(
        "SELECT experiment_id,created_at,metrics_json FROM experiments WHERE decision='relationship_requires_external_validation' ORDER BY created_at DESC LIMIT 30"
    )
    with st.expander("已保存的关系研究（浏览器刷新后仍保留）"):
        if history:
            index = st.selectbox(
                "历史研究",
                range(len(history)),
                format_func=lambda idx: (
                    f"{history[idx]['created_at']} · {json.loads(history[idx]['metrics_json'])['relationship']}"
                ),
            )
            if st.button("查看历史结果"):
                st.session_state["last_relationship_result"] = json.loads(
                    history[index]["metrics_json"]
                )
                st.rerun()
        else:
            st.caption("尚未运行新版关系研究。")


def _membrane_result(report: dict) -> None:
    interpretation = report["interpretation"]
    st.subheader("该样品的条件性解释")
    render_model_output(interpretation["summary"])
    st.caption("以下结果需要人工核验，不自动写入 NF／RO 或进入 ML；不是定量膜性能预测。")
    with st.expander("逐图可见读数和不确定性", expanded=True):
        st.json(report["figure_reading"])
    labels = {
        "findings": "化学组成与图像发现",
        "conditional_structure": "可能的结构关联",
        "conditional_performance": "可能的性能趋势",
        "alternatives": "其他解释",
        "missing_measurements": "还需要哪些测量",
        "suggested_experiments": "建议验证实验",
    }
    for key, label in labels.items():
        if interpretation[key]:
            st.markdown(f"### {label}")
            for item in interpretation[key]:
                render_model_output("- " + item)
    with st.expander("检索到的真实文献证据（含 DOI 与页码，模型抽取需核验）"):
        st.dataframe(pd.DataFrame(report["evidence"]), hide_index=True, width="stretch")
    directory = Path(report["run_dir"])
    for filename, mime in (("report.md", "text/markdown"), ("report.json", "application/json")):
        if (directory / filename).exists():
            st.download_button(
                f"下载 {filename}",
                normalize_math_markdown((directory / filename).read_text(encoding="utf-8"))
                if filename.endswith(".md")
                else (directory / filename).read_bytes(),
                file_name=filename,
                mime=mime,
            )


def page_images(settings, db, manager, owner) -> None:
    st.title("上传 XPS · 分析这个膜")
    st.caption(
        "便宜的视觉模型读图 → 主模型结合已读取文献解释。支持同一样品的全谱、N1s、O1s、C1s；建议上传带坐标和拟合标注的清晰图片。"
    )
    uploads = st.file_uploader(
        "XPS 图（每次最多 4 张，每张 20 MiB）",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        max_upload_size=20,
        key="xps_images",
    )
    preview_valid = True
    images = []
    if uploads:
        for upload in uploads[:4]:
            try:
                image = save_xps_image(settings, upload.name, upload.getvalue())
                images.append(image)
                st.image(image["model_input"], caption=upload.name, width=500)
            except Exception as exc:
                preview_valid = False
                st.error(safe_error(exc))
    a, b = st.columns(2)
    kind = a.selectbox("膜体系", ["NF", "RO", "其他／未知"])
    material = b.text_input("样品名称／聚合物与改性材料")
    preparation = st.text_area("制备信息（单体、浓度单位、IP 与后处理等）")
    test = st.text_area("已有独立测量／测试条件（可选，注明单位；未知留空）")
    purpose = st.text_input("最想了解的问题", "可能的化学结构变化、性能趋势和还需要验证什么？")
    confirmed = st.checkbox("确认调用视觉与主模型分析（可能计费，结果需要核验）")
    if st.button(
        "分析这组 XPS 图",
        type="primary",
        disabled=not uploads
        or not preview_valid
        or len(uploads) > 4
        or not confirmed
        or not settings.siliconflow_api_key,
    ):
        try:
            context = {
                "membrane_type": kind,
                "sample": material,
                "preparation": preparation,
                "independent_measurements": test,
                "question": purpose,
            }

            def worker(task):
                with SiliconFlowClient(
                    settings, db, progress=task.update, check_cancelled=task.check
                ) as llm:
                    return MembraneAnalyzer(
                        settings, db, llm, progress=task.update, check_cancelled=task.check
                    ).analyze(images, context)

            _start(settings, manager, owner, "上传 XPS 膜分析", "membrane", worker)
        except Exception as exc:
            st.error(safe_error(exc))
    if "last_membrane_result" in st.session_state:
        _membrane_result(st.session_state["last_membrane_result"])
    history = db.rows(
        "SELECT path,created_at FROM artifacts WHERE kind='membrane_analysis' ORDER BY created_at DESC LIMIT 30"
    )
    with st.expander("已保存的膜分析"):
        if history:
            index = st.selectbox(
                "历史分析", range(len(history)), format_func=lambda idx: history[idx]["created_at"]
            )
            if st.button("打开已保存报告"):
                path = Path(history[index]["path"])
                if path.is_relative_to(settings.workspace_root / "runs") and path.exists():
                    st.session_state["last_membrane_result"] = json.loads(
                        path.read_text(encoding="utf-8")
                    )
                    st.rerun()
        else:
            st.caption("尚无上传图像分析报告。")
