from __future__ import annotations

import html
import json
from pathlib import Path

import streamlit as st

from aic2026.models import Candidate, Query
from aic2026.submission import competition_answer, validate_candidates

st.set_page_config(page_title="PEGASUS Agent Console", layout="wide")
st.markdown("""
<style>
.keyframe-name { font-size: 1.7rem; font-weight: 800; color: #0f172a; padding: .7rem 1rem;
  background: #f1f5f9; border-left: 7px solid #2563eb; border-radius: .35rem; word-break: break-all; }
.gallery-caption { font-size: .8rem; color: #475569; word-break: break-all; }
</style>
""", unsafe_allow_html=True)


def run_backend(task_type: str, query: Query, runtime: dict, backend_url: str):
    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("Chạy `uv sync` để cài httpx.") from exc
    response = httpx.post(
        f"{backend_url.rstrip('/')}/tasks/{task_type}/run",
        json={"query_id": query.query_id, "text": query.text, "question": query.question, "events": query.events, "runtime": runtime},
        timeout=300,
    )
    response.raise_for_status()
    from aic2026.agent.types import AgentResult
    return AgentResult.model_validate(response.json())


def build_query(query_type: str, text: str, question: str, events_text: str) -> Query:
    events = [line.strip(" -•\t") for line in events_text.splitlines() if line.strip()]
    return Query(query_id="live-query", type=query_type, text=text.strip(), question=question.strip() or None, events=events)


def keyframe_file(candidate: Candidate, root: Path) -> Path | None:
    if not candidate.keyframe_path:
        return None
    path = Path(candidate.keyframe_path)
    return path if path.is_absolute() else root / path


# def file_link(path: Path | None) -> str | None:
#     return path.as_uri() if path and path.exists() else None

def file_link(path: Path | None) -> str | None:
    if not path or not path.exists():
        return None
    return path.resolve().as_uri()

def select_candidate(index: int) -> None:
    st.session_state["selected_candidate"] = index


st.title("PEGASUS - Agent tìm kiếm video")
st.caption("Konichiwaiiii.")

controls, gallery = st.columns([1, 3], gap="large")

with controls:
    st.subheader("1. Chọn luồng")
    if "query_type" not in st.session_state:
        st.session_state["query_type"] = "kis"
    for task, label in [("kis", "KIS"), ("qa", "Q&A"), ("trake", "TRAKE")]:
        if st.button(label, use_container_width=True, type="primary" if st.session_state["query_type"] == task else "secondary"):
            st.session_state["query_type"] = task
    query_type = st.session_state["query_type"]
    st.caption({"kis": "Tìm frame theo mô tả.", "qa": "Tìm frame rồi trả lời.", "trake": "Căn chỉnh chuỗi event."}[query_type])

    st.subheader("2. Nhập query")
    text = st.text_area("Đề BTC", height=115, placeholder="Dán nguyên văn câu hỏi vào đây", label_visibility="collapsed")
    question = ""
    events_text = ""
    if query_type == "qa":
        question = st.text_input("Câu hỏi Q&A", placeholder="Ví dụ: Có bao nhiêu người?")
    if query_type == "trake":
        events_text = st.text_area("Event theo thứ tự", height=115, placeholder="Chạy đà\nGiậm nhảy\nBay qua xà\nTiếp đất")

    with st.expander("Cấu hình backend/index", expanded=False):
        backend_url = st.text_input("Backend API", "http://127.0.0.1:8000")
        manifest_path = st.text_input("Manifest", "data/processed/derived_manifest.jsonl")
        features_path = st.text_input("Feature .npy", "data/processed/derived_features.npy")
        raw_root = Path(st.text_input("Root keyframe", "data/raw")).resolve()
        encoder_name = st.selectbox("Text encoder", ["siglip2", "clip"])
        encoder_model = st.text_input("SigLIP model", "google/siglip2-base-patch16-224")
        llm_model = st.text_input("Ollama model", "qwen3:8b")
        ollama_url = st.text_input("Ollama URL", "http://127.0.0.1:11434")
    if "raw_root" not in locals():
        # Values rendered inside an expander are still initialized on every Streamlit pass.
        raw_root = Path("data/processed")

    if st.button("Chạy Agent", type="primary", use_container_width=True, disabled=not text.strip()):
        try:
            query = build_query(query_type, text, question, events_text)
            if query_type == "trake" and not query.events:
                st.error("TRAKE cần ít nhất một event theo thứ tự.")
            else:
                runtime = {"manifest_path": manifest_path, "features_path": features_path, "encoder": encoder_name, "encoder_model": encoder_model, "llm_model": llm_model, "ollama_url": ollama_url}
                with st.spinner("Agent đang retrieval và điều phối task..."):
                    result = run_backend(query_type, query, runtime, backend_url)
                st.session_state.update(query=query, candidates=result.candidates, trace=result.trace, selected_candidate=0, raw_root=str(raw_root))
        except Exception as error:
            st.error(f"Không chạy được agent: {error}")

    if "candidates" in st.session_state:
        query: Query = st.session_state["query"]
        candidates: list[Candidate] = st.session_state["candidates"]
        if query.type == "qa" and not all(item.answer for item in candidates):
            st.divider()
            manual_answer = st.text_input("Answer Q&A thủ công")
            if manual_answer and st.button("Áp dụng answer", use_container_width=True):
                for item in candidates:
                    if not item.answer:
                        item.answer = manual_answer
                st.rerun()
        try:
            ranked = validate_candidates(query, candidates)
            payload = {"query_id": query.query_id, "type": query.type, "answers": [competition_answer(query, item) for item in ranked]}
            st.download_button("Tải submission preview", json.dumps(payload, ensure_ascii=False, indent=2), file_name=f"{query.type}_answers.json", mime="application/json", use_container_width=True)
        except ValueError as error:
            st.caption(f"Chưa thể export: {error}")

with gallery:
    if "candidates" not in st.session_state:
        st.info("Nhập query ở cột trái rồi bấm **Chạy Agent**. Kết quả hình ảnh sẽ chiếm phần lớn màn hình này.")
    else:
        query: Query = st.session_state["query"]
        candidates: list[Candidate] = st.session_state["candidates"]
        root = Path(st.session_state.get("raw_root", "data/processed"))
        if not candidates:
            st.warning("Agent không trả candidate. Hãy kiểm tra index hoặc đổi query.")
        else:
            selected_index = min(st.session_state.get("selected_candidate", 0), len(candidates) - 1)
            selected = candidates[selected_index]
            selected_path = keyframe_file(selected, root)
            st.subheader(f"Đang kiểm tra candidate #{selected_index + 1}")
            st.markdown(f'<div class="keyframe-name">{html.escape(selected_path.name if selected_path else "Không có keyframe path")}</div>', unsafe_allow_html=True)
            st.caption(str(selected_path) if selected_path else "")
            if file_link(selected_path):
                st.link_button("Mở file keyframe", file_link(selected_path), use_container_width=False)
            if selected_path and selected_path.exists():
                st.image(str(selected_path), caption=f"{selected.video_id} | frame {selected.frame_id}", use_container_width=True)
            else:
                st.warning("Không tìm thấy ảnh keyframe theo path trong manifest.")
            st.json(competition_answer(query, selected) if query.type != "qa" or selected.answer else {"video_id": selected.video_id, "frame_id": selected.frame_id, "answer": "CẦN VLM/nhập tay"})

            st.divider()
            st.subheader(f"Gallery kết quả ({len(candidates)} candidate)")
            for offset in range(0, len(candidates), 4):
                cards = st.columns(4)
                for column, index in zip(cards, range(offset, min(offset + 4, len(candidates)))):
                    item = candidates[index]
                    path = keyframe_file(item, root)
                    with column:
                        if path and path.exists():
                            st.image(str(path), use_container_width=True)
                        else:
                            st.caption("Không có ảnh")
                        st.caption(f"#{index + 1} | {item.video_id}\nframe {item.frame_id}")
                        if file_link(path):
                            st.link_button("Link ảnh", file_link(path), use_container_width=True)
                        st.button("Kiểm tra", key=f"pick-{index}", on_click=select_candidate, args=(index,), use_container_width=True)

            with st.expander("Trace agent"):
                st.json([item.model_dump() for item in st.session_state["trace"]])
