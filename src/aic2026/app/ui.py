from __future__ import annotations

import html
import json
from pathlib import Path

import httpx
import streamlit as st
from pydantic import ValidationError

from aic2026.ingestion import load_manifest
from aic2026.models import Candidate, Query
from aic2026.submission import competition_answer, validate_candidates


class BackendRequestError(RuntimeError):
    """Raised when the UI cannot obtain a valid result from the backend."""
st.set_page_config(page_title="PEGASUS Agent Console", layout="wide")
st.markdown("""
<style>
.keyframe-name { font-size: 1.7rem; font-weight: 800; color: #0f172a; padding: .7rem 1rem;
  background: #f1f5f9; border-left: 7px solid #2563eb; border-radius: .35rem; word-break: break-all; }
.gallery-caption { font-size: .8rem; color: #475569; word-break: break-all; }
</style>
""", unsafe_allow_html=True)


def run_backend(
    task_type: str,
    query: Query,
    runtime: dict,
    backend_url: str,
):
    from aic2026.agent.types import AgentResult

    try:
        response = httpx.post(
            (
                f"{backend_url.rstrip('/')}"
                f"/tasks/{task_type}/run"
            ),
            json={
                "query_id": query.query_id,
                "text": query.text,
                "question": query.question,
                "events": query.events,
                "runtime": runtime,
            },
            timeout=300,
        )

        response.raise_for_status()
        payload = response.json()

        return AgentResult.model_validate(payload)

    except httpx.TimeoutException as exc:
        raise BackendRequestError(
            "Backend xử lý quá thời gian cho phép."
        ) from exc

    except httpx.HTTPStatusError as exc:
        status_code = exc.response.status_code

        try:
            detail = exc.response.json()
        except json.JSONDecodeError:
            detail = exc.response.text

        raise BackendRequestError(
            f"Backend trả HTTP {status_code}: {detail}"
        ) from exc

    except httpx.RequestError as exc:
        raise BackendRequestError(
            "Không thể kết nối tới backend. "
            "Hãy kiểm tra FastAPI có đang chạy không."
        ) from exc

    except json.JSONDecodeError as exc:
        raise BackendRequestError(
            "Backend không trả về JSON hợp lệ."
        ) from exc

    except ValidationError as exc:
        raise BackendRequestError(
            "Dữ liệu backend không đúng schema AgentResult."
        ) from exc


def parse_events(events_text: str) -> list[str]:
    """Parse ordered TRAKE events from one event per line."""

    events = [
        line.strip(" -•\t")
        for line in events_text.splitlines()
        if line.strip()
    ]

    # Also support semicolon-separated events entered on one line.
    # Do not split by commas because commas may belong to one event.
    if len(events) == 1 and ";" in events[0]:
        events = [
            part.strip()
            for part in events[0].split(";")
            if part.strip()
        ]

    return events


def build_query(
    query_type: str,
    text: str,
    question: str,
    events_text: str,
) -> Query:
    return Query(
        query_id="live-query",
        type=query_type,
        text=text.strip(),
        question=question.strip() or None,
        events=parse_events(events_text),
    )

@st.cache_data(show_spinner=False)
def load_keyframe_lookup(
    manifest_path: str,
) -> dict[tuple[str, int], str]:
    """Map each video/frame pair to its keyframe path."""

    records = load_manifest(
        Path(manifest_path)
    )

    return {
        (
            record.video_id,
            record.frame_id,
        ): record.keyframe_path
        for record in records
    }

def resolve_keyframe_path(
    stored_path: str,
    root: Path,
) -> Path:
    """Resolve an absolute or root-relative keyframe path."""

    path = Path(stored_path)

    if path.is_absolute():
        return path

    return root / path

def keyframe_file(candidate: Candidate, root: Path) -> Path | None:
    if not candidate.keyframe_path:
        return None
    path = Path(candidate.keyframe_path)
    return path if path.is_absolute() else root / path

def event_keyframe_file(
    video_id: str,
    frame_id: int,
    lookup: dict[tuple[str, int], str],
    root: Path,
) -> Path | None:
    """Resolve one TRAKE event frame through the active manifest."""

    stored_path = lookup.get(
        (
            video_id,
            frame_id,
        )
    )

    if stored_path is None:
        return None

    return resolve_keyframe_path(
        stored_path=stored_path,
        root=root,
    )

def render_trake_timeline(
    query: Query,
    candidate: Candidate,
    root: Path,
    manifest_path: str,
) -> None:
    """Render one ordered image for each TRAKE event."""

    event_frames = candidate.event_frames or []

    st.subheader(
        f"TRAKE timeline — {candidate.video_id}"
    )

    if not event_frames:
        st.error(
            "Candidate TRAKE không chứa event_frames."
        )
        return

    if len(event_frames) != len(query.events):
        st.error(
            "Số event frame không khớp với số event đầu vào: "
            f"{len(event_frames)} frame / "
            f"{len(query.events)} event."
        )
        return

    try:
        lookup = load_keyframe_lookup(
            manifest_path
        )
    except (
        FileNotFoundError,
        ValueError,
    ) as error:
        st.error(
            f"Không đọc được manifest: {error}"
        )
        return

    for offset in range(
        0,
        len(event_frames),
        3,
    ):
        end = min(
            offset + 3,
            len(event_frames),
        )

        columns = st.columns(
            end - offset
        )

        for column, event_index in zip(
            columns,
            range(offset, end),
            strict=True,
        ):
            frame_id = event_frames[event_index]
            event_text = query.events[event_index]

            path = event_keyframe_file(
                video_id=candidate.video_id,
                frame_id=frame_id,
                lookup=lookup,
                root=root,
            )

            with column:
                st.markdown(
                    f"### Event {event_index + 1}"
                )

                st.caption(event_text)

                st.markdown(
                    f"**Frame `{frame_id}`**"
                )

                if path and path.exists():
                    st.image(
                        str(path),
                        caption=(
                            f"{candidate.video_id} "
                            f"| frame {frame_id}"
                        ),
                        use_container_width=True,
                    )

                    link = file_link(path)

                    if link:
                        st.link_button(
                            "Mở ảnh",
                            link,
                            use_container_width=True,
                        )
                else:
                    st.warning(
                        "Không tìm thấy ảnh trong manifest."
                    )

                    if path:
                        st.caption(str(path))

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
        events_text = st.text_area(
            "Event theo thứ tự",
            height=115,
            placeholder=(
                "Chạy đà\n"
                "Giậm nhảy\n"
                "Bay qua xà\n"
                "Tiếp đất"
            ),
        )

        parsed_events = parse_events(events_text)

        st.caption(
            f"Đã nhận {len(parsed_events)} event."
        )

        if len(parsed_events) == 1:
            st.warning(
                "TRAKE hiện chỉ nhận một event. "
                "Hãy đặt mỗi event trên một dòng riêng "
                "hoặc phân cách bằng dấu chấm phẩy (;)."
            )

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
                st.session_state.update(
                    query=query,
                    candidates=result.candidates,
                    trace=result.trace,
                    selected_candidate=0,
                    raw_root=str(raw_root),
                    manifest_path=manifest_path,
                )
        except (
            BackendRequestError,
            ValidationError,
        ) as error:
            st.error(
                f"Không chạy được agent: {error}"
            )

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
        st.info(
            "Nhập query ở cột trái rồi bấm **Chạy Agent**. "
            "Kết quả hình ảnh sẽ chiếm phần lớn màn hình này."
        )

    else:
        query: Query = st.session_state["query"]
        candidates: list[Candidate] = st.session_state[
            "candidates"
        ]

        root = Path(
            st.session_state.get(
                "raw_root",
                "data/processed",
            )
        )

        if not candidates:
            st.warning(
                "Agent không trả candidate. "
                "Hãy kiểm tra index hoặc đổi query."
            )

        else:
            selected_index = min(
                st.session_state.get(
                    "selected_candidate",
                    0,
                ),
                len(candidates) - 1,
            )

            selected = candidates[selected_index]

            selected_path = keyframe_file(
                selected,
                root,
            )

            st.subheader(
                f"Đang kiểm tra candidate "
                f"#{selected_index + 1}"
            )

            # TRAKE candidate represents one video and contains
            # one ordered frame ID for each input event.
            if query.type == "trake":
                manifest_path = st.session_state.get(
                    "manifest_path"
                )

                if not manifest_path:
                    st.error(
                        "Không có manifest_path trong session. "
                        "Hãy chạy lại Agent."
                    )
                else:
                    render_trake_timeline(
                        query=query,
                        candidate=selected,
                        root=root,
                        manifest_path=str(
                            manifest_path
                        ),
                    )

            # Keep the existing single-frame interface
            # unchanged for KIS and QA.
            else:
                keyframe_name = (
                    selected_path.name
                    if selected_path
                    else "Không có keyframe path"
                )

                st.markdown(
                    (
                        '<div class="keyframe-name">'
                        f"{html.escape(keyframe_name)}"
                        "</div>"
                    ),
                    unsafe_allow_html=True,
                )

                st.caption(
                    str(selected_path)
                    if selected_path
                    else ""
                )

                selected_link = file_link(
                    selected_path
                )

                if selected_link:
                    st.link_button(
                        "Mở file keyframe",
                        selected_link,
                        use_container_width=False,
                    )

                if (
                    selected_path
                    and selected_path.exists()
                ):
                    st.image(
                        str(selected_path),
                        caption=(
                            f"{selected.video_id} "
                            f"| frame "
                            f"{selected.frame_id}"
                        ),
                        use_container_width=True,
                    )
                else:
                    st.warning(
                        "Không tìm thấy ảnh keyframe "
                        "theo path trong manifest."
                    )

            # Build the competition-format preview.
            if (
                query.type == "qa"
                and not selected.answer
            ):
                preview = {
                    "video_id": selected.video_id,
                    "frame_id": selected.frame_id,
                    "answer": "CẦN VLM/nhập tay",
                }
            else:
                preview = competition_answer(
                    query,
                    selected,
                )

            st.subheader(
                "Submission preview"
            )
            st.json(preview)

            st.divider()

            st.subheader(
                f"Gallery kết quả "
                f"({len(candidates)} candidate)"
            )

            for offset in range(
                0,
                len(candidates),
                4,
            ):
                end = min(
                    offset + 4,
                    len(candidates),
                )

                cards = st.columns(
                    end - offset
                )

                for column, index in zip(
                    cards,
                    range(offset, end),
                    strict=True,
                ):
                    item = candidates[index]

                    path = keyframe_file(
                        item,
                        root,
                    )

                    with column:
                        if (
                            path
                            and path.exists()
                        ):
                            st.image(
                                str(path),
                                use_container_width=True,
                            )
                        else:
                            st.caption(
                                "Không có ảnh đại diện"
                            )

                        if query.type == "trake":
                            event_frames = (
                                item.event_frames
                                or []
                            )

                            st.caption(
                                f"#{index + 1} "
                                f"| {item.video_id}\n"
                                f"{len(event_frames)} "
                                "event frames"
                            )

                            if event_frames:
                                st.caption(
                                    " → ".join(
                                        str(frame_id)
                                        for frame_id
                                        in event_frames
                                    )
                                )
                        else:
                            st.caption(
                                f"#{index + 1} "
                                f"| {item.video_id}\n"
                                f"frame {item.frame_id}"
                            )

                        item_link = file_link(
                            path
                        )

                        if item_link:
                            st.link_button(
                                "Link ảnh",
                                item_link,
                                use_container_width=True,
                            )

                        st.button(
                            "Kiểm tra",
                            key=f"pick-{index}",
                            on_click=select_candidate,
                            args=(index,),
                            use_container_width=True,
                        )

            with st.expander(
                "Trace agent"
            ):
                st.json(
                    [
                        item.model_dump()
                        for item
                        in st.session_state[
                            "trace"
                        ]
                    ]
                )
