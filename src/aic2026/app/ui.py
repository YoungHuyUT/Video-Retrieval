from __future__ import annotations

import html
import json
import urllib.parse
from pathlib import Path

import httpx
import streamlit as st
from pydantic import ValidationError

from aic2026.ingestion import load_manifest
from aic2026.models import Candidate, Query
from aic2026.submission import competition_answer, csv_row

import re as _re

# Ký tự có dấu tiếng Việt → dùng để nhận diện query tiếng Việt (cần dịch).
_VI_MARK_RE = _re.compile(
    r"[àáạảãâầấậẩẫăằắặẳẵđèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹ]",
    _re.IGNORECASE,
)


def _is_english(text: str) -> bool:
    """True nếu text KHÔNG chứa ký tự tiếng Việt có dấu."""
    return not bool(_VI_MARK_RE.search(text or ""))


class BackendRequestError(RuntimeError):
    """Raised when the UI cannot obtain a valid result from the backend."""


st.set_page_config(page_title="PEGASUS", layout="wide")

st.markdown(
    """
<style>
:root{
  --peg-violet:#a78bda; --peg-pink:#e3a3c4; --peg-cyan:#7fcdd9; --peg-amber:#f59e0b;
  --peg-ink:#3b3660; --peg-muted:#7c7894; --peg-line:rgba(167,139,218,.20);
}
/* Nền gradient tím - hồng - xanh đã hạ bão hòa (pastel) + glow nhẹ */
html, body, [data-testid="stAppViewContainer"] { font-family: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
[data-testid="stAppViewContainer"] {
  background:
    radial-gradient(900px 500px at 12% -5%, rgba(227,163,196,.16), transparent 60%),
    radial-gradient(800px 500px at 95% 0%, rgba(127,205,217,.14), transparent 60%),
    radial-gradient(700px 600px at 50% 110%, rgba(167,139,218,.16), transparent 60%),
    linear-gradient(160deg, #f7f4fc 0%, #fbf3f8 45%, #f1fafc 100%);
  background-attachment: fixed;
}

/* ---- Header (gradient pastel + sparkle) ---- */
.peg-header { position:relative; display:flex; align-items:center; gap:.7rem; overflow:hidden;
  background: linear-gradient(110deg, #a78bda 0%, #e3a3c4 50%, #7fcdd9 100%);
  padding:.6rem 1rem; border-radius:.8rem;
  box-shadow: 0 8px 20px rgba(167,139,218,.28); margin-bottom:.7rem; }
.peg-header::after { content:"✦ ✧ ❉ ✦"; position:absolute; top:.35rem; right:.7rem;
  font-size:.7rem; color:rgba(255,255,255,.7); letter-spacing:.3rem; }
.peg-logo { display:flex; align-items:center; justify-content:center;
  width:46px; height:46px; flex:0 0 auto; background:rgba(255,255,255,.30);
  border-radius:.65rem; border:1px solid rgba(255,255,255,.55);
  box-shadow: inset 0 0 8px rgba(255,255,255,.35); }
.peg-horse { width:34px; height:34px; display:block; }
.peg-titles { display:flex; flex-direction:column; line-height:1.05; flex:1 1 auto; }
.peg-word { font-size:1.35rem; font-weight:900; color:#fff; letter-spacing:.16em; margin:0;
  text-shadow: 0 2px 8px rgba(80,60,120,.25); }
.peg-sub { font-size:.66rem; color:rgba(255,255,255,.92); letter-spacing:.03em; margin-top:.1rem; }
.peg-badge { background:rgba(255,255,255,.28); color:#fff; font-size:.66rem; font-weight:700;
  padding:.28rem .55rem; border-radius:999px; border:1px solid rgba(255,255,255,.55);
  white-space:nowrap; box-shadow:0 1px 6px rgba(80,60,120,.12); }

/* ---- Topbar (glassmorphism) ---- */
.topbar { background:rgba(255,255,255,.55); backdrop-filter:blur(14px); -webkit-backdrop-filter:blur(14px);
  border:1px solid rgba(255,255,255,.7); border-radius:.8rem;
  padding:.6rem .8rem; margin-bottom:.7rem; box-shadow:0 4px 14px rgba(167,139,218,.12); }
.mini-label { font-size:.64rem; text-transform:uppercase; letter-spacing:.07em;
  color:#8b6fb5; margin-bottom:.15rem; font-weight:800; }

/* ---- Gallery cards (glass + gradient accent) ---- */
.hero-wrap { border-radius:.6rem; overflow:hidden; border:1px solid rgba(255,255,255,.8);
  box-shadow:0 2px 8px rgba(167,139,218,.14); transition: transform .2s ease, box-shadow .2s ease; }
.hero-wrap:hover { transform:translateY(-4px) scale(1.02); box-shadow:0 12px 24px rgba(227,163,196,.30); }
.hero-wrap img { display:block; width:100%; border-radius:0; }
.card-meta { font-size:.72rem; color:#4a4668; line-height:1.3; margin-top:.35rem;
  background:rgba(255,255,255,.7); backdrop-filter:blur(6px); -webkit-backdrop-filter:blur(6px);
  border:1px solid rgba(255,255,255,.8); border-left:3px solid #b9a0e0;
  border-radius:.5rem; padding:.4rem .5rem; }
.card-rank { font-weight:800; color:#c97aa6; }
.card-video { font-weight:700; color:var(--peg-ink); }
.card-answer { font-size:.7rem; background:linear-gradient(100deg,#f1fbfd,#fbeef7); color:#8a5a80;
  border-radius:.35rem; padding:.2rem .45rem; margin-top:.3rem; display:inline-block;
  max-width:100%; word-break:break-word; border:1px solid rgba(227,163,196,.30); }
.small-muted { font-size:.75rem; color:var(--peg-muted); }
.peg-pill { background:linear-gradient(100deg,#a78bda,#7fcdd9); color:#fff; font-size:.7rem; font-weight:700;
  padding:.28rem .55rem; border-radius:999px; box-shadow:0 2px 8px rgba(167,139,218,.28); }

/* ---- Grid / form elements (dịu màu) ---- */
.stButton > button[data-testid="baseButton-primary"] {
  background: linear-gradient(100deg,#a78bda 0%, #e3a3c4 55%, #7fcdd9 100%) !important;
  color:#fff !important; font-weight:800 !important; letter-spacing:.04em;
  border:none !important; border-radius:.65rem !important; padding:.55rem .5rem !important;
  box-shadow: 0 6px 16px rgba(227,163,196,.30) !important;
  transition: transform .15s ease, box-shadow .15s ease !important; }
.stButton > button[data-testid="baseButton-primary"]:hover {
  transform: translateY(-2px) !important;
  box-shadow: 0 10px 22px rgba(167,139,218,.42) !important; }

.stCheckbox > label { font-weight:700; color:var(--peg-ink); }
input[type="checkbox"]:checked { accent-color:#c97aa6; }

/* ---- Card clickable: toàn bộ ảnh là vùng bấm chọn (nút trong suốt phủ lên) ---- */
.frame-card { position: relative; border-radius:.6rem; }
.frame-card .stButton > button {
  position:absolute; top:0; left:0; width:100%; height:100%; min-height:100%;
  opacity:0; border:none; background:transparent; padding:0; margin:0; cursor:pointer; }
.frame-card.selected .hero-wrap {
  border:3px solid #c97aa6;
  box-shadow:0 0 0 3px rgba(201,122,166,.35), 0 12px 24px rgba(227,163,196,.30); }
.frame-card .frame-pick-hint {
  position:absolute; top:.35rem; right:.35rem; z-index:2;
  background:rgba(201,122,166,.92); color:#fff; font-size:.62rem; font-weight:800;
  padding:.12rem .4rem; border-radius:999px; opacity:0; transition:opacity .15s ease; }
.frame-card.selected .frame-pick-hint { opacity:1; }

[data-testid="stSlider"] { color:#8b6fb5; }

/* ---- Ô nhập liệu (text input) có viền đẹp ---- */
.stTextInput > div > div > input {
  border:1.5px solid #c9b6e8 !important; border-radius:.6rem !important;
  background:rgba(255,255,255,.85) !important; color:var(--peg-ink) !important;
  padding:.55rem .7rem !important; font-size:.9rem !important;
  box-shadow:0 1px 4px rgba(167,139,218,.15) !important;
  transition: border-color .15s ease, box-shadow .15s ease !important; }
.stTextInput > div > div > input::placeholder { color:#a89fc4 !important; }
.stTextInput > div > div > input:focus,
.stTextInput > div > div > input:focus-visible {
  border-color:#a78bda !important; outline:none !important;
  box-shadow:0 0 0 3px rgba(167,139,218,.28) !important; }
</style>
""",
    unsafe_allow_html=True,
)

# Logo PEGASUS: đầu KỲ LÂN (unicorn) — path chuẩn từ Lucide Lab (license ISC),
# vẽ nét (line-art) trắng trên nền xanh. Inline SVG, viewBox 0 0 24 24.
PEGASUS_UNICORN_SVG = """
<svg class="peg-horse" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg"
     stroke="#ffffff" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"
     aria-label="PEGASUS unicorn">
  <path d="m15.6 4.8 2.7 2.3M15.5 10S19 7 22 2c-6 2-10 5-10 5m-.5 5H11"/>
  <path d="M5 15a4 4 0 0 0 4 4h7.8l.3.3a3 3 0 0 0 4-4.46L12 7c0-3-1-5-1-5S8 3 8 7c-4 1-6 3-6 3"/>
  <path d="M2 4.5C4 3 6 3 6 3l2 4M6.14 17.8S4 19 2 22"/>
</svg>
"""


# ---------------------------------------------------------------------------
# Backend call
# ---------------------------------------------------------------------------
def run_backend(task_type: str, query: Query, runtime: dict, backend_url: str):
    from aic2026.agent.types import AgentResult

    try:
        response = httpx.post(
            f"{backend_url.rstrip('/')}/tasks/{task_type}/run",
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
        return AgentResult.model_validate(response.json())
    except httpx.TimeoutException as exc:
        raise BackendRequestError("Backend xử lý quá thời gian cho phép.") from exc
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json()
        except json.JSONDecodeError:
            detail = exc.response.text
        raise BackendRequestError(f"Backend trả HTTP {exc.response.status_code}: {detail}") from exc
    except httpx.RequestError as exc:
        raise BackendRequestError("Không thể kết nối tới backend. Hãy kiểm tra FastAPI có đang chạy không.") from exc
    except (json.JSONDecodeError, ValidationError) as exc:
        raise BackendRequestError("Backend không trả về dữ liệu hợp lệ.") from exc


def run_backend_qa_phase1(query: Query, runtime: dict, backend_url: str):
    """Phase-1 QA: chỉ retrieval, không VLM. Trả candidates ngay để hiển gallery."""
    from aic2026.agent.types import AgentResult

    try:
        response = httpx.post(
            f"{backend_url.rstrip('/')}/tasks/qa/candidates",
            json={
                "query_id": query.query_id,
                "text": query.text,
                "question": query.question,
                "events": query.events,
                "runtime": runtime,
            },
            timeout=120,  # retrieval thường xong trong 5-20s
        )
        response.raise_for_status()
        return AgentResult.model_validate(response.json())
    except httpx.TimeoutException as exc:
        raise BackendRequestError("Retrieval quá thời gian (120s).") from exc
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json()
        except json.JSONDecodeError:
            detail = exc.response.text
        raise BackendRequestError(f"Backend trả HTTP {exc.response.status_code}: {detail}") from exc
    except httpx.RequestError as exc:
        raise BackendRequestError("Không thể kết nối tới backend.") from exc
    except (json.JSONDecodeError, ValidationError) as exc:
        raise BackendRequestError("Backend không trả về dữ liệu hợp lệ.") from exc


def run_backend_qa_answers(
    question: str,
    candidates: list[Candidate],
    runtime: dict,
    backend_url: str,
) -> dict[int, str]:
    """Phase-2 QA: gửi candidates lên backend, nhận {vector_id: answer} từ VLM song song."""
    try:
        response = httpx.post(
            f"{backend_url.rstrip('/')}/tasks/qa/answers",
            json={
                "question": question,
                "candidates": [c.model_dump() for c in candidates],
                "runtime": runtime,
            },
            timeout=600,  # VLM có thể mất 2-5 phút với nhiều video
        )
        response.raise_for_status()
        # API trả {str(vector_id): answer} → chuyển thành {int: answer}
        raw = response.json()
        return {int(k): v for k, v in raw.items()}
    except httpx.TimeoutException as exc:
        raise BackendRequestError("VLM quá thời gian (600s).") from exc
    except httpx.HTTPStatusError as exc:
        try:
            detail = exc.response.json()
        except json.JSONDecodeError:
            detail = exc.response.text
        raise BackendRequestError(f"Backend trả HTTP {exc.response.status_code}: {detail}") from exc
    except httpx.RequestError as exc:
        raise BackendRequestError("Không thể kết nối tới backend.") from exc
    except (json.JSONDecodeError, ValidationError) as exc:
        raise BackendRequestError("Backend không trả về dữ liệu hợp lệ.") from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def parse_events(events_text: str) -> list[str]:
    events = [line.strip(" -•\t") for line in events_text.splitlines() if line.strip()]
    if len(events) == 1 and ";" in events[0]:
        events = [part.strip() for part in events[0].split(";") if part.strip()]
    return events


def build_query(query_type: str, text: str, question: str, events_text: str) -> Query:
    return Query(
        query_id="live-query",
        type=query_type or "kis",
        text=text.strip(),
        question=question.strip() or None,
        events=parse_events(events_text),
    )


@st.cache_data(show_spinner=False)
def load_keyframe_lookup(manifest_path: str) -> dict[tuple[str, int], str]:
    records = load_manifest(Path(manifest_path))
    return {(record.video_id, record.frame_id): record.keyframe_path for record in records}


@st.cache_data(show_spinner=False)
def load_asr_lookup(manifest_path: str) -> dict[tuple[str, int], list[str]]:
    try:
        records = load_manifest(Path(manifest_path))
        return {
            (record.video_id, record.frame_id): record.asr_text
            for record in records
            if getattr(record, "asr_text", None)
        }
    except Exception:
        return {}


def resolve_keyframe_path(stored_path: str, root: Path) -> Path:
    path = Path(stored_path)
    if path.is_absolute() or path.exists():
        return path
    for candidate in (
        root / path,
        Path.cwd() / path,
        Path.cwd() / "data" / "raw" / path,
        Path.cwd() / "data" / "processed" / path,
    ):
        if candidate.exists():
            return candidate
    return root / path


def keyframe_file(candidate: Candidate, root: Path) -> Path | None:
    if not candidate.keyframe_path:
        return None
    return resolve_keyframe_path(candidate.keyframe_path, root)


def event_keyframe_file(video_id: str, frame_id: int, lookup: dict[tuple[str, int], str], root: Path) -> Path | None:
    stored_path = lookup.get((video_id, frame_id))
    if stored_path is None:
        return None
    return resolve_keyframe_path(stored_path, root)


def file_link(path: Path | None) -> str | None:
    if not path or not path.exists():
        return None
    return path.resolve().as_uri()


def build_btc_json(query: Query, candidates: list[Candidate], selected_idx: list[int]) -> list[dict]:
    """Chỉ lấy candidate được tick, GIỮ NGUYÊN THỨ TỰ CHỌN (không sort score).

    Thứ tự xuất = thứ tự user click (frame mới click nhất ở đầu).
    """
    picked = [candidates[i] for i in selected_idx if 0 <= i < len(candidates)]
    return [competition_answer(query, c) for c in picked]


def build_btc_csv(query: Query, candidates: list[Candidate], selected_idx: list[int]) -> str:
    """Chỉ lấy candidate được tick, GIỮ NGUYÊN THỨ TỰ CHỌN (không sort score).

    Thứ tự xuất = thứ tự user click (frame mới click nhất ở đầu). Trả về
    text CSV thuần túy (không header row, LF), sẵn sàng tải về nộp bài.
    """
    picked = [candidates[i] for i in selected_idx if 0 <= i < len(candidates)]
    return "\n".join(csv_row(query, c) for c in picked)


# ---------------------------------------------------------------------------
# Gallery renderers (với checkbox chọn) — định nghĩa TRƯỚC phần gọi gallery
# để tránh NameError khi Streamlit thực thi tuần tự từ trên xuống.
# ---------------------------------------------------------------------------
def _toggle(idx: int, nonce: int):
    # Checkbox tự quản lý trạng thái qua key "sel_{nonce}_{idx}";
    # mình đồng bộ ngược lại vào danh sách "selected" CÓ THỨ TỰ.
    # Frame CHỌN ĐẦU TIÊN nằm TRÊN CÙNG (top 1) của danh sách xuất ra — vì nó
    # là ưu tiên cao nhất. Các frame chọn sau nối TIẾP xuống dưới (giữ nguyên
    # thứ tự chọn đầu→cuối = trên→dưới). Bỏ tick thì loại khỏi danh sách.
    # `nonce` thay đổi mỗi lần chạy query mới → key checkbox hoàn toàn mới →
    # không bao giờ nhớ tick của query trước (reset cứng, không phụ thuộc xóa state).
    checked = st.session_state.get(f"sel_{nonce}_{idx}", False)
    s = st.session_state["selected"]
    if checked:
        if idx not in s:
            if not s:
                s.insert(0, idx)      # frame đầu chọn → trên cùng (top 1)
            else:
                s.append(idx)          # chọn sau → nối tiếp phía dưới
    else:
        if idx in s:
            s.remove(idx)


def _render_frame_gallery(query: Query, visible: list[tuple[int, Candidate]], root: Path, selected: list[int]) -> None:
    # nonce thay đổi mỗi query mới → key checkbox "sel_{nonce}_{idx}" luôn mới,
    # không nhớ tick của query trước (reset cứng).
    nonce = int(st.session_state.get("_query_nonce", 0))
    manifest_path = st.session_state.get("manifest_path")
    asr_lookup = load_asr_lookup(manifest_path) if manifest_path else {}
    st.subheader(f"Gallery ({len(visible)} ảnh)")
    grid = st.columns(5)
    for pos, (idx, item) in enumerate(visible):
        path = keyframe_file(item, root)
        with grid[pos % 5]:
            if path and path.exists():
                st.markdown(f'<div class="gallery-card-container" data-video-id="{html.escape(item.video_id)}">', unsafe_allow_html=True)
                st.markdown('<div class="hero-wrap">', unsafe_allow_html=True)
                img_url = f"http://127.0.0.1:8000/images?path={urllib.parse.quote(str(path))}"
                st.image(img_url, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="gallery-card-container" data-video-id="{html.escape(item.video_id)}">', unsafe_allow_html=True)
                st.caption("Không có ảnh")
            answer_html = ""
            if query.type == "qa":
                ans = item.answer or "⏳ đang phân tích..."
                answer_html = f'<div class="card-answer">Đáp: {html.escape(str(ans))}</div>'
            asr_snippets = asr_lookup.get((item.video_id, item.frame_id), [])
            asr_html = ""
            if asr_snippets:
                joined_asr = " ".join(asr_snippets)
                short_asr = (joined_asr[:65] + "...") if len(joined_asr) > 65 else joined_asr
                asr_html = (
                    f'<div style="font-size:0.75rem; color:#3730a3; background:#e0e7ff; '
                    f'border-radius:4px; padding:2px 4px; margin-top:3px; line-height:1.2;" '
                    f'title="{html.escape(joined_asr)}">🗣️ {html.escape(short_asr)}</div>'
                )
            st.markdown(
                f'<div class="card-meta">'
                f'<span class="card-rank">#{idx + 1}</span> '
                f'<span class="card-video">{html.escape(item.video_id)}</span><br>'
                f"frame {item.frame_id} · score {item.score:.3f}"
                f"{asr_html}"
                f"{answer_html}</div>",
                unsafe_allow_html=True,
            )
            # Checkbox chọn nằm DƯỚI CÙNG, cách xa ảnh (tránh bấm nhầm khi lướt).
            st.checkbox("Chọn", key=f"sel_{nonce}_{idx}", value=(idx in selected), on_change=_toggle, args=(idx, nonce), label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)


def _render_trake_gallery(query: Query, visible: list[tuple[int, Candidate]], root: Path, selected: list[int]) -> None:
    try:
        manifest_path = st.session_state.get("manifest_path")
        lookup = load_keyframe_lookup(manifest_path) if manifest_path else {}
    except (FileNotFoundError, ValueError):
        lookup = {}

    st.subheader(f"Gallery TRAKE ({len(visible)} video)")
    for idx, item in visible:
        is_sel = idx in selected
        st.markdown(
            f'<div class="card-meta"><span class="card-rank">#{idx + 1}</span> '
            f'<span class="card-video">{html.escape(item.video_id)}</span> · '
            f"score {item.score:.3f}</div>",
            unsafe_allow_html=True,
        )
        event_frames = item.event_frames or []
        if not event_frames:
            st.caption("Candidate này không có event_frames.")
        else:
            if len(event_frames) != len(query.events):
                st.caption(f"Số event frame không khớp: {len(event_frames)}/{len(query.events)}.")
            rows = [event_frames[i : i + 4] for i in range(0, len(event_frames), 4)]
            for row in rows:
                cols = st.columns(len(row))
                for col, frame_id in zip(cols, row):
                    path = event_keyframe_file(item.video_id, frame_id, lookup, root)
                    ev_idx = event_frames.index(frame_id)
                    ev_text = query.events[ev_idx] if ev_idx < len(query.events) else f"event {ev_idx + 1}"
                    with col:
                        if path and path.exists():
                            st.markdown('<div class="hero-wrap">', unsafe_allow_html=True)
                            st.image(str(path), use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                        else:
                            st.caption("Không có ảnh")
                        st.caption(f"**{ev_text}**\nframe {frame_id}")
        nonce = int(st.session_state.get("_query_nonce", 0))
        st.checkbox("Chọn video này", key=f"sel_{nonce}_{idx}", value=(idx in selected), on_change=_toggle, args=(idx, nonce))
        st.divider()


# ---------------------------------------------------------------------------
# HEADER
# ---------------------------------------------------------------------------
st.markdown(
    '<div class="peg-header">'
    '<div class="peg-logo">'
    f"{PEGASUS_UNICORN_SVG}"
    "</div>"
    '<div class="peg-titles">'
    '<h1 class="peg-word">PEGASUS</h1>'
    '<span class="peg-sub">AI Challenge 2026</span>'
    "</div>"
    '<span class="peg-badge">KHÔNG THẮNG THÌ THÔI</span>'
    "</div>",
    unsafe_allow_html=True,
)


# ===========================================================================
# HÀNG TRÊN: NHẬP & CHẠY (3 khung cân đối: Luồng – Query – Chạy Agent)
# ===========================================================================
st.markdown('<div class="topbar">', unsafe_allow_html=True)

# --- Chọn luồng (NGOÀI form để Enter luôn hoạt động) ---
q_col, rest_col = st.columns([1, 4.4], gap="small")
with q_col:
    st.markdown('<div class="mini-label">Luồng</div>', unsafe_allow_html=True)
    query_type = st.segmented_control(
        "Luồng",
        ["kis", "qa", "trake"],
        default="kis",
        key="query_type",
        label_visibility="collapsed",
    )

# Input phụ (Q&A / TRAKE) — cũng ngoài form để cập nhật ngay khi gõ
question = ""
events_text = ""
if query_type == "qa":
    question = st.text_input("Câu hỏi Q&A", placeholder="Ví dụ: Có bao nhiêu người?", label_visibility="collapsed",
                             key="query_question")
if query_type == "trake":
    events_text = st.text_area(
        "Event TRAKE (mỗi event 1 dòng, theo thứ tự thời gian trong video)",
        placeholder=(
            "A person enters the room\n"
            "The person sits down\n"
            "The person starts speaking"
        ),
        height=140,
        label_visibility="collapsed",
        key="query_events",
        help=(
            "TRAKE tìm một CHUỖI sự kiện theo thứ tự thời gian (khác KIS chỉ 1 khung). "
            "Nhập mỗi event trên 1 dòng, từ sớm → muộn. Số dòng = số frame xuất ra CSV. "
            "Viết tiếng Anh, ngắn gọn: 'a person opens the fridge'."
        ),
    )

# --- Ô Query + nút Chạy trong CÙNG một form (Enter = submit) ---
with rest_col:
    with st.form("query_form", border=False):
        t_col, run_col = st.columns([3.4, 1], gap="small")
        with t_col:
            st.markdown('<div class="mini-label">Query</div>', unsafe_allow_html=True)
            text = st.text_input("Query", placeholder="Dán đề BTC vào đây (Enter để chạy)",
                                 label_visibility="collapsed", key="query_text")
        with run_col:
            st.markdown('<div class="mini-label">&nbsp;</div>', unsafe_allow_html=True)
            submitted = st.form_submit_button("Chạy Agent", type="primary", use_container_width=True)
            if submitted:
                st.session_state["_run_agent"] = True

    # Nút tùy chọn: dịch VI→EN trước khi retrieval. Mặc định TẮT (dùng query nguyên bản).
    # Bật khi query nhập bằng tiếng Việt và muốn đảm bảo đầu vào CLIP (tiếng Anh).
    translate_query = st.checkbox(
        "🌐 Dịch VI→EN",
        value=st.session_state.get("cfg_translate", False),
        key="cfg_translate",
        help=(
            "Dịch truy vấn tiếng Việt sang tiếng Anh trước khi retrieval (CLIP text "
            "encoder là tiếng Anh nên đầu vào tiếng Anh cho kết quả tốt hơn). "
            "Mặc định TẮT — dùng query nguyên bản. Bật nếu bạn nhập tiếng Việt và "
            "muốn hệ thống tự dịch. Khi tắt mà vẫn nhập tiếng Việt, retrieval có thể "
            "ra frame sai do CLIP không hiểu tiếng Việt."
        ),
    )

if st.session_state.get("_run_agent"):
    st.session_state["_run_agent"] = False
    if not text.strip():
        st.warning("Hãy nhập query trước khi chạy Agent.")
    else:
        try:
            query = build_query(query_type, text, question, events_text)
            if query_type == "trake" and not query.events:
                st.error("TRAKE cần ít nhất một event.")
            else:
                runtime = {
                    "manifest_path": st.session_state.get("cfg_manifest", "data/processed/official_manifest.jsonl"),
                    "features_path": st.session_state.get("cfg_features", "data/processed/official_features.npy"),
                    "clip_model": st.session_state.get("cfg_clip_model", "ViT-B-32"),
                    "clip_pretrained": st.session_state.get("cfg_clip", "openai"),
                    "llm_model": st.session_state.get("cfg_llm", "qwen3.5:4b"),
                    "ollama_url": st.session_state.get("cfg_ollama", "http://127.0.0.1:11434"),
                    "metadata_filter": st.session_state.get("cfg_filter", ""),
                    "translate_query": bool(st.session_state.get("cfg_translate", False)),
                    "coarse_top_k": int(st.session_state.get("cfg_coarse_top_k", 200)),
                    "vlm_backend": st.session_state.get("cfg_vlm_backend", "florence"),
                    "vlm_model": st.session_state.get("cfg_vlm_model", "microsoft/Florence-2-base-ft"),
                    "vlm_top_videos": int(st.session_state.get("cfg_vlm_top_videos", 20)),
                    "vlm_max_workers": int(st.session_state.get("cfg_vlm_max_workers", 8)),
                    "vlm_timeout": int(st.session_state.get("cfg_vlm_timeout", 120)),
                }
                backend_url = st.session_state.get("cfg_backend", "http://127.0.0.1:8000")

                if query_type == "qa":
                    # ---- Phase 1: retrieval nhanh (giống KIS), hiển gallery ảnh ngay ----
                    with st.spinner("🔍 Đang tìm kiếm... (sẽ thấy ảnh ngay)"):
                        result = run_backend_qa_phase1(query, runtime, backend_url)
                    st.session_state.update(
                        query=query,
                        candidates=result.candidates,
                        trace=result.trace,
                        raw_root=st.session_state.get("cfg_root", "data/processed"),
                        manifest_path=runtime["manifest_path"],
                        selected=[],
                        qa_answers_ready=False,   # phase 2 chưa chạy
                        qa_question=question,
                        qa_runtime=runtime,
                        qa_backend_url=backend_url,
                        _query_nonce=st.session_state.get("_query_nonce", 0) + 1,
                    )
                    st.rerun()
                else:
                    # KIS / TRAKE: giữ nguyên luồng cũ
                    with st.spinner("Agent đang retrieval..."):
                        result = run_backend(query_type, query, runtime, backend_url)
                    st.session_state.update(
                        query=query,
                        candidates=result.candidates,
                        trace=result.trace,
                        raw_root=st.session_state.get("cfg_root", "data/processed"),
                        manifest_path=runtime["manifest_path"],
                        selected=[],
                        _query_nonce=st.session_state.get("_query_nonce", 0) + 1,
                    )

                # Bản dịch (giữ nguyên cho mọi luồng)
                translated = None
                translation_source = None
                for step in result.trace:
                    if step.step == "translate":
                        try:
                            _td = json.loads(step.detail)
                            translation_source = _td.get("source")
                            _changes = _td.get("changes") or {}
                            _text_change = _changes.get("text") or {}
                            translated = _text_change.get("to") or _td.get("to")
                        except (json.JSONDecodeError, ValueError, AttributeError, TypeError):
                            translated = None
                        break
                st.session_state["translated_query"] = translated
                if translation_source == "offline_fallback":
                    st.toast(
                        f"⚠️ LLM dịch lỗi — đang dùng bản dịch offline: {translated or text}",
                        icon="🌐",
                    )
                elif translated:
                    st.toast(f"✅ Đã dịch VI→EN: {translated}", icon="🌐")
                elif text.strip() and _is_english(text):
                    st.toast("ℹ️ Query đã là tiếng Anh — không cần dịch", icon="🌐")
                elif not bool(st.session_state.get("cfg_translate", False)):
                    st.toast("ℹ️ Chưa bật 'Dịch VI→EN' — dùng query nguyên bản", icon="🌐")
                else:
                    st.toast("⚠️ Không dịch được (LLM lỗi?) — dùng nguyên bản tiếng Việt", icon="🌐")
                    del st.session_state[k]
        except (BackendRequestError, ValidationError) as error:
            st.error(f"Không chạy được agent: {error}")

with st.expander("Cấu hình nâng cao"):
    st.text_input("Backend API", "http://127.0.0.1:8000", key="cfg_backend")
    st.text_input("Manifest", "data/processed/official_manifest.jsonl", key="cfg_manifest")
    st.text_input("Feature .npy", "data/processed/official_features.npy", key="cfg_features")
    st.text_input("Root keyframe", "data/processed", key="cfg_root")
    st.text_input("CLIP model", "ViT-B-32", key="cfg_clip_model", help="ViT-B-32, ViT-L-14, ViT-L-14-quickgelu, ViT-B-16-SigLIP, ViT-SO400M-14-SigLIP-384")
    st.text_input("CLIP pretrained", "openai", key="cfg_clip", help="openai, metaclip_fullcc, metaclip_400m, laion2b_s32b_b82k, webli")
    st.text_input("Ollama model", "qwen3.5:4b", key="cfg_llm")
    st.text_input("Ollama URL", "http://127.0.0.1:11434", key="cfg_ollama")
    st.text_input(
        "Metadata / Object filter (phân cách dấu phẩy)",
        "",
        key="cfg_filter",
        help=(
            "Lọc video theo nhãn Object và Metadata (object_labels, title, description "
            "trong manifest). Ví dụ: 'trong cửa hàng,máy tính' sẽ chỉ giữ các video có "
            "chứa những nhãn này trước khi chạy retrieval/TRAKE. Để trống = không lọc."
        ),
    )
    st.number_input(
        "TRAKE: số video lọc nhanh (coarse_top_k)",
        min_value=0,
        max_value=5000,
        value=200,
        step=50,
        key="cfg_coarse_top_k",
        help=(
            "Giới hạn số video đưa vào DP alignment cho luồng TRAKE. "
            "Lọc nhanh (coarse) theo độ tương đồng video-level trước, chỉ top-K "
            "video liên quan nhất mới xếp hạng chi tiết. Nhỏ = nhanh (dataset lớn), "
            "lớn = đầy đủ nhưng chậm/OOM. 0 = xét hết mọi video."
        ),
    )
    # Cấu hình VLM phục vụ Phase-2 QA
    st.selectbox("QA VLM Backend", ["florence", "none"], index=0, key="cfg_vlm_backend",
                 help="Florence-2 thay thế QwenVLM/Ollama (chạy CPU, không cần GPU/Ollama server).")
    st.text_input("QA VLM Model", "microsoft/Florence-2-base-ft", key="cfg_vlm_model")
    st.number_input("QA VLM: Số video giới hạn gửi (vlm_top_videos)", min_value=1, max_value=200, value=20, key="cfg_vlm_top_videos")
    st.number_input("QA VLM: Số thread song song (vlm_max_workers)", min_value=1, max_value=32, value=8, key="cfg_vlm_max_workers")
    st.number_input("QA VLM: Timeout mỗi video (giây)", min_value=10, max_value=600, value=120, key="cfg_vlm_timeout")

st.markdown("</div>", unsafe_allow_html=True)  # đóng .topbar

# Hiển thị bản dịch tiếng Anh (nếu có) ngay dưới hàng nhập để dễ thấy bước trans
_translated = st.session_state.get("translated_query")
_query_text_raw = st.session_state.get("query_text", "")
if _translated:
    st.markdown(
        f'<div style="margin:.1rem 0 .5rem;">'
        f'<span class="peg-pill">🌐 Đã dịch (VI→EN)</span> '
        f'<span class="small-muted">{html.escape(_translated)}</span></div>',
        unsafe_allow_html=True,
    )
elif _query_text_raw and _is_english(_query_text_raw):
    # Đã chạy, query gốc thực sự là tiếng Anh (không có dấu tiếng Việt)
    st.markdown(
        f'<div style="margin:.1rem 0 .5rem;">'
        f'<span class="peg-pill">🌐 Đã kiểm tra dịch</span> '
        f'<span class="small-muted">Query đã là tiếng Anh — không cần dịch.</span></div>',
        unsafe_allow_html=True,
    )

st.markdown("<hr style='margin:.3rem 0 1rem;'>", unsafe_allow_html=True)


# ===========================================================================
# PHẦN DƯỚI: GALLERY ẢNH (TRUNG TÂM) + CHỌN + XUẤT JSON
# ===========================================================================
if "candidates" not in st.session_state:
    st.info("Nhập query ở hàng trên rồi bấm **Chạy Agent**. Kết quả (tới 100 keyframe) hiện nguyên màn hình bên dưới để bạn xem và chọn.")
else:
    query: Query = st.session_state["query"]
    candidates: list[Candidate] = st.session_state["candidates"]
    root = Path(st.session_state.get("raw_root", "data/processed"))
    # Đảm bảo session selected là list có thứ tự hợp lệ (không phải set cũ)
    if "selected" not in st.session_state or not isinstance(st.session_state["selected"], list):
        st.session_state["selected"] = []
    selected: list[int] = st.session_state["selected"]

    if not candidates:
        st.warning("Agent không trả candidate. Hãy kiểm tra index hoặc đổi query.")
    else:
        total = len(candidates)
        # candidates đã sort score giảm dần từ agent (hoặc sort lại cho chắc)
        candidates.sort(key=lambda c: c.score, reverse=True)

        st.markdown(
            f'<span class="peg-pill">Đã trả {total} candidate</span> '
            f'<span class="small-muted">— tick ảnh để chọn, xuất JSON chuẩn BTC. '
            f'Score cao nhất đứng đầu.</span>',
            unsafe_allow_html=True,
        )

        # Thanh công cụ: lọc video + giới hạn + chọn tất cả
        col_filter, col_limit, col_all = st.columns([2, 1, 1])
        with col_filter:
            if query.type == "TRAKE":
                video_options = ["Tất cả"] + sorted({c.video_id for c in candidates})
                selected_video = st.selectbox("Lọc theo video", video_options, index=0)
            else:
                selected_video = "Tất cả"
                video_options = sorted({c.video_id for c in candidates})
                opts = "".join([f'<option value="{html.escape(vid)}">{html.escape(vid)}</option>' for vid in video_options])
                filter_html = f"""
                <div style="margin-bottom: 0px;">
                    <label for="client-video-filter" style="font-weight: bold; color: var(--peg-ink); font-size: 0.85rem; display: block; margin-bottom: 4px;">Lọc theo video (Tức thì)</label>
                    <select id="client-video-filter" onchange="window.filterKeyframes(this.value)" style="
                        width: 100%;
                        padding: 6px 10px;
                        border-radius: 0.5rem;
                        border: 1.5px solid #c9b6e8;
                        background: rgba(255,255,255,0.85);
                        color: var(--peg-ink);
                        font-size: 0.85rem;
                        height: 38px;
                    ">
                        <option value="Tất cả">Tất cả</option>
                        {opts}
                    </select>
                </div>
                
                <script>
                window.filterKeyframes = function(selectedVideo) {{
                    const containers = document.querySelectorAll('.gallery-card-container');
                    containers.forEach(container => {{
                        const column = container.closest('[data-testid="column"]');
                        if (!column) return;
                        if (selectedVideo === 'Tất cả' || container.getAttribute('data-video-id') === selectedVideo) {{
                            column.style.display = 'block';
                        }} else {{
                            column.style.display = 'none';
                        }}
                    }});
                }};
                // Gọi lại bộ lọc khi Streamlit rerun để đồng bộ trạng thái hiển thị
                setTimeout(() => {{
                    const dropdown = document.getElementById('client-video-filter');
                    if (dropdown) {{
                        window.filterKeyframes(dropdown.value);
                    }}
                }}, 100);
                </script>
                """
                st.markdown(filter_html, unsafe_allow_html=True)
        with col_limit:
            if total <= 1:
                max_show = total
            else:
                max_show = st.number_input(
                    "Số ảnh hiển thị",
                    min_value=1,
                    max_value=total,
                    value=min(20, total),
                    step=1,
                    key="max_show_input",
                )
        with col_all:
            if st.button("Chọn tất cả hiển thị", use_container_width=True):
                # Giữ NGUYÊN thứ tự những gì user đã chọn trước đó (đầu danh sách).
                # Chỉ nối thêm các frame đang hiển thị mà CHƯA được chọn, theo
                # thứ tự hiển thị (score giảm dần). Không xáo trộn, không đẩy
                # các lựa chọn cũ xuống dưới.
                vis_idx = [
                    i for i, c in enumerate(candidates)
                    if (selected_video == "Tất cả" or c.video_id == selected_video)
                ][:max_show]
                for i in vis_idx:
                    if i not in selected:
                        selected.append(i)
                st.rerun()

        # Hiển thị: ĐƯA CÁC FRAME ĐÃ CHỌN LÊN ĐẦU (theo thứ tự chọn), phần còn lại
        # giữ nguyên thứ tự score bên dưới. Nhờ đó khi user tick vài frame, chúng
        # nhảy lên trên cùng ngay, dễ thấy; và danh sách xuất ra cũng bắt đầu bằng
        # những frame đã chọn (frame chọn đầu = trên cùng = top 1).
        filtered = [
            (i, c) for i, c in enumerate(candidates)
            if (selected_video == "Tất cả" or c.video_id == selected_video)
        ]
        sel_set = set(selected)
        chosen = [(i, c) for (i, c) in filtered if i in sel_set]
        rest = [(i, c) for (i, c) in filtered if i not in sel_set]
        # Sắp xếp lại chosen theo thứ tự trong `selected` (đầu tiên = trên cùng)
        order = {idx: pos for pos, idx in enumerate(selected)}
        chosen.sort(key=lambda ic: order.get(ic[0], len(selected)))
        visible = (chosen + rest)[:max_show]

        # ---- Gallery ảnh ----
        if query.type == "TRAKE":
            _render_trake_gallery(query, visible, root, selected)
        else:
            _render_frame_gallery(query, visible, root, selected)

        # ---- Phase 2: Chạy VLM song song cho QA ----
        if query.type == "qa" and not st.session_state.get("qa_answers_ready", False):
            vlm_q = st.session_state.get("qa_question")
            vlm_rt = st.session_state.get("qa_runtime")
            vlm_url = st.session_state.get("qa_backend_url")
            if vlm_q and vlm_rt and vlm_url:
                with st.spinner("🤖 VLM đang phân tích các video song song để tìm đáp án..."):
                    try:
                        answers = run_backend_qa_answers(
                            question=vlm_q,
                            candidates=candidates,
                            runtime=vlm_rt,
                            backend_url=vlm_url,
                        )
                        for c in candidates:
                            if c.vector_id is not None and c.vector_id in answers:
                                c.answer = answers[c.vector_id]
                        st.session_state["qa_answers_ready"] = True
                        st.rerun()
                    except Exception as e:
                        st.error(f"Lỗi khi chạy VLM song song: {e}")

        # ---- Thanh xuất CSV chuẩn nộp bài BTC ----
        st.divider()
        n_sel = len(selected)
        st.markdown(f"**Đã chọn {n_sel} ảnh** (xuất theo thứ tự chọn — frame click trước ở đầu).")
        if n_sel > 0:
            btc_csv = build_btc_csv(query, candidates, selected)
            col_dl, col_cl = st.columns([1, 2])
            with col_dl:
                st.download_button(
                    "📥 Tải CSV chuẩn BTC",
                    btc_csv,
                    file_name=f"query-{query.query_id}-{query.type}.csv",
                    mime="text/csv",
                    use_container_width=True,
                )
            with col_cl:
                st.code(btc_csv, language="text")
        else:
            st.caption("Chưa chọn ảnh nào. Tick vào ảnh ở gallery để xuất.")

        with st.expander("Trace agent (plan / retrieve / judge)"):
            st.json([item.model_dump() for item in st.session_state["trace"]])

