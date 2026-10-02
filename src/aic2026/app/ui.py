from __future__ import annotations

from pathlib import Path

import streamlit as st
from streamlit.components.v1 import html as st_html

# PEGASUS Retrieval Console — giao diện chính được nhúng từ dashboard.html.
# File này nằm cùng thư mục với ui.py (src/aic2026/app/dashboard.html).
# Logo con ngựa (PEGASUS unicorn) nằm bên trong dashboard.html (thẻ .brand).

st.set_page_config(page_title="PEGASUS", layout="wide")

# Bỏ giới hạn chiều rộng và padding mặc định của Streamlit để dashboard
# (iframe) dàn tận viền, không còn chừa lề 2 bên.
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] > .main { padding: 0 !important; }
    [data-testid="stAppViewContainer"] .block-container {
        max-width: 100% !important;
        padding: 0 !important;
        padding-top: 0 !important;
    }
    /* iframe nhúng dashboard sát viền */
    .stCustomComponentV1 iframe { width: 100% !important; border: 0 !important; }
    header[data-testid="stHeader"] { display: none !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

_DASHBOARD_PATH = Path(__file__).with_name("dashboard.html")


def _load_dashboard() -> str:
    if _DASHBOARD_PATH.exists():
        return _DASHBOARD_PATH.read_text(encoding="utf-8")
    return (
        "<!doctype html><html><body style='font-family:system-ui;padding:2rem'>"
        "<h1>PEGASUS</h1><p>Không tìm thấy dashboard.html kế bên ui.py.</p></body></html>"
    )


def main() -> None:
    # Nhúng toàn bộ dashboard (tự chứa, cuộn độc lập 2 bên bên trong).
    # Đặt iframe cao đúng bằng cửa sổ trình duyệt; dashboard tự quản lý scroll
    # của sidebar và workspace, nên trang Streamlit không cần cuộn.
    st_html(_load_dashboard(), height=800, scrolling=False)


if __name__ == "__main__":
    main()
