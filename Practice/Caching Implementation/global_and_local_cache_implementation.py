import sys

import streamlit as st

from rag_cache import (
    ask,
    get_redis_status,
    redis_reset,
    LOCAL_CACHE_ENABLED,
    GLOBAL_CACHE_ENABLED,
    L1_CACHE_TTL,
    L2_CACHE_TTL,
)

# =========================================================
# Page Configuration
# =========================================================

st.set_page_config(
    page_title="RAG Cache Demo",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# Custom Styles
# =========================================================

st.markdown(
    """
    <style>
    .cache-pipeline {
        display: flex;
        align-items: center;
        gap: 8px;
        margin-bottom: 1rem;
    }
    .cache-step {
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        border: 1px solid #ddd;
        background: #f8f9fa;
        color: #555;
    }
    .cache-step.active {
        background: #0d6efd;
        color: white;
        border-color: #0d6efd;
    }
    .cache-step.hit-l1 {
        background: #198754;
        color: white;
        border-color: #198754;
    }
    .cache-step.hit-l2 {
        background: #0dcaf0;
        color: #000;
        border-color: #0dcaf0;
    }
    .cache-step.miss {
        background: #ffc107;
        color: #000;
        border-color: #ffc107;
    }
  .answer-box {
        background: #f0f4ff;
        border: 1px solid #c8d8ff;
        border-radius: 10px;
        padding: 16px 20px;
        margin-top: 8px;
        line-height: 1.6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# Session State
# =========================================================

if "history" not in st.session_state:
    st.session_state.history = []

if "clear_question_input" not in st.session_state:
    st.session_state.clear_question_input = False

# =========================================================
# Helpers
# =========================================================

SOURCE_META = {
    "L1 Local Cache": {
        "badge": "⚡ L1 HIT",
        "color": "green",
        "pipeline_class": "hit-l1",
        "message": "Answer served from L1 Local Cache (fastest)",
    },
    "L2 Redis Cache": {
        "badge": "🚀 L2 HIT",
        "color": "blue",
        "pipeline_class": "hit-l2",
        "message": "Answer served from L2 Redis Cache",
    },
    "RAG + LLM": {
        "badge": "🤖 CACHE MISS",
        "color": "orange",
        "pipeline_class": "miss",
        "message": "Answer generated using RAG + LLM (full pipeline)",
    },
}


def render_pipeline(active_step: str):
    steps = [
        ("L1 Local Cache", "hit-l1"),
        ("L2 Redis Cache", "hit-l2"),
        ("RAG + LLM", "miss"),
    ]

    html_parts = ["<div class='cache-pipeline'>"]
    for label, css_class in steps:
        is_active = label == active_step
        cls = f"cache-step {css_class}" if is_active else "cache-step"
        html_parts.append(f"<span class='{cls}'>{label}</span>")
        if label != "RAG + LLM":
            html_parts.append("<span>→</span>")
    html_parts.append("</div>")
    st.markdown("".join(html_parts), unsafe_allow_html=True)


def process_question(question: str):
    with st.spinner("Processing your question..."):
        result = ask(question)
    st.session_state.history.insert(0, {"question": question, **result})


# =========================================================
# Sidebar
# =========================================================

with st.sidebar:
    st.header("⚙️ Cache Configuration")

    st.checkbox("L1 Local Cache", value=LOCAL_CACHE_ENABLED, disabled=True)
    st.checkbox("L2 Redis Cache", value=GLOBAL_CACHE_ENABLED, disabled=True)

    st.divider()

    st.markdown(f"**L1 TTL:** {L1_CACHE_TTL} seconds")
    st.markdown(f"**L2 TTL:** {L2_CACHE_TTL} seconds")

    st.divider()

    redis_status = get_redis_status()

    if redis_status["connected"]:
        st.success("🟢 Redis Connected")
        st.caption(f"URL: `{redis_status['url']}`")
    else:
        st.error("🔴 Redis Disconnected")
        st.caption("L2 cache is skipped. L1 and RAG still work.")
        if redis_status["error"]:
            st.warning(f"Error: {redis_status['error']}")
            if "No module named 'redis'" in redis_status["error"]:
                st.code(
                    f"{sys.executable} -m pip install redis",
                    language="powershell",
                )
                st.caption(
                    "This is a missing Python package, not an ENV variable. "
                    "Run the command above, then restart Streamlit."
                )
        if st.button("Retry Redis connection", use_container_width=True):
            redis_reset()
            st.rerun()

    st.divider()

    st.subheader("🧪 How to test caching")
    st.markdown(
        """
        1. Ask a question → expect **CACHE MISS** (RAG + LLM)
        2. Ask the **same question** again → expect **L1 HIT** (fast)
        3. Restart the app, ask again → **L2 HIT** if Redis is running
        """
    )

    if st.session_state.history:
        if st.button("Clear history", use_container_width=True):
            st.session_state.history = []
            st.rerun()

# =========================================================
# Header
# =========================================================

st.title("🚀 RAG Multi-Level Cache")
st.caption("Ask questions and watch L1 → L2 → RAG + LLM in action")

# =========================================================
# Ask Section
# =========================================================

st.subheader("💬 Ask a Question")

sample_questions = [
    "What technologies does the company use?",
    "What is the refund policy?",
    "Who are the employees?",
]

with st.expander("Sample questions (click to use)", expanded=False):
    cols = st.columns(len(sample_questions))
    for i, sample in enumerate(sample_questions):
        if cols[i].button(sample, key=f"sample_{i}", use_container_width=True):
            process_question(sample)
            st.rerun()

# Clear widget state before the text area is created
if st.session_state.clear_question_input:
    st.session_state.question_input = ""
    st.session_state.clear_question_input = False

question = st.text_area(
    "Your question",
    placeholder="Example: What is the refund policy?",
    height=80,
    label_visibility="collapsed",
    key="question_input",
)

btn_col1, btn_col2, _ = st.columns([1, 1, 4])

with btn_col1:
    ask_clicked = st.button("Ask", type="primary", use_container_width=True)

with btn_col2:
    clear_input = st.button("Clear input", use_container_width=True)

if clear_input:
    st.session_state.clear_question_input = True
    st.rerun()

if ask_clicked:
    if not question.strip():
        st.warning("Please enter a question.")
    else:
        process_question(question.strip())
        st.session_state.clear_question_input = True
        st.rerun()

# =========================================================
# Latest Result
# =========================================================

if st.session_state.history:
    latest = st.session_state.history[0]
    meta = SOURCE_META[latest["source"]]

    st.divider()
    st.subheader("📋 Latest Result")

    render_pipeline(latest["source"])

    st.markdown(f"**Your question:** {latest['question']}")

    if meta["color"] == "green":
        st.success(meta["message"])
    elif meta["color"] == "blue":
        st.info(meta["message"])
    else:
        st.warning(meta["message"])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Source", latest["source"])
    m2.metric("Response Time", f"{latest['time']:.3f} sec")
    m3.metric("Documents", latest["documents"])
    m4.metric("Status", meta["badge"])

    st.markdown("**Answer**")
    st.markdown(
        f"<div class='answer-box'>{latest['answer']}</div>",
        unsafe_allow_html=True,
    )

    with st.expander("🔑 Cache key & details"):
        st.code(latest["cache_key"], language="text")
        st.markdown(f"L1 TTL: {L1_CACHE_TTL}s · L2 TTL: {L2_CACHE_TTL}s")

# =========================================================
# History
# =========================================================

if st.session_state.history:
    st.divider()
    st.subheader("📜 Question History")

    for i, entry in enumerate(st.session_state.history):
        meta = SOURCE_META[entry["source"]]
        with st.container(border=True):
            st.markdown(f"**Q:** {entry['question']}")
            st.markdown(
                f"**A:** {entry['answer'][:300]}{'...' if len(entry['answer']) > 300 else ''}"
            )
            c1, c2, c3 = st.columns(3)
            c1.caption(f"Source: {entry['source']}")
            c2.caption(f"Time: {entry['time']:.3f}s")
            c3.caption(f"Status: {meta['badge']}")

            if st.button("Ask again", key=f"reask_{i}", use_container_width=True):
                process_question(entry["question"])
                st.rerun()
