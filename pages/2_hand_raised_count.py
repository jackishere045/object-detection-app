import logging
import queue
import threading
from pathlib import Path
from typing import List, NamedTuple, Optional

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

WRIST             = 0
MIDDLE_FINGER_TIP = 12
HAND_CONNECTIONS  = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (5,9),(9,10),(10,11),(11,12),
    (9,13),(13,14),(14,15),(15,16),
    (13,17),(17,18),(18,19),(19,20),
    (0,17),
]


class HandRaiseResult(NamedTuple):
    hand_count: int
    raised_count: int
    total_raises: int


st.set_page_config(
    page_title="Hand Raise — JackDev Vision",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }

.main .block-container { padding: 40px 5% 40px 5%; max-width: 100% !important; }

section[data-testid="stSidebar"] {
    background: #0a0a0a;
    border-right: 1px solid #1e1e1e;
}
section[data-testid="stSidebar"] * { color: #c8c8c8 !important; }

.page-label {
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    letter-spacing: 0.2em;
    color: #888;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.page-title {
    font-family: 'Space Mono', monospace;
    font-size: 32px;
    font-weight: 700;
    color: #f0f0f0;
    margin: 0 0 32px 0;
    letter-spacing: -0.02em;
}
.page-title span { color: #e8ff47; }

/* Metrics grid */
.metrics-grid {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 12px;
    margin-bottom: 20px;
}
.metric-box {
    background: #111;
    border: 1px solid #1e1e1e;
    padding: 20px 20px 18px;
}
.metric-label {
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    letter-spacing: 0.18em;
    color: #444;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.metric-value {
    font-family: 'Space Mono', monospace;
    font-size: 36px;
    font-weight: 700;
    line-height: 1;
    color: #f0f0f0;
}
.metric-value.accent { color: #e8ff47; }
.metric-value.raised { color: #ff6b6b; }

/* Reset button */
.stButton > button {
    background: transparent !important;
    border: 1px solid #2a2a2a !important;
    color: #555 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 11px !important;
    letter-spacing: 0.1em !important;
    padding: 8px 20px !important;
    border-radius: 0 !important;
    transition: all 0.2s !important;
}
.stButton > button:hover {
    border-color: #e8ff47 !important;
    color: #e8ff47 !important;
    background: transparent !important;
}

/* History log */
.log-wrap {
    background: #0a0a0a;
    border: 1px solid #1a1a1a;
    padding: 16px;
    max-height: 240px;
    overflow-y: auto;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
}
.log-entry {
    color: #444;
    padding: 3px 0;
    border-bottom: 1px solid #111;
}
.log-entry span { color: #e8ff47; }

.info-block {
    background: #0d0d0d;
    border: 1px solid #1a1a1a;
    border-left: 2px solid #e8ff47;
    padding: 16px 20px;
    margin-top: 20px;
    font-size: 12px;
    color: #555;
    line-height: 1.8;
    font-family: 'Space Mono', monospace;
}

/* Raised indicator overlay */
.raised-badge {
    display: inline-block;
    background: #e8ff47;
    color: #0a0a0a;
    font-family: 'Space Mono', monospace;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.1em;
    padding: 3px 10px;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

MEDIAPIPE_AVAILABLE = False
mp_hands_module = None

try:
    import mediapipe as mp
    mp_hands_module     = mp.solutions.hands
    MEDIAPIPE_AVAILABLE = True
except Exception:
    pass

if not MEDIAPIPE_AVAILABLE:
    st.error("MediaPipe tidak tersedia. Jalankan: pip install mediapipe==0.10.14")
    st.stop()


@st.cache_resource
def load_hands():
    return mp_hands_module.Hands(
        static_image_mode=False,
        max_num_hands=6,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )


hands_model = load_hands()

st.markdown('<div class="page-label">02 / Module</div>', unsafe_allow_html=True)
st.markdown('<h1 class="page-title">Hand Raise <span>Counter</span></h1>', unsafe_allow_html=True)

st.sidebar.markdown("### Settings")
raise_threshold = st.sidebar.slider("Raise sensitivity", 0.0, 0.3, 0.08, 0.01,
                                    help="Lebih kecil = lebih sensitif")
show_landmarks  = st.sidebar.checkbox("Show hand landmarks", value=True)

result_queue: "queue.Queue[HandRaiseResult]" = queue.Queue()

_lock        = threading.Lock()
_raise_state = {"total": 0, "prev": set(), "log": []}


def is_hand_raised(hand_landmarks, threshold: float) -> bool:
    wrist      = hand_landmarks.landmark[WRIST]
    middle_tip = hand_landmarks.landmark[MIDDLE_FINGER_TIP]
    return (wrist.y < 0.5 + threshold) and (middle_tip.y < wrist.y)


def draw_landmarks(image, hand_landmarks):
    h, w = image.shape[:2]
    lm   = hand_landmarks.landmark
    pts  = [(int(lm[i].x * w), int(lm[i].y * h)) for i in range(21)]
    for a, b in HAND_CONNECTIONS:
        cv2.line(image, pts[a], pts[b], (232, 255, 71), 1)
    for pt in pts:
        cv2.circle(image, pt, 4, (232, 255, 71), -1)
        cv2.circle(image, pt, 4, (0, 0, 0), 1)


def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    h, w  = image.shape[:2]

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results   = hands_model.process(image_rgb)

    current_raised = set()
    num_hands      = 0
    num_raised     = 0

    if results.multi_hand_landmarks:
        num_hands = len(results.multi_hand_landmarks)

        for idx, hand_lm in enumerate(results.multi_hand_landmarks):
            raised = is_hand_raised(hand_lm, raise_threshold)
            if show_landmarks:
                draw_landmarks(image, hand_lm)

            wx = int(hand_lm.landmark[WRIST].x * w)
            wy = int(hand_lm.landmark[WRIST].y * h)

            if raised:
                current_raised.add(idx)
                num_raised += 1
                # Kotak label raised
                cv2.rectangle(image, (wx - 4, wy - 36), (wx + 84, wy - 10), (232, 255, 71), -1)
                cv2.putText(image, "RAISED", (wx, wy - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 2)
            else:
                cv2.putText(image, "hand", (wx, wy - 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    with _lock:
        new_raises = current_raised - _raise_state["prev"]
        _raise_state["total"] += len(new_raises)
        _raise_state["prev"]   = current_raised
        if new_raises:
            import time
            _raise_state["log"].append(
                f"+{len(new_raises)} raise  [{time.strftime('%H:%M:%S')}]"
            )
            if len(_raise_state["log"]) > 50:
                _raise_state["log"].pop(0)
        total = _raise_state["total"]

    cv2.putText(image, f"hands:{num_hands}  raised:{num_raised}  total:{total}",
                (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (232, 255, 71), 1)

    result_queue.put(HandRaiseResult(num_hands, num_raised, total))
    return av.VideoFrame.from_ndarray(image, format="bgr24")


col_video, col_info = st.columns([3, 2], gap="large")

with col_video:
    webrtc_ctx = webrtc_streamer(
        key="hand-raise",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_info:
    metrics_ph = st.empty()
    
    col_reset, _ = st.columns([1, 2])
    with col_reset:
        if st.button("Reset counter"):
            with _lock:
                _raise_state["total"] = 0
                _raise_state["prev"]  = set()
                _raise_state["log"]   = []

    log_ph = st.empty()

    st.markdown("""
    <div class="info-block">
        Model: MediaPipe Hands<br>
        Landmark: 21 titik per tangan<br>
        Max hands: 6<br>
        Logic: wrist.y &lt; 0.5 + threshold<br>
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;AND finger_tip.y &lt; wrist.y
    </div>
    """, unsafe_allow_html=True)

    if webrtc_ctx.state.playing:
        while True:
            try:
                hr = result_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            metrics_ph.markdown(f"""
            <div class="metrics-grid">
                <div class="metric-box">
                    <div class="metric-label">Detected</div>
                    <div class="metric-value">{hr.hand_count:02d}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Raised now</div>
                    <div class="metric-value raised">{hr.raised_count:02d}</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Total events</div>
                    <div class="metric-value accent">{hr.total_raises:03d}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with _lock:
                log_entries = list(reversed(_raise_state["log"][-12:]))

            if log_entries:
                rows = "".join(
                    f'<div class="log-entry"><span>&gt;</span> {e}</div>'
                    for e in log_entries
                )
                log_ph.markdown(
                    f'<div class="log-wrap">{rows}</div>',
                    unsafe_allow_html=True,
                )
            else:
                log_ph.markdown(
                    '<div class="log-wrap" style="color:#222;">-- no events yet --</div>',
                    unsafe_allow_html=True,
                )