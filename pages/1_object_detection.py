import logging
import queue
from pathlib import Path
from typing import List, NamedTuple

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

HERE = Path(__file__).parent
ROOT = HERE.parent

logger = logging.getLogger(__name__)

MODEL_URL           = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.caffemodel"
MODEL_LOCAL_PATH    = ROOT / "models/MobileNetSSD_deploy.caffemodel"
PROTOTXT_URL        = "https://github.com/robmarkcole/object-detection-app/raw/master/model/MobileNetSSD_deploy.prototxt.txt"
PROTOTXT_LOCAL_PATH = ROOT / "models/MobileNetSSD_deploy.prototxt.txt"

CLASSES = [
    "background","aeroplane","bicycle","bird","boat",
    "bottle","bus","car","cat","chair","cow",
    "diningtable","dog","horse","motorbike","person",
    "pottedplant","sheep","sofa","train","tvmonitor",
]


class Detection(NamedTuple):
    class_id: int
    label: str
    score: float
    box: np.ndarray


st.set_page_config(
    page_title="Object Detection — JackDev Vision",
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

/* Detection table */
.det-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    font-family: 'Space Mono', monospace;
}
.det-table th {
    text-align: left;
    padding: 10px 14px;
    background: #111;
    color: #555;
    font-size: 10px;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    border-bottom: 1px solid #1e1e1e;
}
.det-table td {
    padding: 10px 14px;
    border-bottom: 1px solid #151515;
    color: #ccc;
}
.det-table tr:hover td { background: #111; }

.conf-bar-wrap {
    background: #1a1a1a;
    border-radius: 2px;
    height: 4px;
    width: 100px;
    display: inline-block;
    vertical-align: middle;
    margin-right: 8px;
}
.conf-bar {
    background: #e8ff47;
    height: 100%;
    border-radius: 2px;
}

.stat-box {
    background: #111;
    border: 1px solid #1e1e1e;
    padding: 20px 24px;
    margin-bottom: 16px;
}
.stat-label {
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 0.15em;
    color: #444;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 28px;
    color: #e8ff47;
    font-weight: 700;
}

.info-block {
    background: #0d0d0d;
    border: 1px solid #1a1a1a;
    border-left: 2px solid #e8ff47;
    padding: 16px 20px;
    margin-top: 24px;
    font-size: 13px;
    color: #555;
    line-height: 1.7;
    font-family: 'Space Mono', monospace;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="page-label">01 / Module</div>', unsafe_allow_html=True)
st.markdown('<h1 class="page-title">Object <span>Detection</span></h1>', unsafe_allow_html=True)

st.sidebar.markdown("### Settings")
score_threshold = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)
show_table      = st.sidebar.checkbox("Show detection table", value=True)

def download_file(url, dest, expected_size=0):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and (not expected_size or dest.stat().st_size >= expected_size):
        return
    import urllib.request
    with st.spinner(f"Downloading {dest.name}..."):
        urllib.request.urlretrieve(url, dest)


@st.cache_resource
def generate_label_colors():
    np.random.seed(42)
    return np.random.uniform(0, 255, size=(len(CLASSES), 3))


@st.cache_resource
def load_model():
    download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=23147564)
    download_file(PROTOTXT_URL, PROTOTXT_LOCAL_PATH, expected_size=29353)
    return cv2.dnn.readNetFromCaffe(str(PROTOTXT_LOCAL_PATH), str(MODEL_LOCAL_PATH))


net    = load_model()
COLORS = generate_label_colors()

result_queue: "queue.Queue[List[Detection]]" = queue.Queue()

def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    image = frame.to_ndarray(format="bgr24")
    h, w  = image.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(image, (300, 300)),
        scalefactor=0.007843,
        size=(300, 300),
        mean=(127.5, 127.5, 127.5),
    )
    net.setInput(blob)
    output = net.forward().squeeze()
    output = output[output[:, 2] >= score_threshold]

    detections = [
        Detection(
            class_id=int(d[1]),
            label=CLASSES[int(d[1])],
            score=float(d[2]),
            box=(d[3:7] * np.array([w, h, w, h])),
        )
        for d in output
    ]

    for det in detections:
        color = COLORS[det.class_id].tolist()
        xmin, ymin, xmax, ymax = det.box.astype("int")
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
        label_y = ymin - 10 if ymin - 10 > 10 else ymin + 18
        cv2.putText(image, f"{det.label} {det.score*100:.0f}%",
                    (xmin + 4, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

    result_queue.put(detections)
    return av.VideoFrame.from_ndarray(image, format="bgr24")

col_video, col_info = st.columns([3, 2], gap="large")

with col_video:
    webrtc_ctx = webrtc_streamer(
        key="object-detection",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_info:
    obj_count_ph = st.empty()
    table_ph     = st.empty()

    st.markdown("""
    <div class="info-block">
        Model: MobileNet SSD (Caffe)<br>
        Classes: 20 object categories<br>
        Input: 300 x 300 px blob<br>
        Runtime: OpenCV DNN
    </div>
    """, unsafe_allow_html=True)

    if webrtc_ctx.state.playing and show_table:
        while True:
            try:
                dets = result_queue.get(timeout=1.0)
            except queue.Empty:
                continue

            obj_count_ph.markdown(f"""
            <div class="stat-box">
                <div class="stat-label">Objects detected</div>
                <div class="stat-value">{len(dets):02d}</div>
            </div>
            """, unsafe_allow_html=True)

            if dets:
                rows = ""
                for d in dets:
                    pct = d.score * 100
                    rows += f"""
                    <tr>
                        <td>{d.label}</td>
                        <td>
                            <span class="conf-bar-wrap">
                                <span class="conf-bar" style="width:{pct}%"></span>
                            </span>
                            {pct:.1f}%
                        </td>
                    </tr>"""
                table_ph.markdown(f"""
                <table class="det-table">
                    <thead><tr><th>Label</th><th>Confidence</th></tr></thead>
                    <tbody>{rows}</tbody>
                </table>
                """, unsafe_allow_html=True)
            else:
                table_ph.markdown(
                    '<p style="color:#333;font-family:Space Mono,monospace;font-size:13px;'
                    'margin-top:16px;">No objects in frame</p>',
                    unsafe_allow_html=True,
                )