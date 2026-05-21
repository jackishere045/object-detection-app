import streamlit as st

st.set_page_config(
    page_title="JackDev Task",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("JackDev Task Test")
st.caption("Real-time Computer Vision — WebRTC + OpenCV + MediaPipe")

st.divider()

col1, col2 = st.columns(2, gap="large")

with col1:
    st.subheader("01 — Object Detection")
    st.write(
        "Deteksi 20 kelas objek secara real-time menggunakan MobileNetSSD. "
        "Bounding box dengan confidence score langsung di video."
    )
    st.markdown("**Model:** MobileNet SSD (Caffe)")
    st.markdown("**Classes:** aeroplane, bicycle, bird, boat, bottle, bus, car, cat, chair, cow, diningtable, dog, horse, motorbike, person, pottedplant, sheep, sofa, train, tvmonitor")

with col2:
    st.subheader("02 — Hand Raise Counter")
    st.write(
        "Lacak dan hitung tangan terangkat secara real-time menggunakan MediaPipe Hands. "
        "Mendukung hingga 6 tangan sekaligus."
    )
    st.markdown("**Model:** MediaPipe Hands")
    st.markdown("**Landmark:** 21 titik per tangan, counter event raise terakumulasi")

st.divider()

st.caption("Stack: Streamlit · streamlit-webrtc · OpenCV · MediaPipe · aiortc")