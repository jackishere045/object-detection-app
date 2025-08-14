# app.py

import streamlit as st
from streamlit_webrtc import webrtc_streamer

st.title("Web RTC by JackDev")

webrtc_streamer(key="sample")
