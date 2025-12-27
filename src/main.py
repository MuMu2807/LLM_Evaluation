import html
import json
import streamlit as st
import random
import time
import threading
import boto3
import hashlib
import bleach
import pandas as pd
from botocore.exceptions import ClientError
from datetime import datetime
from utils import (
    create_logger,
    run_query_and_log,
    log_feedback,
)

# Prepare CloudWatch logger (shared)
cw_logger = create_logger()

# -------------------------------------------------------------------
# GLOBAL BACKGROUND IMAGE
# -------------------------------------------------------------------
import base64

def set_background(image_file):
    with open(image_file, "rb") as f:
        data = f.read()
    encoded_img = base64.b64encode(data).decode()

    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded_img}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


st.set_page_config(layout="wide") 

# ---- GLOBAL BACKGROUND IMAGE ----

set_background("image_bg.png")   

st.set_page_config(layout="wide") 

                
#  Chat Application
if "chat_log" not in st.session_state:
    st.session_state.chat_log = []
        
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None
        
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""

if "last" not in st.session_state:
    st.session_state.last = {
        "request_id": None,
        "query": None,
        "answer": None,
        "tokens": None,
        "latency_ms": None,
            "feedback_done": False,
    }  
USER_AVATAR = "https://cdn-icons-png.flaticon.com/512/1077/1077012.png"
BOT_AVATAR  = "https://cdn-icons-png.flaticon.com/512/4712/4712109.png"
        
st.markdown("""
        <style>
        .flow-chat { background:#fafafa; padding:1.5rem; }
        .chat-row { display:flex; margin:12px 0; }
        .chat-row.user { justify-content:flex-end; }
        .chat-avatar { width:36px; border-radius:50%; margin:0 10px; }
        .msg { max-width:60%; padding:14px; border-radius:16px; }
        .user .msg { background:#f3f4f7; }
        .bot .msg { background:#0066ff; color:white; }
        </style>
        """, unsafe_allow_html=True)
        
def render_msg(text, author):
    avatar = USER_AVATAR if author == "user" else BOT_AVATAR
    st.markdown(
        f"""
        <div class="chat-row {author}">
            <img src="{avatar}" class="chat-avatar">
            <div class="msg">{html.escape(text)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def handle_feedback(rating: str):
    last = st.session_state.last

    # Guard: ensure we have the CloudWatchLogger
    if not cw_logger or not getattr(cw_logger, "cloudwatch_logger", None):
        st.error("Feedback error: CloudWatch logger is not available.")
        return

    try:
        # tokens_detail should be a dict: {"input": ..., "output": ..., "total": ...}
        tokens_detail = last.get("token_detail")
        if not isinstance(tokens_detail, dict):
            # Fallback if only total is present
            tokens_detail = {
                "input": None,
                "output": None,
                "total": last.get("tokens"),
            }

        # latencies should be a dict
        latencies = last.get("latency_ms")
        if isinstance(latencies, int):
            latencies = {"total_ms": latencies}

        log_feedback(
            cw_logger=cw_logger.cloudwatch_logger,  # <-- pass CloudWatchLogger
            request_id=last["request_id"],
            rating=rating,  # "like" or "dislike"
            query=last["query"],
            answer=last["answer"],
            tokens=tokens_detail,
            latencies=latencies,
        )
        st.session_state.last["feedback_done"] = True
        st.success("Thanks for your feedback!")
    except Exception as e:
        st.error(f"Feedback error: {e}")

        
def submit_chat():
    q = st.session_state.chat_input.strip()
    if q:
        st.session_state.chat_log.append({"author": "user", "text": q})
        st.session_state.pending_query = q
        st.session_state.chat_input = ""
        
st.markdown("---")
st.markdown("### 🪄 Ask Chatbot")
        
        #st.markdown("<div class='flow-chat'>", unsafe_allow_html=True)
        
for m in st.session_state.chat_log:
    render_msg(m["text"], m["author"])
        
if st.session_state.pending_query:
    with st.spinner("Thinking…"):
        res = run_query_and_log(st.session_state.pending_query, cw_logger)
        answer_text = res.get("answer", "")
    st.session_state.chat_log.append({
        "author": "bot",
        "text": answer_text
    })
    st.session_state.last.update({
        "request_id": res.get("request_id"),
        "query": st.session_state.pending_query,
        "answer": answer_text,
        "tokens": res.get("tokens"),
        "latency_ms": res.get("latency_ms"),
        "feedback_done": False,
    })
    st.session_state.pending_query = None
    st.rerun()
  
# -------------------------------------------------
# FEEDBACK SECTION
 # -------------------------------------------------
last = st.session_state.last
        
if last.get("request_id") and not last.get("feedback_done"):
    st.markdown("---")
    st.markdown("**Was this answer helpful?**")
        
    c1, c2 = st.columns(2)
        
    with c1:
        if st.button("👍 Like", use_container_width=True):
            handle_feedback("like")
        
    with c2:
         if st.button("👎 Dislike", use_container_width=True):
            handle_feedback("dislike")
   
st.text_input(
    " Need Instant Answers? Our Chatbot is here to assist you ",
    key="chat_input",
    placeholder="Enter your question here",
    on_change=submit_chat
)
        #st.markdown("</div>", unsafe_allow_html=True)
