import os, base64, json, sqlite3
from datetime import datetime, timedelta, timezone, date
from typing import List, Dict, Any, Optional
import streamlit as st
import pandas as pd
import numpy as np
import boto3
import matplotlib.pyplot as plt
from botocore.config import Config

# ---------------------------------------------------------------------
# import autorefresh helpers (works if either package is installed)
# ---------------------------------------------------------------------
st_autorefresh = None
try:
    from streamlit_extras.app_utils import st_autorefresh as _st_autorefresh
    st_autorefresh = _st_autorefresh
except Exception:
    try:
        from streamlit_autorefresh import st_autorefresh as _st_autorefresh2
        st_autorefresh = _st_autorefresh2
    except Exception:
        st_autorefresh = None  # Fallback to manual refresh buttons

# ---------------------------------------------------------------------
# Page setup & Background image
# ---------------------------------------------------------------------
st.set_page_config(page_title="Unified LLM & KB Logs Dashboard", layout="wide")

def set_bg_image(image_path: str):
    """Optional: Set a full-page background image (PNG recommended)."""
    abs_path = os.path.abspath(image_path)
    if not os.path.exists(abs_path):
        return
    with open(abs_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        [data-testid="stAppViewContainer"] {{
            background-image: url("data:image/png;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            background: rgba(255,255,255,0.65);
            backdrop-filter: blur(2px);
        }}
        </style>
        """,
        unsafe_allow_html=True
    )
set_bg_image("image_bg.png")

# ---------------------------------------------------------------------
# Sidebar: dashboard selector
# ---------------------------------------------------------------------
st.sidebar.title("Navigation")
mode = st.sidebar.radio(
    "Choose dashboard",
    ["Evaluation Dashboard", "Observability Dashboard"],
    index=0
)
# =====================================================================
# DASHBOARD A: SQLite — LLM Evaluation Dashboard
# =====================================================================
if mode == "Evaluation Dashboard":
    st.title("Evaluation Results")

    # -----------------------------
    # Sidebar: DB & Filters
    # -----------------------------
    st.sidebar.header("Eval Settings")
    default_db = os.environ.get("EVAL_DB_PATH", "logs.db")
    db_path = st.sidebar.text_input("SQLite DB path", default_db)

    st.sidebar.caption("Auto refresh every 15s")
    if st_autorefresh:
        st_autorefresh(interval=15_000, key="sqlite_refresh")
    else:
        if st.sidebar.button("Refresh now (SQLite)"):
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

    st.sidebar.header("Filters")
    time_preset = st.sidebar.selectbox(
        "Time range",
        [
            "All time",
            "Past 5 minutes",
            "Past 15 minutes",
            "Past 30 minutes",
            "Past 1 hour",
            "Past 24 hours",
            "Custom range",
        ],
        index=0,
    )

    def get_time_bounds_utc(preset: str):
        """Return (start, end) as timezone-aware UTC datetimes, or (None, None) for All time."""
        if preset == "All time":
            return None, None
        now_utc = datetime.now(timezone.utc)
        if preset == "Past 5 minutes":
            return now_utc - timedelta(minutes=5), now_utc
        if preset == "Past 15 minutes":
            return now_utc - timedelta(minutes=15), now_utc
        if preset == "Past 30 minutes":
            return now_utc - timedelta(minutes=30), now_utc
        if preset == "Past 1 hour":
            return now_utc - timedelta(hours=1), now_utc
        if preset == "Past 24 hours":
            return now_utc - timedelta(hours=24), now_utc
        return None, None

    start_ts, end_ts = get_time_bounds_utc(time_preset)
    
    if time_preset == "Custom range":
        col_a, col_b = st.sidebar.columns(2)

        # Pick date range (From → To)
        today_utc = datetime.now(timezone.utc).date()
        default_start = today_utc - timedelta(days=1)

        date_range = st.sidebar.date_input(
            "Select date range (From → To)",
            value=(default_start, today_utc),
            help="Pick From and To dates. Time is set to cover full days in UTC.",
        )
        # Ensure tuple (from, to)
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = date_range
            end_date = date_range

        if start_date > end_date:
            st.sidebar.error("Start date must be on or before end date.")
            st.stop()

        # Convert to full day UTC datetimes
        start_ts = datetime.combine(start_date, datetime.min.time()).replace(tzinfo=timezone.utc)
        end_ts = datetime.combine(end_date, datetime.max.time()).replace(tzinfo=timezone.utc).replace(microsecond=999000)

    # Text & boolean filters
    query_filter = st.sidebar.text_input("Search in query/answer", "")
    evaluator_filter = st.sidebar.text_input("Evaluator ID contains", "")
    category_filter = st.sidebar.text_input("Category contains", "")
    hallucinated_only = st.sidebar.checkbox("Only hallucinated (system)")
    tool_correct_only = st.sidebar.checkbox("Only tool correct")

    # Reset filters quickly
    if st.sidebar.button("Reset LLM filters"):
        st.session_state.clear()
        try:
            st.rerun()
        except Exception:
            st.experimental_rerun()

    # Optional CSV export
    export_csv = st.sidebar.checkbox("Enable CSV export button")

    st.divider()

    # -----------------------------
    # Data Load
    # -----------------------------
    def load_data(db_path: str) -> pd.DataFrame:
        if not os.path.exists(db_path):
            return pd.DataFrame()  # empty df
        conn = sqlite3.connect(db_path)
        try:
            df = pd.read_sql_query("SELECT * FROM evaluations ORDER BY id DESC", conn)
        finally:
            conn.close()
        return df

    df_raw = load_data(db_path)

    # If empty DB
    if df_raw.empty:
        st.info("No evaluations found yet. Ensure `cw_logger.log_evaluation(record)` wrote to the DB and the path is correct.")
        st.stop()

    # Normalize potential NaNs for text filters
    df = df_raw.copy()
    for col in ["query", "answer", "evaluator_id", "category", "judge_raw"]:
        if col in df.columns:
            df[col] = df[col].fillna("")

    # Parse JSON columns safely
    def try_json_loads(x):
        if isinstance(x, str) and x:
            try:
                return json.loads(x)
            except Exception:
                return []
        return x if isinstance(x, list) else []

    for col in ["expected_keywords", "retrieved_docs", "tool_usage"]:
        if col in df.columns:
            df[col] = df[col].apply(try_json_loads)

    # Parse timestamps as UTC-aware
    if "ts" in df.columns:
        ts_series = df["ts"].astype(str).str.replace("Z", "+00:00", regex=False)
        df["ts_parsed"] = pd.to_datetime(ts_series, errors="coerce", utc=True)
    else:
        df["ts_parsed"] = pd.to_datetime(pd.Series([None] * len(df)), utc=True)

    # -----------------------------
    # Apply Filters
    # -----------------------------
    def to_utc_timestamp(dt: datetime):
        if dt is None:
            return None
        ts = pd.Timestamp(dt)
        if ts.tz is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        return ts

    start_ts_pd = to_utc_timestamp(start_ts)
    end_ts_pd = to_utc_timestamp(end_ts)

    if start_ts_pd is not None and end_ts_pd is not None:
        df = df[(df["ts_parsed"] >= start_ts_pd) & (df["ts_parsed"] <= end_ts_pd)]

    # Text search (case-insensitive)
    if query_filter.strip():
        qf = query_filter.lower()
        df = df[
            df["query"].str.lower().str.contains(qf) |
            df["answer"].str.lower().str.contains(qf)
        ]

    # Evaluator filter
    if evaluator_filter.strip() and "evaluator_id" in df.columns:
        ef = evaluator_filter.lower()
        df = df[df["evaluator_id"].str.lower().str.contains(ef)]

    # Category filter
    if category_filter.strip() and "category" in df.columns:
        cf = category_filter.lower()
        df = df[df["category"].str.lower().str.contains(cf)]

    # Hallucinated toggle
    if hallucinated_only and "hallucination_system" in df.columns:
        df = df[(df["hallucination_system"].fillna(0).astype(int)) == 1]

    # Tool correctness toggle
    if tool_correct_only and "tool_correctness" in df.columns:
        df = df[(df["tool_correctness"].fillna(0).astype(int)) == 1]

    # -----------------------------
    # KPIs
    # -----------------------------
    col1, col2, col3, col4, col5 = st.columns(5)

    count_evals = len(df)
    avg_latency = f"{df['latency_ms'].mean():.1f}" if "latency_ms" in df.columns and count_evals else "—"
    hallucinated_sum = int(df["hallucination_system"].fillna(0).astype(int).sum()) if "hallucination_system" in df.columns else 0
    avg_judge_accuracy = f"{df['judge_accuracy'].mean():.2f}" if "judge_accuracy" in df.columns and count_evals else "—"
    avg_harmfulness = f"{df['judge_harmfulness'].mean():.2f}" if "judge_harmfulness" in df.columns and count_evals else "—"

    col1.metric("Evaluations", count_evals)
    col2.metric("Avg Latency (ms)", avg_latency)
    col3.metric("Hallucinated (sys)", hallucinated_sum)
    col4.metric("Avg Judge Accuracy", avg_judge_accuracy)
    col5.metric("Avg Harmfulness", avg_harmfulness)

    st.divider()

    # -----------------------------
    # Table
    # -----------------------------
    show_cols = [
        "ts", "evaluator_id", "test_id", "category", "latency_ms",
        "correctness", "tokens_used", "task_completion", "tool_correctness",
        "hallucination_system", "word_overlap_score", "semantic_similarity",
        "judge_accuracy", "judge_relevance", "judge_completeness", "judge_faithfulness",
        "judge_helpfulness", "judge_coherence", "judge_follow_instructions",
        "judge_professional_tone", "judge_readability", "judge_harmfulness",
        "judge_stereotype", "request_id", "query", "answer",
    ]

    existing_cols = [c for c in show_cols if c in df.columns]
    st.subheader("Evaluations")
    if len(df):
        st.dataframe(df[existing_cols], use_container_width=True, height=420)
    else:
        st.write("No rows to display with current filters.")

    # -----------------------------
    # Details per record
    # -----------------------------
    st.subheader("Details per record")
    if len(df):
        for idx, row in df.head(50).iterrows():
            header = f"[{row.get('test_id', '—')}] {row.get('category', '')} | {row.get('ts','')} | req={row.get('request_id','—')}"
            with st.expander(header):
                st.write("**Query**")
                st.code(row.get("query", ""), language="text")

                st.write("**Answer**")
                st.code(row.get("answer", ""), language="text")

                st.write("**Expected Keywords**")
                ek = row.get("expected_keywords", [])
                st.write(ek if isinstance(ek, list) else "—")

                st.write("**Tool Usage**")
                tu = row.get("tool_usage", [])
                st.write(tu if isinstance(tu, list) else "—")

                st.write("**Metrics (selected)**")
                mcols = st.columns(3)
                mcols[0].write(f"Latency (ms): {row.get('latency_ms', '—')}")
                mcols[1].write(f"Correctness: {row.get('correctness', '—')}")
                mcols[2].write(f"Task Completion: {row.get('task_completion', '—')}")

                mcols = st.columns(3)
                mcols[0].write(f"Hallucination (Sys): {row.get('hallucination_system', '—')}")
                mcols[1].write(f"Word Overlap: {row.get('word_overlap_score', '—')}")
                mcols[2].write(f"Semantic Similarity: {row.get('semantic_similarity', '—')}")

                st.write("**Judge Scores**")
                jcols1 = st.columns(3)
                jcols2 = st.columns(3)
                jcols3 = st.columns(3)
                jcols1[0].write(f"Accuracy: {row.get('judge_accuracy', '—')}")
                jcols1[1].write(f"Relevance: {row.get('judge_relevance', '—')}")
                jcols1[2].write(f"Completeness: {row.get('judge_completeness', '—')}")
                jcols2[0].write(f"Faithfulness: {row.get('judge_faithfulness', '—')}")
                jcols2[1].write(f"Helpfulness: {row.get('judge_helpfulness', '—')}")
                jcols2[2].write(f"Coherence: {row.get('judge_coherence', '—')}")
                jcols3[0].write(f"Follow Instructions: {row.get('judge_follow_instructions', '—')}")
                jcols3[1].write(f"Professional Tone: {row.get('judge_professional_tone', '—')}")
                jcols3[2].write(f"Readability: {row.get('judge_readability', '—')}")

                st.write("**Risk Scores**")
                rcols = st.columns(2)
                rcols[0].write(f"Harmfulness: {row.get('judge_harmfulness', '—')}")
                rcols[1].write(f"Stereotype: {row.get('judge_stereotype', '—')}")

    # -----------------------------
    # CSV Export
    # -----------------------------
    if export_csv and len(df):
        df_export = df.copy()
        for c in ["expected_keywords", "retrieved_docs", "tool_usage"]:
            if c in df_export.columns:
                df_export[c] = df_export[c].apply(
                    lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else (v or "")
                )
        csv = df_export[existing_cols].to_csv(index=False).encode("utf-8")
        st.download_button("Download filtered CSV", data=csv, file_name="evaluations_filtered.csv", mime="text/csv")

# =====================================================================
# DASHBOARD B: CloudWatch — KB Agent Logs
# =====================================================================
else:
    st.title("CloudWatch Logs")

    # ------------------ Constants --------------------
    DEFAULT_LOG_GROUP = "<>"

    # ------------------ AWS Helper -------------------
    def get_client(region_name: Optional[str] = None):
        region = region_name or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"
        return boto3.client(
            "logs",
            region_name=region,
            config=Config(retries={"max_attempts": 10, "mode": "standard"})
        )

    def millis(dt: datetime) -> int:
        return int(dt.timestamp() * 1000)

    def to_dt(ms: int) -> datetime:
        return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)

    def parse_log_message(message: str) -> Dict[str, Any]:
        """Try to parse JSON log message; fallback to plain text under 'message'."""
        try:
            obj = json.loads(message)
            if isinstance(obj, dict):
                return obj
            return {"message": obj}
        except json.JSONDecodeError:
            return {"message": message}

    def extract_feedback_status(msg: Dict[str, Any]) -> str:
        """
        Return 'like', 'dislike', 'neutral', or 'unknown' based on fields in the log payload.
        Designed for payloads like:
          { "event_type": "feedback", "rating": "like" }
        """
        if not isinstance(msg, dict):
            return "unknown"

        # Primary mapping: top-level string rating
        r = msg.get("rating")
        if isinstance(r, str):
            v = r.strip().lower()
            if v in {"like", "liked", "thumbs_up", "upvote", "positive"}:
                return "like"
            if v in {"dislike", "disliked", "thumbs_down", "downvote", "negative"}:
                return "dislike"
            if v == "neutral":
                return "neutral"

        # Optional nested feedback object
        fb = msg.get("feedback")
        if isinstance(fb, dict):
            t = fb.get("type") or fb.get("status") or fb.get("reaction") or fb.get("rating")
            if isinstance(t, str):
                t_norm = t.strip().lower()
                if t_norm in {"like", "liked", "thumbs_up", "upvote", "positive"}:
                    return "like"
                if t_norm in {"dislike", "disliked", "thumbs_down", "downvote", "negative"}:
                    return "dislike"
                if t_norm == "neutral":
                    return "neutral"

            liked = fb.get("liked")
            if isinstance(liked, bool):
                return "like" if liked else "dislike"

            # numeric rating variant (>=4 like, <=2 dislike)
            try:
                rv = float(fb.get("rating"))
                if rv >= 4:
                    return "like"
                if rv <= 2:
                    return "dislike"
                return "neutral"
            except (TypeError, ValueError):
                pass

        # numeric rating at top-level
        try:
            rv = float(r)
            if rv >= 4:
                return "like"
            if rv <= 2:
                return "dislike"
            return "neutral"
        except (TypeError, ValueError):
            pass

        return "unknown"

    def summarize_event(e: Dict[str, Any]) -> Dict[str, Any]:
        """Flatten common fields for tabular view."""
        msg = parse_log_message(e.get("message", ""))
        tokens = msg.get("tokens") or {}

        # Robust event_type detection
        evt = (
            msg.get("event_type")
            or msg.get("type")
            or msg.get("eventType")
            or None
        )

        # If see rating-like fields, assume feedback
        if not evt:
            r = msg.get("rating")
            if isinstance(r, str) and r.strip().lower() in {"like", "dislike", "neutral"}:
                evt = "feedback"
            elif isinstance(msg.get("feedback"), dict):
                evt = "feedback"

        return {
            "timestamp_dt": to_dt(e.get("timestamp")),
            "timestamp": to_dt(e.get("timestamp")).isoformat(),
            "logStreamName": e.get("logStreamName"),
            "event_type": evt or "unknown",
            "request_id": msg.get("request_id"),
            "query": msg.get("query"),
            "answer": msg.get("answer"),
            "answer_len": len(msg.get("answer", "")) if isinstance(msg.get("answer"), str) else None,
            "retrieved_docs_len": len(msg.get("retrieved_docs", [])) if isinstance(msg.get("retrieved_docs"), list) else None,
            "latency_ms": msg.get("latency_ms"),
            "tokens_input": tokens.get("input"),
            "tokens_output": tokens.get("output"),
            "tokens_total": tokens.get("total"),
            "cost_estimate_usd": msg.get("cost_estimate_usd"),
            "model": msg.get("model"),
            "kb_id": msg.get("kb_id"),
            "feedback_status": extract_feedback_status(msg),
            "raw": msg,
            "raw_text": e.get("message", ""),
        }

    @st.cache_data(ttl=10, show_spinner=False)
    def fetch_events(
        region: str,
        log_group_name: str,
        filter_pattern: str,
        start_time_ms: int,
        end_time_ms: int,
        max_events: int = 2000,
    ) -> List[Dict[str, Any]]:
        """
        Fetch events from CloudWatch Logs using FilterLogEvents across all streams in a log group.
        Pagination is handled until max_events or nextToken exhaustion.
        """
        client = get_client(region)
        events = []
        next_token = None
        while True:
            kwargs = {
                "logGroupName": log_group_name,
                "startTime": start_time_ms,
                "endTime": end_time_ms,
                "limit": min(1000, max_events - len(events)),
            }
            if filter_pattern:
                kwargs["filterPattern"] = filter_pattern
            if next_token:
                kwargs["nextToken"] = next_token

            resp = client.filter_log_events(**kwargs)
            batch = resp.get("events", [])
            events.extend(batch)
            next_token = resp.get("nextToken")

            if not next_token or len(events) >= max_events:
                break

        events.sort(key=lambda x: x.get("timestamp", 0))
        return events

    # ------------------ Sidebar -------------------
    st.sidebar.header("KB Agent Settings")
    region_input = st.sidebar.text_input("AWS Region", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    log_group_name = st.sidebar.text_input("Log Group Name", DEFAULT_LOG_GROUP)

    st.sidebar.caption("Server-side filter pattern (CloudWatch):")
    filter_pattern = st.sidebar.text_input(
        "Filter Pattern",
        placeholder='Examples:\n{ $.event_type = "chat" }\nADP\n"gpt-4o"',
        value=""
    )

    # --- Date-only time range controls ---
    time_mode = st.sidebar.radio("Time Range", ["Last N days", "Date range"], index=0)
    today_utc = datetime.now(timezone.utc).date()

    if time_mode == "Last N days":
        n_days = st.sidebar.slider("Days", min_value=1, max_value=30, value=7)
        start_date = today_utc - timedelta(days=n_days - 1)
        end_date = today_utc
    else:
        default_start = today_utc - timedelta(days=6)
        date_range = st.sidebar.date_input(
            "Select date range (From → To)",
            value=(default_start, today_utc),
            help="Pick From and To dates. Time is set to cover full days in UTC.",
        )
        if isinstance(date_range, tuple) and len(date_range) == 2:
            start_date, end_date = date_range
        else:
            start_date = date_range
            end_date = date_range

        if start_date > end_date:
            st.sidebar.error("Start date must be on or before end date.")
            st.stop()

    # Convert selected dates to full-day UTC datetimes
    start_dt = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end_dt = datetime.combine(end_date, datetime.max.time(), tzinfo=timezone.utc).replace(microsecond=999000)

    max_events = st.sidebar.slider("Max events to fetch", 200, 10000, 2000, step=200)
    auto_refresh = st.sidebar.checkbox("Auto-refresh every 10s", value=True)

    # Client-side filters
    st.sidebar.markdown("---")
    st.sidebar.caption("Client-side filters (applied after fetch):")
    latency_thresh = st.sidebar.number_input("Minimum latency (ms)", min_value=0, value=0, step=100)
    cost_thresh = st.sidebar.number_input("Minimum cost (USD)", min_value=0.0, value=0.0, step=0.001, format="%.5f")
    keyword = st.sidebar.text_input("Keyword in query/answer/raw", value="")
    request_id_contains = st.sidebar.text_input("Request ID contains", value="")
    apply_only_json = st.sidebar.checkbox("Only events with JSON payload", value=True)

    # Optional autorefresh
    if auto_refresh and st_autorefresh:
        st_autorefresh(interval=10_000, key="cw_refresh")
    elif auto_refresh:
        # Manual refresh fallback
        if st.sidebar.button("Refresh now (CloudWatch)"):
            try:
                st.rerun()
            except Exception:
                st.experimental_rerun()

    # ------------------ Main -------------------
    st.write(f"Viewing log group: **{log_group_name}**")
    if not log_group_name:
        st.info("Enter a Log Group Name to begin.")
        st.stop()

    with st.spinner("Fetching logs…"):
        raw_events = fetch_events(
            region=region_input,
            log_group_name=log_group_name,
            filter_pattern=filter_pattern.strip(),
            start_time_ms=millis(start_dt),
            end_time_ms=millis(end_dt),
            max_events=max_events,
        )

    summaries = [summarize_event(e) for e in raw_events]
    df = pd.DataFrame(summaries)

    if df.empty:
        st.warning("No events found in the selected window/filters.")
        st.stop()

    # Derived categorical filters from data
    if "event_type" in df.columns:
        event_types = sorted([x for x in df["event_type"].dropna().unique()])
    else:
        event_types = []

    # If we observed any rows that look like feedback (by feedback_status),
    # make sure "feedback" is offered as a selectable event type.
    if "feedback_status" in df.columns:
        has_feedback = df["feedback_status"].isin(["like", "dislike", "neutral"]).any()
        if has_feedback and "feedback" not in event_types:
            event_types.append("feedback")

    # Now render UI
    col_ft1, col_ft2 = st.columns(2)
    with col_ft1:
        sel_event_types = st.multiselect("Event types", event_types, default=event_types)
    with col_ft2:
        st.markdown("**Feedback filter**")
        only_feedback = st.checkbox("Show only feedback events", value=False)
        fb_choices = ["like", "dislike", "neutral", "unknown"]
        sel_feedback = st.multiselect(
            "Feedback status",
            fb_choices,
            default=fb_choices if not only_feedback else ["like", "dislike"]
        )

    # Apply client-side filters
    f = df.copy()

    if apply_only_json:
        f = f[~f["raw"].isna()]

    # Event type filter
    if sel_event_types:
        f = f[f["event_type"].isin(sel_event_types)]

    # Feedback filters
    if "feedback_status" in f.columns:
        if only_feedback:
            f = f[f["event_type"] == "feedback"]
            if sel_feedback:
                f = f[f["feedback_status"].isin(sel_feedback)]
        else:
            if sel_feedback and set(sel_feedback) != set(["like", "dislike", "neutral", "unknown"]):
                f = f[f["feedback_status"].isin(sel_feedback)]

    # Numeric filters
    def _to_numeric(val):
        """Return a float if val looks numeric; else None."""
        if isinstance(val, (int, float)):
            return float(val)
        if isinstance(val, str):
            try:
                return float(val)
            except ValueError:
                return None
        if isinstance(val, dict):
            for k in ("value", "ms", "latency", "total"):
                v = val.get(k)
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, str):
                    try:
                        return float(v)
                    except ValueError:
                        pass
            for v in val.values():
                if isinstance(v, (int, float)):
                    return float(v)
                if isinstance(v, str):
                    try:
                        return float(v)
                    except ValueError:
                        continue
            return None
        return None

    def _series_numeric(s: pd.Series) -> pd.Series:
        """Coerce a Series with mixed types to numeric (floats), dropping invalids."""
        return pd.to_numeric(s.apply(_to_numeric), errors="coerce")

    if latency_thresh and "latency_ms" in f.columns:
        f = f[(f["latency_ms"].fillna(0) >= latency_thresh)]

    if cost_thresh and "cost_estimate_usd" in f.columns:
        f = f[(f["cost_estimate_usd"].fillna(0.0) >= cost_thresh)]

    # Keyword filters
    if keyword:
        kw = keyword.lower()
        f = f[
            f["query"].fillna("").str.lower().str.contains(kw) |
            f["answer"].fillna("").str.lower().str.contains(kw) |
            f["raw_text"].fillna("").str.lower().str.contains(kw)
        ]

    if request_id_contains:
        rid = request_id_contains.lower()
        f = f[f["request_id"].fillna("").str.lower().str.contains(rid)]

    # Sort newest first
    f = f.sort_values("timestamp_dt", ascending=False)

    # ------------------ Metrics -------------------
    latencies = _series_numeric(f["latency_ms"]) if "latency_ms" in f.columns else pd.Series(dtype=float)
    tokens_total = _series_numeric(f["tokens_total"]) if "tokens_total" in f.columns else pd.Series(dtype=float)
    costs = _series_numeric(f["cost_estimate_usd"]) if "cost_estimate_usd" in f.columns else pd.Series(dtype=float)

    def percentile(series: pd.Series, p: float) -> float:
        return float(np.percentile(series.dropna(), p)) if not series.dropna().empty else 0.0

    col_m1, col_m2, col_m3, col_m4, col_m5 = st.columns(5)
    col_m1.metric("Events", f.shape[0])
    col_m2.metric("Avg latency (ms)", round(latencies.mean(), 2) if not latencies.dropna().empty else 0)
    col_m3.metric("P95 latency (ms)", round(percentile(latencies, 95), 2))
    col_m4.metric("Avg tokens (total)", round(tokens_total.mean(), 2) if not tokens_total.dropna().empty else 0)
    col_m5.metric("Total cost (USD)", round(costs.sum(), 5) if not costs.dropna().empty else 0.0)

    # ------------------ Table -------------------
    st.subheader("Event Summary")
    summary_cols = [
        "timestamp", "event_type", "request_id",
        "query", "answer",
        "latency_ms", "tokens_total",
        "cost_estimate_usd", "kb_id",
        "feedback_status"
    ]
    available_cols = [c for c in summary_cols if c in f.columns]
    st.dataframe(f[available_cols], use_container_width=True)

    # ------------------ Charts -------------------
    st.subheader("Charts: Tokens & Cost")

    ft = f.set_index("timestamp_dt").sort_index()

    st.markdown("**Total Tokens Over Time**")
    tok_series = _series_numeric(ft["tokens_total"]) if "tokens_total" in ft.columns else pd.Series(dtype=float)
    if not tok_series.dropna().empty:
        st.line_chart(tok_series.dropna())
    else:
        st.info("No token data to plot.")

    st.markdown("**Total cost over time (sum per 5 minutes)**")
    cost_series = _series_numeric(ft["cost_estimate_usd"]) if "cost_estimate_usd" in ft.columns else pd.Series(dtype=float)
    if not cost_series.dropna().empty:
        cost_resampled = cost_series.dropna().resample("5T").sum()
        st.line_chart(cost_resampled)
    else:
        st.info("No cost data to plot.")

    # ------------------ Export -------------------
    st.subheader("Export")
    csv_bytes = f.to_csv(index=False).encode("utf-8")
    st.download_button("Download filtered events (CSV)", data=csv_bytes, file_name="kb_agent_logs.csv", mime="text/csv")

    json_bytes = json.dumps(f["raw"].tolist(), indent=2).encode("utf-8")
    st.download_button("Download raw JSON payloads", data=json_bytes, file_name="kb_agent_logs_raw.json", mime="application/json")
