from __future__ import annotations

import os
import json
import time
import uuid
import sqlite3
from typing import Dict, Optional, Any, List, Set

import boto3
from botocore.exceptions import ClientError
from strands import Agent, tool
from strands.models.litellm import LiteLLMModel
from sat_token import get_sat_token

# =====================================================
# CONFIG
# =====================================================
REGION = "us-east-1"
KB_ID = os.getenv("BEDROCK_KB_ID")

MODEL_ID = os.getenv("MODEL")
API_BASE = os.getenv("API_BASE")

INPUT_PRICE_PER_1K = float(os.getenv("MODEL_INPUT_USD_PER_1K", "0.01"))
OUTPUT_PRICE_PER_1K = float(os.getenv("MODEL_OUTPUT_USD_PER_1K", "0.03"))

DB_DEFAULT_PATH = "logs.db"

LAST_RETRIEVED_DOCS: List[Dict[str, Any]] = []

LOG_GROUP_NAME = os.getenv("LOG_GROUP_NAME")


def todays_log_stream() -> str:
    return f"kb-agent-{time.strftime('%Y-%m-%d')}"

# =====================================================
# CLOUDWATCH LOGGER
# =====================================================
class CloudWatchLogger:
    def __init__(self, log_group: str, log_stream: str, region: str = REGION):
        self.log_group = log_group
        self.log_stream = log_stream
        self.client = boto3.client("logs", region_name=region)
        self.sequence_token = None

    def ensure_log_group_and_stream(self):
        try:
            self.client.create_log_group(logGroupName=self.log_group)
        except ClientError as e:
            if e.response["Error"]["Code"] != "ResourceAlreadyExistsException":
                raise

        try:
            self.client.create_log_stream(
                logGroupName=self.log_group,
                logStreamName=self.log_stream,
            )
            self.sequence_token = None
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceAlreadyExistsException":
                self._refresh_sequence_token()
            else:
                raise

    def _refresh_sequence_token(self):
        resp = self.client.describe_log_streams(
            logGroupName=self.log_group,
            logStreamNamePrefix=self.log_stream,
        )
        streams = resp.get("logStreams", [])
        if not streams:
            raise RuntimeError("Log stream not found")
        self.sequence_token = streams[0].get("uploadSequenceToken")

    def put_event(self, message: dict):
        event = {
            "timestamp": int(time.time() * 1000),
            "message": json.dumps(message, ensure_ascii=False),
        }

        kwargs = {
            "logGroupName": self.log_group,
            "logStreamName": self.log_stream,
            "logEvents": [event],
        }
        if self.sequence_token:
            kwargs["sequenceToken"] = self.sequence_token

        try:
            resp = self.client.put_log_events(**kwargs)
            self.sequence_token = resp.get("nextSequenceToken")
        except ClientError:
            # refresh token and retry once
            self._refresh_sequence_token()
            kwargs["sequenceToken"] = self.sequence_token
            resp = self.client.put_log_events(**kwargs)
            self.sequence_token = resp.get("nextSequenceToken")

# =====================================================
# SQLITE LOGGER
# =====================================================
class EvalLogger:
    def __init__(self, db_path: str = DB_DEFAULT_PATH):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute("""
            CREATE TABLE IF NOT EXISTS evaluations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ts TEXT NOT NULL,
                evaluator_id TEXT,
                test_id TEXT,
                category TEXT,
                query TEXT,
                expected_keywords TEXT,
                answer TEXT,
                latency_ms REAL,
                correctness INTEGER,
                tokens_used INTEGER,
                task_completion INTEGER,
                tool_correctness INTEGER,
                hallucination_system INTEGER,
                word_overlap_score REAL,
                semantic_similarity REAL,
                hallucination_llm_judge INTEGER,
                judge_accuracy INTEGER,
                judge_relevance INTEGER,
                judge_completeness INTEGER,
                judge_faithfulness INTEGER,
                judge_helpfulness INTEGER,
                judge_coherence INTEGER,
                judge_follow_instructions INTEGER,
                judge_professional_tone INTEGER,
                judge_readability INTEGER,
                judge_harmfulness INTEGER,
                judge_stereotype INTEGER,
                judge_raw TEXT,
                retrieved_docs TEXT,
                tool_usage TEXT,
                request_id TEXT
            );
            """)
            conn.commit()
        finally:
            conn.close()

    def log_evaluation(self, record: Dict[str, Any]):
        def to_json_or_none(v):
            if v is None:
                return None
            try:
                return json.dumps(v, ensure_ascii=False)
            except Exception:
                return json.dumps(str(v), ensure_ascii=False)

        row = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "evaluator_id": record.get("evaluator_id"),
            "test_id": record.get("test_id"),
            "category": record.get("category"),
            "query": record.get("query"),
            "expected_keywords": to_json_or_none(record.get("expected_keywords")),
            "answer": record.get("answer"),
            "latency_ms": record.get("latency_ms"),
            "correctness": int(bool(record.get("correctness"))) if record.get("correctness") is not None else None,
            "tokens_used": record.get("tokens_used"),
            "task_completion": int(bool(record.get("task_completion"))) if record.get("task_completion") is not None else None,
            "tool_correctness": int(bool(record.get("tool_correctness"))) if record.get("tool_correctness") is not None else None,
            "hallucination_system": int(bool(record.get("hallucination_system"))) if record.get("hallucination_system") is not None else None,
            "word_overlap_score": record.get("word_overlap_score"),
            "semantic_similarity": record.get("semantic_similarity"),
            "hallucination_llm_judge": int(bool(record.get("hallucination_llm_judge"))) if record.get("hallucination_llm_judge") is not None else None,
            "judge_accuracy": record.get("judge_accuracy"),
            "judge_relevance": record.get("judge_relevance"),
            "judge_completeness": record.get("judge_completeness"),
            "judge_faithfulness": record.get("judge_faithfulness"),
            "judge_helpfulness": record.get("judge_helpfulness"),
            "judge_coherence": record.get("judge_coherence"),
            "judge_follow_instructions": record.get("judge_follow_instructions"),
            "judge_professional_tone": record.get("judge_professional_tone"),
            "judge_readability": record.get("judge_readability"),
            "judge_harmfulness": record.get("judge_harmfulness"),
            "judge_stereotype": record.get("judge_stereotype"),
            "judge_raw": record.get("judge_raw"),
            "retrieved_docs": to_json_or_none(record.get("retrieved_docs")),
            "tool_usage": to_json_or_none(record.get("tool_usage")),
            "request_id": record.get("request_id"),
        }

        conn = sqlite3.connect(self.db_path)
        try:
            placeholders = ",".join(["?"] * len(row))
            columns = ",".join(row.keys())
            sql = f"INSERT INTO evaluations ({columns}) VALUES ({placeholders})"
            conn.execute(sql, tuple(row.values()))
            conn.commit()
        finally:
            conn.close()

# =====================================================
# COMPOSITE LOGGER
# =====================================================
class CompositeLogger:
    def __init__(self, sqlite_logger: EvalLogger, cloudwatch_logger: Optional[CloudWatchLogger] = None):
        self.sqlite_logger = sqlite_logger
        self.cloudwatch_logger = cloudwatch_logger

    def log_evaluation(self, record: Dict[str, Any]):
        if self.sqlite_logger:
            self.sqlite_logger.log_evaluation(record)
        if self.cloudwatch_logger:
            try:
                self.cloudwatch_logger.put_event({"type": "evaluation", **record})
            except Exception as e:
                print(f"[CW WARN] Failed to put_event: {e}")

def create_logger(db_path: str = DB_DEFAULT_PATH) -> "CompositeLogger":
    """
    Creates (or ensures) CloudWatch logger, and returns a CompositeLogger that
    also writes to SQLite for Streamlit dashboard.
    """
    cw = CloudWatchLogger(LOG_GROUP_NAME, todays_log_stream())
    cw.ensure_log_group_and_stream()

    sqlite_logger = EvalLogger(db_path=db_path)
    return CompositeLogger(sqlite_logger=sqlite_logger, cloudwatch_logger=cw)

# =====================================================
# MODEL / AGENT
# =====================================================
def build_model():
    return LiteLLMModel(
        client_args={
            "api_key": get_sat_token(),
            "base_url": API_BASE,
        },
        model_id=MODEL_ID,
    )

SYSTEM_PROMPT = """
You are a STRICT knowledge base QA assistant.

Rules:
- ALWAYS use kb_retrieve for every question.
- Use ONLY the retrieved documents.
- NEVER use outside knowledge.
- NEVER hallucinate.
- Answer clearly and concisely.
"""

@tool
def kb_retrieve(query: str, top_k: int = 30) -> List[Dict[str, Any]]:
    global LAST_RETRIEVED_DOCS
    LAST_RETRIEVED_DOCS = []  # reset per run

    kb = boto3.client("bedrock-agent-runtime", region_name=REGION)
    resp = kb.retrieve(
        knowledgeBaseId=KB_ID,
        retrievalQuery={"text": query},
        retrievalConfiguration={"vectorSearchConfiguration": {"numberOfResults": top_k}},
    )

    records: List[Dict[str, Any]] = []
    for d in resp.get("retrievalResults", []):
        text = d.get("content", {}).get("text")
        if not text:
            continue

        doc = {
            "text": text,
            "uri": d.get("location", {}).get("s3Location", {}).get("uri"),
        }

        LAST_RETRIEVED_DOCS.append(doc)
        records.append(doc)

    return records

def build_agent():
    return Agent(
        model=build_model(),
        tools=[kb_retrieve],
        system_prompt=SYSTEM_PROMPT,
        record_direct_tool_call=True,
    )

def extract_answer_text(resp: Any) -> str:
    if hasattr(resp, "message"):
        return resp.message["content"][0]["text"]
    return str(resp)

def extract_token_usage(resp: Any) -> Dict[str, Optional[int]]:
    usage = {"input": None, "output": None, "total": None}
    if hasattr(resp, "metrics"):
        au = resp.metrics.accumulated_usage
        usage["input"] = au.get("inputTokens")
        usage["output"] = au.get("outputTokens")
        usage["total"] = au.get("totalTokens")
    return usage

def estimate_cost(tokens: Dict[str, Optional[int]]) -> Optional[float]:
    if not tokens.get("input") or not tokens.get("output"):
        return None
    return round(
        (tokens["input"] / 1000.0) * INPUT_PRICE_PER_1K
        + (tokens["output"] / 1000.0) * OUTPUT_PRICE_PER_1K,
        6,
    )

def extract_tool_usage(resp: Any) -> Set[str]:
    if not hasattr(resp, "metrics"):
        return set()
    tool_metrics = getattr(resp.metrics, "tool_metrics", None)
    if not isinstance(tool_metrics, dict):
        return set()
    return set(tool_metrics.keys())

def run_query_and_log(query: str, cw_logger: CompositeLogger) -> Dict[str, Any]:
    global LAST_RETRIEVED_DOCS

    start = time.time()
    agent = build_agent()
    resp = agent(query.strip())
    latency_ms = int((time.time() - start) * 1000)

    answer = extract_answer_text(resp)
    tokens_detail = extract_token_usage(resp)
    cost = estimate_cost(tokens_detail)

    retrieved_docs = LAST_RETRIEVED_DOCS.copy()
    tool_usage = extract_tool_usage(resp)
    tokens_total = tokens_detail.get("total")

    result = {
        "request_id": str(uuid.uuid4()),
        "query": query,
        "answer": answer,
        "retrieved_docs": retrieved_docs,
        "retrieved_doc_count": len(retrieved_docs),
        "tokens": tokens_total,
        "token_detail": tokens_detail,
        "latency_ms": latency_ms,
        "cost_estimate_usd": cost,
        "model": MODEL_ID,
        "kb_id": KB_ID,
        "tool_usage": list(tool_usage),
    }

    if isinstance(cw_logger, CompositeLogger) and cw_logger.cloudwatch_logger:
        try:
            cw_logger.cloudwatch_logger.put_event({
                "type": "request_response",
                "request_id": result["request_id"],
                "query": query,
                "answer": answer,
                "latency_ms": latency_ms,
                "model": MODEL_ID,
                "kb_id": KB_ID,
                "retrieved_doc_count": result["retrieved_doc_count"],
                "tool_usage": result["tool_usage"],
                "tokens_total": tokens_total,
                "token_detail": tokens_detail,
                "cost_estimate_usd": cost,
                "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
            })
        except Exception as e:
            print(f"[CW WARN] Failed to send request_response: {e}")

    return result

def log_feedback(
    cw_logger: CloudWatchLogger,
    request_id: str,
    rating: str,
    query: str,
    answer: str,
    tokens: Dict[str, Optional[int]],
    latencies: Dict[str, Any],
) -> None:
    event = {
        "event_type": "feedback",
        "request_id": request_id,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model": MODEL_ID,
        "region": REGION,
        "kb_id": KB_ID,
        "rating": rating,
        "query": query,
        "answer": answer,
        "tokens": tokens,
        "latency_ms": latencies,
        "pricing_usd_per_1k": {
            "input": INPUT_PRICE_PER_1K,
            "output": OUTPUT_PRICE_PER_1K,
        },
        "log_origin": "streamlit-ui",
    }
    cw_logger.put_event(event)
