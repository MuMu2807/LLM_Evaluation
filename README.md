# Knowledge-Base Chatbot with Evaluation, Observability & Monitoring

This repository implements a Knowledge-Base (KB) driven AI chatbot with end-to-end observability and evaluation


## Architecture Overview

<img width="1402" height="556" alt="image" src="https://github.com/user-attachments/assets/98264833-950a-4118-8f74-4b13877faa4f" />


- Users interact with the system through a Streamlit-based chat UI.

- Each user query is processed by an AI agent built using AWS Strands, with LiteLLM handling model invocation.

- The agent mandatorily retrieves context from an AWS Bedrock KB & generates responses strictly from the retrieved documents.

- All user queries, model responses, token usage, latency, cost estimates, and feedback are logged to Amazon CloudWatch for observability.

- A dedicated evaluation framework automatically assesses response quality, grounding, and safety.

- Evaluation metrics, hallucination signals, and judge scores are persisted in SQLite for auditability.

- Observability dashboards provide real-time insights into runtime performance, cost, and user feedback.

- Evaluation dashboards enable structured analysis of accuracy, hallucination rates, and quality metrics over time.

## 📁 Repository Structure

```
.
├── main.py                     # Streamlit chat application
├── utils.py                    # Core utilities: agent, logging, KB retrieval
├── evaluation.py               # Structured evaluation & LLM-as-a-Judge
├── evaluation_config.json      # Test cases for evaluation runs
├── monitoring_dashboards.py    # Evaluation & CloudWatch dashboards
├── image_bg.png                # UI background image
└── README.md                   # Documentation
```

### 1. main.py — User Facing Chat Application

  It is the entry point of the chatbot UI. It runs a Streamlit web app that users interact with to ask questions and receive answers from the knowledge-base-backed AI agent.

#### Query Execution

When a user submits a question:

*  The query is passed to run_query_and_log() which uses AI agent to retrieve data from the knowledge base and llm model to refine the answer

*  The answer is returned and displayed

*  Uses st.session_state to store Chat History and other Metadata

#### Observability & Logging

* Uses a CloudWatch logger created via create_logger()

* Logs: Requests, Responses, Token usage, Latency, Ensures traceability via request IDs, User Feedback Collection [Users can rate answers as Like or Dislike]

  <img width="1345" height="722" alt="image" src="https://github.com/user-attachments/assets/f4e09e64-f949-4463-a4c5-0aca6e52e6f1" />



### 2. evaluation.py — Automated Evaluation & Safety Engine

  It is the quality assurance backbone of the system. It Runs predefined test cases, Calls the chatbot programmatically, Detects hallucinations, Uses an LLM to judge answer quality and Logs structured evaluation results

  * Reads test cases from evaluation_config.json

  * Sends each query to the chatbot and Measures: Latency, Token usage, Tool usage, Answer correctness, Hallucination Detection

  #### Uses three independent signals to detect hallucination

  **i) Word overlap** : Measures lexical overlap between the generated answer and retrieved knowledge-base documents to verify explicit grounding.

  **ii) Semantic similarity** : Uses sentence embeddings from the model all-MiniLM-L6-v2 to compare the semantic meaning of the answer against retrieved content, capturing paraphrased or implicitly grounded responses.

  **iii) LLM grounding judge** : A separate AI [Here used anthropic claude model] decides if the answer is supported by documents

  These signals are combined conservatively to avoid false trust.

  #### ⚖️ LLM-as-a-Judge - Evaluation Metrics

  A dedicated AI judge scores each response on:
  
  Accuracy, Relevance, Completeness, Faithfulness, Helpfulness, Coherence, Instruction-following, Professional tone, Readability, Harmfulness, Bias / stereotypes
  
  All scores use a 1–5 scale.

  #### 🧾 Structured Evaluation Records - Easy for Manual Evaluation

  Each test produces a structured record containing: Query and answer, Hallucination results, Quality scores, Tool correctness
  These records are logged to dashboards for easy evaluation

  *All evaluation results are persisted to a local SQLite DB and reflected in the Streamlit dashboard in near real time. In production or product-based environment, this storage layer can be seamlessly replaced with a managed solution such as Amazon RDS for long-term analytics or Amazon CloudWatch for centralized logging, without changing the evaluation logic itself*

  <img width="1432" height="579" alt="image" src="https://github.com/user-attachments/assets/02774752-9cee-4302-a084-fee499df97b7" />

  
### 3. evaluation_config.json — Test Case Definitions

  It is a configuration file that defines the evaluation test cases.
    
  Each test includes:
    
    id → Unique test identifier
    
    query → User question to test
    
    expected_contains → Keywords expected in the answer
    
    expected_tool → Tools the agent should use (e.g., KB retrieval)
    
    category → Logical grouping (e.g., compliance, product info)

  It separates evaluation logic from evaluation data, making the system: Easier to maintain, Easier to audit and Easier to scale

### 4. utils.py — Helper Functions 

  This file is referenced by other files and acts as a shared utility module.

  i) create_logger() → Initializes CloudWatch + DB logging
    
  ii) run_query_and_log() → Executes AI query and logs results
    
  iii) log_feedback() → Records user feedback events

  Also has other functions to calculate the metrics

  i)   extract_answer_text() --> To format the answer
    
  ii)  extract_token_usage() --> To calculate token usage
    
  iii) estimate_cost() --> To calculate the cost

  iv)  extract_tool_usage() --> To get the tool details

### 4. monitoring_dashboards.py

  This file implements a Unified Streamlit Dashboard that combines:
  
  * LLM Evaluation Results stored in a local SQLite database
  
  * KB Agent observability logs streamed from AWS CloudWatch Logs

## Configuration
i) Environment Variables

    - Knowledge_base ID 
    - API_BASE
    - LOG_GROUP_NAME
    - MODEL_INPUT_USD_PER_1K=0.01
    - MODEL_OUTPUT_USD_PER_1K=0.03
    - AWS_DEFAULT_REGION=us-east-1

ii)  Required AWS Resources & Permission

iii) AWS Bedrock Knowledge Base

iv)  LLM Model Access

v)   CloudWatch Logs permissions


## ▶️ How to Run

#### 1. Install Dependencies
  pip install streamlit boto3 pandas numpy strands sentence-transformers

#### 2. Run Chat UI
  streamlit run main.py

#### 3. Run Evaluation
  python evaluation.py

#### 4. Run Dashboards
  streamlit run monitoring_dashboards.py


