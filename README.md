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

## Configuration
Environment Variables
- MODEL_INPUT_USD_PER_1K=0.01
- MODEL_OUTPUT_USD_PER_1K=0.03
- AWS_DEFAULT_REGION=us-east-1

Required AWS Resources

AWS Bedrock Knowledge Base

CloudWatch Logs permissions

Valid SAT token provider

## ▶️ How to Run

#### 1. Install Dependencies
  pip install streamlit boto3 pandas numpy strands sentence-transformers

#### 2. Run Chat UI
  streamlit run main.py

#### 3. Run Evaluation
  python evaluation.py

#### 4. Run Dashboards
  streamlit run monitoring_dashboards.py


