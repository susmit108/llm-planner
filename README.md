# MedQA Graph Chatbot

This repo now includes a MedQA-powered chatbot that combines:

- a graph neural network trained on `GBaker/MedQA-USMLE-4-options`
- lexical retrieval over MedQA cases for RAG-style evidence
- a Groq LLM layer for balanced final responses
- a Streamlit chat UI for symptom-based questions

The model predicts the best matching medical action/answer from MedQA. Depending on the case, that answer can be a treatment, test, diagnosis, or next-best step.

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Create a `.env` file:

```bash
GROQ_API_KEY="your_groq_api_key"
```

## Train

```bash
.venv/bin/python train_medqa_graph.py --epochs 10
```

Artifacts are written to `artifacts/medqa_graph/`.

## Run

```bash
.venv/bin/streamlit run app.py
```

## Benchmark

Quick smoke test:

```bash
.venv/bin/python benchmark_medqa.py --limit 100
```

Full MedQA test split:

```bash
.venv/bin/python benchmark_medqa.py
```

## Reset chat history

```bash
python3 reset.py
```

## Notes

- The training script downloads the MedQA dataset from Hugging Face at runtime.
- Retrieval is BM25-style lexical search over the MedQA training questions plus extracted symptom phrases.
- The GNN is a lightweight two-layer graph convolution network over symptom and action nodes.
- Groq is required for chatbot responses; the app will not answer without `GROQ_API_KEY`.
- This is a research prototype, not a clinical decision system.
