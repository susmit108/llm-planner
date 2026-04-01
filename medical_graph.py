from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from utils import (
    ARTIFACT_DIR,
    ChatGroq,
    DownloadConfig,
    ROOT_DIR,
    dedupe_keep_order,
    ensure_dir,
    hashed_bow,
    load_dataset,
    load_dotenv,
    load_pickle,
    mean,
    normalize_text,
    os,
    save_pickle,
    tokenize,
    torch,
    nn,
    F,
)


DATASET_NAME = "GBaker/MedQA-USMLE-4-options"
MODEL_FILE = ARTIFACT_DIR / "medqa_graph_bundle.pt"
RAG_FILE = ARTIFACT_DIR / "medqa_rag_index.pkl"
DEFAULT_HASH_DIM = 384
DEFAULT_HIDDEN_DIM = 128
HF_CACHE_DIR = ROOT_DIR / ".hf_cache"


class MissingDependencyError(RuntimeError):
    pass


class GroqConfigurationError(RuntimeError):
    pass


def require_training_dependencies() -> None:
    missing: list[str] = []
    if load_dataset is None:
        missing.append("datasets")
    if torch is None or nn is None or F is None:
        missing.append("torch")
    if missing:
        joined = ", ".join(missing)
        raise MissingDependencyError(
            f"Missing required packages: {joined}. Install the repo requirements first."
        )
    torch.sparse.check_sparse_tensor_invariants.disable()


TorchModuleBase = nn.Module if nn is not None else object


def extract_symptoms(example: dict[str, Any]) -> list[str]:
    phrases = example.get("metamap_phrases") or []
    normalized = [normalize_text(phrase) for phrase in phrases]
    normalized = [phrase for phrase in normalized if phrase]
    if normalized:
        return dedupe_keep_order(normalized[:20])

    tokens = tokenize(example.get("question", ""))
    fallback = [" ".join(tokens[index : index + 3]) for index in range(0, max(len(tokens) - 2, 0), 3)]
    fallback = [phrase for phrase in fallback if phrase]
    return dedupe_keep_order(fallback[:12])


def normalize_options(options: Any) -> dict[str, str]:
    if isinstance(options, dict):
        return {str(key): str(value) for key, value in options.items()}
    return {}


def build_doc_text(question: str, symptoms: list[str], answer: str, meta_info: str) -> str:
    parts = [question, " ".join(symptoms), answer, meta_info]
    return " ".join(part for part in parts if part)


def build_sparse_adjacency(num_nodes: int, base_edges: set[tuple[int, int]]) -> Any:
    undirected_edges = set()
    for source, target in base_edges:
        undirected_edges.add((source, target))
        undirected_edges.add((target, source))
    for node_index in range(num_nodes):
        undirected_edges.add((node_index, node_index))

    degrees = [0] * num_nodes
    for source, _ in undirected_edges:
        degrees[source] += 1

    rows: list[int] = []
    cols: list[int] = []
    values: list[float] = []
    for source, target in undirected_edges:
        rows.append(source)
        cols.append(target)
        values.append(1.0 / math.sqrt(degrees[source] * degrees[target]))

    indices = torch.tensor([rows, cols], dtype=torch.long)
    tensor_values = torch.tensor(values, dtype=torch.float32)
    adjacency = torch.sparse_coo_tensor(
        indices,
        tensor_values,
        (num_nodes, num_nodes),
        check_invariants=False,
    )
    return adjacency.coalesce()


class GraphConvolution(TorchModuleBase):
    def __init__(self, input_dim: int, output_dim: int):
        require_training_dependencies()
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, features: Any, adjacency: Any) -> Any:
        propagated = torch.sparse.mm(adjacency, features)
        return self.linear(propagated)


class MedQAGraphNet(TorchModuleBase):
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float):
        require_training_dependencies()
        super().__init__()
        self.gcn_1 = GraphConvolution(input_dim, hidden_dim)
        self.gcn_2 = GraphConvolution(hidden_dim, hidden_dim)
        self.query_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.dropout = nn.Dropout(dropout)

    def encode_nodes(self, node_features: Any, adjacency: Any) -> Any:
        hidden = self.gcn_1(node_features, adjacency)
        hidden = F.relu(hidden)
        hidden = self.dropout(hidden)
        hidden = self.gcn_2(hidden, adjacency)
        return F.normalize(hidden, dim=-1)

    def encode_queries(self, query_features: Any, symptom_context: Any) -> Any:
        encoded = self.query_encoder(query_features)
        encoded = encoded + symptom_context
        return F.normalize(encoded, dim=-1)


def build_retrieval_index(cases: list[dict[str, Any]]) -> dict[str, Any]:
    tokenized_docs: list[list[str]] = []
    doc_frequencies: Counter[str] = Counter()
    average_lengths: list[float] = []

    for case in cases:
        tokens = tokenize(case["doc_text"])
        case["tokens"] = tokens
        case["term_counts"] = dict(Counter(tokens))
        tokenized_docs.append(tokens)
        average_lengths.append(float(len(tokens)))
        for token in set(tokens):
            doc_frequencies[token] += 1

    total_docs = max(len(cases), 1)
    idf = {
        token: math.log(1.0 + (total_docs - freq + 0.5) / (freq + 0.5))
        for token, freq in doc_frequencies.items()
    }
    return {
        "cases": cases,
        "idf": idf,
        "avg_doc_len": mean(average_lengths),
        "k1": 1.5,
        "b": 0.75,
    }


def bm25_search(index: dict[str, Any], query: str, top_k: int = 5) -> list[dict[str, Any]]:
    query_tokens = tokenize(query)
    if not query_tokens:
        return []

    results: list[dict[str, Any]] = []
    for case in index["cases"]:
        score = 0.0
        doc_len = max(len(case["tokens"]), 1)
        for token in query_tokens:
            term_frequency = case["term_counts"].get(token, 0)
            if term_frequency == 0:
                continue
            idf = index["idf"].get(token, 0.0)
            denominator = term_frequency + index["k1"] * (
                1.0 - index["b"] + index["b"] * doc_len / max(index["avg_doc_len"], 1.0)
            )
            score += idf * (term_frequency * (index["k1"] + 1.0)) / denominator
        if score > 0:
            payload = dict(case)
            payload["score"] = score
            results.append(payload)

    results.sort(key=lambda item: item["score"], reverse=True)
    return results[:top_k]


def select_matched_symptoms(
    query: str,
    retrieved_cases: list[dict[str, Any]],
    symptom_to_index: dict[str, int],
    top_k: int = 8,
) -> list[str]:
    query_tokens = set(tokenize(query))
    if not query_tokens:
        return []

    scores: dict[str, float] = defaultdict(float)
    for rank, case in enumerate(retrieved_cases, start=1):
        case_weight = 1.0 / rank
        for phrase in case["symptoms"]:
            phrase_tokens = set(tokenize(phrase))
            if not phrase_tokens:
                continue
            overlap = len(query_tokens & phrase_tokens)
            if overlap == 0:
                continue
            phrase_score = case_weight * overlap / len(phrase_tokens)
            scores[phrase] += phrase_score

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [phrase for phrase, _ in ranked if phrase in symptom_to_index][:top_k]


def symptom_context_from_ids(node_embeddings: Any, symptom_ids: list[int]) -> Any:
    hidden_dim = node_embeddings.shape[1]
    if not symptom_ids:
        return torch.zeros((1, hidden_dim), dtype=node_embeddings.dtype, device=node_embeddings.device)
    context = node_embeddings[symptom_ids].mean(dim=0, keepdim=True)
    return context


def tensor_from_vectors(vectors: list[list[float]]) -> Any:
    return torch.tensor(vectors, dtype=torch.float32)


def load_medqa_split(split: str) -> Any:
    cache_dir = str(ensure_dir(HF_CACHE_DIR))
    try:
        return load_dataset(
            DATASET_NAME,
            split=split,
            cache_dir=cache_dir,
        )
    except Exception:
        os.environ["HF_DATASETS_OFFLINE"] = "1"
        os.environ["HF_HUB_OFFLINE"] = "1"
        return load_dataset(
            DATASET_NAME,
            split=split,
            cache_dir=cache_dir,
            download_config=DownloadConfig(local_files_only=True) if DownloadConfig is not None else None,
        )


def prepare_training_bundle(hash_dim: int) -> dict[str, Any]:
    require_training_dependencies()

    train_split = load_medqa_split("train")
    test_split = load_medqa_split("test")

    symptom_to_index: dict[str, int] = {}
    action_to_index: dict[str, int] = {}
    symptom_texts: list[str] = []
    action_texts: list[str] = []
    examples: list[dict[str, Any]] = []
    cases: list[dict[str, Any]] = []

    for row in train_split:
        question = str(row.get("question", "")).strip()
        answer = normalize_text(str(row.get("answer", "")))
        meta_info = str(row.get("meta_info", "")).strip()
        options = normalize_options(row.get("options"))
        symptoms = extract_symptoms(row)

        if not question or not answer:
            continue

        for symptom in symptoms:
            if symptom not in symptom_to_index:
                symptom_to_index[symptom] = len(symptom_texts)
                symptom_texts.append(symptom)

        for option_text in options.values():
            normalized_option = normalize_text(option_text)
            if normalized_option and normalized_option not in action_to_index:
                action_to_index[normalized_option] = len(action_texts)
                action_texts.append(normalized_option)

        if answer not in action_to_index:
            action_to_index[answer] = len(action_texts)
            action_texts.append(answer)

        label = action_to_index[answer]
        symptom_ids = [symptom_to_index[symptom] for symptom in symptoms if symptom in symptom_to_index]
        query_text = " ".join(part for part in [question, " ".join(symptoms)] if part)

        cases.append(
            {
                "question": question,
                "answer": answer,
                "answer_text": str(row.get("answer", "")).strip(),
                "options": options,
                "meta_info": meta_info,
                "symptoms": symptoms,
                "action_label": label,
                "doc_text": build_doc_text(question, symptoms, answer, meta_info),
            }
        )
        examples.append(
            {
                "query_text": query_text,
                "symptom_ids": symptom_ids,
                "label": label,
            }
        )

    action_offset = len(symptom_texts)
    base_edges: set[tuple[int, int]] = set()
    for example in examples:
        unique_symptoms = dedupe_keep_order(
            [symptom_texts[index] for index in example["symptom_ids"] if index < len(symptom_texts)]
        )
        unique_ids = [symptom_to_index[symptom] for symptom in unique_symptoms]
        action_node = action_offset + example["label"]

        for symptom_id in unique_ids:
            base_edges.add((symptom_id, action_node))

        limited_ids = unique_ids[:10]
        for left_index, source in enumerate(limited_ids):
            for target in limited_ids[left_index + 1 : left_index + 4]:
                base_edges.add((source, target))

    node_texts = symptom_texts + action_texts
    node_features = tensor_from_vectors([hashed_bow(text, hash_dim) for text in node_texts])
    query_features = tensor_from_vectors([hashed_bow(example["query_text"], hash_dim) for example in examples])
    labels = torch.tensor([example["label"] for example in examples], dtype=torch.long)
    adjacency = build_sparse_adjacency(len(node_texts), base_edges)
    action_node_ids = torch.tensor(
        [action_offset + index for index in range(len(action_texts))],
        dtype=torch.long,
    )

    test_examples: list[dict[str, Any]] = []
    for row in test_split:
        answer = normalize_text(str(row.get("answer", "")))
        if answer not in action_to_index:
            continue
        symptoms = extract_symptoms(row)
        mapped_symptoms = [symptom_to_index[s] for s in symptoms if s in symptom_to_index]
        test_examples.append(
            {
                "query_text": " ".join(
                    part for part in [str(row.get("question", "")).strip(), " ".join(symptoms)] if part
                ),
                "symptom_ids": mapped_symptoms,
                "label": action_to_index[answer],
            }
        )

    retrieval_index = build_retrieval_index(cases)

    return {
        "node_features": node_features,
        "adjacency": adjacency,
        "query_features": query_features,
        "labels": labels,
        "symptom_sets": [example["symptom_ids"] for example in examples],
        "test_examples": test_examples,
        "action_node_ids": action_node_ids,
        "symptom_to_index": symptom_to_index,
        "action_texts": action_texts,
        "hash_dim": hash_dim,
        "action_offset": action_offset,
        "retrieval_index": retrieval_index,
    }


def evaluate_model(model: Any, bundle: dict[str, Any]) -> dict[str, float]:
    if not bundle["test_examples"]:
        return {"top1_accuracy": 0.0, "top3_accuracy": 0.0}

    model.eval()
    with torch.no_grad():
        node_embeddings = model.encode_nodes(bundle["node_features"], bundle["adjacency"])
        action_embeddings = node_embeddings[bundle["action_node_ids"]]

        top1 = 0
        top3 = 0
        for example in bundle["test_examples"]:
            query_vector = torch.tensor(
                [hashed_bow(example["query_text"], bundle["hash_dim"])],
                dtype=torch.float32,
            )
            context = symptom_context_from_ids(node_embeddings, example["symptom_ids"])
            query_embedding = model.encode_queries(query_vector, context)
            logits = torch.matmul(query_embedding, action_embeddings.T).squeeze(0)
            ranked = torch.topk(logits, k=min(3, logits.shape[0])).indices.tolist()
            if example["label"] == ranked[0]:
                top1 += 1
            if example["label"] in ranked:
                top3 += 1

    total = len(bundle["test_examples"])
    return {
        "top1_accuracy": top1 / total,
        "top3_accuracy": top3 / total,
    }


def train_medqa_graph(
    output_dir: Path | None = None,
    epochs: int = 10,
    learning_rate: float = 1e-3,
    batch_size: int = 128,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    hash_dim: int = DEFAULT_HASH_DIM,
    dropout: float = 0.15,
) -> dict[str, Any]:
    bundle = prepare_training_bundle(hash_dim=hash_dim)

    output_path = ensure_dir(output_dir or ARTIFACT_DIR)
    model = MedQAGraphNet(input_dim=hash_dim, hidden_dim=hidden_dim, dropout=dropout)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)

    num_examples = bundle["labels"].shape[0]
    batch_size = max(1, min(batch_size, num_examples))
    best_loss = float("inf")
    history: list[dict[str, float]] = []

    for epoch in range(epochs):
        model.train()
        permutation = torch.randperm(num_examples)
        total_loss = 0.0

        for start in range(0, num_examples, batch_size):
            batch_indices = permutation[start : start + batch_size]
            node_embeddings = model.encode_nodes(bundle["node_features"], bundle["adjacency"])
            action_embeddings = node_embeddings[bundle["action_node_ids"]]

            contexts = torch.cat(
                [
                    symptom_context_from_ids(node_embeddings, bundle["symptom_sets"][index])
                    for index in batch_indices.tolist()
                ],
                dim=0,
            )
            query_vectors = bundle["query_features"][batch_indices]
            query_embeddings = model.encode_queries(query_vectors, contexts)
            logits = torch.matmul(query_embeddings, action_embeddings.T)
            loss = F.cross_entropy(logits, bundle["labels"][batch_indices])

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_indices.shape[0]

        average_loss = total_loss / num_examples
        metrics = evaluate_model(model, bundle)
        history.append({"epoch": float(epoch + 1), "loss": average_loss, **metrics})

        if average_loss < best_loss:
            best_loss = average_loss
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "hash_dim": hash_dim,
                    "hidden_dim": hidden_dim,
                    "dropout": dropout,
                    "action_texts": bundle["action_texts"],
                    "symptom_to_index": bundle["symptom_to_index"],
                    "node_features": bundle["node_features"],
                    "adjacency_indices": bundle["adjacency"].indices(),
                    "adjacency_values": bundle["adjacency"].values(),
                    "adjacency_size": bundle["adjacency"].size(),
                    "action_node_ids": bundle["action_node_ids"],
                    "history": history,
                    "metrics": metrics,
                },
                output_path / MODEL_FILE.name,
            )

    save_pickle(output_path / RAG_FILE.name, bundle["retrieval_index"])
    return {
        "model_path": str(output_path / MODEL_FILE.name),
        "rag_path": str(output_path / RAG_FILE.name),
        "history": history,
    }


def benchmark_saved_model(
    artifact_dir: Path | None = None,
    limit: int | None = None,
) -> dict[str, Any]:
    require_training_dependencies()
    assistant = MedQAGraphAssistant(artifact_dir=artifact_dir or ARTIFACT_DIR)
    test_split = load_medqa_split("test")

    total = 0
    top1 = 0
    top3 = 0
    retrieval_top1 = 0
    retrieval_top3 = 0
    evaluated_examples: list[dict[str, Any]] = []

    for row in test_split:
        gold_answer_raw = str(row.get("answer", "")).strip()
        gold_answer = normalize_text(gold_answer_raw)
        if gold_answer not in assistant.action_texts:
            continue

        symptoms = extract_symptoms(row)
        question = str(row.get("question", "")).strip()
        query = " ".join(part for part in [question, " ".join(symptoms)] if part)
        retrieved_cases = bm25_search(assistant.retrieval_index, query, top_k=5)
        predictions = assistant.rank_actions(query, retrieved_cases, top_k=3)

        total += 1
        predicted_texts = [normalize_text(item.text) for item in predictions]
        retrieved_texts = [normalize_text(item["answer"]) for item in retrieved_cases[:3]]

        if predicted_texts and predicted_texts[0] == gold_answer:
            top1 += 1
        if gold_answer in predicted_texts:
            top3 += 1
        if retrieved_texts and retrieved_texts[0] == gold_answer:
            retrieval_top1 += 1
        if gold_answer in retrieved_texts:
            retrieval_top3 += 1

        if len(evaluated_examples) < 5:
            evaluated_examples.append(
                {
                    "question": question,
                    "gold_answer": gold_answer_raw,
                    "predictions": [item.text for item in predictions],
                    "retrieved_answers": [item["answer"] for item in retrieved_cases[:3]],
                }
            )

        if limit is not None and total >= limit:
            break

    if total == 0:
        return {
            "num_examples": 0,
            "graph_top1_accuracy": 0.0,
            "graph_top3_accuracy": 0.0,
            "retrieval_top1_accuracy": 0.0,
            "retrieval_top3_accuracy": 0.0,
            "examples": [],
        }

    return {
        "num_examples": total,
        "graph_top1_accuracy": top1 / total,
        "graph_top3_accuracy": top3 / total,
        "retrieval_top1_accuracy": retrieval_top1 / total,
        "retrieval_top3_accuracy": retrieval_top3 / total,
        "examples": evaluated_examples,
    }


@dataclass
class Prediction:
    text: str
    score: float
    source: str


class MedQAGraphAssistant:
    def __init__(self, artifact_dir: Path | None = None):
        require_training_dependencies()
        self.artifact_dir = artifact_dir or ARTIFACT_DIR
        model_payload = torch.load(self.artifact_dir / MODEL_FILE.name, map_location="cpu")
        self.retrieval_index = load_pickle(self.artifact_dir / RAG_FILE.name)
        self.model = MedQAGraphNet(
            input_dim=model_payload["hash_dim"],
            hidden_dim=model_payload["hidden_dim"],
            dropout=model_payload["dropout"],
        )
        self.model.load_state_dict(model_payload["state_dict"])
        self.model.eval()

        adjacency = torch.sparse_coo_tensor(
            model_payload["adjacency_indices"],
            model_payload["adjacency_values"],
            tuple(model_payload["adjacency_size"]),
            check_invariants=False,
        ).coalesce()

        self.hash_dim = model_payload["hash_dim"]
        self.action_texts = model_payload["action_texts"]
        self.symptom_to_index = model_payload["symptom_to_index"]
        self.node_features = model_payload["node_features"]
        self.adjacency = adjacency
        self.action_node_ids = model_payload["action_node_ids"]
        self.training_history = model_payload.get("history", [])
        self.training_metrics = model_payload.get("metrics", {})
        self.groq_model_name = "llama-3.3-70b-versatile"

    @classmethod
    def artifacts_ready(cls, artifact_dir: Path | None = None) -> bool:
        base_dir = artifact_dir or ARTIFACT_DIR
        return (base_dir / MODEL_FILE.name).exists() and (base_dir / RAG_FILE.name).exists()

    @staticmethod
    def groq_ready() -> bool:
        load_dotenv()
        return ChatGroq is not None and bool(os.getenv("GROQ_API_KEY"))

    @classmethod
    def require_groq(cls) -> None:
        load_dotenv()
        missing: list[str] = []
        if ChatGroq is None:
            missing.append("langchain-groq")
        if not os.getenv("GROQ_API_KEY"):
            missing.append("GROQ_API_KEY")
        if missing:
            raise GroqConfigurationError(
                "Groq is required for chatbot responses. Missing: " + ", ".join(missing)
            )

    def rank_actions(self, question: str, retrieved_cases: list[dict[str, Any]], top_k: int = 5) -> list[Prediction]:
        matched_symptoms = select_matched_symptoms(question, retrieved_cases, self.symptom_to_index)
        matched_ids = [self.symptom_to_index[symptom] for symptom in matched_symptoms]
        query_text = " ".join([question, " ".join(matched_symptoms)]).strip()

        with torch.no_grad():
            node_embeddings = self.model.encode_nodes(self.node_features, self.adjacency)
            action_embeddings = node_embeddings[self.action_node_ids]
            query_vector = torch.tensor([hashed_bow(query_text, self.hash_dim)], dtype=torch.float32)
            context = symptom_context_from_ids(node_embeddings, matched_ids)
            query_embedding = self.model.encode_queries(query_vector, context)
            logits = torch.matmul(query_embedding, action_embeddings.T).squeeze(0)
            probabilities = torch.softmax(logits, dim=0)

        graph_ranked = torch.topk(probabilities, k=min(top_k * 2, probabilities.shape[0]))
        retrieval_scores: dict[str, float] = defaultdict(float)
        for rank, case in enumerate(retrieved_cases, start=1):
            retrieval_scores[case["answer"]] += 1.0 / rank

        predictions: list[Prediction] = []
        for action_index, probability in zip(
            graph_ranked.indices.tolist(),
            graph_ranked.values.tolist(),
            strict=False,
        ):
            action_text = self.action_texts[action_index]
            combined = 0.75 * probability + 0.25 * retrieval_scores.get(action_text, 0.0)
            predictions.append(Prediction(text=action_text, score=combined, source="graph+rag"))

        seen = {prediction.text for prediction in predictions}
        for action_text, score in sorted(retrieval_scores.items(), key=lambda item: item[1], reverse=True):
            if action_text in seen:
                continue
            predictions.append(Prediction(text=action_text, score=score, source="rag"))

        predictions.sort(key=lambda item: item.score, reverse=True)
        return predictions[:top_k]

    def answer_question(self, question: str, top_k: int = 3) -> dict[str, Any]:
        self.require_groq()
        retrieved_cases = bm25_search(self.retrieval_index, question, top_k=5)
        predictions = self.rank_actions(question, retrieved_cases, top_k=top_k)
        matched_symptoms = select_matched_symptoms(question, retrieved_cases, self.symptom_to_index)

        if predictions:
            primary = predictions[0].text
        else:
            primary = "No confident action found"

        response_lines = [
            f"Best matching action/answer: {primary}.",
            "This is a MedQA-trained suggestion and can represent a treatment, diagnostic step, or next-best action depending on the case.",
        ]
        if matched_symptoms:
            response_lines.append(f"Matched symptom phrases: {', '.join(matched_symptoms[:6])}.")
        if retrieved_cases:
            top_case = retrieved_cases[0]
            response_lines.append(
                "Closest retrieved case: "
                f"{top_case['question']} -> {top_case['answer_text']}."
            )

        base_response = " ".join(response_lines)
        result = {
            "response": base_response,
            "base_response": base_response,
            "predictions": [prediction.__dict__ for prediction in predictions],
            "retrieved_cases": [
                {
                    "question": case["question"],
                    "answer": case["answer_text"],
                    "meta_info": case["meta_info"],
                    "score": round(case["score"], 4),
                    "symptoms": case["symptoms"][:8],
                }
                for case in retrieved_cases
            ],
            "matched_symptoms": matched_symptoms,
            "response_source": "groq+graph+rag",
        }
        groq_response = self.synthesize_with_groq(question, result)
        if not groq_response:
            raise GroqConfigurationError(
                "Groq response generation failed. Check your API key, package installation, and network access."
            )
        result["response"] = groq_response
        return result

    def synthesize_with_groq(self, question: str, result: dict[str, Any]) -> str | None:
        self.require_groq()

        predictions = result.get("predictions", [])
        retrieved_cases = result.get("retrieved_cases", [])
        matched_symptoms = result.get("matched_symptoms", [])

        prediction_block = "\n".join(
            [
                f"- {item['text']} | score={item['score']:.4f} | source={item['source']}"
                for item in predictions[:5]
            ]
        ) or "- none"
        retrieval_block = "\n".join(
            [
                (
                    f"- score={case['score']:.4f} | meta={case['meta_info']} | "
                    f"question={case['question']} | answer={case['answer']}"
                )
                for case in retrieved_cases[:3]
            ]
        ) or "- none"
        symptom_block = ", ".join(matched_symptoms[:8]) if matched_symptoms else "none"

        llm = ChatGroq(
            model=self.groq_model_name,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0.2,
        )

        prompt = f"""
You are the response-balancing layer for a medical QA prototype.

Your job:
- rewrite the raw graph and retrieval output into a balanced, careful answer
- stay grounded in the provided evidence only
- do not invent a diagnosis or treatment not supported by the evidence
- if the retrieved case looks only partially related, say that clearly
- if the user question is broad or underspecified, ask for a few concise follow-up details
- keep the tone helpful and calm
- do not mention scores
- do not mention internal implementation details unless useful in one short phrase
- do not prescribe medication doses

Return plain text with:
1. a short best-fit interpretation
2. one short explanation tied to the evidence
3. if needed, 2-4 concise follow-up questions
4. a brief safety note if symptoms like shortness of breath could need urgent evaluation

User question:
{question}

Matched symptom phrases:
{symptom_block}

Top candidate actions:
{prediction_block}

Closest retrieved cases:
{retrieval_block}

Raw fallback answer:
{result.get("base_response", "")}
""".strip()

        try:
            response = llm.invoke(prompt)
        except Exception:
            return None

        content = getattr(response, "content", "")
        if isinstance(content, str):
            return content.strip() or None
        return None
