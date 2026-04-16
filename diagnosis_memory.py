"""
diagnosis_memory.py
-------------------
Lightweight diagnosis-memory adapter for the planner pipeline.

It reuses the symptom and disease/task knowledge shipped with the
`Disease-Diagnosis` research code, but keeps runtime dependencies minimal:
- symptom extraction is lexical + fuzzy over the canonical symptom list
- disease ranking is knowledge-graph score aggregation
- task recommendation is disease-task score propagation

The resulting hypotheses are stored separately from persona.json so we do not
mix inferred diagnoses with explicitly stated medical facts.
"""

from __future__ import annotations

import json
import math
import pickle
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher, get_close_matches
from pathlib import Path


CONVERSATION_FILE = Path("conversation.json")
DIAGNOSIS_MEMORY_FILE = Path("diagnosis_memory.json")

DIAGNOSIS_DIR = Path("Disease-Diagnosis")
SYMPTOM_DIR = DIAGNOSIS_DIR / "symptom_extraction"
TASK_KG_FILE = DIAGNOSIS_DIR / "data" / "task_knowledge" / "disease_task_map.json"

SID_FILE = SYMPTOM_DIR / "SID.p"
DID_FILE = SYMPTOM_DIR / "DID.p"
SYMPTOM_DISEASE_GRAPH_FILE = SYMPTOM_DIR / "SD++.json"

TOP_K_DISEASES = 3
TOP_K_TASKS = 3
DIAGNOSIS_TASK_PREFIX = "[DX]"
MIN_DIRECT_DISEASE_PROBABILITY = 0.60
MIN_DISEASE_MARGIN = 0.15
SAFE_DIAGNOSIS_TASK_PREFIXES = (
    "Assess",
    "Check",
    "Consult",
    "Evaluate",
    "Monitor",
    "Obtain",
    "Order",
    "Perform",
    "Refer",
    "Rule out",
    "Schedule",
)


DEFAULT_MEMORY = {
    "patient_messages": [],
    "symptoms_raw": [],
    "symptoms_normalized": [],
    "unmatched_symptoms": [],
    "candidate_diseases": [],
    "recommended_tasks": [],
    "metadata": {
        "updated_at": None,
        "patient_message_count": 0,
        "extractor": "lightweight-kiddi-adapter",
    },
}


@dataclass(frozen=True)
class DiseaseCandidate:
    name: str
    probability: float
    score: float
    triggered_by: list[str]


@dataclass(frozen=True)
class TaskCandidate:
    task: str
    score: float
    triggered_by: list[str]


def _load_pickle(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def load_json(path: Path, default):
    if not path.exists():
        return default
    with path.open("r") as fh:
        return json.load(fh)


def save_json(path: Path, data) -> None:
    with path.open("w") as fh:
        json.dump(data, fh, indent=4)


def load_patient_messages() -> list[str]:
    conversation = load_json(CONVERSATION_FILE, {"Conversation": {}})
    values = conversation.get("Conversation", {}).values()
    messages = []
    for value in values:
        if not isinstance(value, str):
            continue
        stripped = value.strip()
        if not stripped:
            continue
        if stripped.startswith("[MarIA]:"):
            continue
        messages.append(stripped)
    return messages


class DiagnosisMemoryBuilder:
    def __init__(self):
        self.sid = _load_pickle(SID_FILE)
        self.did = _load_pickle(DID_FILE)
        self.symptom_graph = load_json(SYMPTOM_DISEASE_GRAPH_FILE, {})
        self.disease_task_map = load_json(TASK_KG_FILE, {})

        self.symptom_id_to_name = {value: key for key, value in self.sid.items()}
        self.disease_id_to_name = {value: key for key, value in self.did.items()}
        self.symptom_names = list(self.sid.keys())
        self.symptom_names_lower = {name.lower(): name for name in self.symptom_names}

    def build(self, patient_messages: list[str]) -> dict:
        if not patient_messages:
            memory = dict(DEFAULT_MEMORY)
            memory["metadata"] = {
                **DEFAULT_MEMORY["metadata"],
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "patient_message_count": 0,
            }
            return memory

        conversation_text = "\n".join(patient_messages)
        raw_symptoms, normalized_symptoms, unmatched = self.extract_symptoms(conversation_text)
        disease_candidates = self.rank_diseases(normalized_symptoms)
        recommended_tasks = self.recommend_tasks(disease_candidates)

        return {
            "patient_messages": patient_messages,
            "symptoms_raw": raw_symptoms,
            "symptoms_normalized": normalized_symptoms,
            "unmatched_symptoms": unmatched,
            "candidate_diseases": [
                {
                    "name": candidate.name,
                    "probability": round(candidate.probability, 4),
                    "score": round(candidate.score, 4),
                    "triggered_by": candidate.triggered_by,
                }
                for candidate in disease_candidates
            ],
            "recommended_tasks": [
                {
                    "task": task.task,
                    "score": round(task.score, 4),
                    "triggered_by": task.triggered_by,
                }
                for task in recommended_tasks
            ],
            "metadata": {
                "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "patient_message_count": len(patient_messages),
                "extractor": "lightweight-kiddi-adapter",
            },
        }

    def extract_symptoms(self, text: str) -> tuple[list[str], list[str], list[str]]:
        lowered = text.lower()
        exact_matches: set[str] = set()

        for symptom in sorted(self.symptom_names, key=len, reverse=True):
            symptom_lower = symptom.lower()
            if symptom_lower in lowered:
                exact_matches.add(symptom)

        sentence_candidates = re.split(r"[.!?\n]+", text)
        fuzzy_raw: set[str] = set()
        fuzzy_norm: set[str] = set()

        for sentence in sentence_candidates:
            cleaned = self._normalize_sentence(sentence)
            if not cleaned:
                continue

            close = get_close_matches(cleaned, list(self.symptom_names_lower.keys()), n=2, cutoff=0.85)
            for match in close:
                fuzzy_raw.add(cleaned)
                fuzzy_norm.add(self.symptom_names_lower[match])

            tokens = cleaned.split()
            for size in range(2, min(7, len(tokens) + 1)):
                for start in range(0, len(tokens) - size + 1):
                    phrase = " ".join(tokens[start:start + size])
                    close = get_close_matches(phrase, list(self.symptom_names_lower.keys()), n=1, cutoff=0.92)
                    if close:
                        fuzzy_raw.add(phrase)
                        fuzzy_norm.add(self.symptom_names_lower[close[0]])

        normalized = sorted(exact_matches | fuzzy_norm)
        raw = sorted(exact_matches | fuzzy_raw)
        unmatched = sorted(
            symptom for symptom in raw
            if self._best_normalized_match(symptom) is None
        )
        return raw, normalized, unmatched

    def rank_diseases(self, normalized_symptoms: list[str]) -> list[DiseaseCandidate]:
        symptom_support: dict[str, set[str]] = {}
        disease_scores: dict[str, float] = {}

        for symptom in normalized_symptoms:
            symptom_id = self.sid.get(symptom)
            if not symptom_id:
                continue

            for target_id, weight in self.symptom_graph.get(symptom_id, {}).items():
                if not str(target_id).startswith("D_"):
                    continue

                weight_value = float(weight)
                disease_scores[target_id] = disease_scores.get(target_id, 0.0) + weight_value
                symptom_support.setdefault(target_id, set()).add(symptom)

        if not disease_scores:
            return []

        ranked = []
        for disease_id, base_score in disease_scores.items():
            support_count = len(symptom_support.get(disease_id, set()))
            boosted_score = base_score + max(0, support_count - 1) * 0.15
            ranked.append((disease_id, boosted_score))

        ranked.sort(key=lambda item: item[1], reverse=True)
        top_ranked = ranked[:TOP_K_DISEASES]
        probabilities = self._softmax([score for _, score in top_ranked])

        return [
            DiseaseCandidate(
                name=self.disease_id_to_name.get(disease_id, disease_id),
                probability=probability,
                score=score,
                triggered_by=sorted(symptom_support.get(disease_id, set())),
            )
            for (disease_id, score), probability in zip(top_ranked, probabilities)
        ]

    def recommend_tasks(self, disease_candidates: list[DiseaseCandidate]) -> list[TaskCandidate]:
        if not self._has_high_confidence_diagnosis(disease_candidates):
            return []

        aggregated: dict[str, dict] = {}

        for disease in disease_candidates:
            if disease.probability < MIN_DIRECT_DISEASE_PROBABILITY:
                continue

            for task_info in self.disease_task_map.get(disease.name, []):
                task_name = task_info["task"].strip()
                if not self._is_safe_diagnosis_task(task_name):
                    continue

                task_weight = float(task_info.get("weight", 0.0))
                score = disease.probability * task_weight

                if score < 0.15:
                    continue

                bucket = aggregated.setdefault(
                    task_name,
                    {"score": 0.0, "triggered_by": set()},
                )
                bucket["score"] += score
                bucket["triggered_by"].add(disease.name)

        ranked_tasks = sorted(
            aggregated.items(),
            key=lambda item: item[1]["score"],
            reverse=True,
        )[:TOP_K_TASKS]

        return [
            TaskCandidate(
                task=task_name,
                score=data["score"],
                triggered_by=sorted(data["triggered_by"]),
            )
            for task_name, data in ranked_tasks
        ]

    @staticmethod
    def _has_high_confidence_diagnosis(disease_candidates: list[DiseaseCandidate]) -> bool:
        if not disease_candidates:
            return False

        top_probability = disease_candidates[0].probability
        second_probability = disease_candidates[1].probability if len(disease_candidates) > 1 else 0.0
        return (
            top_probability >= MIN_DIRECT_DISEASE_PROBABILITY
            and (top_probability - second_probability) >= MIN_DISEASE_MARGIN
        )

    @staticmethod
    def _is_safe_diagnosis_task(task_name: str) -> bool:
        return task_name.startswith(SAFE_DIAGNOSIS_TASK_PREFIXES)

    def _best_normalized_match(self, raw_symptom: str) -> str | None:
        lowered = raw_symptom.lower()
        if lowered in self.symptom_names_lower:
            return self.symptom_names_lower[lowered]

        match = get_close_matches(lowered, list(self.symptom_names_lower.keys()), n=1, cutoff=0.92)
        if match:
            return self.symptom_names_lower[match[0]]

        best_name = None
        best_ratio = 0.0
        for candidate_lower, canonical in self.symptom_names_lower.items():
            ratio = SequenceMatcher(None, lowered, candidate_lower).ratio()
            if ratio > best_ratio:
                best_ratio = ratio
                best_name = canonical
        if best_ratio >= 0.92:
            return best_name
        return None

    @staticmethod
    def _normalize_sentence(sentence: str) -> str:
        lowered = sentence.lower()
        lowered = re.sub(r"[^a-z0-9\s]+", " ", lowered)
        lowered = re.sub(r"\s+", " ", lowered).strip()
        return lowered

    @staticmethod
    def _softmax(scores: list[float]) -> list[float]:
        if not scores:
            return []
        max_score = max(scores)
        exps = [math.exp(score - max_score) for score in scores]
        total = sum(exps)
        return [exp_value / total for exp_value in exps]


def load_diagnosis_memory() -> dict:
    return load_json(DIAGNOSIS_MEMORY_FILE, DEFAULT_MEMORY)


def refresh_diagnosis_memory() -> dict:
    builder = DiagnosisMemoryBuilder()
    patient_messages = load_patient_messages()
    memory = builder.build(patient_messages)
    save_json(DIAGNOSIS_MEMORY_FILE, memory)
    return memory


def add_diagnosis_tasks(existing_tasks: dict[str, str], diagnosis_memory: dict | None = None) -> dict[str, str]:
    updated_tasks = dict(existing_tasks)
    memory = diagnosis_memory or load_diagnosis_memory()

    existing_values = set(updated_tasks.values())
    next_index = len(updated_tasks) + 1

    for recommendation in memory.get("recommended_tasks", []):
        task_text = recommendation.get("task", "").strip()
        if not task_text:
            continue

        full_text = f"{DIAGNOSIS_TASK_PREFIX} {task_text}"
        if full_text in existing_values:
            continue

        updated_tasks[str(next_index)] = full_text
        existing_values.add(full_text)
        next_index += 1

    return updated_tasks


def format_diagnosis_context(memory: dict | None = None) -> str:
    data = memory or load_diagnosis_memory()
    symptoms = data.get("symptoms_normalized", [])
    diseases = data.get("candidate_diseases", [])
    tasks = data.get("recommended_tasks", [])

    symptom_line = ", ".join(symptoms[:8]) if symptoms else "(none inferred yet)"
    disease_line = (
        "\n".join(
            f"  - {item['name']} ({item['probability']:.0%}, triggered by: {', '.join(item.get('triggered_by', [])) or 'n/a'})"
            for item in diseases[:3]
        )
        if diseases else
        "  (no disease hypotheses yet)"
    )
    task_line = (
        "\n".join(
            f"  - {item['task']} (score {item['score']:.2f})"
            for item in tasks[:3]
        )
        if tasks else
        "  (no diagnosis-informed remedies yet)"
    )

    return (
        "DIAGNOSIS MEMORY (hypotheses from patient wording; not confirmed diagnoses):\n"
        f"  Symptoms: {symptom_line}\n"
        f"  Probable diseases:\n{disease_line}\n"
        f"  Diagnosis-informed tasks:\n{task_line}"
    )


if __name__ == "__main__":
    memory = refresh_diagnosis_memory()
    print(json.dumps(memory, indent=2))
