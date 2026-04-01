from __future__ import annotations

import json
import math
import os
import pickle
import re
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

try:
    import streamlit as st
except ImportError:  # pragma: no cover - optional at import time
    st = None

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - optional at import time
    torch = None
    nn = None
    F = None

try:
    from datasets import DownloadConfig, load_dataset
except ImportError:  # pragma: no cover - optional at import time
    DownloadConfig = None
    load_dataset = None

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional at import time
    def load_dotenv(dotenv_path: str | os.PathLike[str] | None = None, *_args: Any, **_kwargs: Any) -> bool:
        path = Path(dotenv_path) if dotenv_path is not None else ROOT_DIR / ".env"
        if not path.exists():
            return False
        loaded = False
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded = True
        return loaded

try:
    from langchain_groq import ChatGroq
except ImportError:  # pragma: no cover - optional at import time
    ChatGroq = None

try:
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover - optional at import time
    BaseModel = object

    def Field(*_args: Any, **kwargs: Any) -> Any:
        return kwargs.get("default")

try:
    from langchain_core.documents import Document
except ImportError:  # pragma: no cover - optional at import time
    Document = None

try:
    from langchain_community.vectorstores import FAISS
except ImportError:  # pragma: no cover - optional at import time
    FAISS = None

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:  # pragma: no cover - optional at import time
    HuggingFaceEmbeddings = None


ROOT_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = ROOT_DIR / "artifacts" / "medqa_graph"
TOKEN_RE = re.compile(r"[a-z0-9]+")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path: str | Path, default: Any | None = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        if default is None:
            raise FileNotFoundError(file_path)
        return default
    with file_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: Any) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=4)


def load_pickle(path: str | Path, default: Any | None = None) -> Any:
    file_path = Path(path)
    if not file_path.exists():
        if default is None:
            raise FileNotFoundError(file_path)
        return default
    with file_path.open("rb") as handle:
        return pickle.load(handle)


def save_pickle(path: str | Path, payload: Any) -> None:
    file_path = Path(path)
    ensure_dir(file_path.parent)
    with file_path.open("wb") as handle:
        pickle.dump(payload, handle)


def normalize_text(text: str) -> str:
    return " ".join(TOKEN_RE.findall((text or "").lower()))


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall((text or "").lower())


def dedupe_keep_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            ordered.append(item)
    return ordered


def stable_hash(text: str) -> int:
    value = 0
    for char in text:
        value = (value * 131 + ord(char)) % 2_147_483_647
    return value


def hashed_bow(text: str, dim: int) -> list[float]:
    vector = [0.0] * dim
    for token in tokenize(text):
        index = stable_hash(token) % dim
        vector[index] += 1.0
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
