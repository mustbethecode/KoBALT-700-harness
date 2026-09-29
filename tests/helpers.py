"""Shared test helpers (offline, no network/GPU).

Importable as ``from helpers import ...`` under pytest's default
rootdir insertion (tests/ has no __init__.py, so its parent dir is on
sys.path when collecting).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kobalt_eval.backends.base import Backend, GenerationResult

VALID_CLASSES = [
    "Syntax",
    "Semantics",
    "Pragmatics",
    "Phonetics/Phonology",
    "Morphology",
]

PREDICTION_FIELDS = {
    "id",
    "model",
    "backend",
    "raw_output",
    "predicted_answer",
    "ground_truth",
    "correct",
    "class",
    "subclass",
    "level",
    "latency_ms",
    "usage",
}


class StubBackend(Backend):
    """Canned-output stub backend.

    Returns ``outputs`` in call order (one GenerationResult per messages
    list), once exhausted repeats the last output, and pairs each call with
    the matching entry from ``usages`` (None beyond the list). Counts calls
    so resume tests can assert only-missing-items were inferred.
    """

    def __init__(
        self,
        outputs: list[str],
        usages: list[dict[str, Any] | None] | None = None,
    ) -> None:
        if not outputs:
            raise ValueError("StubBackend needs at least one canned output")
        self.outputs = list(outputs)
        self.usages = list(usages) if usages else []
        self.calls = 0

    def generate(
        self,
        messages_list: list[list[dict]],
        config: Any = None,
    ) -> list[GenerationResult]:
        out: list[GenerationResult] = []
        for _ in messages_list:
            idx = self.calls
            text = self.outputs[idx] if idx < len(self.outputs) else self.outputs[-1]
            usage = self.usages[idx] if idx < len(self.usages) else None
            out.append(GenerationResult(text=text, usage=usage))
            self.calls += 1
        return out


# Three-item fixture: one correct single-letter CoT, one no-phrase
# response, one multi-letter response (incorrect by construction).
STUB_OUTPUT_CORRECT_H = "문제를 분석하면 보기가 H가 타당하다. 정답은 H입니다."
STUB_OUTPUT_NO_PHRASE = "잘 모르겠습니다. (no answer phrase here)"
STUB_OUTPUT_MULTI = "처음에는 정답은 A입니다 라고 생각했지만, 다시 보니 정답은 C입니다"


def make_items() -> list[dict[str, Any]]:
    """Three small normalized items aligned with the canned stub outputs."""
    return [
        {
            "id": "test-id-0001",
            "class": "Syntax",
            "subclass": "Binding",
            "question": "다음 중 옳은 문장은? (A) ... (H) ...",
            "ground_truth": "H",
            "level": 2,
            "sampling_yn": 0,
        },
        {
            "id": "test-id-0002",
            "class": "Semantics",
            "subclass": "Implicature",
            "question": "함축 의미 문제 ... (A) ... (J) ...",
            "ground_truth": "A",
            "level": 1,
            "sampling_yn": 0,
        },
        {
            "id": "test-id-0003",
            "class": "Morphology",
            "subclass": "Agreement",
            "question": "형태론 문제 ... (A) ... (J) ...",
            "ground_truth": "B",
            "level": 3,
            "sampling_yn": 1,
        },
    ]


def make_raw_items() -> list[dict[str, Any]]:
    """Same three items in raw HF schema (ID/Class/Subclass/...)."""
    return [
        {
            "ID": "test-id-0001",
            "Class": "Syntax",
            "Subclass": "Binding",
            "Question": "다음 중 옳은 문장은? (A) ... (H) ...",
            "Answer": "H",
            "Level": 2,
            "Sampling_YN": 0,
        },
        {
            "ID": "test-id-0002",
            "Class": "Semantics",
            "Subclass": "Implicature",
            "Question": "함축 의미 문제 ... (A) ... (J) ...",
            "Answer": "A",
            "Level": 1,
            "Sampling_YN": 0,
        },
        {
            "ID": "test-id-0003",
            "Class": "Morphology",
            "Subclass": "Agreement",
            "Question": "형태론 문제 ... (A) ... (J) ...",
            "Answer": "B",
            "Level": 3,
            "Sampling_YN": 1,
        },
    ]


def make_valid_items(n: int = 700) -> list[dict[str, Any]]:
    """Build n normalized items satisfying validate_items (all 5 classes)."""
    letters = list("ABCDEFGHIJ")
    items: list[dict[str, Any]] = []
    for i in range(n):
        items.append(
            {
                "id": f"test-id-{i:04d}",
                "class": VALID_CLASSES[i % len(VALID_CLASSES)],
                "subclass": f"Phenomenon-{i % 24}",
                "question": f"테스트 질문 {i}번. 보기 (A) ... (J) ...",
                "ground_truth": letters[i % len(letters)],
                "level": (i % 3) + 1,
                "sampling_yn": i % 2,
            }
        )
    return items


def write_run_config(
    path: str | Path,
    *,
    family: str = "api",
    model: str = "stub-model",
    api_provider: str = "openai",
    concurrency: int = 1,
) -> Path:
    """Write a minimal run.yaml for CLI/factory tests."""
    import yaml

    payload = {
        "backend": {
            "family": family,
            "model": model,
            "api_provider": api_provider,
            "concurrency": concurrency,
        },
    }
    p = Path(path)
    p.write_text(yaml.safe_dump(payload, allow_unicode=True), encoding="utf-8")
    return p


def write_results_run(
    path: str | Path,
    *,
    model: str = "model-a",
    backend: str = "api",
    accuracy: float = 0.5,
    num_items: int = 4,
    num_correct: int = 2,
    by_class: dict[str, Any] | None = None,
) -> Path:
    """Create a fake completed run dir holding only results.json."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    if by_class is None:
        by_class = {
            "Syntax": {"n": 2, "correct": 1, "accuracy": 0.5},
            "Semantics": {"n": 2, "correct": 1, "accuracy": 0.5},
        }
    payload = {
        "model": model,
        "backend": backend,
        "accuracy": accuracy,
        "num_items": num_items,
        "num_correct": num_correct,
        "by_class": by_class,
    }
    (p / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return p
