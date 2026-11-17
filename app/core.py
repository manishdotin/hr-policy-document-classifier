"""Dataset preparation and shared helpers for the HR policy classifier."""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "hr_policy_paragraphs.csv"
MODEL_DIR = ROOT / "models" / "hr-policy-gpt2"
BASE_MODEL = "distilgpt2"
MAX_LENGTH = 160

# The label strings are special, single-token outputs learned by the model.
LABELS = {
    "leave_policy": "Leave Policy",
    "expense_claim": "Expense Claim",
    "attendance_policy": "Attendance Policy",
    "remote_work": "Remote Work Policy",
    "recruitment_policy": "Recruitment Policy",
}
LABEL_TOKENS = {label: f"<{label}>" for label in LABELS}


def load_examples(path: str | Path = DATA_PATH) -> list[dict[str, str]]:
    """Read labelled paragraph examples from the project CSV."""
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("The HR training dataset is empty.")
    for row in rows:
        if row.get("label") not in LABELS:
            raise ValueError(f"Unknown HR category: {row.get('label')}")
        if not row.get("text", "").strip():
            raise ValueError("Training paragraphs cannot be empty.")
    return rows


def format_training_text(text: str, label: str) -> str:
    """Reframe classification as next-token generation."""
    if label not in LABEL_TOKENS:
        raise ValueError(f"Unknown HR category: {label}")
    return f"{text.strip()} {LABEL_TOKENS[label]}"


def split_paragraphs(document_text: str) -> list[str]:
    """Split an uploaded text document into non-empty paragraphs/sections."""
    return [part.strip() for part in re.split(r"\n\s*\n+", document_text) if part.strip()]


def category_name(label: str) -> str:
    return LABELS.get(label, label.replace("_", " ").title())


def route_department(label: str) -> str:
    return {
        "leave_policy": "People Operations",
        "expense_claim": "Finance",
        "attendance_policy": "HR Operations",
        "remote_work": "HR / IT Operations",
        "recruitment_policy": "Talent Acquisition",
    }.get(label, "HR Review")
