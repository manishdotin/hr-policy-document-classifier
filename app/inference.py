"""Load the fine-tuned model and classify paragraphs using label-token scores."""
from __future__ import annotations

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from app.core import LABELS, LABEL_TOKENS, MODEL_DIR, category_name, route_department


class HRPolicyClassifier:
    def __init__(self, model_dir: str | Path = MODEL_DIR):
        model_path = Path(model_dir)
        if not (model_path / "config.json").exists():
            raise FileNotFoundError(
                f"Fine-tuned model not found at {model_path}. Run: python -m app.train"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForCausalLM.from_pretrained(str(model_path)).eval()
        self.label_ids = {
            label: self.tokenizer.convert_tokens_to_ids(token)
            for label, token in LABEL_TOKENS.items()
        }
        self.id_to_label = {token_id: label for label, token_id in self.label_ids.items()}
        self.device = next(self.model.parameters()).device

    @torch.no_grad()
    def classify_paragraph(self, paragraph: str) -> dict:
        if not paragraph.strip():
            raise ValueError("Paragraph cannot be empty.")
        inputs = self.tokenizer(paragraph.strip() + " ", return_tensors="pt").to(self.device)
        logits = self.model(**inputs).logits[0, -1]
        ids = list(self.id_to_label)
        scores = torch.softmax(logits[ids], dim=0)
        best_index = int(torch.argmax(scores).item())
        label_id = ids[best_index]
        label = self.id_to_label[label_id]
        return {
            "label": label,
            "category": category_name(label),
            "confidence": float(scores[best_index].item()),
            "department": route_department(label),
        }
