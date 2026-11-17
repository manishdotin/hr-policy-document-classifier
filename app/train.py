"""Fine-tune DistilGPT-2 to generate one HR policy label token per paragraph."""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import Dataset
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from app.core import (
    BASE_MODEL,
    LABEL_TOKENS,
    MAX_LENGTH,
    MODEL_DIR,
    load_examples,
)


def build_tokenized_dataset(tokenizer, rows: list[dict[str, str]]) -> Dataset:
    """Tokenize examples and calculate loss only on the special label token."""
    label_to_id = {
        label: tokenizer.convert_tokens_to_ids(token)
        for label, token in LABEL_TOKENS.items()
    }
    encoded_rows = []
    for row in rows:
        text_ids = tokenizer(
            row["text"].strip(),
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_LENGTH - 2,
        )["input_ids"]
        label_id = label_to_id[row["label"]]
        sequence = text_ids + [label_id]
        if tokenizer.eos_token_id is not None:
            sequence.append(tokenizer.eos_token_id)
        sequence = sequence[:MAX_LENGTH]
        attention = [1] * len(sequence)
        padding = MAX_LENGTH - len(sequence)
        sequence += [tokenizer.pad_token_id] * padding
        attention += [0] * padding
        targets = [-100] * MAX_LENGTH
        # CausalLM shifts labels internally, so this target is predicted
        # from the final paragraph token immediately before it.
        targets[len(text_ids)] = label_id
        encoded_rows.append({
            "input_ids": sequence,
            "attention_mask": attention,
            "labels": targets,
        })
    return Dataset.from_list(encoded_rows)


def train_model(
    data_path: str | Path | None = None,
    output_dir: str | Path = MODEL_DIR,
    model_name: str = BASE_MODEL,
    epochs: int = 5,
    batch_size: int = 4,
) -> Path:
    rows = load_examples(data_path) if data_path else load_examples()
    train_rows, eval_rows = train_test_split(
        rows, test_size=0.2, random_state=42, stratify=[row["label"] for row in rows]
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.add_special_tokens({"additional_special_tokens": list(LABEL_TOKENS.values())})
    tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.resize_token_embeddings(len(tokenizer))
    model.config.pad_token_id = tokenizer.pad_token_id

    train_data = build_tokenized_dataset(tokenizer, train_rows)
    eval_data = build_tokenized_dataset(tokenizer, eval_rows)

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=5e-4,
        weight_decay=0.01,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=42,
        use_cpu=not torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=lambda features: {
            key: torch.tensor([item[key] for item in features], dtype=torch.long)
            for key in ("input_ids", "attention_mask", "labels")
        },
    )
    trainer.train()

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    print(f"Fine-tuned model saved to: {output.resolve()}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--model-name", default=BASE_MODEL)
    parser.add_argument("--data", default=None, help="Optional path to a compatible CSV dataset")
    parser.add_argument("--output", default=str(MODEL_DIR))
    args = parser.parse_args()
    train_model(args.data, args.output, args.model_name, args.epochs, args.batch_size)


if __name__ == "__main__":
    main()
