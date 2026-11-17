"""Evaluate the saved model on a held-out portion of the labelled CSV."""
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from app.core import DATA_PATH, LABELS, load_examples
from app.inference import HRPolicyClassifier


def main():
    rows = load_examples(DATA_PATH)
    _, test_rows = train_test_split(
        rows, test_size=0.2, random_state=42, stratify=[row["label"] for row in rows]
    )
    classifier = HRPolicyClassifier()
    truth, predicted = [], []
    for row in test_rows:
        truth.append(row["label"])
        predicted.append(classifier.classify_paragraph(row["text"])["label"])
    labels = list(LABELS)
    print(f"Held-out accuracy: {accuracy_score(truth, predicted):.1%}")
    print(classification_report(truth, predicted, labels=labels, zero_division=0))
    print("Confusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(truth, predicted, labels=labels))


if __name__ == "__main__":
    main()
