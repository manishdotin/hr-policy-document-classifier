from app.core import LABELS, load_examples


def test_every_category_has_multiple_training_examples():
    rows = load_examples()
    counts = {label: sum(row["label"] == label for row in rows) for label in LABELS}
    assert all(count >= 8 for count in counts.values())
