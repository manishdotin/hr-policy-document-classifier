import pytest

from app.core import LABELS, LABEL_TOKENS, category_name, format_training_text, load_examples, route_department, split_paragraphs


def test_dataset_has_five_categories_and_examples():
    rows = load_examples()
    assert len(LABELS) == 5
    assert len(rows) >= 40
    assert {row["label"] for row in rows} == set(LABELS)


def test_each_category_has_a_special_label_token():
    assert len(set(LABEL_TOKENS.values())) == 5
    assert all(token.startswith("<") and token.endswith(">") for token in LABEL_TOKENS.values())


def test_training_text_appends_label_token():
    result = format_training_text("  Apply for annual leave.  ", "leave_policy")
    assert result == "Apply for annual leave. <leave_policy>"


def test_unknown_label_is_rejected():
    with pytest.raises(ValueError):
        format_training_text("Paragraph", "unknown")


def test_document_is_split_into_nonempty_paragraphs():
    assert split_paragraphs("First section.\n\n  Second section. \n\n") == [
        "First section.", "Second section."
    ]


def test_department_routing():
    assert route_department("expense_claim") == "Finance"
    assert route_department("recruitment_policy") == "Talent Acquisition"
    assert category_name("remote_work") == "Remote Work Policy"
