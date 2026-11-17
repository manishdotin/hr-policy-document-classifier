# HR Policy Document Classifier

A local Streamlit application that classifies HR policy paragraphs into five categories using a fine-tuned DistilGPT-2 model. Classification is implemented as text generation: each training paragraph is followed by one special category token, and inference selects the highest-scoring category token.

> **Prototype notice:** The included CSV contains synthetic example paragraphs for learning and demonstration. It is not a real company's policy dataset. Model confidence is not a guarantee; an HR professional should review results before routing or taking action.

## Categories

| Label token | Category | Suggested routing |
|---|---|---|
| `<leave_policy>` | Leave Policy | People Operations |
| `<expense_claim>` | Expense Claim | Finance |
| `<attendance_policy>` | Attendance Policy | HR Operations |
| `<remote_work>` | Remote Work Policy | HR / IT Operations |
| `<recruitment_policy>` | Recruitment Policy | Talent Acquisition |

## Project structure

```text
hr-policy-document-classifier/
├── app/
│   ├── core.py          # labels, data loading, paragraph splitting
│   ├── train.py         # tokenizer extension and Trainer fine-tuning
│   ├── inference.py     # label-token inference pipeline
│   ├── evaluate.py      # held-out accuracy and confusion matrix
│   └── main.py          # Streamlit upload and results UI
├── data/hr_policy_paragraphs.csv
├── examples/sample_hr_policy.txt
├── models/              # created after training; not committed
├── tests/
└── requirements.txt
```

## Setup

Python 3.10–3.12 is recommended. The first training run downloads the pretrained `distilgpt2` weights from Hugging Face, so internet access is required for that step.

### Windows PowerShell

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the model

From the project root:

```bash
python -m app.train
```

The default configuration uses `distilgpt2`, five epochs, batch size four, and CPU automatically when CUDA is unavailable. On a CPU this may take several minutes or longer. To shorten a first test run:

```bash
python -m app.train --epochs 1 --batch-size 2
```

The tokenizer and fine-tuned model are saved in `models/hr-policy-gpt2/`. Do not delete that folder if you want to use the app without retraining.

The training code adds the five category tokens to the tokenizer, resizes the model embeddings, and masks the training loss so only the category token is scored. The Trainer evaluates on a held-out 20% split each epoch.

## Run the app

```bash
streamlit run app/main.py
```

Open the local URL shown in the terminal. Upload a UTF-8 `.txt` file. Separate policy sections with blank lines, then select **Classify sections**. The app displays the predicted category, suggested department, model score, and paragraph. You can download the results as CSV.

A sample file is provided at `examples/sample_hr_policy.txt`.

## Evaluate

After training:

```bash
python -m app.evaluate
```

This evaluates the saved model on the same deterministic held-out split and reports accuracy, per-class precision/recall/F1, and a confusion matrix. Since the sample dataset is small and synthetic, these metrics are educational rather than evidence of real-world HR performance.

## Tests

```bash
pytest -q
```

Tests cover the category dataset, label-token formatting, paragraph splitting, and department routing. They do not download or train GPT-2.

## Add more training data

Edit `data/hr_policy_paragraphs.csv`. Keep the columns `text,label` and use one of the five labels in `app/core.py`. Aim for a balanced number of varied paragraphs per category. Do not include confidential employee information or internal company documents unless you have permission to use them.

## Limitations

- The repository contains training code and example data, not pretrained or fine-tuned model weights. Run the training command to create the local model.
- The model predicts only the five configured categories. It cannot reliably identify categories outside that set.
- A softmax score over five labels is a relative model score, not a calibrated probability.
- This is a demonstration prototype, not an HR compliance or legal decision system.
