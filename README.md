## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- Python 3.11+
- An OpenAI API key

## 1. Set your API key

```bash
set OPENAI_API_KEY="your-key-here"
```

## 2. Install dependencies

```bash
uv sync
```

## 3. Start the classifier

```bash
uv run uvicorn classifier.main:app --port 8000
```

The classifier is now available at `http://localhost:8000`. Leave this running.

## 4. Verify it works

In a second terminal:

```bash
curl -s http://localhost:8000/health
curl -s -X POST http://localhost:8000/classify_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I hope you have a great day", "I hate you"]}'
```

Expected:
```json
{"status": "ok"}
{"results": [{"label": "not_toxic"}, {"label": "toxic"}]}
```

## 5. Run Evaluation Tests

Once the classifier is running, you can execute the evaluation tests in a separate terminal.

### Correctness Test
Evaluates if the classifier reliably distinguishes harmful from non-harmful content.
```bash
uv run eval-correctness --sample_size 4096 --mcc_threshold 0.50 --precision_threshold 0.80
```
- `--sample_size`: Number of samples to evaluate (default: 4096).
- `--mcc_threshold`: Matthews Correlation Coefficient threshold (default: 0.50).
- `--precision_threshold`: Precision threshold (default: 0.80).

### Fairness Test
Evaluates if the classifier treats different demographic groups fairly.
```bash
uv run eval-bias --threshold 0.05 --batch_size 32
```
- `--threshold`: Maximum acceptable disparity (FPR/FNR) between demographic categories (default: 0.05).
- `--batch_size`: Number of texts per API request (default: 32).

### Robustness Test
Evaluates if the classifier's behavior remains stable under perturbations.
```bash
uv run eval-robustness --threshold 0.90
```
- `--threshold`: Minimum acceptable invariance rate (default: 0.90).

> **Note:** All tests accept an optional `--url` parameter if you wish to point them to a different API endpoint (default is `http://localhost:8000/classify_batch`) and a `--batch_size` parameter (default: 32).

For all tests, the resulting metrics are printed to the terminal and outputted as JSON files. Information and discussion about the chosen metrics, datasets and how the datasets were chosen & generated is available in the accompanying writeup "Technical documentation".

## 6. Run Unit and Integration Tests

The test suite in the `tests` directory uses mocks and can be run offline without starting the classifier server. To run the tests, execute:

```bash
uv run pytest
```
