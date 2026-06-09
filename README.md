## Prerequisites

- [uv](https://docs.astral.sh/uv/getting-started/installation/) installed
- Python 3.11+
- A Google Gemini API key

## 1. Set your API key

```bash
set GEMINI_API_KEY="your-key-here"
```

## 2. Install dependencies

```bash
uv sync
```

## 3. Start the classifier

By default, the server runs with the Gemini model backend. You can start the server using the entrypoint script:

```bash
uv run start-classifier --port 8000
```

Alternatively, you can run the FastAPI application directly using `uvicorn`:
```bash
uv run uvicorn classifier.main:app --port 8000
```

### Model Backend Configuration
The server supports multiple backend models (Gemini, OpenAI, Hugging Face). You can select the backend using command-line arguments or by setting environment variables in your `.env` file (`CLASSIFIER_TYPE`, `CLASSIFIER_MODEL`, `CLASSIFIER_DEVICE`).

#### Gemini (Default)
```bash
uv run start-classifier --model gemini-3.1-flash-lite
```

#### OpenAI GPT
Ensure you have set `OPENAI_API_KEY` in your environment or `.env` file.
```bash
uv run start-classifier --model gpt-4o-mini --type openai
```

#### Hugging Face Models
You can run local evaluations on Hugging Face models. The Hugging Face classifier includes dedicated, explicit model scripts to handle inference and label mapping:

- **Toxic BERT (`unitary/toxic-bert`)**:
  Utilizes sequence classification and maps toxic labels (such as `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, `identity_hate`) to `toxic` or `not_toxic`.
  ```bash
  uv run start-classifier --model unitary/toxic-bert --device cpu
  ```

- **Llama Guard (`meta-llama/LlamaGuard-7b`)**:
  Loads Llama Guard as a causal text-generation model, formats the prompt for the safety policy task, and maps generated outputs (`unsafe`/`safe`) to `toxic` or `not_toxic`.
  ```bash
  uv run start-classifier --model meta-llama/LlamaGuard-7b --device cpu
  ```

The classifier is now available at `http://localhost:8000`. Leave this running.

## 4. Verify it works

In a second terminal:

**Linux / macOS / Git Bash:**
```bash
curl -s -X POST http://localhost:8000/classify_batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I hope you have a great day", "I hate you"]}'
```

**Windows (Command Prompt):**
```cmd
curl -s -X POST http://localhost:8000/classify_batch -H "Content-Type: application/json" -d "{\"texts\": [\"I hope you have a great day\", \"I hate you\"]}"
```

**Windows (PowerShell):**
```powershell
Invoke-RestMethod -Uri http://localhost:8000/classify_batch -Method Post -ContentType "application/json" -Body '{"texts": ["I hope you have a great day", "I hate you"]}'
```

Expected:
```json
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
