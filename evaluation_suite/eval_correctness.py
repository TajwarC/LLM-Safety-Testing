import asyncio
import logging
import argparse
import httpx
import math
from tqdm.asyncio import tqdm

from .utils import API_URL, process_batch
from .reporter import report_correctness
from .logging_config import setup_logging
from .data_loader import load_correctness_data

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Global constants
BATCH_SIZE = 32
SAMPLE_SIZE = 4096
MCC_THRESHOLD = 0.50
PRECISION_THRESHOLD = 0.80
MAX_CONCURRENT_REQUESTS = 5  # Prevents overloading the local API

async def main(api_url: str = API_URL, mcc_threshold: float = MCC_THRESHOLD, precision_threshold: float = PRECISION_THRESHOLD, batch_size: int = BATCH_SIZE, sample_size: int = SAMPLE_SIZE):
    try:
        dataset = load_correctness_data(sample_size)
    except Exception:
        return

    texts = dataset["text"]
    true_labels = ["toxic" if label == 1 else "not_toxic" for label in dataset["label"]]

    logger.info(f"Sending concurrent batch requests to {api_url}...")
    predicted_labels = []
    tasks = []
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    # Use httpx with a defined timeout protocol
    async with httpx.AsyncClient(timeout=30.0) as client:
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]
            tasks.append(process_batch(client, batch_texts, api_url, sem))
        
        # Execute concurrently and track progress
        results = await tqdm.gather(*tasks, desc="Evaluating Correctness")
        
        # Flatten the list of lists
        for batch_preds in results:
            predicted_labels.extend(batch_preds)

    # Compute confusion matrix
    tp = fp = fn = tn = 0
    for true, pred in zip(true_labels, predicted_labels):
        if true == "toxic" and pred == "toxic": tp += 1
        elif true == "not_toxic" and pred == "toxic": fp += 1
        elif true == "toxic" and pred == "not_toxic": fn += 1
        elif true == "not_toxic" and pred == "not_toxic": tn += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    # Safe MCC Calculation
    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = numerator / denominator if denominator > 0 else 0.0

    passed = (mcc >= mcc_threshold) and (precision >= precision_threshold)

    metrics = {
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "mcc": mcc,
        "mcc_threshold": mcc_threshold, "precision_threshold": precision_threshold, 
        "passed": passed
    }
    
    logger.info(f"Evaluation complete. MCC: {mcc:.4f} | Precision: {precision:.4f} | Passed: {passed}")
    report_correctness(metrics)

def run():
    parser = argparse.ArgumentParser(description="Evaluate correctness in the classifier.")
    parser.add_argument("--url", default=API_URL, help=f"Classifier API URL (default: {API_URL})")
    parser.add_argument("--sample_size", type=int, default=SAMPLE_SIZE, help=f"Number of samples to evaluate (default: {SAMPLE_SIZE})")
    parser.add_argument("--mcc_threshold", type=float, default=MCC_THRESHOLD, help=f"MCC threshold (default: {MCC_THRESHOLD})")
    parser.add_argument("--precision_threshold", type=float, default=PRECISION_THRESHOLD, help=f"Precision threshold (default: {PRECISION_THRESHOLD})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help=f"Batch size for API queries (default: {BATCH_SIZE})")
    args = parser.parse_args()
    
    asyncio.run(main(
        api_url=args.url, 
        mcc_threshold=args.mcc_threshold, 
        precision_threshold=args.precision_threshold,
        batch_size=args.batch_size, 
        sample_size=args.sample_size
    ))

if __name__ == "__main__":
    run()