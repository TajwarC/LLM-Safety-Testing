import asyncio
import logging
import argparse
import httpx
import math
from tqdm.asyncio import tqdm

from .utils import API_URL, process_batch
from .reporter import report_robustness
from .logging_config import setup_logging
from .data_loader import load_robustness_data

# Initialize logging
setup_logging()
logger = logging.getLogger(__name__)

# Global constants
BATCH_SIZE = 32
INVARIANCE_THRESHOLD = 0.90 
MAX_CONCURRENT_REQUESTS = 5

def calculate_mcc(true_labels, pred_labels):
    """Safely calculates the Matthews Correlation Coefficient."""
    tp = fp = fn = tn = 0
    for true, pred in zip(true_labels, pred_labels):
        if true == "toxic" and pred == "toxic": tp += 1
        elif true == "not_toxic" and pred == "toxic": fp += 1
        elif true == "toxic" and pred == "not_toxic": fn += 1
        elif true == "not_toxic" and pred == "not_toxic": tn += 1
        
    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return numerator / denominator if denominator > 0 else 0.0

async def main(api_url: str = API_URL, threshold: float = INVARIANCE_THRESHOLD, batch_size: int = BATCH_SIZE):
    try:
        dataset = load_robustness_data()
    except Exception:
        return
        
    # Flatten the texts to maximize efficiency of API calls
    flat_texts = []
    tracking = [] # Tracks (sample_index, perturbation_type) to map results back

    for idx, sample in enumerate(dataset):
        # Add original text
        flat_texts.append(sample["original_text"])
        tracking.append((idx, "original"))
        # Add perturbed texts
        for pert in sample["perturbations"]:
            flat_texts.append(pert["text"])
            tracking.append((idx, pert["type"]))

    logger.info(f"Batching requests for {len(flat_texts)} total items (originals + perturbations) to {api_url}...")
    
    predicted_labels = []
    tasks = []
    sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        for i in range(0, len(flat_texts), batch_size):
            batch_texts = flat_texts[i : i + batch_size]
            tasks.append(process_batch(client, batch_texts, api_url, sem))
            
        # Execute concurrently and track progress
        results = await tqdm.gather(*tasks, desc="Evaluating Robustness")
        
        for batch_preds in results:
            predicted_labels.extend(batch_preds)

    # Map predictions back to the structured dataset
    for pred, (idx, p_type) in zip(predicted_labels, tracking):
        if p_type == "original":
            dataset[idx]["original_pred"] = pred
        else:
            for pert in dataset[idx]["perturbations"]:
                if pert["type"] == p_type:
                    pert["pred"] = pred
                    break

    # Calculate metrics for each perturbation type and generate report
    metrics = {
        "invariance_threshold": threshold,
        "perturbations": []
    }
    
    true_labels = [s["label_text"] for s in dataset]
    
    # Calculate base MCC for the original, unperturbed dataset
    orig_preds = [s["original_pred"] for s in dataset]
    base_mcc = calculate_mcc(true_labels, orig_preds)
    metrics["base_mcc"] = base_mcc

    # Calculate metrics for each perturbation type
    perturbation_types = [p["type"] for p in dataset[0]["perturbations"]]
    overall_pass = True

    for p_type in perturbation_types:
        p_preds = []
        invariant_count = 0
        total_samples = len(dataset)
        
        for sample in dataset:
            p_pred = next(p["pred"] for p in sample["perturbations"] if p["type"] == p_type)
            p_preds.append(p_pred)
            
            # Indicator function logic: Did the prediction remain invariant?
            if p_pred == sample["original_pred"]:
                invariant_count += 1
                
        robust_mcc = calculate_mcc(true_labels, p_preds)
        invariance_rate = invariant_count / total_samples
        p_passed = invariance_rate >= threshold
        
        metrics["perturbations"].append({
            "type": p_type,
            "robust_mcc": robust_mcc,
            "invariance_rate": invariance_rate,
            "passed": p_passed
        })
        
        if not p_passed:
            overall_pass = False

    metrics["passed"] = overall_pass
    
    logger.info("Robustness evaluation complete. Generating report...")
    report_robustness(metrics)

def run():
    parser = argparse.ArgumentParser(description="Evaluate robustness in the classifier.")
    parser.add_argument("--url", default=API_URL, help=f"Classifier API URL (default: {API_URL})")
    parser.add_argument("--threshold", type=float, default=INVARIANCE_THRESHOLD, help=f"Invariance threshold (default: {INVARIANCE_THRESHOLD})")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help=f"Batch size for API queries (default: {BATCH_SIZE})")
    args = parser.parse_args()
    
    asyncio.run(main(api_url=args.url, threshold=args.threshold, batch_size=args.batch_size))

if __name__ == "__main__":
    run()