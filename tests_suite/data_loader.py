import logging
from datasets import load_dataset

logger = logging.getLogger(__name__)

# Dataset Identifiers
BIAS_DATASET = "TajwarC/Social-Media-Counterfactuals"
ROBUSTNESS_DATASET = "TajwarC/mteb_toxic_conversations_2.5k_robustness"
CORRECTNESS_DATASET = "mteb/toxic_conversations_50k"

def load_bias_data():
    """Loads the Social Media Counterfactuals dataset for bias evaluation."""
    logger.info(f"Loading bias evaluation dataset from {BIAS_DATASET}...")
    try:
        ds = load_dataset(BIAS_DATASET)
        return ds["train"]
    except Exception as e:
        logger.critical(f"Failed to load bias dataset: {e}")
        raise

def load_robustness_data():
    """Loads the robustness dataset and returns it as a list of samples."""
    logger.info(f"Loading robustness evaluation dataset from {ROBUSTNESS_DATASET}...")
    try:
        ds = load_dataset(ROBUSTNESS_DATASET)
        return list(ds["train"])
    except Exception as e:
        logger.critical(f"Failed to load robustness dataset: {e}")
        raise

def load_correctness_data(sample_size: int):
    """Loads a sampled portion of the toxic conversations dataset for correctness evaluation."""
    logger.info(f"Loading correctness evaluation dataset from {CORRECTNESS_DATASET} (Sample size: {sample_size})...")
    try:
        dataset = load_dataset(CORRECTNESS_DATASET, split="train")
        actual_sample_size = min(sample_size, len(dataset))
        return dataset.select(range(actual_sample_size))
    except Exception as e:
        logger.critical(f"Failed to load correctness dataset: {e}")
        raise
