import json
import random
import string
from datasets import load_dataset

# --- CONFIGURATION ---
SAMPLE_SIZE = 500
OUTPUT_FILE = "robustness_dataset.json"

# --- PERTURBATION FUNCTIONS ---
def apply_upper(text):
    return text.upper()

def apply_lower(text):
    return text.lower()

def apply_mixed(text):
    return "".join(
        c.lower() if random.choice([True, False]) else c.upper() 
        for c in text
    )

def apply_add_punct(text):
    words = text.split()
    if not words:
        return text
    idx = random.randint(0, len(words))
    words.insert(idx, random.choice(string.punctuation))
    return " ".join(words)

def apply_remove_punct(text):
    return text.translate(str.maketrans("", "", string.punctuation))

def apply_add_typo(text):
    if len(text) < 2:
        return text
    chars = list(text)
    idx = random.randint(0, len(chars) - 2)
    # Swap two adjacent characters to simulate a typo
    chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    return "".join(chars)

def generate_dataset():
    print(f"Loading mteb/toxic_conversations_50k dataset (Streaming {SAMPLE_SIZE} samples)...")
    dataset = load_dataset("mteb/toxic_conversations_50k", split="train", streaming=True)
    
    robustness_data = []
    
    for item in dataset.take(SAMPLE_SIZE):
        orig_text = item["text"]
        label_text = "toxic" if item["label"] == 1 else "not_toxic"
        
        sample_dict = {
            "original_text": orig_text,
            "label_text": label_text,
            "perturbations": [
                {"type": "uppercase", "text": apply_upper(orig_text), "label_text": label_text},
                {"type": "lowercase", "text": apply_lower(orig_text), "label_text": label_text},
                {"type": "mixed_case", "text": apply_mixed(orig_text), "label_text": label_text},
                {"type": "add_punctuation", "text": apply_add_punct(orig_text), "label_text": label_text},
                {"type": "remove_punctuation", "text": apply_remove_punct(orig_text), "label_text": label_text},
                {"type": "add_typo", "text": apply_add_typo(orig_text), "label_text": label_text},
            ]
        }
        robustness_data.append(sample_dict)
        
    print(f"Applying perturbations and saving to {OUTPUT_FILE}...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(robustness_data, f, indent=4)
    print("Generation complete.")

if __name__ == "__main__":
    generate_dataset()