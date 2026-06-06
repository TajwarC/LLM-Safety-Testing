import json
import os

'''
Script to generate a structured counterfactual fairness evaluation dataset.

The script inserts identity terms into benign and toxic seeds. For each seed, the [IDENTITY] placeholder is replaced with terms from IDENTITY_VECTORS.

The IDENTITY_VECTORS are sweapped across each seed, such that the sentiment (toxic vs not toxic) for each term can be evaluated independently.

For example, we expect a fair model to give the same toxicity label for each seed, irrespective of the identity term used. This allows us to evaluate if certain identity terms are disproportionately associated with toxicity predictions, which would indicate bias.

'''

# Identity Vectors and Evaluation Control Group
IDENTITY_VECTORS = {
    "religion": ["Muslim", "Christian", "Jewish", "Atheist", "Hindu", "Buddhist", "Sikh"],
    "race_ethnicity": ["Black","African", "White","European", "Asian", "Hispanic","South American", "Indian", "Middle Eastern", "Native American", "Arab"],
    "control": ["people"] # Establishes baseline plain toxicity rate
}

# Load seeds from JSON file
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
seeds_file_path = os.path.join(SCRIPT_DIRECTORY, "balanced_social_media_seeds.json")

with open(seeds_file_path, "r", encoding="utf-8") as f:
    seeds_data = json.load(f)

# Benign refers to non-toxic
benign_seeds = [item["text"] for item in seeds_data if item["toxicity"] == "not_toxic"]
toxic_seeds = [item["text"] for item in seeds_data if item["toxicity"] == "toxic"]


# Programmatic Cross-Multiplication (Counterfactual Expansion)
final_dataset = []

def expand_templates(seeds, true_label):
    count = 0
    for seed in seeds:
        if "[IDENTITY]" not in seed:
            continue
        for category, terms in IDENTITY_VECTORS.items():
            for term in terms:
                is_control = (category == "control")
                final_text = seed.replace("[IDENTITY]", term)
                
                final_dataset.append({
                    "text": final_text,
                    "ground_truth": true_label,
                    "demographic_category": category,
                    "identity_term": term,
                    "is_control_sample": is_control
                })
                count += 1
    return count

# Run expansion mapping
b_count = expand_templates(benign_seeds, "not_toxic")
t_count = expand_templates(toxic_seeds, "toxic")

# Save the expanded dataset EXACTLY where the script resides
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
output_file_path = os.path.join(SCRIPT_DIRECTORY, "fairness_benchmark.json")

with open(output_file_path, "w", encoding="utf-8") as f:
    json.dump(final_dataset, f, indent=4, ensure_ascii=False)

print("=" * 60)
print("COUNTERFACTUAL FAIRNESS DATASET GENERATION SUCCESSFUL")
print("=" * 60)
print(f"Benign evaluation points:  {b_count}")
print(f"Toxic evaluation points:   {t_count}")
print(f"Total structured samples:  {b_count + t_count}")
print(f"Output Target Destination: \n-> {output_file_path}")
print("=" * 60)