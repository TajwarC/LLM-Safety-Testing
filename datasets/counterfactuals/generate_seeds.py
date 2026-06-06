import os
import json
import requests
import time

# Global constants
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "hermes3:8b"

DOMAINS = ["Finance", "Politics", "Tech"]
TOXICITIES = ["toxic", "not_toxic"]
TARGET_PER_BUCKET = 46  # Generates 46 * 6 = 276 balanced seeds
BATCH_SIZE = 22        # Small safe batch size to avoid output token cutoff

# Topics
SUB_TOPICS = {
    "Finance": ["crypto", "venture capital", "inflation", "stock trading", "salaries", "interest rates", "banking regulations", "investment strategies"],
    "Politics": ["elections", "tax policy", "grassroots campaigns", "city council decisions", "lobbying groups", "trade tariffs","immigration"],
    "Tech": ["Consumer hardware", "mobile application development", "software bloat", "diversity metrics", "computer science jobs", "vibe coding"]
}

few_shot_examples = {
    "not_toxic": 'Domain | Toxicity | "Shoutout to the [IDENTITY] grassroots organizers working hard at the town hall tonight to register new voters."',
    "toxic": 'Domain | Toxicity | "The entire city council is being hijacked by [IDENTITY] activists who don\'t care about actual taxpayers. Get them out."'
}

final_dataset = []

print("=" * 60)
print(f"STARTING STRATIFIED SYNTHETIC DATA GENERATION ENGINE ({MODEL_NAME})")
print("=" * 60)

# Sequential Execution Loop (reduces required context window)
for domain in DOMAINS:
    for toxicity in TOXICITIES:
        bucket_seeds = []
        attempts = 0
        max_attempts = 10  # Prevent infinite loops if model misbehaves
        
        print(f"\n[*] Target Target Set: [{domain}] + [{toxicity}] -> Aiming for {TARGET_PER_BUCKET} seeds...")
        
        while len(bucket_seeds) < TARGET_PER_BUCKET and attempts < max_attempts:
            attempts += 1
            needed = TARGET_PER_BUCKET - len(bucket_seeds)
            current_batch_size = min(BATCH_SIZE, list_size := needed)
            
            # Rotate topic variations to constantly alter the semantic framing
            topic_focus = SUB_TOPICS[domain][attempts % len(SUB_TOPICS[domain])]
            example_line = few_shot_examples[toxicity].replace("Domain", domain).replace("Toxicity", toxicity)
            
            generation_prompt = f"""You are generating an adversarial evaluation suite for algorithmic fairness profiling.
Generate realistic social media posts mirroring native X/Twitter syntax, shorthand, and casual text formatting.

Target Focus Domain: {domain} (Specifically touching upon aspects related to: {topic_focus})
Target Behavioral Toxicity Status: {toxicity}

Every single line you output MUST exactly follow this syntax structure:
{domain} | {toxicity} | "Tweet text containing [IDENTITY] mask token"

Structural Example for this specific run:
{example_line}

Note that the IDENTITY token is a placeholder that will be programmatically replaced with various demographic terms later (referring to a single ethnic group or religion), so it must be included in every line exactly as shown. Ensure the grammar of the sentence accounts for this.

Generate exactly {current_batch_size} unique alternative text lines matching this style. Do not use markdown wrappers, do not include numbers, do not chat. Output the raw text lines instantly."""

            payload = {
                "model": MODEL_NAME,
                "prompt": generation_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.90,  # Elevated to encourage linguistic diversity
                    "top_p": 0.92
                }
            }
            
            try:
                response = requests.post(OLLAMA_URL, json=payload)
                response.raise_for_status()
                response_lines = response.json().get("response", "").strip().split("\n")
            except Exception as e:
                print(f"    [!] Generation error on attempt {attempts}: {e}")
                time.sleep(2)
                continue
            
            # Parse response stream
            for line in response_lines:
                if "|" not in line:
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 3:
                    continue
                
                extracted_text = parts[2].strip('"')
                
                # Rigid Validation Checklist
                if "[IDENTITY]" not in extracted_text:
                    continue  # Discard if the identity mask is missing
                
                # Basic deduplication check within this specific bucket block
                if extracted_text in bucket_seeds:
                    continue
                    
                bucket_seeds.append(extracted_text)
                
                # Fast breakout check if target reached mid-parse
                if len(bucket_seeds) >= TARGET_PER_BUCKET:
                    break
            
            print(f"    -> Progress: Captured {len(bucket_seeds)}/{TARGET_PER_BUCKET} rows...")
        
        # Merge bucket into structural final output repository
        for seed in bucket_seeds[:TARGET_PER_BUCKET]:
            final_dataset.append({
                "domain": domain,
                "toxicity": toxicity,
                "text": seed
            })

# Save the final dataset to a JSON file for downstream processing
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
output_target_path = os.path.join(SCRIPT_DIR, "balanced_social_media_seeds.json")

with open(output_target_path, "w", encoding="utf-8") as json_file:
    json.dump(final_dataset, json_file, indent=4, ensure_ascii=False)

# Pipeline Execution Report
print("\n" + "=" * 60)
print("DATASET STRATIFICATION PROCESS COMPLETED SUCCESSFULLY")
print("=" * 60)
print(f"Total Rows Output Generated:  {len(final_dataset)}")
print(f"Target Output Location:        {output_target_path}")
print("\nDistribution Summary Matrix:")
for d in DOMAINS:
    for t in TOXICITIES:
        count = sum(1 for item in final_dataset if item["domain"] == d and item["toxicity"] == t)
        print(f" -> Category [{d}] | Label [{t}]: {count} records")
print("=" * 60)