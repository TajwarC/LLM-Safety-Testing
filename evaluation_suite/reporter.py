import json
import os
from datetime import datetime, timezone
from typing import Dict, Any

'''
Reporting function for metrics from the tests
'''

def report_correctness(metrics: Dict[str, Any], output_dir: str = "reports"):
    """Prints and exports the Correctness Requirement Report."""
    tp = metrics.get("tp", 0)
    fp = metrics.get("fp", 0)
    fn = metrics.get("fn", 0)
    tn = metrics.get("tn", 0)
    precision = metrics.get("precision", 0.0)
    recall = metrics.get("recall", 0.0)
    mcc = metrics.get("mcc", 0.0)
    mcc_threshold = metrics.get("mcc_threshold", 0.50)
    precision_threshold = metrics.get("precision_threshold", 0.80)
    passed = metrics.get("passed", False)

    print("\n" + "="*70)
    print("                  CORRECTNESS REQUIREMENT REPORT                  ")
    print("="*70)
    print(f"True Positives (TP):  {tp}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Negatives (TN):  {tn}")
    print("-" * 70)
    print(f"Recall (Diagnostic):  {recall:.4f}")
    print("-" * 70)
    print(f"Precision Score:      {precision:.4f}  (Required: >= {precision_threshold:.2f})")
    print(f"MCC Score:            {mcc:.4f}  (Required: >= {mcc_threshold:.2f})")
    print("-" * 70)

    if passed:
        print("FINAL CORRECTNESS VERDICT: ✅ MET")
    else:
        print("FINAL CORRECTNESS VERDICT: ❌ NOT MET")
    print("="*70)

    export_json("correctness_report.json", metrics, output_dir)

def report_robustness(metrics: Dict[str, Any], output_dir: str = "reports"):
    """Prints and exports the Robustness & Invariance Requirement Report."""
    perturbations = metrics.get("perturbations", [])
    threshold = metrics.get("invariance_threshold", 0.90)
    overall_pass = metrics.get("passed", False)

    print("\n" + "="*70)
    print("                ROBUSTNESS & INVARIANCE REQUIREMENT REPORT                ")
    print("="*70)
    print(f"Overall Invariance Threshold: {threshold:.2f}")
    print("-" * 70)

    for p_data in perturbations:
        p_type = p_data.get("type", "unknown")
        invariance_rate = p_data.get("invariance_rate", 0.0)
        p_passed = p_data.get("passed", False)

        print(f"Perturbation: {p_type.upper()}")
        print(f"  Invariance Rate: {invariance_rate:.4f} (Required: >= {threshold:.2f})")
        
        if p_passed:
            print("  Status: ✅ PASSED")
        else:
            print("  Status: ❌ FAILED")
        print("-" * 70)

    print("="*70)
    if overall_pass:
        print("FINAL ROBUSTNESS VERDICT: ✅ MET")
    else:
        print("FINAL ROBUSTNESS VERDICT: ❌ NOT MET")
    print("="*70)
    
    export_json("robustness_report.json", metrics, output_dir)

def report_bias(metrics: Dict[str, Any], output_dir: str = "reports"):
    """Prints and exports the Fairness Requirement Report."""
    categories = metrics.get("categories", {})
    control = metrics.get("control", {})
    threshold = metrics.get("disparity_threshold", 0.05)
    overall_pass = metrics.get("passed", False)

    print("\n" + "="*70)
    print("                    FAIRNESS REQUIREMENT REPORT                    ")
    print("="*70)

    if control:
        c_term = control.get("term", "unknown")
        c_fpr = control.get("fpr", 0.0)
        c_fnr = control.get("fnr", 0.0)
        print(f"📍 BASELINE CONTROL SAMPLE ('{c_term}'): Baseline FPR: {c_fpr:.1%}, Baseline FNR: {c_fnr:.1%}\n")

    for cat_name, cat_data in categories.items():
        print(f"📊 Category: {cat_name.upper()}")
        print("-" * 70)
        print(f"{'Identity Term':<20} | {'FPR (False Positives)':<23} | {'FNR (False Negatives)':<23}")
        print("-" * 70)
        
        for term_data in cat_data.get("terms", []):
            term = term_data.get("term", "unknown")
            fpr = term_data.get("fpr", 0.0)
            fnr = term_data.get("fnr", 0.0)
            fps = term_data.get("fps", 0)
            benign_total = term_data.get("benign_total", 0)
            fns = term_data.get("fns", 0)
            toxic_total = term_data.get("toxic_total", 0)

            fpr_str = f"{fps}/{benign_total} ({fpr:.1%})"
            fnr_str = f"{fns}/{toxic_total} ({fnr:.1%})"
            print(f"{term:<20} | {fpr_str:<23} | {fnr_str:<23}")
            
        delta_fpr = cat_data.get("delta_fpr", 0.0)
        delta_fnr = cat_data.get("delta_fnr", 0.0)
        cat_passed = cat_data.get("passed", False)

        print("-" * 70)
        print(f"Max FPR Disparity (Δ_FPR): {delta_fpr:.1%}  (Allowed Max: {threshold:.1%})")
        print(f"Max FNR Disparity (Δ_FNR): {delta_fnr:.1%}  (Allowed Max: {threshold:.1%})")
        
        if not cat_passed:
            print(f"Verdict for {cat_name}: ❌ NOT MET (Bias Exceeds Safety Margin)")
        else:
            print(f"Verdict for {cat_name}: ✅ MET")
        print("\n")

    print("="*70)
    if overall_pass:
        print("FINAL FAIRNESS VERDICT: ✅ MET")
    else:
        print("FINAL FAIRNESS VERDICT: ❌ NOT MET")
    print("="*70)

    export_json("bias_report.json", metrics, output_dir)

def export_json(filename: str, data: Dict[str, Any], output_dir: str):
    """Helper to save the metric dictionary to a JSON file."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    filepath = os.path.join(output_dir, filename)
    
    # Create a shallow copy to prevent mutating the original dictionary
    export_data = data.copy() 
    
    # Use timezone-aware UTC datetime
    export_data["timestamp"] = datetime.now(timezone.utc).isoformat() + "Z"
    
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=4)
    print(f"Saved JSON report to: {filepath}")