import re
from collections import defaultdict

def verify_points(file_path: str):
    """
    Parses the benchmark log file and verifies that the sum of points for each question
    matches the total points reported in the final standings table.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    # 1. Parse Standings Table
    # The table follows the header: "MODEL RANKINGS (Automated Evaluation Only)"
    if "MODEL RANKINGS (Automated Evaluation Only)" not in content:
        print("Error: Could not find 'MODEL RANKINGS (Automated Evaluation Only)' header.")
        return
        
    standings_section = content.split("MODEL RANKINGS (Automated Evaluation Only)")[-1]
    # Match: "1. model/name: 72.48/100.0 points"
    standings_pattern = re.compile(r'^\d+\.\s+(.*?):\s+(\d+\.\d+)/\d+\.\d+\s+points', re.MULTILINE)
    
    standings = {}
    for match in standings_pattern.finditer(standings_section):
        model_name = match.group(1).strip()
        total_points = float(match.group(2))
        standings[model_name] = total_points

    if not standings:
        print("Error: Found the standings header but could not parse any rankings.")
        return

    # 2. Parse Question Scores
    # Match aggregated model scores per question:
    # MODEL: model/name
    # SCORE: X.XX/Y.YY
    score_pattern = re.compile(r'MODEL:\s+(.*?)\nSCORE:\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)')
    
    model_summed_scores = defaultdict(float)
    matches = list(score_pattern.finditer(content))
    
    for match in matches:
        model_name = match.group(1).strip()
        points_awarded = float(match.group(2))
        model_summed_scores[model_name] += points_awarded

    # 3. Output Results Table
    print(f"\n{'Model Name':<50} | {'Table Total':<12} | {'Summed Total':<12} | {'Diff':<8} | {'Status'}")
    print("-" * 95)
    
    all_models = sorted(set(standings.keys()) | set(model_summed_scores.keys()))
    mismatch_found = False
    
    for model in all_models:
        table_val = standings.get(model, 0.0)
        summed_val = model_summed_scores.get(model, 0.0)
        diff = abs(table_val - summed_val)
        # Using a small epsilon for float comparison
        status = "PASS" if diff < 0.001 else "FAIL"
        if status == "FAIL":
            mismatch_found = True
        
        print(f"{model:<50} | {table_val:<12.2f} | {summed_val:<12.2f} | {diff:<8.4f} | {status}")

    if mismatch_found:
        print("\n[!] ERROR: Discrepancies detected between question sums and final standings!")
    else:
        print("\n[+] SUCCESS: All model totals match the sum of their question points.")

if __name__ == "__main__":
    LOG_FILE = "/home/kubilay/Projects/Kubis-Benchmark/results_advanced/benchmark_results_advanced_20260130_000507_final_all.txt"
    verify_points(LOG_FILE)
