import re
import sys
from collections import defaultdict

def verify_benchmark_scores(file_path: str):
    """
    Verifies that:
    1. The aggregated SCORE for each model on each question matches the average of the points from its individual runs.
    2. The sum of these calculated question scores matches the Final Score reported in the Final Standings.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    print(f"Verifying {file_path}...")
    
    # --- 1. Parse Final Standings ---
    print("\n--- Parsing Final Standings ---")
    final_standings_match = re.search(r'MODEL RANKINGS \(Automated Evaluation Only\)(.*)', content, re.DOTALL)
    reported_totals = {}
    
    if final_standings_match:
        standings_block = final_standings_match.group(1)
        # Regex for lines like: 1. google/gemini-3-flash-preview: 72.48/100.0 points (72.5%) - ...
        ranking_lines = re.findall(r'\d+\.\s+(.+?):\s+(\d+(?:\.\d+)?)/', standings_block)
        for model, score in ranking_lines:
            reported_totals[model.strip()] = float(score)
            # print(f"Found Final Score: {model.strip()} = {score}")
    else:
        print("Warning: Could not find Final Standings section.")

    # --- 2. Parse Detailed Results ---
    print("\n--- Verifying Detailed Scores ---")
    
    # Split by questions
    question_blocks = re.split(r'#{80,}\nQUESTION \d+:', content)
    
    # Store calculated totals
    calculated_totals = defaultdict(float)
    
    # For reporting discrepancies
    run_consistency_errors = []
    
    # Iterate question blocks
    for i, block in enumerate(question_blocks[1:], 1):
        lines = block.split('\n')
        q_id = lines[0].strip() # e.g. "A1"
        
        # Get Max Points for Question
        points_match = re.search(r'POINTS:\s+(\d+(?:\.\d+)?)', block)
        if not points_match:
            continue
        max_points = float(points_match.group(1))

        # Split by MODEL
        model_blocks = re.split(r'MODEL:\s+', block)
        
        for m_block in model_blocks[1:]:
            model_name_end = m_block.find('\n')
            model_name = m_block[:model_name_end].strip()
            
            # Extract Reported Question Score (for model/question pair)
            score_match = re.search(r'SCORE:\s+(\d+(?:\.\d+)?)/', m_block)
            reported_q_score = 0.0
            if score_match:
                reported_q_score = float(score_match.group(1))
            
            # Find all runs
            run_splits = re.split(r'---\s+RUN\s+#\d+\s+---', m_block)
            
            if len(run_splits) < 2:
                # No runs found? If reported score > 0 this is an issue, but if 0 it might be skipped.
                continue
                
            runs = run_splits[1:]
            run_points_list = []
            
            for run_content in runs:
                r_p = None
                
                # Priority 1: Explicit RUN RESULT as fraction
                # RUN RESULT: 0.50/0.50 pts
                rr_fraction_match = re.search(r'RUN RESULT:\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\s+pts', run_content)
                if rr_fraction_match:
                    num = float(rr_fraction_match.group(1))
                    den = float(rr_fraction_match.group(2))
                    # Normalization: (obtained / possible) * Question_Points
                    # Note: usually denominator in RUN RESULT is the max points for that question.
                    # But strict logic: Run Points are absolute.
                    # If we trust "X/Y pts" literally, X is the points obtained. 
                    # Assuming X is already scaled to the question points if Y matches Max Points? 
                    # Let's rely on X/Y. If Y == MaxPoints, then X is score.
                    # If Y != MaxPoints, maybe scaling needed? Usually Y == MaxPoints.
                    # Let's just take 'num' as the points obtained, assuming 'den' matches max_points or it's implicitly correct.
                    # Actually standardizing on: (num/den) * max_points seems safest if format varies, 
                    # but if 'pts' implies absolute points, then just 'num'. 
                    # Looking at file: "RUN RESULT: 1.00/1.00 pts" where POINTS: 1.
                    # "RUN RESULT: 0.00/3.00 pts" where POINTS: 3.
                    # So 'num' is the points.
                    r_p = num
                    
                # Priority 2: JUDGE EVALUATION with SCORE line
                if r_p is None:
                    je_score_match = re.search(r'JUDGE EVALUATION:[\s\S]*?SCORE:(\d+(?:\.\d+)?)/', run_content)
                    if je_score_match:
                         r_p = float(je_score_match.group(1))
                
                # Fallback: JUDGE VERDICT
                if r_p is None:
                    jv_match = re.search(r'JUDGE VERDICT:\s+(Pass|Fail|Success)', run_content)
                    if jv_match:
                        verdict = jv_match.group(1)
                        if verdict in ["Pass", "Success"]:
                            r_p = max_points
                        else:
                            r_p = 0.0
                
                if r_p is not None:
                    run_points_list.append(r_p)
            
            if not run_points_list:
                continue
                
            avg_run_score = sum(run_points_list) / len(run_points_list)
            
            # Check 1: Run Consistency (Avg vs Reported Question Score)
            if abs(reported_q_score - avg_run_score) > 0.01:
                run_consistency_errors.append(
                    f"Question {q_id} | Model {model_name}: Reported={reported_q_score:.2f}, Calc Avg={avg_run_score:.2f}"
                )
            
            # Add to total
            calculated_totals[model_name] += avg_run_score

    # --- 3. Compare Finals ---
    print("\n--- Final Score Consistency Check ---")
    print(f"{'Model':<50} | {'Reported':<10} | {'Calculated':<10} | {'Diff':<10} | {'Status'}")
    print("-" * 105)
    
    final_errors = []
    
    # Check all models found in reported totals
    all_models = set(reported_totals.keys()) | set(calculated_totals.keys())
    
    for model in sorted(all_models):
        reported = reported_totals.get(model, 0.0)
        calculated = calculated_totals.get(model, 0.0)
        diff = abs(reported - calculated)
        
        status = "PASS" if diff < 0.02 else "FAIL" # Tolerance
        
        print(f"{model:<50} | {reported:<10.2f} | {calculated:<10.2f} | {diff:<10.4f} | {status}")
        
        if status == "FAIL":
            final_errors.append(f"{model}: Reported {reported} != Calc {calculated}")

    # --- 4. Report Summary ---
    print("\n" + "="*50)
    print("VERIFICATION SUMMARY")
    print("="*50)
    
    if run_consistency_errors:
        print("\n[!] RUN CONSISTENCY ERRORS (Reported Question Score != Run Average):")
        for err in run_consistency_errors:
            print(f"  - {err}")
    else:
        print("\n[+] Run Consistency: OK")

    if final_errors:
        print("\n[!] FINAL SCORE ERRORS (Reported Total != Sum of Question Scores):")
        for err in final_errors:
            print(f"  - {err}")
    else:
        print("\n[+] Final Score Consistency: OK")

    if not run_consistency_errors and not final_errors:
        print("\nSUCCESS: All scores are consistent.")
    else:
        print("\nFAILURE: Discrepancies found.")

if __name__ == "__main__":
    LOG_FILE = "/home/kubilay/Projects/Kubis-Benchmark/results_advanced/benchmark_results_advanced_20260130_000507_final_all.txt"
    if len(sys.argv) > 1:
        LOG_FILE = sys.argv[1]
    
    verify_benchmark_scores(LOG_FILE)
