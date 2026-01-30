import re
import math

def verify_run_consistency(file_path: str):
    """
    Verifies that the aggregated SCORE for each model on each question
    matches the average of the points from its individual runs.
    
    Logic for Run Points:
    1. Check for `RUN RESULT: X/Y pts`. Points = (X/Y) * Question_Points.
    2. Check for `JUDGE EVALUATION` ... `SCORE: X/Y`. Points = X.
    3. Fallback to `JUDGE VERDICT`: Pass -> Question_Points, Fail -> 0.
    """
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    # Split by questions
    question_blocks = re.split(r'#{80,}\nQUESTION \d+:', content)
    
    if len(question_blocks) < 2:
        print("Error: Could not find question blocks.")
        return

    print(f"{'Question':<10} | {'Model':<50} | {'Reported':<8} | {'Calc Avg':<8} | {'Diff':<8} | {'Status'}")
    print("-" * 110)

    discrepancy_count = 0
    total_checks = 0

    for i, block in enumerate(question_blocks[1:], 1):
        lines = block.split('\n')
        q_id = lines[0].strip()
        
        points_match = re.search(r'POINTS:\s+(\d+(?:\.\d+)?)', block)
        if not points_match:
            # Maybe it's at the end of the block? 
            # Or formatted differently.
            # print(f"Warning: Could not find POINTS for question {q_id}")
            continue
            
        max_points = float(points_match.group(1))

        # Split by MODEL
        model_blocks = re.split(r'MODEL:\s+', block)
        
        for m_block in model_blocks[1:]:
            model_name_end = m_block.find('\n')
            model_name = m_block[:model_name_end].strip()
            
            # Extract Reported Score
            score_match = re.search(r'SCORE:\s+(\d+(?:\.\d+)?)/', m_block)
            if not score_match:
                continue
            reported_score = float(score_match.group(1))
            
            # Find all runs
            run_splits = re.split(r'---\s+RUN\s+#\d+\s+---', m_block)
            
            if len(run_splits) < 2:
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
                    if den != 0:
                        r_p = (num / den) * max_points
                    else:
                        r_p = 0.0 # Should not happen

                # Priority 2: JUDGE EVALUATION with SCORE line
                if r_p is None:
                    # Look for SCORE: X/Y inside JUDGE EVALUATION section
                    # We need to be careful not to match the Model's global score if regex is too broad.
                    # run_content is just the run part.
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
                else:
                    # Could not determine points
                    # This implies incomplete run or unknown format
                    pass

            if not run_points_list:
                continue
                
            avg_score = sum(run_points_list) / len(run_points_list)
            total_checks += 1
            
            diff = abs(reported_score - avg_score)
            status = "PASS" if diff < 0.01 else "FAIL" # Relaxed tolerance slightly for floating point
            
            if status == "FAIL":
                discrepancy_count += 1
                print(f"{q_id:<10} | {model_name:<50} | {reported_score:<8.2f} | {avg_score:<8.2f} | {diff:<8.4f} | {status}")

    if discrepancy_count == 0:
        print(f"\n[+] SUCCESS: All {total_checks} model/question pairs match (within tolerance).")
    else:
        print(f"\n[!] ERROR: {discrepancy_count} discrepancies found out of {total_checks} checks.")

if __name__ == "__main__":
    LOG_FILE = "/home/kubilay/Projects/Kubis-Benchmark/results_advanced/benchmark_results_advanced_20260130_000507_final_all.txt"
    verify_run_consistency(LOG_FILE)
