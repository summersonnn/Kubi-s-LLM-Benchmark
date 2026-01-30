"""
Rebuilds the basic log file from the advanced log file to fix structural corruption.
"""

import re
from pathlib import Path
from collections import defaultdict

def parse_question_index(q_id: str) -> tuple[int, int]:
    match = re.match(r"A(\d+)(?:\.(\d+))?", q_id)
    if match:
        main = int(match.group(1))
        sub = int(match.group(2)) if match.group(2) else 0
        return (main, sub)
    return (999, 999)

def main():
    project_root = Path(__file__).parent.parent
    adv_path = project_root / "results_advanced" / "benchmark_results_advanced_20260130_172227_final.txt"
    bas_path = project_root / "results" / "benchmark_results_20260130_172227_final.txt"
    
    if not adv_path.exists():
        print("Advanced log not found.")
        return
        
    print("Parsing advanced log...")
    content = adv_path.read_text(encoding="utf-8")
    
    # 1. Split into Intro, Question blocks, and Standings
    q_blocks = re.split(r"#{100}\nQUESTION \d+: (A\d+(?:\.\d+)?)\n#{100}", content)
    intro = q_blocks[0]
    
    # Extract standings from the last chunk
    last_chunk = q_blocks[-1]
    standings_match = re.search(r"#{100}\nMODEL RANKINGS", last_chunk)
    if standings_match:
        standings_text = last_chunk[standings_match.start():]
        q_blocks[-1] = last_chunk[:standings_match.start()]
    else:
        standings_text = ""

    # Parse questions and results
    # Data: model_name -> { q_id -> {expected, ave_score, max_pts, runs} }
    model_results = defaultdict(dict)
    
    for i in range(1, len(q_blocks), 2):
        q_id = q_blocks[i]
        q_content = q_blocks[i+1]
        
        expected_match = re.search(r"EXPECTED ANSWER: (.*)", q_content)
        expected = expected_match.group(1).strip() if expected_match else "N/A"
        
        pts_match = re.search(r"POINTS: (\d+\.?\d*)", q_content)
        max_pts = float(pts_match.group(1)) if pts_match else 1.0
        
        # Split by Model
        model_blocks = re.split(r"MODEL: ([a-zA-Z0-9_/.\-@]+)\n", q_content)
        for j in range(1, len(model_blocks), 2):
            model_name = model_blocks[j]
            model_content = model_blocks[j+1]
            
            # Average score
            score_match = re.search(r"SCORE: (\d+\.?\d*)/(\d+\.?\d*)", model_content)
            ave_score = score_match.group(1) if score_match else "0.00"
            
            # Individual runs
            run_parts = re.split(r"--- RUN #\d+ ---\n", model_content)
            runs = []
            for run_c in run_parts[1:]:
                actual_s = 0.0
                sm = re.search(r"SCORE:(\d+\.?\d*)/(\d+\.?\d*)", run_c)
                if sm:
                    s, m = float(sm.group(1)), float(sm.group(2))
                    actual_s = (s / m) * max_pts if m > 0 else 0.0
                else:
                    vm = re.search(r"JUDGE VERDICT: (Pass|Success)", run_c, re.IGNORECASE)
                    actual_s = max_pts if vm else 0.0
                runs.append(f"{actual_s:g}/{max_pts:g}")
            
            model_results[model_name][q_id] = {
                'expected': expected,
                'ave_score': ave_score,
                'max_pts': max_pts,
                'runs': runs
            }

    print("Reconstructing basic log...")
    output = []
    
    # Fix intro for basic log (convert # to = if appropriate, but keep original style)
    # The advanced intro uses #################### for header. 
    # Let's keep it consistent with basic log if possible.
    clean_intro = intro.replace("#" * 100, "=" * 80).strip()
    output.append(clean_intro + "\n\n")
    
    # Get sorted model list from intro
    models_section = re.search(r"Models benchmarked \(\d+\):\n(.*?)\n\n", intro, re.DOTALL)
    if models_section:
        models_list = re.findall(r"^- ([a-zA-Z0-9_/.\-@]+)", models_section.group(1), re.MULTILINE)
    else:
        models_list = sorted(model_results.keys())
    print(f"Total models parsed from question blocks: {len(model_results)}")
    # Check for mismatches
    mismatches = set(models_list) - set(model_results.keys())
    if mismatches:
        print(f"Warning: Models in list but not parsed: {mismatches}")

    for model in models_list:
        output.append("=" * 80 + "\n")
        output.append(f"{model.upper()} RESULTS:\n")
        output.append("-" * 80 + "\n\n")
        
        # Sort questions
        qs = sorted(model_results[model].keys(), key=parse_question_index)
        for k, q_id in enumerate(qs):
            res = model_results[model][q_id]
            output.append(f"Question {k+1} ({q_id}):\n")
            output.append(f"  Expected: {res['expected']}\n")
            output.append(f"  Score: {float(res['ave_score']):.2f}/{f'{res['max_pts']:g}'}\n")
            output.append(f"  Runs: {', '.join(res['runs'])}\n\n")
            
        output.append("\n")

    # Final Standings
    if standings_text:
        # Standardize the standings header for basic log
        standings_clean = standings_text.replace("#" * 100, "=" * 80)
        output.append(standings_clean)
    
    bas_path.write_text("".join(output), encoding="utf-8")
    print(f"Basic log reconstructed at: {bas_path}")

if __name__ == "__main__":
    main()
