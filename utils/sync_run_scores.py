"""
Synchronizes run-level scores from advanced logs to basic logs.
"""

import re
from pathlib import Path
from collections import defaultdict

def parse_advanced_log(filepath: Path) -> dict:
    content = filepath.read_text(encoding="utf-8")
    q_blocks = re.split(r"#{100}\nQUESTION \d+: (A\d+(?:\.\d+)?)\n#{100}", content)
    scores_map = defaultdict(dict)
    
    for i in range(1, len(q_blocks), 2):
        q_id = q_blocks[i]
        q_content = q_blocks[i+1]
        pts_match = re.search(r"POINTS: (\d+\.?\d*)", q_content)
        max_pts = float(pts_match.group(1)) if pts_match else 1.0
        
        model_blocks = re.split(r"MODEL: ([a-zA-Z0-9_/.-]+)\n", q_content)
        for j in range(1, len(model_blocks), 2):
            model_name = model_blocks[j].lower()
            model_content = model_blocks[j+1]
            run_blocks = re.split(r"--- RUN #\d+ ---\n", model_content)
            run_results = []
            for run_c in run_blocks[1:]:
                actual_s = 0.0
                score_match = re.search(r"SCORE:(\d+\.?\d*)/(\d+\.?\d*)", run_c)
                if score_match:
                    s, m = float(score_match.group(1)), float(score_match.group(2))
                    actual_s = (s / m) * max_pts if m > 0 else 0.0
                else:
                    verdict_match = re.search(r"JUDGE VERDICT: (Pass|Success)", run_c, re.IGNORECASE)
                    actual_s = max_pts if verdict_match else 0.0
                run_results.append(f"{actual_s:g}/{max_pts:g}")
            
            if q_id not in scores_map[model_name] or len(run_results) > len(scores_map[model_name][q_id]):
                scores_map[model_name][q_id] = run_results
    return scores_map

def update_basic_log(filepath: Path, scores_map: dict) -> None:
    content = filepath.read_text(encoding="utf-8")
    model_regex = r"([A-Z0-9_/.-]+ RESULTS:)\n(--------------------------------------------------------------------------------)"
    new_parts = []
    matches = list(re.finditer(model_regex, content))
    if not matches: return
    new_parts.append(content[:matches[0].start()])
    for i, match in enumerate(matches):
        header = match.group(0)
        model_name = match.group(1).replace(" RESULTS:", "").lower()
        end_idx = matches[i+1].start() if i+1 < len(matches) else len(content)
        section = content[match.end():end_idx]
        q_regex = r"(Question \d+ \((A\d+(?:\.\d+)?)\):)"
        sub_matches = list(re.finditer(q_regex, section))
        if sub_matches:
            new_section = [section[:sub_matches[0].start()]]
            for j, sub_match in enumerate(sub_matches):
                sub_header, q_id = sub_match.group(0), sub_match.group(2)
                sub_end = sub_matches[j+1].start() if j+1 < len(sub_matches) else len(section)
                sub_content = section[sub_match.end():sub_end]
                if model_name in scores_map and q_id in scores_map[model_name]:
                    runs_str = ", ".join(scores_map[model_name][q_id])
                    sub_content = re.sub(r"^\s+Runs: .*$", f"  Runs: {runs_str}", sub_content, flags=re.MULTILINE)
                new_section.append(sub_header + sub_content)
            new_parts.append(header + "".join(new_section))
        else:
            new_parts.append(header + section)
    filepath.write_text("".join(new_parts), encoding="utf-8")

def main():
    root = Path(__file__).parent.parent
    adv, bas = root/"results_advanced"/"benchmark_results_advanced_20260130_172227_final.txt", root/"results"/"benchmark_results_20260130_172227_final.txt"
    if adv.exists() and bas.exists():
        print("Syncing scores...")
        update_basic_log(bas, parse_advanced_log(adv))
        print("Done.")

if __name__ == "__main__":
    main()
