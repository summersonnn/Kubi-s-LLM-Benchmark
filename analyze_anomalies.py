"""
Analyze benchmark results to find strange success and failure instances.
A better model performing worse than a significantly lower ranked model.
"""
import polars as pl
from bs4 import BeautifulSoup
import re
import sys

def parse_score(score_str: str, max_points: float) -> float:
    if score_str == "PASS":
        return max_points
    if score_str == "FAIL":
        return 0.0
    try:
        # Handle cases like "0.75"
        return float(score_str)
    except ValueError:
        return 0.0

def main():
    file_path = "/home/kubilay/Projects/Kubis-Benchmark/results/performance_table_20260130_000507_final_all.html"
    try:
        with open(file_path, "r") as f:
            html_content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    soup = BeautifulSoup(html_content, "html.parser")

    # Get model names from the first header row
    header_row = soup.find("thead").find("tr")
    model_headers = header_row.find_all("th", class_="model-header")
    model_names = [th.get_text(strip=True) for th in model_headers]
    
    if not model_names:
        print("Error: Could not find model headers.")
        return

    # Find the TOTAL row to get final rankings
    # It's usually in tfoot, but let's be robust
    total_row = None
    tfoot = soup.find("tfoot")
    if tfoot:
        total_row = tfoot.find("tr")
    else:
        # Search all rows for "TOTAL"
        for tr in soup.find_all("tr"):
            if "TOTAL" in tr.get_text():
                total_row = tr
                break
    
    if not total_row:
        print("Error: Could not find TOTAL row.")
        return

    total_scores_cells = total_row.find_all("td", class_="score")
    model_totals = {}
    for i, cell in enumerate(total_scores_cells):
        if i < len(model_names):
            score_text = cell.get_text(strip=True).split('/')[0]
            try:
                model_totals[model_names[i]] = float(score_text)
            except ValueError:
                model_totals[model_names[i]] = 0.0

    # Rank models (1 is best)
    ranked_models = sorted(model_names, key=lambda m: model_totals.get(m, 0.0), reverse=True)
    rank_map = {model: i + 1 for i, model in enumerate(ranked_models)}

    # Parse questions from tbody
    results = []
    tbody = soup.find("tbody")
    if not tbody:
        print("Error: Could not find tbody.")
        return

    rows = tbody.find_all("tr")
    for row in rows:
        q_idx_cell = row.find("td", class_="q-col")
        if not q_idx_cell:
            continue
        q_idx = q_idx_cell.get_text(strip=True)
        
        tds = row.find_all("td")
        if len(tds) < 2:
            continue
            
        try:
            max_points = float(tds[1].get_text(strip=True))
        except ValueError:
            max_points = 1.0 # Default
        
        q_scores = {}
        for i, model in enumerate(model_names):
            # Model i starts at tds[2 + i*3]
            score_index = 2 + (i * 3)
            if score_index < len(tds):
                score_str = tds[score_index].get_text(strip=True)
                q_scores[model] = parse_score(score_str, max_points)
            else:
                q_scores[model] = 0.0
        
        results.append({
            "question": q_idx,
            "max_points": max_points,
            "scores": q_scores
        })

    # Prepare for analysis using Polars (as required)
    # We'll create a DataFrame where each row is a question and columns are model scores
    df_data = {"question": [r["question"] for r in results], "max_points": [r["max_points"] for r in results]}
    for model in model_names:
        df_data[model] = [r["scores"][model] for r in results]
    
    df = pl.DataFrame(df_data)

    # Analysis
    print("# Benchmark Anomalies Report")
    print(f"Total Models analyzed: {len(model_names)}")
    print(f"Top 3 Models: {', '.join(ranked_models[:3])}")
    print(f"Bottom 3 Models: {', '.join(ranked_models[-3:])}\n")

    top_n = min(5, len(ranked_models))
    bottom_n = min(5, len(ranked_models))
    
    top_models = ranked_models[:top_n]
    bottom_models = ranked_models[-bottom_n:]

    for r in results:
        q = r["question"]
        mx = r["max_points"]
        scores = r["scores"]
        
        # Rankings (1 is best)
        # Top 5 overall
        top_5_models = ranked_models[:5]
        # Bottom 10 overall
        bottom_10_models = ranked_models[-10:]

        findings = []
        
        # For each top model, if it failed (scored 0)
        # Check if any bottom model scored significantly better
        for tm in top_5_models:
            if scores[tm] == 0:
                # Find the best performing bottom model for this Q
                best_bottom = max(bottom_10_models, key=lambda m: scores[m])
                if scores[best_bottom] >= 0.5 * mx:
                    findings.append(f"❌ **Strange Failure**: Top-5 model {tm} (Rank {rank_map[tm]}) Got 0 while Bottom-10 model {best_bottom} (Rank {rank_map[best_bottom]}) Got {scores[best_bottom]}!")

        # For each bottom model, if it scored high (>= 80%)
        # Check if top models mostly failed
        for bm in bottom_10_models:
            if scores[bm] >= 0.8 * mx:
                # How many top models failed?
                failed_tops = [tm for tm in top_5_models if scores[tm] <= 0.2 * mx]
                if len(failed_tops) >= 3:
                     findings.append(f"✨ **Strange Success**: Bottom-10 model {bm} (Rank {rank_map[bm]}) Got {scores[bm]} while {len(failed_tops)} of Top-5 models failed!")

        if findings:
            print(f"### {q} (Points: {mx})")
            for f in findings:
                print(f"- {f}")
            print()

if __name__ == "__main__":
    main()
