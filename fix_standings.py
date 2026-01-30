import re
import sys
from collections import defaultdict

def fix_standings(file_path: str):
    print(f"fixing standings for {file_path}...")
    try:
        with open(file_path, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return

    # --- 1. Calculate Scores from Detailed Runs ---
    print("Calculating scores from runs...")
    
    # Split by questions
    question_blocks = re.split(r'#{80,}\nQUESTION \d+:', content)
    
    # Store calculated totals
    calculated_totals = defaultdict(float)
    
    # Iterate question blocks
    for i, block in enumerate(question_blocks[1:], 1):
        # Get Max Points (just in case needed, but we rely on runs mostly)
        points_match = re.search(r'POINTS:\s+(\d+(?:\.\d+)?)', block)
        max_points = float(points_match.group(1)) if points_match else 1.0

        # Split by MODEL
        model_blocks = re.split(r'MODEL:\s+', block)
        
        for m_block in model_blocks[1:]:
            model_name_end = m_block.find('\n')
            model_name = m_block[:model_name_end].strip()
            
            # Find all runs
            run_splits = re.split(r'---\s+RUN\s+#\d+\s+---', m_block)
            if len(run_splits) < 2:
                continue
                
            runs = run_splits[1:]
            run_points_list = []
            
            for run_content in runs:
                r_p = None
                
                # Priority 1: Explicit RUN RESULT as fraction
                rr_fraction_match = re.search(r'RUN RESULT:\s+(\d+(?:\.\d+)?)/(\d+(?:\.\d+)?)\s+pts', run_content)
                if rr_fraction_match:
                    num = float(rr_fraction_match.group(1))
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
            
            if run_points_list:
                avg_run_score = sum(run_points_list) / len(run_points_list)
                calculated_totals[model_name] += avg_run_score

    # --- 2. Parse Existing Standings to get Metadata (Tokens, Cost) ---
    print("Parsing existing standings for metadata...")
    final_standings_match = re.search(r'(MODEL RANKINGS \(Automated Evaluation Only\)\n#+\n)([\s\S]*?)(\n=)', content)
    
    if not final_standings_match:
        print("Error: Could not find MODEL RANKINGS section.")
        return

    header = final_standings_match.group(1)
    old_body = final_standings_match.group(2)
    footer_lines = content[final_standings_match.end(2):] 
    
    # Extract metadata: Tokens, Cost
    # Format: 1. model_name: SCORE/100.0 points (PCT%) - TOKENS tokens - $COST
    model_metadata = {}
    
    for line in old_body.strip().split('\n'):
        if not line.strip(): continue
        
        # Regex to capture Model, Tokens, Cost
        # 1. google/gemini-...: 72.48/100.0 points (72.5%) - 1591284 tokens - $4.785
        # We want to be robust.
        match = re.match(r'\d+\.\s+(.+?):\s+[\d\.]+/[\d\.]+ points \([\d\.]+%\) - (\d+) tokens - \$([\d\.]+)', line)
        if match:
            m_name = match.group(1)
            tokens = match.group(2)
            cost = match.group(3)
            model_metadata[m_name] = {'tokens': tokens, 'cost': cost}
        else:
            print(f"Warning: Could not parse line in table: {line}")
            # Try looser match if model name has weird chars?
            # Actually just continue, if we calculated a score for it, we might lose tokens/cost if parse fails.
            pass

    # --- 3. Reconstruct Table ---
    print("Reconstructing table...")
    
    new_table_lines = []
    
    # Rank models by calculated score (descending)
    ranked_models = sorted(calculated_totals.items(), key=lambda x: x[1], reverse=True)
    
    for rank, (model, score) in enumerate(ranked_models, 1):
        meta = model_metadata.get(model, {'tokens': '0', 'cost': '0.000'})
        tokens = meta['tokens']
        cost = meta['cost']
        
        percent = score # Assuming 100 points total. File says "/100.0 points". 
        # The file header assumes 100 max points.
        
        # Format line
        # 1. model: 72.48/100.0 points (72.5%) - ...
        line = f"{rank}. {model}: {score:.2f}/100.0 points ({percent:.1f}%) - {tokens} tokens - ${cost}"
        new_table_lines.append(line)
        
    new_body = "\n".join(new_table_lines) + "\n"
    
    # --- 4. Replace in Content ---
    new_content = content[:final_standings_match.start(2)] + new_body + content[final_standings_match.end(2):]
    
    # --- 5. Write File ---
    with open(file_path, 'w') as f:
        f.write(new_content)
        
    print("Successfully updated standings.")

if __name__ == "__main__":
    LOG_FILE = "/home/kubilay/Projects/Kubis-Benchmark/results_advanced/benchmark_results_advanced_20260130_000507_final_all.txt"
    if len(sys.argv) > 1:
        LOG_FILE = sys.argv[1]
    
    fix_standings(LOG_FILE)
