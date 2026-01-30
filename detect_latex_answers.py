"""
Script to detect LaTeX formatted answers where \\frac is inside \\boxed in benchmark results.
"""

import re
from pathlib import Path

def detect_frac_in_boxed(text: str) -> bool:
    """
    Detects if \\frac is contained within a \\boxed{...} command.
    """
    # Pattern to find \boxed{...} and then check if it contains \frac
    # This regex looks for \boxed{ followed by any characters (including newlines) 
    # until a closing } that contains \frac inside.
    # Note: This is a heuristic for non-nested \boxed.
    boxed_blocks = re.findall(r"\\boxed\{(.*?)\}", text, flags=re.DOTALL)
    for block in boxed_blocks:
        if "\\frac" in block:
            return True
    return False

def process_logs(file_path: Path):
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return

    content = file_path.read_text()
    
    # Split by questions
    questions = re.split(r"####################################################################################################\nQUESTION \d+: ", content)
    
    results = []

    for q_block in questions[1:]:  # Skip header
        lines = q_block.splitlines()
        q_name = lines[0].strip()
        
        # Split by models within question
        model_blocks = re.split(r"====================================================================================================\n\nMODEL: ", q_block)
        
        for m_block in model_blocks[1:]:
            m_lines = m_block.splitlines()
            model_name = m_lines[0].strip()
            
            # Extract SCORE
            score_match = re.search(r"SCORE: ([\d.]+)/([\d.]+)", m_block)
            score = score_match.group(1) if score_match else "0.00"
            total_score = score_match.group(2) if score_match else "0.00"
            
            # Extract RUNs
            run_blocks = re.split(r"--- RUN #\d+ ---", m_block)
            
            for r_block in run_blocks[1:]:
                # Extract MODEL RESPONSE
                response_match = re.search(r"MODEL RESPONSE:\n(.*?)\n\nJUDGE EVALUATION:", r_block, re.DOTALL)
                if not response_match:
                    continue
                
                response = response_match.group(1).strip()
                
                if detect_frac_in_boxed(response):
                    # Potential fault if score is 0 or low
                    is_faulty = "Yes" if score == "0.00" or (float(score) < float(total_score) * 0.5 if total_score != "0.00" else False) else "No"
                    
                    # Extract the specific boxed content for display
                    boxed_match = re.search(r"(\\boxed\{.*?\\frac.*?\})", response, flags=re.DOTALL)
                    detected_box = boxed_match.group(1).replace("\n", " ") if boxed_match else "N/A"
                    
                    results.append({
                        "question": q_name,
                        "model": model_name,
                        "boxed_content": detected_box,
                        "points": f"{score}/{total_score}",
                        "faulty": is_faulty
                    })

    # Generate Markdown Table
    print("| Question | Model | Boxed Content with \\frac | Points | Potential Fault? |")
    print("|----------|-------|--------------------------|--------|------------------|")
    for res in results:
        print(f"| {res['question']} | {res['model']} | {res['boxed_content']} | {res['points']} | {res['faulty']} |")

if __name__ == "__main__":
    log_file = Path("results_advanced/benchmark_results_advanced_20260130_000507_final_all.txt")
    process_logs(log_file)
