"""
Script to combine results from multiple benchmark runs into a single report.
Parses existing 'results_advanced' text files, merges them, and regenerates
the HTML performance table and text reports.

Usage:
    uv run utils/combine_results.py
    
    # Or with arguments (not yet implemented, interactive mode for now)
"""

import os
import sys
import re
import glob
from datetime import datetime
from typing import Dict, Any, List, Set, Tuple

import os
import sys
import re
import glob
from datetime import datetime
from typing import Dict, Any, List, Set, Tuple

# Fix path to ensure we import 'utils' as a package from project root
# and NOT 'utils.py' from the local directory.
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Change CWD to project root so relative paths (results/, results_advanced/) work
# os.chdir(project_root) # REMOVED per user request

# Remove the script directory from path if it allows importing 'utils' as a module
script_dir = os.path.dirname(os.path.abspath(__file__))
if script_dir in sys.path:
    sys.path.remove(script_dir)

from utils.reporting import (
    write_results_file,
    write_advanced_results_file,
    generate_performance_html,
    print_final_rankings
)
from utils.question_loader import load_questions_data as original_load_questions_data

# Mock setup_logging since we are in a standalone script or just use print
import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

def clean_ansi(text: str) -> str:
    """Removes ANSI escape codes coming from colored output in files (if any)."""
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    return ansi_escape.sub('', text)

def parse_advanced_file(filepath: str) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    """
    Parses a benchmark_results_advanced_*.txt file.
    Returns:
        question_codes: List[str] - The list of questions found in the file.
        all_results: Dict - structure compatible with reporting.py
                     {question_code: {model_name: {stats...}}}
    """
    logger.info(f"Parsing file: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split by Question headers
    # Format: ####################################################################################################
    # QUESTION 1: A1-QuestionName
    # ####################################################################################################
    
    question_sections = re.split(r'#{80,}\nQUESTION \d+: ', content)
    
    # The first split is the header/summary, ignore it
    if len(question_sections) < 2:
        logger.error("No questions found in file parsing.")
        return [], {}
        
    question_codes = []
    all_results = {}
    
    # Process each question section
    for section in question_sections[1:]: # Skip header
        # Extract Question Code (e.g. "A1-QuestionName")
        # The split consumed "QUESTION N: ", so we are looking at "A1-Foo\n#####..."
        lines = section.split('\n')
        q_code_line = lines[0].strip()
        
        # Sometimes there might be trailing #'s or newlines
        q_code = q_code_line.split('\n')[0].strip()
        question_codes.append(q_code)
        all_results[q_code] = {}
        
        # Split by Model separators
        # Format: ====================================================================================================
        # (Wait, actually usually "----------------------------------------------------------------------------------------------------" or just explicit "MODEL: " headers?)
        # Looking at reporting.py:
        # f.write(f"MODEL: {model}\n")
        
        # We can split by "\nMODEL: "
        # BUT we must be careful not to split inside a response.
        # However, "MODEL: " at the start of a line is a strong signal.
        
        model_sections = re.split(r'\nMODEL: ', section)
        
        # model_sections[0] is the question text/ground truth part.
        
        for m_section in model_sections[1:]:
            # First line is the model name
            m_lines = m_section.split('\n')
            model_name = m_lines[0].strip()
            
            # Initialize model data
            all_results[q_code][model_name] = {
                "score": 0.0,
                "total_tokens": 0,
                "total_cost": 0.0,
                "runs": []
            }
            
            # Extract Score, Tokens, Cost
            # SCORE: 1.00/1.00
            # TOKENS USED: 123
            # COST INCURRED: $0.001234
            
            score_match = re.search(r'SCORE: ([\d\.]+)/[\d\.]+', m_section)
            if score_match:
                all_results[q_code][model_name]["score"] = float(score_match.group(1))
                
            tokens_match = re.search(r'TOKENS USED: (\d+)', m_section)
            if tokens_match:
                all_results[q_code][model_name]["total_tokens"] = int(tokens_match.group(1))
                
            cost_match = re.search(r'COST INCURRED: \$([\d\.]+)', m_section)
            if cost_match:
                all_results[q_code][model_name]["total_cost"] = float(cost_match.group(1))
            
            # Parse Runs
            # --- RUN #1 ---
            run_splits = re.split(r'--- RUN #\d+ ---', m_section)
            
            # run_splits[0] is the summary stats we just parsed
            for run_text in run_splits[1:]:
                run_data = {
                    "success": False,
                    "response": "",
                    "model_reasoning": None,
                    "judge_reasoning": "",
                    "judge_verdict": "",
                    "run_score": None, # Optional granular score
                    "run_max": None
                }
                
                # Extract Response
                # MODEL RESPONSE:
                # ...
                resp_match = re.search(r'MODEL RESPONSE:\n(.*?)(?=\nJUDGE EVALUATION:|\n\n--- RUN|$)', run_text, re.DOTALL)
                if resp_match:
                    run_data["response"] = resp_match.group(1).strip()
                
                # Extract Reasoning (if any)
                reasoning_match = re.search(r'MODEL THINKING/REASONING:\n(.*?)(?=\nMODEL RESPONSE:)', run_text, re.DOTALL)
                if reasoning_match:
                    run_data["model_reasoning"] = reasoning_match.group(1).strip()
                    
                # Extract Judge Info
                # JUDGE EVALUATION:
                # ...
                # JUDGE VERDICT: Pass
                
                judge_eval_match = re.search(r'JUDGE EVALUATION:\n(.*?)(?=\nJUDGE VERDICT:)', run_text, re.DOTALL)
                if judge_eval_match:
                    run_data["judge_reasoning"] = judge_eval_match.group(1).strip()
                    
                verdict_match = re.search(r'JUDGE VERDICT: (.*)', run_text)
                if verdict_match:
                    run_data["judge_verdict"] = verdict_match.group(1).strip()
                
                # Extract granular result if available
                # RUN RESULT: 1/5 pts  OR  RUN RESULT: PASS
                run_res_match = re.search(r'RUN RESULT: (.*)', run_text)
                if run_res_match:
                    res_str = run_res_match.group(1).strip()
                    if "PASS" in res_str:
                        run_data["success"] = True
                    elif "FAIL" in res_str:
                        run_data["success"] = False
                    
                    # Check for "X/Y pts"
                    pts_match = re.match(r'(\d+)/(\d+)', res_str)
                    if pts_match:
                        run_data["run_score"] = int(pts_match.group(1))
                        run_data["run_max"] = int(pts_match.group(2))
                        run_data["success"] = (run_data["run_score"] > 0)
                
                all_results[q_code][model_name]["runs"].append(run_data)

    return question_codes, all_results

def get_available_files() -> List[str]:
    """Returns sorted list of available advanced result files."""
    results_dir = os.path.join(project_root, "results_advanced")
    files = glob.glob(os.path.join(results_dir, "benchmark_results_advanced_*.txt"))
    # Sort by timestamp (descending)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files

def main():
    print("="*60)
    print("      BENCHMARK RESULT COMBINER")
    print("="*60)
    
    import argparse
    parser = argparse.ArgumentParser(description="Combine benchmark results.")
    parser.add_argument("--all", action="store_true", help="Combine ALL available files without prompting")
    parser.add_argument("--files", type=str, help="Comma-separated list of file indices to combine (e.g. '1,2')")
    args = parser.parse_args()

    files = get_available_files()
    
    if not files:
        print("No advanced result files found in 'results_advanced/'")
        return

    print("\nAvailable Files:")
    for idx, f in enumerate(files, 1):
        timestamp = os.path.basename(f).split('_')[-1].replace('.txt', '')
        # Try to format timestamp nicely
        try:
            ts_obj = datetime.strptime(timestamp, "%Y%m%d%H%M%S") # 2026 01 25 ...
            # Actually format is %Y%m%d_%H%M%S usually, but let's be robust
        except:
             # Try standard format
            try:
                ts_obj = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
                ts_str = ts_obj.strftime("%Y-%m-%d %H:%M:%S")
            except:
                ts_str = timestamp
        else:
            ts_str = ts_obj.strftime("%Y-%m-%d %H:%M:%S")

        print(f"{idx}. {os.path.basename(f)}  (Date: {ts_str})")
    
    selected_files = []
    
    if args.all:
        selected_files = files
    elif args.files:
        try:
            indices = [int(x.strip()) for x in args.files.split(',') if x.strip()]
            for i in indices:
                if 1 <= i <= len(files):
                    selected_files.append(files[i-1])
                else:
                    print(f"Warning: Index {i} out of range, skipping.")
        except ValueError:
            print("Invalid input in --files.")
            return
    else:
        # Interactive Mode
        print("\nSelect files to combine (comma-separated IDs, e.g., '1,2,3' or 'all'):")
        selection = input("> ").strip().lower()
        
        if selection == 'all':
            selected_files = files
        else:
            try:
                indices = [int(x.strip()) for x in selection.split(',') if x.strip()]
                for i in indices:
                    if 1 <= i <= len(files):
                        selected_files.append(files[i-1])
                    else:
                        print(f"Warning: Index {i} out of range, skipping.")
            except ValueError:
                print("Invalid input.")
                return

    if not selected_files:
        print("No files selected.")
        return

    print(f"\nProcessing {len(selected_files)} files...")
    
    # MASTER DATA STRUCTURES
    combined_question_codes = None
    combined_all_results = {} # {code: {model: data}}
    
    file_models = {} # {filename: [models]}
    
    for fpath in selected_files:
        q_codes, results = parse_advanced_file(fpath)
        
        if not q_codes:
            print(f"Skipping {os.path.basename(fpath)}: Failed to parse or empty.")
            continue
            
        # Validation: Question sets must enable merging
        # Optimally, they should be IDENTICAL. 
        # But user might have run subset. 
        # Requirement: "Even if a single question is different between benchmarks, they should not be combined."
        # strict check:
        
        if combined_question_codes is None:
            combined_question_codes = q_codes
            # Check for dupes?
            # q_codes = sorted(q_codes)
        else:
            # Compare sets
            if set(q_codes) != set(combined_question_codes):
                print(f"ERROR: File {os.path.basename(fpath)} has a different set of questions!")
                print(f"Expected: {len(combined_question_codes)} questions")
                print(f"Found:    {len(q_codes)} questions")
                diff = set(combined_question_codes) ^ set(q_codes)
                print(f"Difference: {list(diff)[:5]}...")
                print("ABORTING COMBINATION.")
                return

        # Merge Results
        filename = os.path.basename(fpath)
        file_models[filename] = []
        
        for q_code, model_dict in results.items():
            if q_code not in combined_all_results:
                combined_all_results[q_code] = {}
            
            for model_name, data in model_dict.items():
                if model_name in combined_all_results[q_code]:
                    print(f"WARNING: Model '{model_name}' appears in multiple files! Overwriting with data from {filename}.")
                
                combined_all_results[q_code][model_name] = data
                if model_name not in file_models[filename]:
                    file_models[filename].append(model_name)

    if not combined_all_results:
        print("No data extracted.")
        return

    # Verify we have models
    all_models = set()
    for q_code in combined_all_results:
        all_models.update(combined_all_results[q_code].keys())
    
    all_models = sorted(list(all_models))
    print(f"\nCombined {len(all_models)} models: {', '.join(all_models)}")
    
    # Re-hydrate Questions Data (Category, Subcategory, Points, Truth)
    print("\nLoading canonical question metadata...")
    # Fix import issue by creating a wrapper load_questions that doesn't rely on global logger if needed,
    # but we imported original_load_questions_data
    
    try:
        # We need the full question objects
        questions_data, valid_codes = original_load_questions_data(combined_question_codes)
        
        if len(valid_codes) != len(combined_question_codes):
            print("WARNING: Some questions could not be re-loaded from disk (files missing?). Metadata might be incomplete.")
    except Exception as e:
        print(f"Error loading question data: {e}")
        return

    # Generate Output
    print("\nGenerating Reports...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_combined_models")
    
    # 1. HTML Performance Table
    html_path = generate_performance_html(
        all_models,
        valid_codes,
        combined_all_results,
        questions_data,
        timestamp,
        output_dir=os.path.join(project_root, "results")
    )
    print(f"[SUCCESS] Generated HTML: file://{html_path}")
    
    # 2. Results Text File
    txt_path = write_results_file(
        all_models,
        valid_codes,
        combined_all_results,
        questions_data,
        timestamp,
        output_dir=os.path.join(project_root, "results")
    )
    print(f"[SUCCESS] Generated Results: {txt_path}")

    # 3. Advanced Results Text File
    adv_path = write_advanced_results_file(
        all_models,
        valid_codes,
        combined_all_results,
        questions_data,
        timestamp,
        output_dir=os.path.join(project_root, "results_advanced")
    )
    print(f"[SUCCESS] Generated Advanced Results: {adv_path}")
    
    # 4. Console Summary
    print_final_rankings(
        all_models,
        valid_codes,
        combined_all_results,
        questions_data
    )

if __name__ == "__main__":
    main()
