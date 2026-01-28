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

def parse_human_eval_addendum(content: str) -> Dict[str, Dict[str, float]]:
    """
    Parses the Human Evaluation Addendum section to extract final human eval scores.
    
    Returns:
        scores: {question_code: {model_name: score}}
    """
    scores: Dict[str, Dict[str, float]] = {}
    
    # Find the Human Evaluation Addendum section
    addendum_match = re.search(
        r'={60,}\nHUMAN EVALUATION ADDENDUM \([^)]+\)\n={60,}\n(.*?)(?:={60,}|$)',
        content,
        re.DOTALL
    )
    
    if not addendum_match:
        return scores
    
    addendum_text = addendum_match.group(1)
    
    # Parse each question block
    # Format:
    # Question A21-H-Leetcode-3788 (Points: 1):
    #   google/gemma-3-27b-it@preset/fp8: 0.82/1 pts
    question_pattern = re.compile(
        r'Question ([\w\-]+) \(Points: \d+\):\n((?:  .+\n?)+)',
        re.MULTILINE
    )
    
    for q_match in question_pattern.finditer(addendum_text):
        q_code = q_match.group(1)
        model_lines = q_match.group(2)
        scores[q_code] = {}
        
        # Parse model scores within this question
        # Format: "  model_name: 0.82/1 pts"
        model_pattern = re.compile(r'  (.+): ([\d.]+)/[\d.]+ pts')
        for m_match in model_pattern.finditer(model_lines):
            model_name = m_match.group(1)
            score = float(m_match.group(2))
            scores[q_code][model_name] = score
    

    return scores


def parse_source_rankings(content: str) -> Dict[str, Tuple[float, float, str]]:
    """
    Parses 'FINAL MODEL RANKINGS' to extract pre-calculated totals.
    Returns: {model_name: (score, possible_points, formatted_string)}
    """
    rankings = {}
    
    # Try different header formats just in case
    header_patterns = [
        r"FINAL MODEL RANKINGS \(Automated \+ Human Evaluation\)",
        r"FINAL MODEL RANKINGS \(Automated Evaluation Only\)",
        r"FINAL MODEL RANKINGS"
    ]
    
    start_idx = -1
    lines = content.split('\n')
    
    for i, line in enumerate(lines):
        for pattern in header_patterns:
            if re.search(pattern, line):
                start_idx = i
                break
        if start_idx != -1:
            break
            
    if start_idx == -1:
        return rankings
        
    # Read until end of section (e.g. # lines or empty lines)
    # Looking for lines like: "1. model: score/max ... "
    rank_pattern = re.compile(r"^\d+\.\s+(.+?):\s+([\d.]+)/([\d.]+)\s+points")
    
    for i in range(start_idx + 1, len(lines)):
        line = lines[i].strip()
        if not line:
            continue
        if line.startswith("#") or line.startswith("="):
            continue
            
        match = rank_pattern.match(line)
        if match:
            model = match.group(1).strip()
            score = float(match.group(2))
            possible = float(match.group(3))
            rankings[model] = (score, possible, line)
        elif line and not rank_pattern.match(line) and i > start_idx + 5:
            # Stop if we hit non-ranking lines after some headers
            break
            
    return rankings



def parse_advanced_file(filepath: str) -> Tuple[List[str], Dict[str, Dict[str, Any]], Dict[str, Tuple[float, float, str]]]:
    """
    Parses a benchmark_results_advanced_*.txt file.
    Returns:
        question_codes: List[str] - The list of questions found in the file.
        all_results: Dict - structure compatible with reporting.py
                     {question_code: {model_name: {stats...}}}
        source_rankings: Dict - extracted Final Model Rankings from source file
                         {model_name: (score, possible, formatted_string)}
    """
    logger.info(f"Parsing file: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Parse Human Evaluation Addendum scores first (if present)
    human_eval_scores = parse_human_eval_addendum(content)

    # Split by Question headers
    # Format: ####################################################################################################
    # QUESTION 1: A1-QuestionName
    # ####################################################################################################
    
    question_sections = re.split(r'#{80,}\nQUESTION \d+: ', content)
    
    # The first split is the header/summary, ignore it
    if len(question_sections) < 2:
        logger.error("No questions found in file parsing.")
        return [], {}, {}
        
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

    # Apply Human Evaluation Addendum scores (overrides automated scores)
    if human_eval_scores:
        logger.info(f"Applying human evaluation scores for {len(human_eval_scores)} questions")
        for q_code, model_scores in human_eval_scores.items():
            if q_code in all_results:
                for model_name, score in model_scores.items():
                    if model_name in all_results[q_code]:
                        all_results[q_code][model_name]["score"] = score
                        
    # Extract source rankings if present
    source_rankings = parse_source_rankings(content)

    return question_codes, all_results, source_rankings

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
    
    # FIRST PASS: Parse all files
    parsed_data = [] # List of (filename, q_codes, results, file_models)
    
    for fpath in selected_files:
        try:
            q_codes, results, src_rankings = parse_advanced_file(fpath)
        except ValueError:
            # Fallback for unexpected return signature if module reload issues (unlikely in script)
            # But just in case
            ret = parse_advanced_file(fpath)
            if len(ret) == 2:
                q_codes, results = ret
                src_rankings = {}
            else:
                q_codes, results, src_rankings = ret

        if not q_codes:
            print(f"Skipping {os.path.basename(fpath)}: Failed to parse or empty.")
            continue
            
        # Extract models for this file
        current_file_models = set()
        for q in results:
            current_file_models.update(results[q].keys())
        current_file_models = sorted(list(current_file_models))
        
        parsed_data.append({
            "filename": os.path.basename(fpath),
            "q_codes": q_codes,
            "results": results,
            "models": current_file_models,
            "q_set": set(q_codes),
            "m_set": set(current_file_models),
            "source_rankings": src_rankings
        })

    if not parsed_data:
        print("No valid data found to process.")
        return

    # Normalize Question Codes to Base ID (to allow merging "A3" with "A3-J-...")
    def extract_base_id(q_code: str) -> str:
        # Extract A<number>[.<number>]
        match = re.match(r'^(A\d+(?:\.\d+)?)', q_code)
        if match:
             return match.group(1)
        return q_code

    # Updates parsed_data effectively
    for i in range(len(parsed_data)):
        data = parsed_data[i]
        old_results = data["results"]
        new_results = {}
        new_q_codes = []
        
        seen_base_ids = set()
        
        for q_code in data["q_codes"]:
            base_id = extract_base_id(q_code)
            
            # Use strict ordering from file
            if base_id not in seen_base_ids:
                new_q_codes.append(base_id)
                seen_base_ids.add(base_id)
                
            # If we somehow have duplicates (A3-v1 and A3-v2), we might overwrite or merge.
            # For now, let's assume we overwrite (last one wins) or keep first?
            # Existing keys in old_results are full strings.
            # We map them to base_id.
            if base_id in new_results:
                 print(f"Warning: Multiple questions map to Base ID '{base_id}' in {data['filename']}. Overwriting...")
            
            new_results[base_id] = old_results[q_code]

        data["q_codes"] = new_q_codes
        data["results"] = new_results
        data["q_set"] = set(new_q_codes)

    # INFERENCE STEP: Determine Mode
    # Check consistency across all parsed files against the first one
    reference = parsed_data[0]
    
    all_same_questions = all(p["q_set"] == reference["q_set"] for p in parsed_data)
    all_same_models = all(p["m_set"] == reference["m_set"] for p in parsed_data)
    
    combine_mode = None
    
    if all_same_questions and not all_same_models:
        combine_mode = "models"
        print(f"\nInference: Detected 'Combine MODELS' mode.")
        print("(All files share the same Question Set, but Models differ)")
        
    elif all_same_models and not all_same_questions:
        combine_mode = "questions"
        print(f"\nInference: Detected 'Combine QUESTIONS' mode.")
        print("(All files share the same Model Set, but Questions differ)")
        
    elif all_same_questions and all_same_models:
        # Check if we are combining just 1 file?
        if len(parsed_data) == 1:
             print("\nNote: Only 1 file selected. Re-generating reports for this single file.")
             combine_mode = "models" # Default fallback
        else:
            print("\nERROR: All files appear to be IDENTICAL (Same Questions AND Same Models).")
            print("There is nothing to merge.")
            return
            
    else:
        # Both differ - mixed bag?
        print("\nERROR: Incompatible files selected.")
        print("Files must either share the EXACT SAME questions (to combine models)")
        print("OR share the EXACT SAME models (to combine questions).")
        print("Your selection has mixed question sets AND mixed model sets.")
        return

    # SECOND PASS: Execute Merge
    combined_question_codes = [] 
    combined_all_results = {} 
    source_metadata = [] 
    all_source_rankings = []

    first_file_questions = reference["q_codes"]
    first_file_models = reference["models"]
    
    # Initialize with first file data if in models mode (preserves order)
    # Actually, let's just loop fresh to be consistent with previous logic
    
    for i, data in enumerate(parsed_data):
        filename = data["filename"]
        q_codes = data["q_codes"]
        results = data["results"]
        current_file_models = data["models"]
        
        # Collect source rankings if available
        if "source_rankings" in data:
            all_source_rankings.append(data["source_rankings"])
        
        # Metadata
        source_metadata.append({
            "filename": filename,
            "models": current_file_models,
            "questions": q_codes
        })
        
        if i == 0:
            combined_question_codes = list(q_codes)
            combined_all_results = results
        else:
            if combine_mode == "models":
                # Warn if overlap
                if not set(current_file_models).isdisjoint(set(first_file_models)):
                     overlap = set(current_file_models).intersection(set(first_file_models))
                     print(f"WARNING: Models {list(overlap)} appear in both files.")
                     print("Proceeding will overwrite earlier entries...")
                
                # Merge
                for q_code, model_dict in results.items():
                    for model_name, res in model_dict.items():
                        combined_all_results[q_code][model_name] = res

            elif combine_mode == "questions":
                # In 'questions' mode, models are same (checked by inference).
                # We need to add NEW questions to the combined set.
                
                # Warn if overlap in questions (shouldn't happen if splitting logic was clean, but possible)
                if not set(q_codes).isdisjoint(set(combined_question_codes)):
                     overlap = set(q_codes).intersection(set(combined_question_codes))
                     if i > 0: # Don't warn on first file
                        print(f"WARNING: Questions {list(overlap)} appear in both files.")
                        print("Proceeding will overwrite earlier entries for overlapping questions...")

                # Merge Results and Update Question List
                for q_code, model_dict in results.items():
                    if q_code not in combined_all_results:
                        # New question found
                        combined_question_codes.append(q_code)
                        combined_all_results[q_code] = model_dict
                    else:
                        # Overlapping question: update/overwrite model data
                        combined_all_results[q_code].update(model_dict)
    
    if not combined_all_results:
        print("No results merged.")
        return

    # Verify we have models
    all_models = set()
    for q_code in combined_all_results:
        all_models.update(combined_all_results[q_code].keys())
    
    all_models = sorted(list(all_models))
    print(f"\nCombined {len(all_models)} models: {', '.join(all_models)}")
    print(f"Total Unique Questions: {len(combined_question_codes)}")
    
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
    
    suffix = "combined_models" if combine_mode == "models" else "combined_questions"
    timestamp = datetime.now().strftime(f"%Y%m%d_%H%M%S_{suffix}")
    
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
        output_dir=os.path.join(project_root, "results"),
        source_metadata=source_metadata
    )
    print(f"[SUCCESS] Generated Results: {txt_path}")

    # 3. Advanced Results Text File
    adv_path = write_advanced_results_file(
        all_models,
        valid_codes,
        combined_all_results,
        questions_data,
        timestamp,
        output_dir=os.path.join(project_root, "results_advanced"),
        source_metadata=source_metadata
    )
    print(f"[SUCCESS] Generated Advanced Results: {adv_path}")
    
    # 4. Append Human Evaluation Addendum if there are human eval questions with scores
    human_eval_codes = [
        code for code in valid_codes 
        if questions_data.get(code, {}).get("is_manual_check", False)
    ]
    
    if human_eval_codes:
        # Check if any human eval question has a non-zero score (indicating completed eval)
        has_human_scores = any(
            combined_all_results.get(code, {}).get(model, {}).get("score", 0) > 0
            for code in human_eval_codes
            for model in all_models
        )
        
        # Always append if we have human eval codes since scores were parsed from source files
        append_human_eval_summary(
            adv_path, txt_path, all_models, valid_codes, 
            combined_all_results, questions_data,
            source_rankings=all_source_rankings,
            combine_mode=combine_mode
        )
    
    # 5. Console Summary
    print_final_rankings(
        all_models,
        valid_codes,
        combined_all_results,
        questions_data
    )


def append_human_eval_summary(
    adv_path: str,
    txt_path: str,
    models: List[str],
    question_codes: List[str],
    all_results: Dict[str, Dict[str, Any]],
    questions_data: Dict[str, Dict[str, Any]],
    source_rankings: List[Dict[str, Tuple[float, float, str]]] | None = None,
    combine_mode: str = "models"
) -> None:
    """
    Appends Human Evaluation Addendum and Final Rankings to result files.
    
    Args:
        source_rankings: List of ranking dicts from source files. 
                         If provided, we aggregate these to form the final list 
                         instead of recalculating.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    human_eval_codes = [
        code for code in question_codes 
        if questions_data.get(code, {}).get("is_manual_check", False)
    ]
    
    # Build the addendum content
    addendum_lines = [
        "",
        "=" * 80,
        f"HUMAN EVALUATION ADDENDUM ({timestamp})",
        "=" * 80,
        "",
        "The following scores reflect human evaluation results:",
        ""
    ]
    
    for code in human_eval_codes:
        points = questions_data.get(code, {}).get("points", 1)
        addendum_lines.append(f"Question {code} (Points: {points}):")
        
        for model in models:
            score = all_results.get(code, {}).get(model, {}).get("score", 0.0)
            addendum_lines.append(f"  {model}: {score:.2f}/{points} pts")
        addendum_lines.append("")
    
    addendum_lines.append("=" * 80)
    addendum_lines.append("")
    
    addendum_lines.append("#" * 80)
    addendum_lines.append("FINAL MODEL RANKINGS (Automated + Human Evaluation)")
    addendum_lines.append("#" * 80)

    # Decide strategy for final rankings
    # Strategy 1: Use sourced rankings if available for ALL models (User Request)
    # Strategy 2: Recalculate (Fallback)
    
    used_sourced_rankings = False
    
    used_sourced_rankings = False
    
    # Only use source rankings if we are combining distinct models (Score A from File 1 + Score B from File 2)
    # If we are combining questions (Score A_part1 + Score A_part2), we MUST sum them up.
    if combine_mode == "models" and source_rankings:
        # Merge all source rankings into one dict
        merged_rankings = {}
        for src in source_rankings:
            merged_rankings.update(src)
            
        # Check if we have rankings for all current models
        # Note: source_rankings[model] = (score, possible, formatted_string)
        if all(m in merged_rankings for m in models):
            # Sort by score descending
            ranked = sorted(merged_rankings.items(), key=lambda x: x[1][0], reverse=True)
            
            for rank, (model, (score, possible, orig_line)) in enumerate(ranked, 1):
                # We reuse the formatted line's suffix if we can, 
                # OR we reconstruct to ensure exact "score" value is used as parsed.
                # The user wants exact match.
                
                percentage = (score / possible * 100) if possible > 0 else 0
                addendum_lines.append(f"{rank}. {model}: {score:.2f}/{possible:.2f} points ({percentage:.1f}%)")
            
            used_sourced_rankings = True
    
    if not used_sourced_rankings:
        # Recalculate (Fallback)
        total_scores = {}
        total_possible = sum(questions_data.get(code, {}).get("points", 1) for code in question_codes)
        
        for model in models:
            total_scores[model] = sum(
                all_results.get(code, {}).get(model, {}).get("score", 0.0)
                for code in question_codes
            )
        
        # Sort by score descending
        ranked = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
        
        for rank, (model, score) in enumerate(ranked, 1):
            percentage = (score / total_possible * 100) if total_possible > 0 else 0
            addendum_lines.append(f"{rank}. {model}: {score:.2f}/{total_possible} points ({percentage:.1f}%)")
    
    addendum_lines.append("#" * 80)
    addendum_lines.append("")
    
    addendum_text = "\n".join(addendum_lines)
    
    # Append to advanced results file
    with open(adv_path, "a") as f:
        f.write(addendum_text)
    
    # Append to standard results file
    with open(txt_path, "a") as f:
        f.write(addendum_text)


if __name__ == "__main__":
    main()
