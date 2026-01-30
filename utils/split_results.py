"""
Script to split a benchmark result file into two separate files based on selected models or questions.
Parses an existing 'results_advanced' text file, splits it based on user selection (by model or question),
and generates two new result files (HTML, basic results, and advanced results).

Usage:
    uv run utils/split_results.py
"""

import os
import sys
import re
import glob
from datetime import datetime
from typing import Dict, Any, List, Set, Tuple

# Fix path to ensure we import 'utils' as a package from project root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

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
    question_sections = re.split(r'#{80,}\nQUESTION \d+: ', content)
    
    # The first split is the header/summary, ignore it
    if len(question_sections) < 2:
        logger.error("No questions found in file parsing.")
        return [], {}
        
    question_codes = []
    all_results = {}
    
    # Process each question section
    for section in question_sections[1:]:
        lines = section.split('\n')
        q_code_line = lines[0].strip()
        
        q_code = q_code_line.split('\n')[0].strip()
        question_codes.append(q_code)
        all_results[q_code] = {}
        
        # Split by Model separators
        model_sections = re.split(r'\nMODEL: ', section)
        
        for m_section in model_sections[1:]:
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
            run_splits = re.split(r'--- RUN #\d+ ---', m_section)
            
            for run_text in run_splits[1:]:
                run_data = {
                    "success": False,
                    "response": "",
                    "model_reasoning": None,
                    "judge_reasoning": "",
                    "judge_verdict": "",
                    "run_score": None
                }
                
                # Extract Response
                resp_match = re.search(r'MODEL RESPONSE:\n(.*?)(?=\nJUDGE EVALUATION:|\n\n--- RUN|$)', run_text, re.DOTALL)
                if resp_match:
                    run_data["response"] = resp_match.group(1).strip()
                
                # Extract Reasoning (if any)
                reasoning_match = re.search(r'MODEL THINKING/REASONING:\n(.*?)(?=\nMODEL RESPONSE:)', run_text, re.DOTALL)
                if reasoning_match:
                    run_data["model_reasoning"] = reasoning_match.group(1).strip()
                    
                # Extract Judge Info
                judge_eval_match = re.search(r'JUDGE EVALUATION:\n(.*?)(?=\nJUDGE VERDICT:)', run_text, re.DOTALL)
                if judge_eval_match:
                    run_data["judge_reasoning"] = judge_eval_match.group(1).strip()
                    
                verdict_match = re.search(r'JUDGE VERDICT: (.*)', run_text)
                if verdict_match:
                    run_data["judge_verdict"] = verdict_match.group(1).strip()
                
                # Derive success from verdict
                run_data["success"] = run_data["judge_verdict"].lower() == "pass"
                
                # Extract granular score from judge_reasoning if available (SCORE:X/Y)
                score_match = re.search(r'SCORE:([\d.]+)/[\d.]+', run_data["judge_reasoning"])
                if score_match:
                    run_data["run_score"] = float(score_match.group(1))
                    run_data["success"] = run_data["run_score"] > 0
                
                all_results[q_code][model_name]["runs"].append(run_data)

    return question_codes, all_results

def get_available_files() -> List[str]:
    """Returns sorted list of available advanced result files."""
    results_dir = os.path.join(project_root, "results_advanced")
    files = glob.glob(os.path.join(results_dir, "benchmark_results_advanced_*.txt"))
    # Sort by timestamp (descending)
    files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
    return files

def generate_split_reports(
    all_models: List[str],
    question_codes: List[str],
    all_results: Dict[str, Dict[str, Any]],
    questions_data: Dict,
    valid_codes: List[str],
    suffix: str
) -> Tuple[str, str, str]:
    """
    Generate HTML, basic results, and advanced results for a split.
    
    Returns:
        Tuple of (html_path, txt_path, adv_path)
    """
    timestamp = datetime.now().strftime(f"%Y%m%d_%H%M%S_{suffix}")
    
    # Filter all_results to include only the selected models and questions
    filtered_results = {}
    for q_code in question_codes:
        if q_code in all_results:
            filtered_results[q_code] = {}
            for model in all_models:
                if model in all_results[q_code]:
                    filtered_results[q_code][model] = all_results[q_code][model]
    
    # HTML Performance Table
    html_path = generate_performance_html(
        all_models,
        valid_codes,
        filtered_results,
        questions_data,
        timestamp,
        output_dir=os.path.join(project_root, "results")
    )
    
    # Results Text File
    txt_path = write_results_file(
        all_models,
        valid_codes,
        filtered_results,
        questions_data,
        timestamp,
        output_dir=os.path.join(project_root, "results")
    )
    
    # Advanced Results Text File
    adv_path = write_advanced_results_file(
        all_models,
        valid_codes,
        filtered_results,
        questions_data,
        timestamp,
        output_dir=os.path.join(project_root, "results_advanced")
    )
    
    return html_path, txt_path, adv_path

def main():
    print("="*60)
    print("      BENCHMARK RESULT SPLITTER")
    print("="*60)
    
    files = get_available_files()
    
    if not files:
        print("No advanced result files found in 'results_advanced/'")
        return

    print("\nAvailable Files:")
    for idx, f in enumerate(files, 1):
        timestamp = os.path.basename(f).split('_')[-1].replace('.txt', '')
        try:
            ts_obj = datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
            ts_str = ts_obj.strftime("%Y-%m-%d %H:%M:%S")
        except:
            try:
                # Handle combined files with suffix
                ts_parts = timestamp.rsplit('_', 1)
                ts_obj = datetime.strptime('_'.join(timestamp.split('_')[:2]), "%Y%m%d_%H%M%S")
                ts_str = ts_obj.strftime("%Y-%m-%d %H:%M:%S")
            except:
                ts_str = timestamp

        print(f"{idx}. {os.path.basename(f)}  (Date: {ts_str})")
    
    # Select file to split
    print("\nSelect file to split (enter number):")
    try:
        selection = int(input("> ").strip())
        if not (1 <= selection <= len(files)):
            print("Invalid selection.")
            return
    except ValueError:
        print("Invalid input.")
        return
    
    selected_file = files[selection - 1]
    print(f"\nProcessing: {os.path.basename(selected_file)}")
    
    # Parse the file
    question_codes, all_results = parse_advanced_file(selected_file)
    
    if not question_codes or not all_results:
        print("Failed to parse file or file is empty.")
        return
    
    # Extract all models
    all_models = set()
    for q_code in all_results:
        all_models.update(all_results[q_code].keys())
    all_models = sorted(list(all_models))
    
    print(f"\nFile contains:")
    print(f"  {len(all_models)} models: {', '.join(all_models)}")
    print(f"  {len(question_codes)} questions: {', '.join(question_codes)}")
    
    # Ask user: split by model or question?
    print("\nHow would you like to split the results?")
    print("1. Split by models")
    print("2. Split by questions")
    
    try:
        split_choice = int(input("> ").strip())
        if split_choice not in [1, 2]:
            print("Invalid selection.")
            return
    except ValueError:
        print("Invalid input.")
        return
    
    split_by_model = (split_choice == 1)
    
    if split_by_model:
        # Show models with indexes
        print("\nAvailable Models:")
        for idx, model in enumerate(all_models, 1):
            print(f"{idx}. {model}")
        
        print("\nSelect models for the FIRST group (comma-separated indexes, e.g., '1,3,4'):")
        selection_str = input("> ").strip()
        
        try:
            indices = [int(x.strip()) for x in selection_str.split(',') if x.strip()]
            group1_models = []
            for i in indices:
                if 1 <= i <= len(all_models):
                    group1_models.append(all_models[i-1])
                else:
                    print(f"Warning: Index {i} out of range, skipping.")
            
            if not group1_models:
                print("No valid models selected.")
                return
            
            # Create group 2 (remaining models)
            group2_models = [m for m in all_models if m not in group1_models]
            
            if not group2_models:
                print("No models remaining for second group.")
                return
            
            print(f"\nGroup 1 models: {', '.join(group1_models)}")
            print(f"Group 2 models: {', '.join(group2_models)}")
            
            # Load question metadata
            questions_data, valid_codes = original_load_questions_data(question_codes)
            
            # Generate reports for Group 1
            print("\n[Generating Group 1 reports...]")
            html1, txt1, adv1 = generate_split_reports(
                group1_models,
                valid_codes,
                all_results,
                questions_data,
                valid_codes,
                "split_models_group1"
            )
            print(f"  HTML: file://{html1}")
            print(f"  Results: {txt1}")
            print(f"  Advanced: {adv1}")
            
            # Generate reports for Group 2
            print("\n[Generating Group 2 reports...]")
            html2, txt2, adv2 = generate_split_reports(
                group2_models,
                valid_codes,
                all_results,
                questions_data,
                valid_codes,
                "split_models_group2"
            )
            print(f"  HTML: file://{html2}")
            print(f"  Results: {txt2}")
            print(f"  Advanced: {adv2}")
            
        except ValueError:
            print("Invalid input.")
            return
    
    else:  # Split by questions
        # Show questions with indexes
        print("\nAvailable Questions:")
        for idx, q in enumerate(question_codes, 1):
            print(f"{idx}. {q}")
        
        print("\nSelect questions for the FIRST group (comma-separated indexes, e.g., '1,3,4'):")
        selection_str = input("> ").strip()
        
        try:
            indices = [int(x.strip()) for x in selection_str.split(',') if x.strip()]
            group1_questions = []
            for i in indices:
                if 1 <= i <= len(question_codes):
                    group1_questions.append(question_codes[i-1])
                else:
                    print(f"Warning: Index {i} out of range, skipping.")
            
            if not group1_questions:
                print("No valid questions selected.")
                return
            
            # Create group 2 (remaining questions)
            group2_questions = [q for q in question_codes if q not in group1_questions]
            
            if not group2_questions:
                print("No questions remaining for second group.")
                return
            
            print(f"\nGroup 1 questions: {', '.join(group1_questions)}")
            print(f"Group 2 questions: {', '.join(group2_questions)}")
            
            # Load question metadata
            questions_data, valid_codes = original_load_questions_data(question_codes)
            
            # Filter valid codes for each group
            valid_codes_group1 = [q for q in group1_questions if q in valid_codes]
            valid_codes_group2 = [q for q in group2_questions if q in valid_codes]
            
            # Generate reports for Group 1
            print("\n[Generating Group 1 reports...]")
            html1, txt1, adv1 = generate_split_reports(
                all_models,
                valid_codes_group1,
                all_results,
                questions_data,
                valid_codes_group1,
                "split_questions_group1"
            )
            print(f"  HTML: file://{html1}")
            print(f"  Results: {txt1}")
            print(f"  Advanced: {adv1}")
            
            # Generate reports for Group 2
            print("\n[Generating Group 2 reports...]")
            html2, txt2, adv2 = generate_split_reports(
                all_models,
                valid_codes_group2,
                all_results,
                questions_data,
                valid_codes_group2,
                "split_questions_group2"
            )
            print(f"  HTML: file://{html2}")
            print(f"  Results: {txt2}")
            print(f"  Advanced: {adv2}")
            
        except ValueError:
            print("Invalid input.")
            return
    
    print("\n" + "="*60)
    print("Split completed successfully!")
    print("="*60)

if __name__ == "__main__":
    main()
