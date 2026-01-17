"""
Benchmarking script to evaluate multiple models on specific questions.
Parses question files, calls models via ModelAPI, and generates reports.
"""

import os
import glob
import re
import subprocess
import sys
import argparse
from datetime import datetime
from typing import Any, Dict, List
from dotenv import load_dotenv
from utils.model_api import ModelAPI
from openai import OpenAIError
from utils.utils import setup_logging, parse_question_file, kill_process_on_port, clear_history
from utils.evaluators import JudgeLLMEvaluator, HumanEvaluator, ValidityEvaluator
from utils.cost_effective import find_model_folder, get_existing_implementations
from concurrent.futures import ThreadPoolExecutor, as_completed

# Load environment variables
load_dotenv()

logger = setup_logging(__name__)

NUM_RUNS = int(os.getenv("NUM_RUNS", "4"))
COST_EFFECTIVE_ENABLED = os.getenv("COST_EFFECTIVE_ENABLED", "true").lower() in ("true", "1", "yes")

def resolve_question_path(question_code: str) -> str | None:
    """
    Finds the file path for a given question code (e.g., 'A2') starting with that code.
    Ensures that 'A2' does not match 'A22' by checking the character following the code.
    """
    # Search recursively in the questions directory
    search_pattern = f"questions/**/{question_code}*.txt"
    matches = glob.glob(search_pattern, recursive=True)

    if not matches:
        return None

    # Filter matches to ensure exact code prefix (e.g., A2 followed by - or .)
    for match in matches:
        basename = os.path.basename(match)
        # Check if basename starts with code and the next char is not a digit
        if basename.startswith(question_code):
            remaining = basename[len(question_code):]
            if not remaining or not remaining[0].isdigit():
                return match

    return None


def print_benchmark_summary(models: List[str], questions_data: Dict[str, Dict[str, Any]], question_codes: List[str]) -> None:
    """
    Prints the benchmark summary to the console.
    """
    print("\nModels benchmarked:")
    for model in models:
        print(f"- {model}")

    print("\nQuestions in the run:")
    for code in question_codes:
        data = questions_data.get(code, {})
        eval_type = data.get("eval_type", "Unknown")
        print(f"- {code} ({eval_type})")
    
def write_advanced_results_file(
    models: List[str],
    question_codes: List[str],
    all_results: Dict[str, Dict[str, Any]],
    questions_data: Dict[str, Dict[str, Any]],
    timestamp: str | None = None
) -> str:
    """
    Writes comprehensive advanced benchmark results to a timestamped file.
    Includes full model responses, reasoning, and all evaluation details for ALL runs.
    Returns the path to the created file.
    """
    # Create results_advanced directory if it doesn't exist
    results_dir = "results_advanced"
    os.makedirs(results_dir, exist_ok=True)

    # Use provided timestamp or generate new one
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_advanced_{timestamp}.txt"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, "w") as f:
        # Header section
        f.write("=" * 100 + "\n")
        f.write("ADVANCED BENCHMARK RESULTS - DETAILED REPORT\n")
        f.write("=" * 100 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary Section
        f.write("Models benchmarked:\n")
        for model in models:
            f.write(f"- {model}\n")
        
        f.write("\nQuestions in the run:\n")
        for code in question_codes:
            eval_type = questions_data.get(code, {}).get("eval_type", "Unknown")
            f.write(f"- {code} ({eval_type})\n")
        f.write("\n")
        
        f.write("=" * 100 + "\n\n")
        
        # Detailed results by question
        for idx, code in enumerate(question_codes, 1):
            question_text = questions_data.get(code, {}).get("question", "N/A")
            ground_truth = questions_data.get(code, {}).get("ground_truth", "N/A")
            points = questions_data.get(code, {}).get("points", 1)
            
            f.write(f"\n{'#' * 100}\n")
            f.write(f"QUESTION {idx}: {code}\n")
            f.write(f"{'#' * 100}\n\n")
            
            f.write("QUESTION TEXT:\n")
            f.write("-" * 100 + "\n")
            # Skip full prompt for very long questions (e.g., A58 BattleShip)
            if len(question_text) > 1000:
                f.write(f"[Prompt omitted - {len(question_text)} chars]\n")
            else:
                f.write(f"{question_text}\n")
            f.write("-" * 100 + "\n\n")
            
            f.write(f"POINTS: {points}\n")
            f.write(f"EXPECTED ANSWER: {ground_truth}\n\n")
            f.write("=" * 100 + "\n\n")
            
            # Results for each model on this question
            for model in models:
                model_data = all_results.get(code, {}).get(model, {})
                runs = model_data.get("runs", [])
                score = model_data.get("score", 0.0)
                total_tokens = model_data.get("total_tokens", 0)
                total_cost = model_data.get("total_cost", 0.0)
                
                f.write(f"MODEL: {model}\n")
                f.write(f"SCORE: {score:.2f}/{points:.2f}\n")
                f.write(f"TOKENS USED: {total_tokens}\n")
                f.write(f"COST INCURRED: ${total_cost:.6f}\n")
                f.write("-" * 100 + "\n\n")

                for run_idx, run_result in enumerate(runs, 1):
                    f.write(f"--- RUN #{run_idx} ---\n")
                    
                    # Model thinking/reasoning (if available)
                    model_reasoning = run_result.get("model_reasoning")
                    if model_reasoning:
                        f.write("MODEL THINKING/REASONING:\n")
                        f.write(f"{model_reasoning}\n\n")
                    
                    # Model response
                    model_response = run_result.get("response", "N/A")
                    f.write("MODEL RESPONSE:\n")
                    # Skip full response if it contains code blocks
                    if "```" in model_response:
                        f.write(f"[Response omitted - contains code ({len(model_response)} chars)]\n\n")
                    else:
                        f.write(f"{model_response}\n\n")
                    
                    # Judge evaluation
                    judge_reasoning = run_result.get("judge_reasoning", "N/A")
                    judge_verdict = run_result.get("judge_verdict", "N/A")
                    
                    f.write("JUDGE EVALUATION:\n")
                    f.write(f"{judge_reasoning}\n\n")
                    
                    f.write(f"JUDGE VERDICT: {judge_verdict}\n\n")
                    
                    # Final result for this run
                    if judge_verdict == "Pending":
                        evaluation = "PENDING (Human Eval Required)"
                    elif "run_score" in run_result:
                        evaluation = f"{run_result['run_score']}/{run_result['run_max']} pts"
                    else:
                        evaluation = "PASS" if run_result.get("success", False) else "FAIL"
                    f.write(f"RUN RESULT: {evaluation}\n\n")
                
                f.write("\n" + "=" * 100 + "\n\n")
        
        # Rankings section (only for non-human-eval questions)
        # Check if there are any non-human-eval questions
        non_human_eval_codes = [code for code in question_codes 
                                if not questions_data.get(code, {}).get("is_manual_check", False)]
        
        if non_human_eval_codes:
            f.write("\n" + "#" * 100 + "\n")
            f.write("MODEL RANKINGS (Automated Evaluation Only)\n")
            f.write("#" * 100 + "\n\n")
            
            # Calculate weighted scores and usage (excluding human eval questions)
            scores = {}
            usage = {} # {model: (tokens, cost)}
            total_possible_points = 0
            for model in models:
                score = 0
                tokens = 0
                cost = 0.0
                for code in non_human_eval_codes:
                    model_data = all_results.get(code, {}).get(model, {})
                    score += model_data.get("score", 0.0)
                    tokens += model_data.get("total_tokens", 0)
                    cost += model_data.get("total_cost", 0.0)
                scores[model] = score
                usage[model] = (tokens, cost)
            
            # Calculate total possible points (excluding human eval)
            for code in non_human_eval_codes:
                total_possible_points += questions_data.get(code, {}).get("points", 1)
            
            # Sort by score descending
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (model, score) in enumerate(ranked, 1):
                percentage = (score / total_possible_points * 100) if total_possible_points > 0 else 0
                tokens, cost = usage[model]
                f.write(f"{rank}. {model}: {score:.2f}/{total_possible_points} points ({percentage:.1f}%) - {tokens} tokens - ${cost:.3f}\n")
            
            f.write("\n" + "=" * 100 + "\n")
        
        # Note about human eval questions if any exist
        human_eval_codes = [code for code in question_codes 
                            if questions_data.get(code, {}).get("is_manual_check", False)]
        if human_eval_codes:
            f.write("\n" + "#" * 100 + "\n")
            f.write("HUMAN EVALUATION PENDING\n")
            f.write("#" * 100 + "\n\n")
            f.write("The following questions require human evaluation:\n")
            for code in human_eval_codes:
                points = questions_data.get(code, {}).get("points", 1)
                f.write(f"  - {code} ({points} points)\n")
            f.write("\nHuman evaluation server was spawned automatically.\n")
            f.write("Complete scoring in the browser windows - this report will be updated.\n")
            f.write("\n" + "=" * 100 + "\n")
    
    return filepath


def write_results_file(
    models: List[str],
    question_codes: List[str],
    all_results: Dict[str, Dict[str, Any]],
    questions_data: Dict[str, Dict[str, Any]],
    timestamp: str | None = None
) -> str:
    """
    Writes benchmark results to a timestamped file in the results directory.
    Returns the path to the created file.
    """
    # Create results directory if it doesn't exist
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    # Use provided timestamp or generate new one
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.txt"
    filepath = os.path.join(results_dir, filename)
    
    with open(filepath, "w") as f:
        # Header section
        f.write("=" * 80 + "\n")
        f.write("BENCHMARK RESULTS\n")
        f.write("=" * 80 + "\n\n")
        
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary Section
        f.write("Models benchmarked:\n")
        for model in models:
            f.write(f"- {model}\n")
        
        f.write("\nQuestions in the run:\n")
        for code in question_codes:
            eval_type = questions_data.get(code, {}).get("eval_type", "Unknown")
            f.write(f"- {code} ({eval_type})\n")
        f.write("\n")
        
        f.write("=" * 80 + "\n\n")
        
        # Results by model
        for model in models:
            f.write(f"{model.upper()} RESULTS:\n")
            f.write("-" * 80 + "\n\n")
            
            for idx, code in enumerate(question_codes, 1):
                model_data = all_results.get(code, {}).get(model, {})
                expected = questions_data.get(code, {}).get("ground_truth", "N/A")
                score = model_data.get("score", 0.0)
                points = questions_data.get(code, {}).get("points", 1)
                is_human_eval = questions_data.get(code, {}).get("is_manual_check", False)
                
                f.write(f"Question {idx} ({code}):\n")
                f.write(f"  Expected: {expected}\n")
                
                if is_human_eval:
                    f.write(f"  Score: PENDING (Human Eval)\n")
                    f.write(f"  Runs: PENDING\n\n")
                else:
                    f.write(f"  Score: {score:.2f}/{points}\n")
                    # List brief verdict for each run (with granular scores if available)
                    runs = model_data.get("runs", [])
                    run_verdicts = []
                    for run in runs:
                        if "run_score" in run:
                            run_verdicts.append(f"{run['run_score']}/{run['run_max']}")
                        elif run.get("success", False):
                            run_verdicts.append("PASS")
                        else:
                            run_verdicts.append("FAIL")
                    f.write(f"  Runs: {', '.join(run_verdicts)}\n\n")
            
            f.write("\n")
        
        # Rankings (only for non-human-eval questions)
        non_human_eval_codes = [code for code in question_codes 
                                if not questions_data.get(code, {}).get("is_manual_check", False)]
        
        if non_human_eval_codes:
            f.write("=" * 80 + "\n")
            f.write("MODEL RANKINGS (Automated Evaluation Only)\n")
            f.write("=" * 80 + "\n\n")
            
            # Calculate weighted scores and usage (excluding human eval)
            scores = {}
            usage = {} # {model: (tokens, cost)}
            total_possible_points = 0
            for model in models:
                score = 0
                tokens = 0
                cost = 0.0
                for code in non_human_eval_codes:
                    model_data = all_results.get(code, {}).get(model, {})
                    score += model_data.get("score", 0.0)
                    tokens += model_data.get("total_tokens", 0)
                    cost += model_data.get("total_cost", 0.0)
                scores[model] = score
                usage[model] = (tokens, cost)
            
            # Calculate total possible points (excluding human eval)
            for code in non_human_eval_codes:
                total_possible_points += questions_data.get(code, {}).get("points", 1)
            
            # Sort by score descending
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            
            for rank, (model, score) in enumerate(ranked, 1):
                percentage = (score / total_possible_points * 100) if total_possible_points > 0 else 0
                tokens, cost = usage[model]
                f.write(f"{rank}. {model}: {score:.2f}/{total_possible_points} points ({percentage:.1f}%) - {tokens} tokens - ${cost:.3f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
        
        # Note about human eval questions
        human_eval_codes = [code for code in question_codes 
                           if questions_data.get(code, {}).get("is_manual_check", False)]
        if human_eval_codes:
            f.write("\n" + "=" * 80 + "\n")
            f.write("HUMAN EVALUATION PENDING\n")
            f.write("=" * 80 + "\n\n")
            f.write("Questions pending human evaluation:\n")
            for code in human_eval_codes:
                points = questions_data.get(code, {}).get("points", 1)
                f.write(f"  - {code} ({points} points)\n")
            f.write("\nHuman evaluation server was spawned automatically.\n")
            f.write("Complete scoring in the browser windows - this report will be updated.\n")
            f.write("\n" + "=" * 80 + "\n")
    
    return filepath


def process_single_run(
    api: ModelAPI,
    model_name: str,
    model_index: int,
    question_code: str,
    question: str,
    ground_truth: str,
    points: int,
    judge: JudgeLLMEvaluator,
    validity_eval: ValidityEvaluator,
    human_eval: HumanEvaluator
) -> Dict[str, Any]:
    """
    Executes a single run for a model on a question and returns the result.
    """
    try:
        # Calculate effective max tokens based on question points
        effective_max_tokens = api.max_tokens * points
        
        # Call the model with ONLY the question and effective max tokens
        response = api.call(question, model_index=model_index, max_tokens=effective_max_tokens)

        # Extract content and reasoning for advanced results
        message = response.choices[0].message
        content = message.content or ""
        reasoning_details = getattr(message, "reasoning_details", None)
        
        # Extract token usage and cost from response
        completion_tokens = 0
        cost = 0.0
        if hasattr(response, 'usage') and response.usage:
            completion_tokens = getattr(response.usage, 'completion_tokens', 0)
            cost = getattr(response.usage, 'cost', 0.0)

        # Evaluation logic
        judge_reasoning = None
        judge_verdict = None

        if ground_truth:
            if ground_truth.upper().strip() == "VALIDITY CHECK":
                eval_result = validity_eval.evaluate(question_code, question, content)
                is_successful = eval_result["success"]
                judge_reasoning = eval_result["reasoning"]
                judge_verdict = eval_result["verdict"]
            else:
                eval_result = judge.evaluate(question, ground_truth, content)
                is_successful = eval_result["success"]
                judge_reasoning = eval_result["reasoning"]
                judge_verdict = eval_result["verdict"]
        else:
            human_eval.evaluate(question, content)
            is_successful = False
            judge_reasoning = "Registered for Human Evaluation"
            judge_verdict = "Pending"

        return {
            "success": is_successful,
            "response": content,
            "model_reasoning": reasoning_details,
            "judge_reasoning": judge_reasoning,
            "judge_verdict": judge_verdict,
            "completion_tokens": completion_tokens,
            "cost": cost
        }

    except (OpenAIError, IndexError, Exception) as e:
        logger.error("Error in run for model %s question %s: %s", model_name, question_code, e)
        return {
            "success": False,
            "response": f"ERROR: {str(e)}",
            "model_reasoning": None,
            "judge_reasoning": f"ERROR: {str(e)}",
            "judge_verdict": "Error",
            "error": str(e)
        }




def generate_performance_html(
    models: List[str],
    question_codes: List[str],
    all_results: Dict[str, Dict[str, Any]],
    questions_data: Dict[str, Dict[str, Any]],
    timestamp: str
) -> str:
    """
    Generates an HTML performance table and saves it to a file.
    Returns the absolute path to the generated HTML file.
    """
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    filename = f"performance_table_{timestamp}.html"
    filepath = os.path.abspath(os.path.join(results_dir, filename))
    
    # Extract short model names - strip @preset/... suffix first, then take name after provider/
    short_models = [m.split("@")[0].split("/")[-1] for m in models]
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Benchmark Performance Summary - {timestamp}</title>
        <style>
            body {{ font-family: sans-serif; margin: 20px; background-color: #f4f4f9; color: #333; }}
            h1 {{ text-align: center; color: #444; }}
            .container {{ max-width: 100%; overflow-x: auto; box-shadow: 0 0 20px rgba(0,0,0,0.1); background-color: #fff; padding: 20px; border-radius: 8px; }}
            table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
            th, td {{ border: 1px solid #ddd; padding: 12px 15px; text-align: center; white-space: nowrap; }}
            th {{ background-color: #009879; color: #ffffff; font-weight: bold; position: sticky; top: 0; }}
            tr:nth-child(even) {{ background-color: #f3f3f3; }}
            tr:hover {{ background-color: #f1f1f1; cursor: default; }}
            .pass {{ color: #27ae60; font-weight: bold; background-color: #eafaf1; }}
            .fail {{ color: #c0392b; font-weight: bold; background-color: #fdedec; }}
            .score {{ color: #2c3e50; font-weight: bold; }}
            .q-col {{ text-align: left; font-weight: bold; background-color: #f8f9fa; }}
            .tokens {{ color: #3498db; font-size: 0.9em; }}
            .cost {{ color: #e67e22; font-size: 0.9em; }}
            .model-header {{ background-color: #006b5a !important; }}
        </style>
    </head>
    <body>
        <h1>Benchmark Performance Summary</h1>
        <p style="text-align: center; color: #666;">Date: {timestamp}</p>
        <div class="container">
            <table>
                <thead>
                    <tr>
                        <th rowspan="2">Question Index</th>
                        <th rowspan="2">Points</th>
    """
    
    # Add model headers (spanning 3 columns: Score, Tokens, Cost)
    for model in short_models:
        html_content += f"                        <th colspan='3' class='model-header'>{model}</th>\n"
    
    html_content += """                    </tr>
                    <tr>
    """
    
    # Add sub-headers for each model (Score, Tokens, Cost)
    for _ in short_models:
        html_content += "                        <th>Score</th>\n"
        html_content += "                        <th>Tokens</th>\n"
        html_content += "                        <th>Cost</th>\n"
        
    html_content += """                    </tr>
                </thead>
                <tbody>
    """
    
    for q_id in question_codes:
        q_data = questions_data.get(q_id, {})
        points = q_data.get("points", 1)
        
        # Format points
        if isinstance(points, float) and points.is_integer():
            p_str = str(int(points))
        else:
            p_str = str(points)
            
        html_content += f"                    <tr>\n                        <td class='q-col'>{q_id}</td>\n                        <td>{p_str}</td>\n"
        
        for model in models:
            model_results = all_results.get(q_id, {}).get(model, {})
            score = model_results.get("score", 0.0)
            total_tokens = model_results.get("total_tokens", 0)
            total_cost = model_results.get("total_cost", 0.0)
            
            val_class = "score"
            val_text = "FAIL"
            
            if score == points and score > 0:
                val_text = "PASS"
                val_class = "pass"
            elif score == 0:
                val_text = "FAIL"
                val_class = "fail"
            else:
                if isinstance(score, float) and score.is_integer():
                    val_text = str(int(score))
                else:
                    # Remove trailing zeros for cleanliness
                    val_text = f"{score:.2f}".rstrip('0').rstrip('.')
            
            # Add score cell
            html_content += f"                        <td class='{val_class}'>{val_text}</td>\n"
            # Add tokens cell
            html_content += f"                        <td class='tokens'>{total_tokens}</td>\n"
            # Add cost cell
            cost_str = f"${total_cost:.6f}".rstrip('0').rstrip('.')
            if cost_str == "$":
                cost_str = "$0"
            html_content += f"                        <td class='cost'>{cost_str}</td>\n"
            
        html_content += "                    </tr>\n"
        
    html_content += """                </tbody>
                <tfoot style="background-color: #f8f9fa; font-weight: bold; border-top: 2px solid #009879;">
                    <tr>
                        <td colspan="2" style="text-align: right; padding-right: 20px;">TOTAL</td>
    """
    
    for model in models:
        total_model_score = 0.0
        total_model_tokens = 0
        total_model_cost = 0.0
        total_possible_points = 0.0
        
        for q_id in question_codes:
            q_data = questions_data.get(q_id, {})
            total_possible_points += q_data.get("points", 1)
            
            model_results = all_results.get(q_id, {}).get(model, {})
            total_model_score += model_results.get("score", 0.0)
            total_model_tokens += model_results.get("total_tokens", 0)
            total_model_cost += model_results.get("total_cost", 0.0)
        
        # Format score
        if total_model_score.is_integer():
            s_str = str(int(total_model_score))
        else:
            s_str = f"{total_model_score:.2f}".rstrip('0').rstrip('.')
            
        if total_possible_points.is_integer():
            tp_str = str(int(total_possible_points))
        else:
            tp_str = str(total_possible_points)
            
        html_content += f"                        <td class='score'>{s_str}/{tp_str}</td>\n"
        html_content += f"                        <td class='tokens'>{total_model_tokens}</td>\n"
        
        cost_str = f"${total_model_cost:.4f}".rstrip('0').rstrip('.')
        if cost_str == "$":
            cost_str = "$0"
        html_content += f"                        <td class='cost'>{cost_str}</td>\n"

    html_content += """                    </tr>
                </tfoot>
            </table>
        </div>
    </body>
    </html>
    """
    
    with open(filepath, "w") as f:
        f.write(html_content)
        
    return filepath



def run_benchmark() -> None:
    """
    Executes the benchmark for questions defined in questions.txt across all loaded models.
    """
    questions_file = "config/questions.txt"
    if not os.path.exists(questions_file):
        logger.error("questions.txt not found.")
        return

    with open(questions_file, "r") as f:
        raw_lines = [line.strip() for line in f if line.strip()]

    if not raw_lines:
        logger.error("No question codes found in questions.txt")
        return

    # Expand subfolder entries (lines ending with "/") to all questions in that subfolder
    question_codes = []
    for line in raw_lines:
        if line.endswith("/"):
            # Treat as subfolder path relative to questions directory
            subfolder_path = os.path.join("questions", line.rstrip("/"))
            if os.path.isdir(subfolder_path):
                subfolder_files = glob.glob(os.path.join(subfolder_path, "*.txt"))
                for fpath in subfolder_files:
                    fname = os.path.basename(fpath)
                    # Exclude known non-question files
                    if fname in ["html_css_js_questions_prefix.txt", "readme.txt", "README.txt"]:
                        continue
                    code = os.path.splitext(fname)[0]
                    question_codes.append(code)
                logger.info("Expanded '%s' to %d questions.", line, len(subfolder_files))
            else:
                logger.warning("Subfolder '%s' not found, skipping.", subfolder_path)
        else:
            question_codes.append(line)

    if not question_codes:
        logger.error("No question codes found after expansion")
        return

    try:
        api = ModelAPI()
        judge = JudgeLLMEvaluator()
        human_eval = HumanEvaluator()
        validity_eval = ValidityEvaluator()
    except (ValueError, Exception) as e:
        logger.error("Failed to initialize system: %s", e)
        return

    # Handle "ALL" keyword
    if any(line.upper() == "ALL" for line in question_codes):
        logger.info("Found 'ALL' in questions.txt. Loading all available questions...")
        all_files = glob.glob("questions/**/*.txt", recursive=True)
        question_codes = []
        for fpath in all_files:
            fname = os.path.basename(fpath)
            # Filter out exclude files
            if fname in ["html_css_js_questions_prefix.txt", "readme.txt", "README.txt"]:
                continue
            
            # Use filename without extension as the code
            # This works with resolve_question_path logic since it matches prefixes
            code = os.path.splitext(fname)[0]
            question_codes.append(code)
        
        # Sort for consistent order
        question_codes.sort()
        logger.info("Discovered %d questions.", len(question_codes))

        # List questions and ask for confirmation
        print("\nQuestions to be run:")
        for idx, code in enumerate(question_codes, 1):
            print(f"{idx}. {code}")
            
        confirmation = input(f"\nAre you sure you want to run these {len(question_codes)} questions? (y/n): ")
        if confirmation.lower() != 'y':
            print("Execution cancelled by user.")
            return

    all_results: Dict[str, Dict[str, Any]] = {} 
    questions_data: Dict[str, Dict[str, Any]] = {}  # {question_code: {question, ground_truth, points, eval_type}}

    # Load prefix for manual checks
    prefix_file = "questions/html_css_js_questions_prefix.txt"
    prefix = ""
    if os.path.exists(prefix_file):
        with open(prefix_file, "r") as f:
            prefix = f.read().strip()
    else:
        logger.warning("Prefix file %s not found.", prefix_file)

    # Pre-load all questions and determine evaluation type
    logger.info("Loading questions...")
    valid_question_codes = []
    
    for code in question_codes:
        question_path = resolve_question_path(code)
        if not question_path:
            logger.error("Could not find question file for code: %s", code)
            continue

        try:
            with open(question_path, "r") as f:
                file_content = f.read()
        except OSError as e:
            logger.error("Failed to read question file at %s: %s", question_path, e)
            continue

        question, ground_truth, points = parse_question_file(file_content)

        # Prepend prefix if question is in manual_checks directory (except Leetcode questions)
        is_leetcode = "Leetcode" in os.path.basename(question_path)
        if "manual_checks" in question_path and not is_leetcode:
            question = f"{prefix}\n\n{question}"

        # Determine Evaluation Type
        if not ground_truth:
            eval_type = "eval by HumanEval"
        elif ground_truth.upper().strip() == "VALIDITY CHECK":
            eval_type = "eval by hardcoded validity checks"
        else:
            eval_type = "eval by Judge LLM"

        questions_data[code] = {
            "question": question,
            "ground_truth": ground_truth if ground_truth else "N/A",
            "points": points,
            "eval_type": eval_type,
            "is_manual_check": "manual_checks" in question_path
        }
        all_results[code] = {}
        valid_question_codes.append(code)

    if not valid_question_codes:
        logger.error("No valid questions found to benchmark.")
        return

    # Generate a single timestamp for all output files in this run
    run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Check if any manual check questions exist and start session
    has_manual_checks = any(
        questions_data[code].get("is_manual_check", False)
        for code in valid_question_codes
    )

    if has_manual_checks:
        human_eval.start_session(run_timestamp)

    # Partition questions: human eval first, then automated
    human_eval_codes = [c for c in valid_question_codes 
                        if questions_data[c].get("is_manual_check", False)]
    non_human_eval_codes = [c for c in valid_question_codes 
                            if not questions_data[c].get("is_manual_check", False)]
    sorted_question_codes = human_eval_codes + non_human_eval_codes
    
    # Print Summary to Console (use original order for display)
    print_benchmark_summary(api.models, questions_data, valid_question_codes)

    # Track whether subprocess has been spawned
    human_eval_server_spawned = False
    
    # Track exception for try/finally - ensures partial results are written on crash
    benchmark_exception = None
    processed_any_question = False

    # ThreadPoolExecutor for parallel runs
    # We want max parallelism, but let's limit to something reasonable
    # (e.g. 7 models * 4 runs = 28 concurrent requests by default)
    max_workers = int(os.getenv("MAX_WORKERS", "28"))
    try:
      with ThreadPoolExecutor(max_workers=max_workers) as executor:
        
        for code in sorted_question_codes:
            data = questions_data[code]
            question = data["question"]
            ground_truth = data["ground_truth"]
            # Convert "N/A" back to None/Empty if needed for logic, but process_single_run handles strings. 
            # Note: parse_question_file returns None for empty GT? Check. 
            # Actually process_single_run expects ground_truth string.
            # If "N/A" was stored, we should be careful. 
            # data["ground_truth"] stores "N/A" if empty.
            # Let's revert to None if "N/A" for the logic check in process_single_run or fix logic.
            # actually parse_question_file returns None or string.
            # Stored as "N/A" in dict.
            # process_single_run logic: `if ground_truth:`
            # If we pass "N/A", it evaluates as true.
            # So we should pass the original ground truth or handle "N/A".
            # Let's clean this up. parse_question_file returns None if missing.
            # In questions_data assignment above: "ground_truth": ground_truth if ground_truth else "N/A"
            # So we lost the None-ness.
            
            # Correction: Let's use the eval_type to determine what to pass or just fix the storage.
            # Simpler: Pass proper ground_truth.
            gt_for_run = ground_truth if ground_truth != "N/A" else None
            points = data["points"]
            
            logger.info("\n" + "=" * 60)
            logger.info("PROCESSING QUESTION: %s", code)
            logger.info("=" * 60)
            # Truncate very long prompts in console logs (e.g., A58 BattleShip)
            if len(question) > 1000:
                logger.info("--- Question [%s] (Points: %d) ---\n[Prompt truncated - %d chars]\n-----------------\n", code, points, len(question))
            else:
                logger.info("--- Question [%s] (Points: %d) ---\n%s\n-----------------\n", code, points, question)
            
            if gt_for_run:
                logger.info("[*] Expected Ground Truth: %s\n", gt_for_run)

            # Dictionary to hold futures for this question
            # Key: future, Value: (model_name, run_index)
            futures_map = {}

            # Submit all runs for all models for this question
            for i, model_name in enumerate(api.models):
                all_results[code][model_name] = {"runs": [], "score": 0.0}
                
                # Check for cached implementations (human_eval questions only)
                cached_impls = []
                is_human_eval_question = questions_data[code].get("is_manual_check", False)
                
                if COST_EFFECTIVE_ENABLED:
                    model_folder = find_model_folder(model_name)
                    if model_folder:
                        cached_impls = get_existing_implementations(
                            model_folder, code, NUM_RUNS
                        )
                
                num_cached = len(cached_impls)
                num_to_submit = NUM_RUNS - num_cached
                
                if num_cached > 0:
                    logger.info(
                        "[*] Model %s: Using %d cached + %d new runs for %s",
                        model_name, num_cached, num_to_submit, code
                    )
                else:
                    logger.info("[*] Submitting %d runs for Model: %s...", NUM_RUNS, model_name)
                
                # Process cached implementations first
                for cached in cached_impls:
                    content = cached["content"]
                    is_manual = questions_data[code].get("is_manual_check", False)
                    
                    if is_manual:
                        judge_reasoning = "Loaded from cost-effective cache"
                        judge_verdict = "Pending"
                        is_successful = False
                        
                        # Save implementation for human eval
                        human_eval.save_implementation(
                            model_name=model_name,
                            question_code=code,
                            run_index=cached["run_index"] - 1,  # Convert 1-indexed to 0-indexed
                            html_content=content,
                            max_points=points
                        )
                        status_log = "PENDING (Human Eval)"
                    else:
                        # Run evaluation on cached content
                        if gt_for_run:
                            if gt_for_run.upper().strip() == "VALIDITY CHECK":
                                eval_result = validity_eval.evaluate(code, question, content)
                                is_successful = eval_result["success"]
                                judge_reasoning = f"{eval_result['reasoning']} (Cached)"
                                judge_verdict = eval_result["verdict"]
                            else:
                                eval_result = judge.evaluate(question, gt_for_run, content)
                                is_successful = eval_result["success"]
                                judge_reasoning = f"{eval_result['reasoning']} (Cached)"
                                judge_verdict = eval_result["verdict"]
                        else:
                            # Should not happen unless config is weird
                            is_successful = False
                            judge_reasoning = "No Ground Truth - Cached"
                            judge_verdict = "Unknown"
                            
                        status_log = "PASS" if is_successful else "FAIL"

                    cached_result = {
                        "success": is_successful,
                        "response": content,
                        "model_reasoning": None,
                        "judge_reasoning": judge_reasoning,
                        "judge_verdict": judge_verdict,
                        "completion_tokens": 0,
                        "cost": 0.0
                    }
                    all_results[code][model_name]["runs"].append(cached_result)
                    
                    logger.info(
                        "    [%s] Run (%d/%d): CACHED (from %s) - %s",
                        model_name, cached["run_index"], NUM_RUNS, cached["source_file"], status_log
                    )
                
                # Submit only the remaining runs needed
                for run_idx in range(num_cached, NUM_RUNS):
                    future = executor.submit(
                        process_single_run,
                        api=api,
                        model_name=model_name,
                        model_index=i,
                        question_code=code,
                        question=question,
                        ground_truth=gt_for_run,
                        points=points,
                        judge=judge,
                        validity_eval=validity_eval,
                        human_eval=human_eval
                    )
                    futures_map[future] = (model_name, run_idx)

            # Collect results as they complete
            # Initialize counts with cached implementations already processed
            completed_counts = {model: len(all_results[code][model]["runs"]) for model in api.models}
            
            for future in as_completed(futures_map):
                model_name, run_idx = futures_map[future]
                try:
                    result = future.result()
                    all_results[code][model_name]["runs"].append(result)
                    
                    completed_counts[model_name] += 1
                    
                    # Extract detailed info from judge_reasoning if available
                    judge_info = result.get("judge_reasoning", "")
                    if result.get("judge_verdict") == "Pending":
                        status = "PENDING"
                    elif result["success"]:
                        status = f"PASS - {judge_info}" if judge_info else "PASS"
                    else:
                        status = f"FAIL - {judge_info}" if judge_info else "FAIL"
                    
                    logger.info("    [%s] Run (%d/%d): %s", 
                                model_name, completed_counts[model_name], NUM_RUNS, status)
                    
                    # Save implementation for manual check questions
                    if questions_data[code].get("is_manual_check", False):
                        human_eval.save_implementation(
                            model_name=model_name,
                            question_code=code,
                            run_index=run_idx,
                            html_content=result.get("response", ""),
                            max_points=points
                        )
                    
                except Exception as e:
                    logger.error("Use-case error collecting future for %s run %d: %s", model_name, run_idx, e)

            # Calculate scores for this question
            logger.info("\n--- Results for Question %s ---", code)
            for model_name in api.models:
                runs = all_results[code][model_name]["runs"]
                
                # Aggregate token usage and cost across all runs
                total_tokens = sum(r.get("completion_tokens", 0) for r in runs)
                total_cost = sum(r.get("cost", 0.0) for r in runs)
                
                # Check if this is a granular scoring question (parse SCORE:X/Y from reasoning)
                total_run_score = 0.0
                has_granular_scores = False
                
                for r in runs:
                    reasoning = r.get("judge_reasoning", "")
                    # Try to parse "SCORE:X/Y" pattern
                    import re
                    score_match = re.search(r'SCORE:(\d+)/(\d+)', reasoning)
                    if score_match:
                        has_granular_scores = True
                        run_score = int(score_match.group(1))
                        run_max = int(score_match.group(2))
                        # Store the run's granular score for display
                        r["run_score"] = run_score
                        r["run_max"] = run_max
                        # Normalize to question's point value
                        total_run_score += (run_score / run_max) * points
                    else:
                        # Fallback: binary success = full points, fail = 0
                        if r.get("success", False):
                            total_run_score += points
                
                # Average across all runs
                score = total_run_score / NUM_RUNS if NUM_RUNS > 0 else 0
                all_results[code][model_name]["score"] = score
                all_results[code][model_name]["total_tokens"] = total_tokens
                all_results[code][model_name]["total_cost"] = total_cost
                
                if has_granular_scores:
                    # Show detailed per-run scores
                    run_details = []
                    for r in runs:
                        if "run_score" in r:
                            run_details.append(f"{r['run_score']}/{r['run_max']}")
                        else:
                            run_details.append("?" if not r.get("success") else "PASS")
                    logger.info("Model: %-30s | Score: %.2f/%d (runs: %s)", 
                                model_name, score, points, ", ".join(run_details))
                else:
                    success_count = sum(1 for r in runs if r.get("success", False))
                    logger.info("Model: %-30s | Score: %.2f/%d (%d/%d PASS)", 
                                model_name, score, points, success_count, NUM_RUNS)
            
            # Mark that we have processable results (for partial writes on crash)
            processed_any_question = True
            
            # After last human eval question completes, finalize session and spawn server
            if (has_manual_checks 
                and human_eval_codes 
                and code == human_eval_codes[-1] 
                and not human_eval_server_spawned):
                
                human_eval.finalize_session()
                session_dir = human_eval.get_session_dir()
                
                logger.info("\n" + "=" * 60)
                logger.info("SPAWNING HUMAN EVALUATION SERVER")
                logger.info("=" * 60)
                logger.info("Human eval questions complete. Spawning server in background.")
                logger.info("Session directory: %s", session_dir)
                logger.info("Automated questions will continue processing.")
                logger.info("=" * 60 + "\n")
                
                # Spawn subprocess detached from parent
                script_dir = os.path.dirname(os.path.abspath(__file__))
                server_script = os.path.join(script_dir, "utils", "human_eval_server.py")
                
                logger.info("Launching: %s %s", server_script, session_dir)
                
                # Kill any existing server instance on this port
                logger.info("Ensuring port 8765 is free...")
                kill_process_on_port(8765)
                
                # Don't use start_new_session - it breaks webbrowser.open() 
                # as it loses access to the display environment
                subprocess.Popen(
                    [sys.executable, server_script, session_dir],
                    cwd=script_dir
                )
                human_eval_server_spawned = True

    except Exception as e:
        benchmark_exception = e
        logger.error("Benchmark crashed: %s", e)
        logger.info("Attempting to save partial results...")

    finally:
        # Always attempt to write results if we processed any questions
        if not processed_any_question:
            logger.warning("No questions were processed. Skipping results file generation.")
            if benchmark_exception:
                raise benchmark_exception
            return

        # Write results files
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING RESULTS FILES")
        logger.info("=" * 60)
        
        results_file_path = write_results_file(
            models=api.models,
            question_codes=valid_question_codes,
            all_results=all_results,
            questions_data=questions_data,
            timestamp=run_timestamp
        )

        logger.info("[+] Results file written to: %s", results_file_path)

        advanced_results_path = write_advanced_results_file(
            models=api.models,
            question_codes=valid_question_codes,
            all_results=all_results,
            questions_data=questions_data,
            timestamp=run_timestamp
        )
        
        logger.info("[+] Advanced results file written to: %s", advanced_results_path)
        
        # Update manifest with results file paths so integrate_scores uses the correct files
        if has_manual_checks and not benchmark_exception:
            human_eval.update_results_paths(results_file_path, advanced_results_path)
            
            # Store data needed for HTML generation (to be done after human eval completes)
            human_eval.store_html_generation_data(
                models=api.models,
                question_codes=valid_question_codes,
                all_results=all_results,
                questions_data=questions_data,
                timestamp=run_timestamp
            )
            
            # Check if human evaluation is already complete
            session_dir = human_eval.get_session_dir()
            manifest_path = os.path.join(session_dir, "manifest.json")
            
            import json
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            
            if manifest.get("scores_collected", False):
                # Human eval finished before benchmark - integrate now
                logger.info("\n" + "=" * 60)
                logger.info("INTEGRATING HUMAN EVALUATION SCORES")
                logger.info("=" * 60)
                
                from utils.integrate_human_scores import integrate_scores
                integrate_scores(session_dir)
                
                logger.info("Human evaluation scores integrated into results files.")
                
                # Generate HTML after integration (scores are now complete)
                html_path = generate_performance_html(
                    api.models, valid_question_codes, all_results, questions_data, run_timestamp
                )
                logger.info(" Performance table generated: file://%s", html_path)
            else:
                # Human eval still in progress - HTML will be generated after integration
                logger.info("\n" + "=" * 60)
                logger.info("HUMAN EVALUATION IN PROGRESS")
                logger.info("=" * 60)
                logger.info("Human evaluation server is running in the background.")
                logger.info("Session directory: %s", session_dir)
                logger.info("Complete scoring in the browser windows.")
                logger.info("Scores will be auto-integrated upon completion.")
                logger.info("Performance table will be generated after integration.")
        elif not has_manual_checks:
            # No human eval questions - generate HTML immediately
            html_path = generate_performance_html(
                api.models, valid_question_codes, all_results, questions_data, run_timestamp
            )
            logger.info(" Performance table generated: file://%s", html_path)
        
        # Log rankings to console at the very end
        if non_human_eval_codes:
            print("\n" + "#" * 60)
            print("FINAL MODEL RANKINGS (Automated Evaluation)")
            print("#" * 60)
            
            scores = {}
            usage = {}
            total_possible_points = 0
            for model in api.models:
                score = 0
                tokens = 0
                cost = 0.0
                for code in non_human_eval_codes:
                    model_data = all_results.get(code, {}).get(model, {})
                    score += model_data.get("score", 0.0)
                    tokens += model_data.get("total_tokens", 0)
                    cost += model_data.get("total_cost", 0.0)
                scores[model] = score
                usage[model] = (tokens, cost)
            
            for code in non_human_eval_codes:
                total_possible_points += questions_data.get(code, {}).get("points", 1)
            
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for rank, (model, score) in enumerate(ranked, 1):
                percentage = (score / total_possible_points * 100) if total_possible_points > 0 else 0
                tokens, cost = usage[model]
                print(f"{rank}. {model}: {score:.2f}/{total_possible_points} points ({percentage:.1f}%) - {tokens} tokens - ${cost:.3f}")
            print("#" * 60 + "\n")

        logger.info("=" * 60)
        
        # Re-raise the exception after saving partial results (preserves original crash behavior)
        if benchmark_exception:
            raise benchmark_exception


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark LLMs across diverse question sets"
    )
    parser.add_argument(
        "--delete_history",
        action="store_true",
        help="Clear all previous benchmark results before running"
    )

    args = parser.parse_args()

    if args.delete_history:
        logger.info("=" * 60)
        logger.info("CLEARING BENCHMARK HISTORY")
        logger.info("=" * 60)
        clear_history()
        logger.info("=" * 60)

    run_benchmark()

