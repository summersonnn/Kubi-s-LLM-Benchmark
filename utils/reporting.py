"""
Benchmark reporting utilities for generating result summaries and HTML tables.
Handles file output for standard and advanced result formats.
"""

import os
import re
from datetime import datetime
from typing import Any, Dict, List, Tuple

def calculate_model_rankings(
    models: List[str],
    question_codes: List[str], # These should be the filtered codes (e.g. non_human_eval_codes)
    all_results: Dict[str, Dict[str, Any]],
    questions_data: Dict[str, Dict[str, Any]]
) -> Tuple[List[Tuple[str, float]], Dict[str, Tuple[int, float]], float]:
    """
    Calculates model rankings from benchmark results.
    Returns: (ranked_models, usage_by_model, total_possible_points)
    """
    scores = {}
    usage = {} # {model: (tokens, cost)}
    total_possible_points = 0.0

    for model in models:
        score = 0.0
        tokens = 0
        cost = 0.0
        for code in question_codes:
            model_data = all_results.get(code, {}).get(model, {})
            score += model_data.get("score", 0.0)
            tokens += model_data.get("total_tokens", 0)
            cost += model_data.get("total_cost", 0.0)
        scores[model] = score
        usage[model] = (tokens, cost)

    # Calculate total possible points
    for code in question_codes:
        total_possible_points += questions_data.get(code, {}).get("points", 1)

    # Sort by score descending
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    return ranked, usage, total_possible_points

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
    timestamp: str | None = None,
    output_dir: str = "results_advanced"
) -> str:
    """
    Writes comprehensive advanced benchmark results to a timestamped file.
    Includes full model responses, reasoning, and all evaluation details for ALL runs.
    Returns the path to the created file.
    """
    # Create results_advanced directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Use provided timestamp or generate new one
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_advanced_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
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
                    # Skip full response if it contains code blocks or is too long
                    if len(model_response) > 2000 or "```" in model_response:
                        f.write("...\n")
                        f.write(f"{model_response[-128:]}\n\n")
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
            
            # Calculate rankings using helper
            ranked, usage, total_possible_points = calculate_model_rankings(
                models, non_human_eval_codes, all_results, questions_data
            )
            
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


def print_final_rankings(
    models: List[str],
    question_codes: List[str],
    all_results: Dict[str, Dict[str, Any]],
    questions_data: Dict[str, Dict[str, Any]]
) -> None:
    """
    Prints the final model rankings to the console.
    """
    # Filter for non-human eval codes if not already filtered, but typically
    # the caller should pass the correct list. We'll handle both cases safely.
    non_human_eval_codes = [
        code for code in question_codes 
        if not questions_data.get(code, {}).get("is_manual_check", False)
    ]

    if non_human_eval_codes:
        print("\n" + "#" * 60)
        print("FINAL MODEL RANKINGS (Automated Evaluation)")
        print("#" * 60)
        
        ranked, usage, total_possible_points = calculate_model_rankings(
            models, non_human_eval_codes, all_results, questions_data
        )
        
        for rank, (model, score) in enumerate(ranked, 1):
            percentage = (score / total_possible_points * 100) if total_possible_points > 0 else 0
            tokens, cost = usage[model]
            print(f"{rank}. {model}: {score:.2f}/{total_possible_points} points ({percentage:.1f}%) - {tokens} tokens - ${cost:.3f}")
        print("#" * 60 + "\n")



def write_results_file(
    models: List[str],
    question_codes: List[str],
    all_results: Dict[str, Dict[str, Any]],
    questions_data: Dict[str, Dict[str, Any]],
    timestamp: str | None = None,
    output_dir: str = "results"
) -> str:
    """
    Writes benchmark results to a timestamped file in the results directory.
    Returns the path to the created file.
    """
    # Create results directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Use provided timestamp or generate new one
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"benchmark_results_{timestamp}.txt"
    filepath = os.path.join(output_dir, filename)
    
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
                    f.write("  Score: PENDING (Human Eval)\n")
                    f.write("  Runs: PENDING\n\n")
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
            
            # Calculate rankings using helper
            ranked, usage, total_possible_points = calculate_model_rankings(
                models, non_human_eval_codes, all_results, questions_data
            )
            
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


def generate_performance_html(
    models: List[str],
    question_codes: List[str],
    all_results: Dict[str, Dict[str, Any]],
    questions_data: Dict[str, Dict[str, Any]],
    timestamp: str,
    output_dir: str = "results"
) -> str:
    """
    Generates an HTML performance table and saves it to a file.
    Returns the absolute path to the generated HTML file.
    """
    os.makedirs(output_dir, exist_ok=True)
    filename = f"performance_table_{timestamp}.html"
    filepath = os.path.abspath(os.path.join(output_dir, filename))
    
    # Extract short model names - strip @preset/... suffix first, then take name after provider/
    short_models = [m.split("@")[0].split("/")[-1] for m in models]

    # Dynamically determine category order
    all_categories = set()
    for q_data in questions_data.values():
        cat = q_data.get("category", "Other")
        all_categories.add(cat)
    
    # Sort categories alphabetically, ensuring "Other" is last if present
    category_list = sorted(list(all_categories))
    if "Other" in category_list:
        category_list.remove("Other")
        category_list.append("Other")
        
    CATEGORY_ORDER = category_list

    def sort_key(q_id):
        data = questions_data.get(q_id, {})
        cat = data.get("category", "Other")
        sub = data.get("subcategory", "Other")

        try:
            cat_idx = CATEGORY_ORDER.index(cat)
        except ValueError:
            cat_idx = len(CATEGORY_ORDER)

        # For numeric part (handling things like A1, A48.1, A48.10)
        id_part = q_id.split('-')[0]
        match = re.match(r'A(\d+(?:\.\d+)?)', id_part)
        num = float(match.group(1)) if match else 999.0

        return (cat_idx, sub, num)

    sorted_q_ids = sorted(question_codes, key=sort_key)

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
    
    for q_id in sorted_q_ids:
        q_data = questions_data.get(q_id, {})
        points = q_data.get("points", 1)
        category = q_data.get("category", "")
        subcategory = q_data.get("subcategory", "")

        # Format points
        if isinstance(points, float) and points.is_integer():
            p_str = str(int(points))
        else:
            p_str = str(points)

        # Use only the ID (e.g., A1, A48.1)
        display_id = q_id.split('-')[0]

        html_content += f"                    <tr data-category='{category}' data-subcategory='{subcategory}'>\n                        <td class='q-col'>{display_id}</td>\n                        <td>{p_str}</td>\n"
        
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
