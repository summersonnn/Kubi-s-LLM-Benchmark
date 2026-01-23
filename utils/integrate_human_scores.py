"""
Integrates human evaluation scores from manifest.json into benchmark results files.
Run this script after completing human evaluation via human_eval_server.py.
"""

import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

from utils.utils import setup_logging

logger = setup_logging(__name__)

NUM_RUNS = 4


def load_manifest(session_dir: str) -> Dict[str, Any]:
    """Loads and validates the manifest.json from a session directory."""
    manifest_path = os.path.join(session_dir, "manifest.json")
    
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    if not manifest.get("scores_collected", False):
        raise ValueError("Human evaluation not yet complete. scores_collected is False.")
    
    return manifest


def calculate_scores(manifest: Dict[str, Any]) -> Dict[str, Dict[str, float]]:
    """
    Calculates average scores per model per question.
    Returns: {question_code: {model_name: average_score}}
    """
    # Group scores by (question_code, model_name)
    grouped: Dict[tuple, List[float]] = {}
    
    for impl in manifest.get("implementations", []):
        key = (impl["question_code"], impl["model_name"])
        score = impl.get("score")
        
        if score is not None:
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(score)
    
    # Calculate averages
    result: Dict[str, Dict[str, float]] = {}
    for (question_code, model_name), scores in grouped.items():
        if question_code not in result:
            result[question_code] = {}
        
        avg = sum(scores) / len(scores) if scores else 0
        result[question_code][model_name] = avg
    
    return result


def find_latest_results_file(results_dir: str) -> str | None:
    """Finds the most recent results file in a directory."""
    if not os.path.exists(results_dir):
        return None
    
    files = [f for f in os.listdir(results_dir) if f.startswith("benchmark_results")]
    if not files:
        return None
    
    # Sort by modification time, newest first
    files.sort(key=lambda f: os.path.getmtime(os.path.join(results_dir, f)), reverse=True)
    return os.path.join(results_dir, files[0])


def update_results_file(
    filepath: str, 
    scores: Dict[str, Dict[str, float]],
    question_points: Dict[str, int]
) -> None:
    """
    Updates an existing results file with human evaluation scores.
    Scores are absolute values (0 to max_points) set directly by human evaluators.
    """
    with open(filepath, "r") as f:
        content = f.read()
    
    # Parse and update scores in the content
    lines = content.split("\n")
    updated_lines = []
    current_model = None
    current_question = None
    in_model_section = False
    
    # Track final totals for leaderboard
    model_totals: Dict[str, float] = {}
    model_possible: Dict[str, float] = {}
    
    # Remove existing FINAL MODEL RANKINGS section if it exists (to avoid duplication on re-runs)
    final_section_start = -1
    for idx, line in enumerate(lines):
        if "HUMAN EVALUATION ADDENDUM" in line or "FINAL MODEL RANKINGS" in line:
            # Look back for the start of the section (markers like === or ###)
            for j in range(idx - 1, max(0, idx - 5), -1):
                if lines[j].startswith("=") or lines[j].startswith("#"):
                    final_section_start = j
                    break
            if final_section_start == -1:
                final_section_start = idx
            break
    
    if final_section_start != -1:
        lines = lines[:final_section_start]

    for i, line in enumerate(lines):
        stripped = line.strip()
        
        # Detect model section header
        if stripped.endswith(" RESULTS:") and " " in stripped:
            current_model_raw = stripped.replace(" RESULTS:", "")
            current_model = current_model_raw.lower()
            if current_model_raw not in model_totals:
                model_totals[current_model_raw] = 0.0
                model_possible[current_model_raw] = 0.0
        elif stripped.startswith("MODEL: "):
            current_model_raw = stripped.replace("MODEL: ", "")
            current_model = current_model_raw.lower()
            if current_model_raw not in model_totals:
                model_totals[current_model_raw] = 0.0
                model_possible[current_model_raw] = 0.0
        
        # Detect question header
        if stripped.lower().startswith("question "):
            if "(" in line and "):" in line: # Regular format
                start = line.find("(") + 1
                end = line.find(")")
                if start > 0 and end > start:
                    current_question = line[start:end]
            elif ":" in line: # Advanced format
                parts = line.split(":")
                id_part = parts[1].strip()
                # Ensure it's not "QUESTION TEXT:" or "QUESTION ID:"
                if id_part and id_part != "TEXT" and not id_part.startswith("TEXT") and id_part != "ID":
                    current_question = id_part
        
        # Update score line
        if stripped.lower().startswith("score:") and current_question:
            indent = len(line) - len(line.lstrip())
            
            # Parse existing points if possible
            total_pts = 1.0
            if "/" in line:
                try:
                    points_part = line.split("/")[1].strip()
                    points_val = points_part.split("(")[0].split()[0].strip()
                    total_pts = float(points_val)
                except (ValueError, IndexError):
                    pass

            if current_question in scores:
                model_scores = scores.get(current_question, {})
                matched_score = None
                
                # Try to match current_model to one of the model names in scores
                # Strict matching: exact or one is a path suffix of the other
                for model_name, avg in model_scores.items():
                    model_lower = model_name.lower()
                    if current_model and (model_lower == current_model or 
                                          model_lower.endswith("/" + current_model) or 
                                          current_model.endswith("/" + model_lower)):
                        matched_score = avg
                        break
                
                if matched_score is not None:
                    # Use provided question_points or fallback to total_pts from file
                    points = question_points.get(current_question, total_pts)
                    
                    # Preserve original casing and spacing
                    score_prefix = "Score" if line.lstrip().startswith("Score") else "SCORE"
                    updated_lines.append(" " * indent + f"{score_prefix}: {matched_score:.2f}/{points:.1f} (Human Eval)")
                    
                    # Update totals
                    for model_raw in model_totals:
                        if model_raw.lower() == current_model:
                            model_totals[model_raw] += matched_score
                            model_possible[model_raw] += points
                    continue
            
            # If not a human eval question, track existing score for leaderboard
            try:
                score_part = line.split(":")[1].split("/")[0].strip()
                existing_score = float(score_part)
                for model_raw in model_totals:
                    if model_raw.lower() == current_model:
                        model_totals[model_raw] += existing_score
                        model_possible[model_raw] += total_pts
            except (ValueError, IndexError):
                pass
        
        # Also update "Runs:" line
        if stripped.lower().startswith("runs:") and current_question and current_question in scores:
            model_scores = scores.get(current_question, {})
            replaced = False
            for model_name, avg in model_scores.items():
                model_lower = model_name.lower()
                if current_model and (model_lower == current_model or 
                                      model_lower.endswith("/" + current_model) or 
                                      current_model.endswith("/" + model_lower)):
                    points = question_points.get(current_question, 1)
                    indent = len(line) - len(line.lstrip())
                    updated_lines.append(" " * indent + f"Runs: Human Eval (avg: {avg:.1f}/{points})")
                    replaced = True
                    break
            if replaced:
                continue
        
        updated_lines.append(line)

    
    # Append Human Evaluation Addendum section
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    addendum = [
        "",
        "=" * 80,
        f"HUMAN EVALUATION ADDENDUM ({timestamp})",
        "=" * 80,
        "",
        "The following scores reflect human evaluation results:",
        ""
    ]
    
    for question_code, model_scores in scores.items():
        points = question_points.get(question_code, 1)
        addendum.append(f"Question {question_code} (Points: {points}):")
        for model_name, avg in sorted(model_scores.items()):
            # Scores are absolute, avg is the final score
            addendum.append(f"  {model_name}: {avg:.2f}/{points} pts")
        addendum.append("")
    
    addendum.append("=" * 80)
    
    # Append Final Leaderboard
    addendum.extend([
        "",
        "#" * 80,
        "FINAL MODEL RANKINGS (Automated + Human Evaluation)",
        "#" * 80,
    ])
    
    # Sort models by total score
    ranked_models = sorted(model_totals.items(), key=lambda x: x[1], reverse=True)
    for i, (model_name, total_score) in enumerate(ranked_models, 1):
        possible = model_possible.get(model_name, 1.0)
        percentage = (total_score / possible * 100) if possible > 0 else 0
        
        # Format score with decimals only if needed
        if total_score.is_integer():
            s_str = f"{int(total_score)}"
        else:
            s_str = f"{total_score:.2f}"
            
        if possible.is_integer():
            p_str = f"{int(possible)}"
        else:
            p_str = f"{possible:.2f}"
            
        addendum.append(f"{i}. {model_name}: {s_str}/{p_str} points ({percentage:.1f}%)")
        
    addendum.append("#" * 80)
    updated_lines.extend(addendum)
    
    # Write updated content
    with open(filepath, "w") as f:
        f.write("\n".join(updated_lines))
    
    logger.info("Updated results file: %s", filepath)


def generate_human_eval_summary(
    session_dir: str,
    scores: Dict[str, Dict[str, float]],
    question_points: Dict[str, int]
) -> str:
    """Generates a summary file of human evaluation results."""
    summary_path = os.path.join(session_dir, "human_eval_summary.txt")
    
    with open(summary_path, "w") as f:
        f.write("=" * 60 + "\n")
        f.write("HUMAN EVALUATION RESULTS SUMMARY\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        for question_code, model_scores in scores.items():
            points = question_points.get(question_code, 1)
            f.write(f"Question: {question_code} (Points: {points})\n")
            f.write("-" * 40 + "\n")
            
            for model_name, avg in sorted(model_scores.items()):
                # Scores are absolute, avg is the final score
                f.write(f"  {model_name}: {avg:.2f}/{points} pts\n")
            
            f.write("\n")
        
        f.write("=" * 60 + "\n")
    
    return summary_path


def generate_standalone_results(
    filepath: str,
    scores: Dict[str, Dict[str, float]],
    question_points: Dict[str, int]
) -> None:
    """
    Generates a standalone results file when no existing benchmark results exist.
    Used for manual server runs where the benchmark wasn't run first.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Collect all unique models
    all_models: set[str] = set()
    for model_scores in scores.values():
        all_models.update(model_scores.keys())

    # Calculate total scores per model
    model_totals: Dict[str, float] = {model: 0.0 for model in all_models}
    total_possible = sum(question_points.values())

    for question_code, model_scores in scores.items():
        points = question_points.get(question_code, 1)
        for model_name, avg in model_scores.items():
            # Scores are absolute, avg is the final score
            model_totals[model_name] += avg

    with open(filepath, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"HUMAN EVALUATION RESULTS ({timestamp})\n")
        f.write("=" * 80 + "\n\n")

        # Summary section
        f.write("SUMMARY\n")
        f.write("-" * 40 + "\n")
        for model_name in sorted(all_models):
            total = model_totals[model_name]
            f.write(f"{model_name}: {total:.2f}/{total_possible} pts\n")
        f.write("\n")

        # Detailed results per model
        for model_name in sorted(all_models):
            f.write("=" * 80 + "\n")
            f.write(f"{model_name} RESULTS:\n")
            f.write("=" * 80 + "\n\n")

            question_num = 1
            for question_code, model_scores in sorted(scores.items()):
                if model_name in model_scores:
                    avg = model_scores[model_name]
                    points = question_points.get(question_code, 1)

                    f.write(f"Question {question_num} ({question_code}):\n")
                    f.write(f"  Runs: Human Eval (avg: {avg:.1f}/{points})\n")
                    f.write(f"  Score: {avg:.2f}/{points} (Human Eval)\n\n")
                    question_num += 1

        f.write("=" * 80 + "\n")

    logger.info("Generated standalone results: %s", filepath)


def integrate_scores(session_dir: str) -> None:
    """Main function to integrate human evaluation scores."""
    logger.info("Loading manifest from: %s", session_dir)
    manifest = load_manifest(session_dir)
    
    logger.info("Calculating scores...")
    scores = calculate_scores(manifest)
    
    # Extract question points from manifest (assuming all runs of same question have same points)
    # For now, default to 1 if not stored. We could also read from question files.
    question_points: Dict[str, int] = {}
    for impl in manifest.get("implementations", []):
        qc = impl["question_code"]
        if qc not in question_points:
            # Try to read from question file
            from main import resolve_question_path
            from utils.utils import parse_question_file
            
            path = resolve_question_path(qc)
            if path:
                with open(path, "r") as f:
                    _, _, pts = parse_question_file(f.read())
                    question_points[qc] = pts
            else:
                question_points[qc] = 1
    
    # Generate summary
    summary_path = generate_human_eval_summary(session_dir, scores, question_points)
    logger.info("Generated summary: %s", summary_path)

    # Create results directories if they don't exist
    os.makedirs("results", exist_ok=True)
    os.makedirs("results_advanced", exist_ok=True)

    # Update results files - only if paths are provided in manifest
    results_file = manifest.get("results_file")
    if results_file and os.path.exists(results_file):
        update_results_file(results_file, scores, question_points)
    elif results_file:
        logger.error("Results file specified in manifest but not found: %s", results_file)
    else:
        # Generate a standalone human eval results file if no benchmark results exist
        # This is for manual server runs outside of the main benchmark
        logger.info("No results file specified in manifest. Generating standalone results.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join("results", f"human_eval_results_{timestamp}.txt")
        generate_standalone_results(results_file, scores, question_points)

    advanced_file = manifest.get("advanced_results_file")
    if advanced_file and os.path.exists(advanced_file):
        update_results_file(advanced_file, scores, question_points)
    elif advanced_file:
        logger.error("Advanced results file specified in manifest but not found: %s", advanced_file)
    else:
        logger.info("No advanced results file to update (this is normal for manual server runs)")
    
    # Generate HTML performance table if generation data is available
    html_data = manifest.get("html_generation_data")
    if html_data:
        # Update all_results with human eval scores before generating HTML
        all_results = html_data["all_results"]
        questions_data = html_data["questions_data"]
        
        # Merge human eval scores into all_results
        for question_code, model_scores in scores.items():
            for model_name, avg in model_scores.items():
                if question_code in all_results and model_name in all_results[question_code]:
                    # Scores are absolute, use avg directly
                    all_results[question_code][model_name]["score"] = avg
        
        # Generate HTML
        from main import generate_performance_html
        html_path = generate_performance_html(
            html_data["models"],
            html_data["question_codes"],
            all_results,
            questions_data,
            html_data["timestamp"]
        )
        logger.info("Performance table generated: file://%s", html_path)
    
    logger.info("Score integration complete!")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python integrate_human_scores.py <session_dir>")
        print("Example: python integrate_human_scores.py manual_run_codes/benchmark_implementation_results_20260107_120000")
        sys.exit(1)
    
    session_dir = sys.argv[1]
    if not os.path.exists(session_dir):
        print(f"Error: Session directory not found: {session_dir}")
        sys.exit(1)
    
    integrate_scores(session_dir)
