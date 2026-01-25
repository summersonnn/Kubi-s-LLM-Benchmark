"""
Benchmarking script to evaluate multiple models on specific questions.
Parses question files, calls models via ModelAPI, and generates reports.
"""

import os
import sys
import re
import argparse
import asyncio
from datetime import datetime
from typing import Any, Dict, List

from dotenv import load_dotenv
from openai import AsyncOpenAI, OpenAIError

# Local Utils
from utils.model_api import ModelAPI
from utils.utils import setup_logging, kill_process_on_port, clear_history
from utils.evaluators import JudgeLLMEvaluator, HumanEvaluator, VerifierEvaluator
from utils.cost_effective import find_model_folder, get_existing_implementations

# New Modules
from utils.reporting import (
    print_benchmark_summary,
    write_advanced_results_file,
    write_results_file,
    generate_performance_html
)
from utils.question_loader import (
    discover_question_codes,
    load_questions_data
)

# Load environment variables
load_dotenv()

logger = setup_logging(__name__)

NUM_RUNS = int(os.getenv("NUM_RUNS", "4"))
COST_EFFECTIVE_ENABLED = os.getenv("COST_EFFECTIVE_ENABLED", "true").lower() in ("true", "1", "yes")


async def process_single_run(
    api: ModelAPI,
    model_name: str,
    model_index: int,
    question_code: str,
    question: str,
    ground_truth: str,
    points: int,
    judge: JudgeLLMEvaluator,
    verifier_eval: VerifierEvaluator,
    human_eval: HumanEvaluator,
    is_verifier_eval: bool = False
) -> Dict[str, Any]:
    """
    Executes a single run for a model on a question and returns the result.
    """
    try:
        # Calculate effective max tokens based on question points
        effective_max_tokens = api.max_tokens * points
        
        # Calculate dynamic timeout based on question points
        # Base timeout is for 1 point. Scale linearly.
        dynamic_timeout = api.timeout * points
        
        response = await api.call(
            question, 
            model_index=model_index, 
            max_tokens=effective_max_tokens,
            timeout=dynamic_timeout
        )

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

        if is_verifier_eval:
            eval_result = await verifier_eval.evaluate(question_code, question, content)
            is_successful = eval_result["success"]
            judge_reasoning = eval_result["reasoning"]
            judge_verdict = eval_result["verdict"]
        elif ground_truth:
            eval_result = await judge.evaluate(question, ground_truth, content, points=points)
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


async def run_benchmark(specific_questions: List[str] | None = None) -> None:
    """
    Executes the benchmark for questions.
    
    Args:
        specific_questions: If provided, runs benchmark ONLY on these question codes.
                            If None, reads question codes from config/questions.txt.
    """
    # 1. Discover Steps
    question_codes = discover_question_codes(specific_questions)
    
    if not question_codes:
        # Might be empty if discovery failed or user cancelled confirmation
        return

    try:
        api = ModelAPI()
        judge = JudgeLLMEvaluator()
        human_eval = HumanEvaluator()
        verifier_eval = VerifierEvaluator()
    except (ValueError, Exception) as e:
        logger.error("Failed to initialize system: %s", e)
        return

    # 2. Load Questions Metadata
    questions_data, valid_question_codes = load_questions_data(question_codes)

    if not valid_question_codes:
        logger.error("No valid questions found to benchmark.")
        return

    # Initialize results container
    all_results: Dict[str, Dict[str, Any]] = {}
    for code in valid_question_codes:
        all_results[code] = {}

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

    # Async concurrency control
    max_workers_count = int(os.getenv("MAX_WORKERS", "28"))
    semaphore = asyncio.Semaphore(max_workers_count)

    async def run_with_semaphore(
        api, model_name, model_index, question_code, question, ground_truth, points, judge, verifier_eval, human_eval, is_verifier_eval
    ):
        async with semaphore:
            return await process_single_run(
                api, model_name, model_index, question_code, question, ground_truth, points, judge, verifier_eval, human_eval, is_verifier_eval
            )

    try:
        for code in sorted_question_codes:
            data = questions_data[code]
            question = data["question"]
            ground_truth = data["ground_truth"]
            gt_for_run = ground_truth if ground_truth != "N/A" else None
            points = data["points"]
            
            logger.info("\n" + "=" * 60)
            logger.info("PROCESSING QUESTION: %s", code)
            logger.info("=" * 60)
            if len(question) > 1000:
                logger.info("--- Question [%s] (Points: %d) ---\n[Prompt truncated - %d chars]\n-----------------\n", code, points, len(question))
            else:
                logger.info("--- Question [%s] (Points: %d) ---\n%s\n-----------------\n", code, points, question)
            
            if gt_for_run:
                logger.info("[*] Expected Ground Truth: %s\n", gt_for_run)

            # Prepare tasks for this question
            tasks = []
            # Keep track of which model/run each task belongs to
            # task_info[task_idx] = (model_name, run_idx)
            task_info = []

            for i, model_name in enumerate(api.models):
                all_results[code][model_name] = {"runs": [], "score": 0.0}
                
                cached_impls = []
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
                
                # Handle cached runs immediately
                for cached in cached_impls:
                    content = cached["content"]
                    is_manual = questions_data[code].get("is_manual_check", False)
                    status_log = "UNKNOWN"
                    
                    if is_manual:
                        judge_reasoning = "Loaded from cost-effective cache"
                        judge_verdict = "Pending"
                        is_successful = False
                        human_eval.save_implementation(
                            model_name=model_name,
                            question_code=code,
                            run_index=cached["run_index"] - 1,
                            html_content=content,
                            max_points=points
                        )
                        status_log = "PENDING (Human Eval)"
                    elif questions_data[code].get("is_verifier_eval", False):
                        # For async verification, we can wait or just run it. 
                        # Since it's cached content, let's just await it properly here.
                        eval_result = await verifier_eval.evaluate(code, question, content)
                        is_successful = eval_result["success"]
                        judge_reasoning = f"{eval_result['reasoning']} (Cached)"
                        judge_verdict = eval_result["verdict"]
                        status_log = "PASS" if is_successful else "FAIL"
                    elif gt_for_run:
                        # For async judge, same deal.
                        eval_result = await judge.evaluate(question, gt_for_run, content)
                        is_successful = eval_result["success"]
                        judge_reasoning = f"{eval_result['reasoning']} (Cached)"
                        judge_verdict = eval_result["verdict"]
                        status_log = "PASS" if is_successful else "FAIL"
                    else:
                        is_successful = False
                        judge_reasoning = "No Ground Truth - Cached"
                        judge_verdict = "Unknown"
                        status_log = "FAIL"

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

                # Queue new runs
                for run_idx in range(num_cached, NUM_RUNS):
                    coroutine = run_with_semaphore(
                        api=api,
                        model_name=model_name,
                        model_index=i,
                        question_code=code,
                        question=question,
                        ground_truth=gt_for_run,
                        points=points,
                        judge=judge,
                        verifier_eval=verifier_eval,
                        human_eval=human_eval,
                        is_verifier_eval=questions_data[code].get("is_verifier_eval", False)
                    )
                    tasks.append(coroutine)
                    task_info.append((model_name, run_idx))

            # Run all tasks for this question
            if tasks:
                completed_counts = {model: len(all_results[code][model]["runs"]) for model in api.models}
                
                async def wrapped_task(name, idx, coro):
                    try:
                        res = await coro
                        return res, name, idx, None
                    except Exception as exc:
                        return None, name, idx, exc

                wrapped_coroutines = [
                    wrapped_task(m_name, r_idx, task) 
                    for (m_name, r_idx), task in zip(task_info, tasks)
                ]
                
                for task_result in asyncio.as_completed(wrapped_coroutines):
                    result, model_name, run_idx, error = await task_result
                    
                    if error:
                        logger.error("Error collecting future for %s run %d: %s", model_name, run_idx, error)
                    else:
                        all_results[code][model_name]["runs"].append(result)
                        completed_counts[model_name] += 1
                        
                        judge_info = result.get("judge_reasoning", "")
                        if result.get("judge_verdict") == "Pending":
                            status = "PENDING"
                        elif result["success"]:
                            status = f"PASS - {judge_info}" if judge_info else "PASS"
                        else:
                            status = f"FAIL - {judge_info}" if judge_info else "FAIL"
                        
                        logger.info("    [%s] Run (%d/%d): %s", 
                                    model_name, completed_counts[model_name], NUM_RUNS, status)
                        
                        if questions_data[code].get("is_manual_check", False):
                            human_eval.save_implementation(
                                model_name=model_name,
                                question_code=code,
                                run_index=run_idx,
                                html_content=result.get("response", ""),
                                max_points=points
                            )
                    
                    processed_any_question = True

            # Calculate scores for this question
            logger.info("\n--- Results for Question %s ---", code)
            for model_name in api.models:
                runs = all_results[code][model_name]["runs"]
                total_tokens = sum(r.get("completion_tokens", 0) for r in runs)
                total_cost = sum(r.get("cost", 0.0) for r in runs)
                
                total_run_score = 0.0
                has_granular_scores = False
                
                for r in runs:
                    reasoning = r.get("judge_reasoning", "")
                    score_match = re.search(r'SCORE:(\d+)/(\d+)', reasoning)
                    if score_match:
                        has_granular_scores = True
                        run_score = int(score_match.group(1))
                        run_max = int(score_match.group(2))
                        r["run_score"] = run_score
                        r["run_max"] = run_max
                        total_run_score += (run_score / run_max) * points
                    else:
                        if r.get("success", False):
                            total_run_score += points
                
                score = total_run_score / NUM_RUNS if NUM_RUNS > 0 else 0
                all_results[code][model_name]["score"] = score
                all_results[code][model_name]["total_tokens"] = total_tokens
                all_results[code][model_name]["total_cost"] = total_cost
                
                if has_granular_scores:
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
            
            # Subprocess spawning logic (human eval)
            if (has_manual_checks 
                and human_eval_codes 
                and code == human_eval_codes[-1] 
                and not human_eval_server_spawned):
                
                human_eval.finalize_session()
                session_dir = human_eval.get_session_dir()
                logger.info("\n" + "=" * 60 + "\nSPAWNING HUMAN EVALUATION SERVER")
                
                script_dir = os.path.dirname(os.path.abspath(__file__))
                server_script = os.path.join(script_dir, "utils", "human_eval_server.py")
                logger.info("Launching: %s %s", server_script, session_dir)
                kill_process_on_port(8765)
                import subprocess
                subprocess.Popen(
                    [sys.executable, server_script, session_dir],
                    cwd=script_dir
                )
                human_eval_server_spawned = True

    except (Exception, KeyboardInterrupt, asyncio.CancelledError) as e:
        benchmark_exception = e
        if isinstance(e, KeyboardInterrupt):
            logger.warning("Benchmark interrupted by user.")
        else:
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
        if has_manual_checks:
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

            else:
                # Human eval still in progress - HTML will be generated after integration
                logger.info("\n" + "=" * 60)
                logger.info("HUMAN EVALUATION IN PROGRESS")
                logger.info("=" * 60)
                logger.info("Human evaluation server is running in the background.")
                logger.info("Session directory: %s", session_dir)
                logger.info("Complete scoring in the browser windows.")
                logger.info("Run 'uv run python utils/integrate_human_scores.py %s' after completion.", session_dir)
                logger.info("The integration script will update results and generate the performance table.")

        else:
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
    
    from utils.question_renamer import ensure_question_naming_convention

    # Check for "!!!" trigger in questions.txt
    questions_file = "config/questions.txt"
    run_only_renamed = False
    
    if os.path.exists(questions_file):
        with open(questions_file, "r") as f:
            content = f.read().strip()
            if content == "!!!":
                run_only_renamed = True

    if run_only_renamed:
        print("Trigger '!!!' detected. Running benchmark ONLY on identified non-conforming questions.")
        
        # 1. Rename and get list
        new_codes = ensure_question_naming_convention()
        
        if not new_codes:
            print("No non-conforming questions were found/renamed. Exiting as requested by '!!!' trigger.")
            # We exit here because the user explicitly asked to run on *renamed* files. 
            # If none were renamed, there's nothing to run.
            sys.exit(0)
            
        print(f"Renamed {len(new_codes)} questions. Starting benchmark for: {new_codes}")
        
        # 2. Run benchmark on specific new codes
        asyncio.run(run_benchmark(specific_questions=new_codes))

    else:
        # Standard behavior
        # Enforce naming conventions (we don't constrain run to these, just tidy up)
        ensure_question_naming_convention()

        if args.delete_history:
            logger.info("=" * 60)
            logger.info("CLEARING BENCHMARK HISTORY")
            logger.info("=" * 60)
            clear_history()
            logger.info("=" * 60)

        asyncio.run(run_benchmark())

    logger.info("Benchmark run complete.")
