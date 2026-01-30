"""
Core benchmark execution engine.
Encapsulates the lifecycle of a benchmark run, including model initialization,
question discovery, and results aggregation.
"""

import os
import sys
import asyncio
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from openai import OpenAIError

from utils.model_api import ModelAPI
from utils.utils import kill_process_on_port
from utils.evaluators import JudgeLLMEvaluator, VerifierEvaluator
from utils.question_loader import discover_question_codes, load_questions_data
from utils.reporting import (
    print_benchmark_summary,
    write_advanced_results_file,
    write_results_file,
    generate_performance_html,
    print_final_rankings
)

from utils.utils import setup_logging

logger = setup_logging(__name__)


class BenchmarkRunner:
    def __init__(self):
        self.num_runs = int(os.getenv("NUM_RUNS", "4"))
        self.max_workers = int(os.getenv("MAX_WORKERS", "28"))
        
        # Initialize components
        try:
            self.api = ModelAPI()
            self.judge = JudgeLLMEvaluator()
            self.verifier_eval = VerifierEvaluator()
        except (ValueError, Exception) as e:
            logger.error("Failed to initialize system: %s", e)
            raise

    async def _process_single_run(
        self,
        model_name: str,
        model_index: int,
        question_code: str,
        question: str,
        ground_truth: Optional[str],
        points: int,
        is_verifier_eval: bool = False
    ) -> Dict[str, Any]:
        """
        Executes a single run for a model on a question and returns the result.
        """
        try:
            # Calculate effective max tokens based on question points
            effective_max_tokens = self.api.max_tokens * points
            
            response = await self.api.call(
                question, 
                model_index=model_index, 
                max_tokens=effective_max_tokens
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
                eval_result = await self.verifier_eval.evaluate(question_code, question, content)
                is_successful = eval_result["success"]
                judge_reasoning = eval_result["reasoning"]
                judge_verdict = eval_result["verdict"]
            elif ground_truth:
                eval_result = await self.judge.evaluate(question, ground_truth, content, points=points)
                is_successful = eval_result["success"]
                judge_reasoning = eval_result["reasoning"]
                judge_verdict = eval_result["verdict"]
            else:
                # Should not happen if logic is correct, but fallback
                is_successful = False
                judge_reasoning = "No Evaluation Method Found"
                judge_verdict = "Error"

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

    async def run(self, specific_questions: List[str] | None = None) -> None:
        """
        Executes the benchmark for questions using a sliding window approach.
        
        Args:
            specific_questions: If provided, runs benchmark ONLY on these question codes.
                                If None, reads question codes from config/questions.txt.
        """
        # 1. Discover Steps
        question_codes = discover_question_codes(specific_questions)
        
        if not question_codes:
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

        sorted_question_codes = valid_question_codes
        
        # Print Summary to Console
        print_benchmark_summary(self.api.models, questions_data, valid_question_codes)

        logger.info(f"Sliding Window execution enabled: Max workers = {self.max_workers}")

        # Track exception for try/finally
        benchmark_exception = None
        processed_any_question = False

        # Async concurrency control
        semaphore = asyncio.Semaphore(self.max_workers)

        async def run_with_semaphore(
            model_name, model_index, question_code, question, ground_truth, points, is_verifier_eval
        ):
            async with semaphore:
                return await self._process_single_run(
                    model_name, model_index, question_code, question, ground_truth, points, is_verifier_eval
                )

        # Task Wrapper
        async def wrapped_task(name, idx, q_code, coro):
            try:
                res = await coro
                return res, name, idx, q_code, None
            except Exception as exc:
                return None, name, idx, q_code, exc

        # Prepare ALL tasks upfront
        all_tasks = []
        # Keep track of expected run count per question to know when to print summary
        # {question_code: remaining_runs}
        runs_remaining = {code: 0 for code in valid_question_codes}
        
        # We also need to initialize the result structure properly for all models/questions first
        for code in sorted_question_codes:
            for model_name in self.api.models:
                all_results[code][model_name] = {"runs": [], "score": 0.0}

        try:
            logger.info("Initializing tasks...")
            
            for code in sorted_question_codes:
                data = questions_data[code]
                question = data["question"]
                ground_truth = data["ground_truth"]
                gt_for_run = ground_truth if ground_truth != "N/A" else None
                points = data["points"]
                
                # We won't log "PROCESSING QUESTION" here for every single one upfront to avoid spam,
                # or maybe we should? The original code logged it per loop. 
                # Let's log a brief summary or just log when we *finish* a question.
                # Actually, logging "Queuing X..." is fine.
                
                
                for i, model_name in enumerate(self.api.models):
                    # Create tasks for all runs
                    for run_idx in range(self.num_runs):
                        coro = run_with_semaphore(
                            model_name=model_name,
                            model_index=i,
                            question_code=code,
                            question=question,
                            ground_truth=gt_for_run,
                            points=points,
                            is_verifier_eval=questions_data[code].get("is_verifier_eval", False)
                        )
                        task = wrapped_task(model_name, run_idx, code, coro)
                        all_tasks.append(task)
                        runs_remaining[code] += 1
            
            logger.info(f"Queued {len(all_tasks)} total tasks across {len(sorted_question_codes)} questions.")
            
            # Helper to calculate and print scores for a single completed question
            def process_completed_question(q_code):
                points = questions_data[q_code]["points"]
                question_text = questions_data[q_code]["question"]
                
                logger.info("\n" + "=" * 60)
                logger.info("COMPLETED QUESTION: %s", q_code)
                if len(question_text) > 200:
                    logger.info("Content: %s...", question_text[:200].replace('\n', ' '))
                else:
                    logger.info("Content: %s", question_text.replace('\n', ' '))
                
                logger.info("--- Results for Question %s ---", q_code)
                
                for model_name in self.api.models:
                    runs = all_results[q_code][model_name]["runs"]
                    total_tokens = sum(r.get("completion_tokens", 0) for r in runs)
                    total_cost = sum(r.get("cost", 0.0) for r in runs)
                    
                    total_run_score = 0.0
                    has_granular_scores = False
                    
                    for r in runs:
                        reasoning = r.get("judge_reasoning", "")
                        score_match = re.search(r'SCORE:([\d.]+)/[\d.]+', reasoning)
                        if score_match:
                            has_granular_scores = True
                            run_score = float(score_match.group(1))
                            r["run_score"] = run_score
                            total_run_score += run_score
                        elif r.get("success", False):
                            total_run_score += points
                        # If neither score nor success, it's a 0 for this run
                    
                    score = total_run_score / self.num_runs if self.num_runs > 0 else 0
                    all_results[q_code][model_name]["score"] = score
                    all_results[q_code][model_name]["total_tokens"] = total_tokens
                    all_results[q_code][model_name]["total_cost"] = total_cost
                    
                    if has_granular_scores:
                        run_details = []
                        for r in runs:
                            if "run_score" in r:
                                run_details.append(f"{r['run_score']}/{points}")
                            else:
                                run_details.append("?" if not r.get("success") else "PASS")
                        logger.info("Model: %-30s | Score: %.2f/%d (runs: %s)", 
                                    model_name, score, points, ", ".join(run_details))
                    else:
                        success_count = sum(1 for r in runs if r.get("success", False))
                        logger.info("Model: %-30s | Score: %.2f/%d (%d/%d PASS)", 
                                    model_name, score, points, success_count, self.num_runs)
                logger.info("=" * 60 + "\n")

            # Process tasks as they complete
            if all_tasks:
                completed_counts = {} # (q_code, model) -> count
                for q_code in valid_question_codes:
                    for m_name in self.api.models:
                         completed_counts[(q_code, m_name)] = 0

                for task_result in asyncio.as_completed(all_tasks):
                    result, model_name, run_idx, q_code, error = await task_result
                    
                    processed_any_question = True
                    runs_remaining[q_code] -= 1
                    
                    if error:
                        logger.error("Error collecting future for %s run %d (Q: %s): %s", model_name, run_idx, q_code, error)
                    else:
                        all_results[q_code][model_name]["runs"].append(result)
                        completed_counts[(q_code, model_name)] += 1
                        
                        judge_info = result.get("judge_reasoning", "")
                        if result.get("judge_verdict") == "Pending":
                            status = "PENDING"
                        elif result["success"]:
                            status = f"PASS - {judge_info}" if judge_info else "PASS"
                        else:
                            status = f"FAIL - {judge_info}" if judge_info else "FAIL"
                        
                        logger.info("    [%s] Run (%d/%d) for %s: %s", 
                                    model_name, completed_counts[(q_code, model_name)], self.num_runs, q_code, status)
                        
                    
                    # Check if this question is now fully complete
                    if runs_remaining[q_code] == 0:
                        process_completed_question(q_code)


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
                models=self.api.models,
                question_codes=valid_question_codes,
                all_results=all_results,
                questions_data=questions_data,
                timestamp=run_timestamp
            )

            logger.info("[+] Results file written to: %s", results_file_path)

            advanced_results_path = write_advanced_results_file(
                models=self.api.models,
                question_codes=valid_question_codes,
                all_results=all_results,
                questions_data=questions_data,
                timestamp=run_timestamp
            )
            
            logger.info("[+] Advanced results file written to: %s", advanced_results_path)
            
            # No human eval questions - generate HTML immediately
            html_path = generate_performance_html(
                self.api.models, valid_question_codes, all_results, questions_data, run_timestamp
            )
            logger.info(" Performance table generated: file://%s", html_path)
            
            # Log rankings to console at the very end
            print_final_rankings(self.api.models, valid_question_codes, all_results, questions_data)

            logger.info("=" * 60)
            
            # Re-raise the exception after saving partial results (preserves original crash behavior)
            if benchmark_exception:
                raise benchmark_exception
