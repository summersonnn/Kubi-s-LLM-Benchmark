"""
Evaluation framework for benchmarking models.
Includes JudgeLLMEvaluator for automated assessment and HumanEvaluator for manual review.
"""

import os
from typing import Any, Dict, Protocol

from utils.cost_effective import extract_base_question_code
from utils.model_api import ModelAPI
from utils.utils import setup_logging

logger = setup_logging(__name__)

class Evaluator(Protocol):
    """Protocol for evaluation strategies."""
    def evaluate(self, question: str, ground_truth: str, answer: str) -> bool:
        ...

class JudgeLLMEvaluator:
    """
    Evaluator that uses a separate LLM to judge the correctness of an answer.
    """

    def __init__(self, judge_model_path: str = "config/judge_model.txt") -> None:
        """
        Initializes the judge evaluator by loading the judge model.
        """
        self.api = ModelAPI()
        self.judge_model_name = self._load_judge_model(judge_model_path)
        
        # Ensure the judge model is NOT in the competing models list to keep ModelAPI clean
        # or we can just specify the model name directly in the call.
        # Actually ModelAPI uses an index. I might need to modify ModelAPI or 
        # add a direct model call method.

    def _load_judge_model(self, path: str) -> str:
        if not os.path.exists(path):
            logger.warning("Judge model file %s not found. Defaulting to gemini-2.0-flash-exp", path)
            return "google/gemini-2.0-flash-exp"
        with open(path, "r") as f:
            return f.read().strip()

    def evaluate(self, question: str, ground_truth: str, answer: str, points: int = 1) -> dict[str, Any]:
        """
        Uses the judge LLM to evaluate the answer against the ground truth.
        Returns a dictionary with 'success', 'reasoning', and 'verdict'.
        """
        prompt = self._build_judge_prompt(question, ground_truth, answer)
        
        # Judge timeout is 1/3 of the competing model's timeout for the same points
        # Competing model timeout = api.timeout * points
        # Judge timeout = (api.timeout * points) / 3.0
        judge_timeout = (self.api.timeout * points) / 3.0
        
        max_retries = 3
        result_text = ""
        
        try:
            for attempt in range(1, max_retries + 1):
                try:
                    response = self.api.call(
                        prompt=prompt,
                        model_name=self.judge_model_name,
                        max_tokens=1024,
                        temperature=0.0,
                        timeout=judge_timeout,
                        reasoning=False # Reasoning not needed for judge extraction
                    )
                    result_text = (response.choices[0].message.content or "").strip()
                    break # Success, exit retry loop
                    
                except (TimeoutError, Exception) as e:
                    # Check if it looks like a timeout
                    is_timeout = isinstance(e, TimeoutError) or "timeout" in str(e).lower()
                    
                    if is_timeout:
                        logger.warning(
                            "Judge LLM timed out (Attempt %d/%d). Retrying...", 
                            attempt, max_retries
                        )
                        if attempt == max_retries:
                            logger.error("Judge LLM failed after %d retries due to timeout.", max_retries)
                            return {
                                "success": False,
                                "reasoning": f"Judge LLM timed out after {max_retries} attempts.",
                                "verdict": "Error"
                            }
                        import time
                        time.sleep(1) # Brief pause before retry
                    else:
                        # Non-timeout error - propagate to outer try/except
                        raise e

            # Parse reasoning and verdict
            # Expecting format: "Reasoning: ... Verdict: Pass/Fail"
            verdict = "Fail"
            reasoning = result_text
            
            if "verdict:" in result_text.lower():
                parts = result_text.lower().rsplit("verdict:", 1)
                reasoning = result_text[:len(parts[0])].strip()
                # Remove "Reasoning: " prefix if present in the reasoning part
                if reasoning.lower().startswith("reasoning:"):
                    reasoning = reasoning[10:].strip()
                
                verdict_part = parts[1].strip()
                if "pass" in verdict_part:
                    verdict = "Pass"
                else:
                    verdict = "Fail"
            elif "pass" in result_text.lower()[-10:]: # fallback if verdict: is missing but pass is at the end
                verdict = "Pass"
            
            return {
                "success": verdict == "Pass",
                "reasoning": reasoning,
                "verdict": verdict
            }

        except Exception as e:
            logger.error("Judge evaluation failed: %s", e)
            return {
                "success": False,
                "reasoning": f"Error during evaluation: {str(e)}",
                "verdict": "Error"
            }

    def _build_judge_prompt(self, question: str, ground_truth: str, answer: str) -> str:
        return f"""You are an impartial judge evaluating the correctness of an AI model's response.
Your task is to compare the [Model Answer] against the [Ground Truth] based on the [Question] provided.

[Question]
{question}

[Ground Truth]
{ground_truth}

[Model Answer]
{answer}

CRITICAL INSTRUCTIONS:
1. Evaluate ONLY the final answer. Ignore the method, reasoning steps, or formatting unless they directly contradict the ground truth.
2. Provide a brief one-sentence reasoning for your decision.
3. Then, conclude with "Verdict: Pass" or "Verdict: Fail".
4. The output must follow this format:
Reasoning: <your_reasoning>
Verdict: <Pass/Fail>
"""

class HumanEvaluator:
    """
    Manages blind human evaluation for HTML/CSS/JS implementations.
    Saves implementations with anonymized filenames and provides evaluation interface.
    """
    def __init__(self) -> None:
        self.base_dir = "manual_run_codes"
        self.session_dir: str | None = None
        self.manifest: Dict[str, Any] = {
            "implementations": [],  # List of {id, model_name, question_code, run_index, filename, score}
            "shuffled_order": [],   # Randomized order of implementation IDs for blind evaluation
            "scores_collected": False
        }
        self._impl_counter = 0

    def start_session(self, timestamp: str) -> str:
        """
        Creates a new session directory for this benchmark run.
        Returns the session directory path.
        """
        self.session_dir = os.path.join(
            self.base_dir, 
            f"benchmark_implementation_results_{timestamp}"
        )
        os.makedirs(self.session_dir, exist_ok=True)
        self._impl_counter = 0
        self.manifest = {
            "implementations": [],
            "shuffled_order": [],
            "scores_collected": False
        }
        logger.info("Human evaluation session started: %s", self.session_dir)
        return self.session_dir

    def save_implementation(
        self,
        model_name: str,
        question_code: str,
        run_index: int,
        html_content: str,
        max_points: int = 1
    ) -> str:
        """
        Saves an implementation with an anonymized filename.
        For Leetcode questions, saves as .txt; for others, saves as .html.
        Returns the implementation ID.
        """
        if not self.session_dir:
            raise ValueError("Session not started. Call start_session() first.")

        self._impl_counter += 1
        impl_id = f"impl_{self._impl_counter:03d}"

        # Determine file extension based on question type
        is_leetcode = "Leetcode" in question_code
        extension = ".txt" if is_leetcode else ".html"
        filename = f"{impl_id}{extension}"
        filepath = os.path.join(self.session_dir, filename)

        with open(filepath, "w") as f:
            f.write(html_content)

        self.manifest["implementations"].append({
            "id": impl_id,
            "model_name": model_name,
            "question_code": question_code,
            "run_index": run_index,
            "filename": filename,
            "score": None,
            "is_leetcode": is_leetcode,
            "max_points": max_points
        })

        logger.info("Saved implementation %s for %s (run %d)", impl_id, model_name, run_index)
        return impl_id

    def finalize_session(self) -> None:
        """
        Finalizes the session by shuffling implementations and saving the manifest.
        """
        import random
        
        if not self.session_dir:
            return
        
        # Create shuffled order
        impl_ids = [impl["id"] for impl in self.manifest["implementations"]]
        random.shuffle(impl_ids)
        self.manifest["shuffled_order"] = impl_ids
        
        # Save manifest
        manifest_path = os.path.join(self.session_dir, "manifest.json")
        with open(manifest_path, "w") as f:
            import json
            json.dump(self.manifest, f, indent=2)
        
        logger.info("Session finalized. %d implementations ready for evaluation.", len(impl_ids))

    def get_session_dir(self) -> str | None:
        """Returns the current session directory path."""
        return self.session_dir

    def update_results_paths(self, results_path: str, advanced_results_path: str) -> None:
        """
        Updates the manifest with the paths to results files.
        This allows integrate_scores to know exactly which files to update.
        """
        import json
        
        if not self.session_dir:
            return
        
        manifest_path = os.path.join(self.session_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            return
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        manifest["results_file"] = results_path
        manifest["advanced_results_file"] = advanced_results_path
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def store_html_generation_data(
        self,
        models: list[str],
        question_codes: list[str],
        all_results: dict,
        questions_data: dict,
        timestamp: str
    ) -> None:
        """
        Stores data required for HTML performance table generation.
        This allows integrate_scores to generate the table after scoring is complete.
        """
        import json
        
        if not self.session_dir:
            return
        
        manifest_path = os.path.join(self.session_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            return
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        manifest["html_generation_data"] = {
            "models": models,
            "question_codes": question_codes,
            "all_results": all_results,
            "questions_data": questions_data,
            "timestamp": timestamp
        }
        
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

    def load_scores(self) -> Dict[str, float]:
        """
        Loads scores from the manifest after human evaluation.
        Returns a dict mapping (model_name, question_code) to average score.
        """
        import json
        
        if not self.session_dir:
            return {}
        
        manifest_path = os.path.join(self.session_dir, "manifest.json")
        if not os.path.exists(manifest_path):
            return {}
        
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        # Group scores by (model_name, question_code)
        scores_by_key: Dict[tuple, list] = {}
        for impl in manifest.get("implementations", []):
            key = (impl["model_name"], impl["question_code"])
            score = impl.get("score")
            if score is not None:
                if key not in scores_by_key:
                    scores_by_key[key] = []
                scores_by_key[key].append(score)
        
        # Calculate averages
        averages = {}
        for key, scores in scores_by_key.items():
            if scores:
                averages[key] = sum(scores) / len(scores)
        
        return averages

    def evaluate(self, question: str, answer: str) -> bool:
        """
        Legacy method for compatibility. Returns False (pending human evaluation).
        """
        return False

class VerifierEvaluator:
    """
    Evaluator that uses hardcoded logic to check if the answer adheres to specific rules.
    Used when Ground Truth is set to 'VALIDITY CHECK'.
    """
    def __init__(self) -> None:
        """Initialize the VerifierEvaluator."""
        self.verifiers_dir = "verifier_scripts"
    
    def _get_verifier_module_name(self, question_code: str) -> str | None:
        """
        Maps question code to the corresponding verifier module name.
        E.g., 'A15' -> 'A15_longest_word_verifier'
        """
        import glob
        
        # Extract base code (e.g., 'A58' from 'A58-BattleShip')
        base_code = extract_base_question_code(question_code)
        
        # Look for a verifier file that starts with the base question code
        pattern = os.path.join(self.verifiers_dir, f"{base_code}_*.py")
        matches = glob.glob(pattern)
        
        if not matches:
            return None
        
        # Return the module name (filename without .py)
        verifier_path = matches[0]
        module_name = os.path.splitext(os.path.basename(verifier_path))[0]
        return module_name
    
    def evaluate(self, question_code: str, question_text: str, answer: str) -> dict[str, Any]:
        """
        Evaluates the answer based on hardcoded rules for the given question code.
        Dynamically imports the appropriate verifier script and invokes it.
        """
        logger.info("[*] VerifierEvaluator: Evaluating question %s", question_code)
        
        # Find the appropriate verifier module
        module_name = self._get_verifier_module_name(question_code)
        
        if not module_name:
            logger.warning("No verifier script found for question code: %s", question_code)
            return {
                "success": False,
                "reasoning": f"No verifier script implemented for question {question_code}",
                "verdict": "Error"
            }
        
        try:
            # Dynamically import the verifier module
            import importlib
            verifier_module = importlib.import_module(f"{self.verifiers_dir}.{module_name}")
            
            # Call the check_validity function
            if not hasattr(verifier_module, "check_validity"):
                logger.error("Verifier module %s does not have check_validity function", module_name)
                return {
                    "success": False,
                    "reasoning": "Invalid verifier module: missing check_validity function",
                    "verdict": "Error"
                }
            
            is_valid, failure_reason = verifier_module.check_validity(answer)
            
            if is_valid:
                return {
                    "success": True,
                    "reasoning": failure_reason,  # Contains detailed score info from checker
                    "verdict": "Pass"
                }
            else:
                return {
                    "success": False,
                    "reasoning": failure_reason,
                    "verdict": "Fail"
                }
        
        except Exception as e:
            logger.error("Error executing verifier script for %s: %s", question_code, e)
            return {
                "success": False,
                "reasoning": f"Error during verification: {str(e)}",
                "verdict": "Error"
            }
