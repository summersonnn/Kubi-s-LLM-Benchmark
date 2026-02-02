"""
Evaluation framework for benchmarking models.
Includes JudgeLLMEvaluator for automated assessment and VerifierEvaluator for rule-based checking.
"""

import json
import os
import re
import asyncio
import random
from typing import Any, Dict, Protocol

from utils.utils import setup_logging, extract_base_question_code
from utils.model_api import ModelAPI

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

    async def evaluate(self, question: str, ground_truth: str, answer: str, points: int = 1) -> dict[str, Any]:
        """
        Uses the judge LLM to evaluate the answer against the ground truth.
        Returns a dictionary with 'success', 'reasoning', and 'verdict'.
        """
        prompt = self._build_judge_prompt(question, ground_truth, answer)
        
        try:
            response = await self.api.call(
                prompt=prompt,
                model_name=self.judge_model_name,
                max_tokens=1024,
                temperature=0.0,
                reasoning=False  # Reasoning not needed for judge extraction
            )
            result_text = (response.choices[0].message.content or "").strip()

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



class VerifierEvaluator:
    """
    Evaluator that uses hardcoded logic to check if the answer adheres to specific rules.
    Used when Ground Truth is set to 'VERIFIER'.
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
        
        # Extract base code
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
    
    async def evaluate(self, question_code: str, question_text: str, answer: str) -> dict[str, Any]:
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
            
            # Call the verify_answer function
            if not hasattr(verifier_module, "verify_answer"):
                logger.error("Verifier module %s does not have verify_answer function", module_name)
                return {
                    "success": False,
                    "reasoning": "Invalid verifier module: missing verify_answer function",
                    "verdict": "Error"
                }
            
            # Run CPU-bound verify_answer in executor
            loop = asyncio.get_running_loop()
            is_valid, failure_reason = await loop.run_in_executor(
                None, 
                verifier_module.verify_answer, 
                answer
            )
            
            # Extract score for verdict if present
            score_match = re.search(r'SCORE:([\d.]+/[\d.]+)', failure_reason)
            verdict = score_match.group(1) if score_match else ("Pass" if is_valid else "Fail")
            
            return {
                "success": is_valid,
                "reasoning": failure_reason,  # Contains detailed score info from checker
                "verdict": verdict
            }
        
        except Exception as e:
            logger.error("Error executing verifier script for %s: %s", question_code, e)
            return {
                "success": False,
                "reasoning": f"Error during verification: {str(e)}",
                "verdict": "Error"
            }
