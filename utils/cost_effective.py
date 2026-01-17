"""
Cost-effective caching utilities for reusing existing model implementations.
Scans the cost-effective folder for pre-generated HTML files to avoid redundant API calls.
"""

import os
import re
from typing import Any

from utils.utils import setup_logging

logger = setup_logging(__name__)

COST_EFFECTIVE_DIR = "cost-effective"


def extract_base_question_code(question_code: str) -> str:
    """
    Extracts the base question code (e.g., 'A17' or 'A23.1') from a full question code
    that may include descriptive suffixes (e.g., 'A17-line-liars' or 'A23.1-some-name').
    
    Args:
        question_code: Full question code potentially with suffix
        
    Returns:
        Base question code (A<number> or A<number>.<subnumber>), or original if no match.
    """
    match = re.match(r"^(A\d+(?:\.\d+)?)", question_code, re.IGNORECASE)
    if match:
        return match.group(1)
    return question_code


def find_model_folder(model_name: str) -> str | None:
    """
    Finds a matching subfolder in cost-effective/ for the given model name.
    Matches if the model_name contains the subfolder name as a substring.
    
    Args:
        model_name: Full model identifier (e.g., 'anthropic/claude-opus-4.5@preset/fp8')
        
    Returns:
        Full path to the matching folder, or None if no match found.
    """
    if not os.path.isdir(COST_EFFECTIVE_DIR):
        return None
    
    for folder_name in os.listdir(COST_EFFECTIVE_DIR):
        folder_path = os.path.join(COST_EFFECTIVE_DIR, folder_name)
        if os.path.isdir(folder_path):
            # Check if model_name contains the folder_name as substring
            if folder_name.lower() in model_name.lower():
                logger.info(
                    "Found cost-effective cache for model '%s' in folder '%s'",
                    model_name, folder_name
                )
                return folder_path
    
    return None



def get_existing_implementations(
    model_folder: str,
    question_code: str,
    num_runs_needed: int
) -> list[dict[str, Any]]:
    """
    Retrieves existing implementations for a question from the cache folder.
    Files must follow the pattern: {question_code}-{run_index}.{ext} (e.g., A43-1.html, A68-1.txt)
    Supported extensions: .html, .txt, .md
    
    Args:
        model_folder: Path to the model's cache folder
        question_code: Question identifier (e.g., 'A43')
        num_runs_needed: Maximum number of implementations to return
        
    Returns:
        List of dicts with 'run_index' and 'content', ordered by run_index.
        Returns at most num_runs_needed items starting from run index 1.
    """
    if not os.path.isdir(model_folder):
        return []
    
    # Extract base question code (e.g., 'A17' from 'A17-line-liars')
    base_code = extract_base_question_code(question_code)
    
    # Pattern: QuestionCode-RunIndex.(html|txt|md) (1-indexed)
    pattern = re.compile(rf"^{re.escape(base_code)}-(\d+)\.(html|txt|md)$", re.IGNORECASE)
    
    implementations = []
    
    for filename in os.listdir(model_folder):
        match = pattern.match(filename)
        if match:
            run_index = int(match.group(1))
            filepath = os.path.join(model_folder, filename)
            
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    content = f.read()
                
                implementations.append({
                    "run_index": run_index,
                    "content": content,
                    "source_file": filepath
                })
            except OSError as e:
                logger.warning(
                    "Failed to read cached implementation %s: %s", filepath, e
                )
    
    # Sort by run_index and take first num_runs_needed
    implementations.sort(key=lambda x: x["run_index"])
    result = implementations[:num_runs_needed]
    
    if result:
        logger.info(
            "Loaded %d cached implementations for %s from %s",
            len(result), question_code, model_folder
        )
    
    return result
