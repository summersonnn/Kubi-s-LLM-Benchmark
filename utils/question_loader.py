"""
Question discovery and loading utilities for the Kubis-Benchmark framework.
Handles question path resolution, metadata extraction, and batch loading.
"""

import os
import glob
import logging
from typing import List, Dict, Tuple, Any, Optional

from utils.utils import parse_question_file


logger = logging.getLogger(__name__)

# Determine project root dynamically to support running from utils/ or other subdirs
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def resolve_question_path(question_code: str) -> str | None:
    """
    Finds the file path for a given question code (e.g., 'A2') starting with that code.
    Ensures that 'A2' does not match 'A22' by checking the character following the code.
    """
    # Search recursively in the questions directory
    # Search recursively in the questions directory
    questions_dir = os.path.join(PROJECT_ROOT, "questions")
    search_pattern = f"{questions_dir}/**/{question_code}*.txt"
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

def get_question_metadata(question_path: str) -> Tuple[str, str]:
    """
    Groups questions into categories and subcategories based on folder structure.
    Infer Category from the top-level folder inside 'questions/'.
    Infer Subcategory from the specific subfolder, or 'General' if directly in Category folder.
    """
    try:
        norm_path = os.path.normpath(question_path)
        parts = norm_path.split(os.sep)
        
        # Find index of 'questions'
        try:
            q_index = parts.index("questions")
        except ValueError:
            return "Other", "Other"
            
        # Structure after 'questions': [Category, Subcategory?, Filename]
        rel_parts = parts[q_index + 1:]
        
        if not rel_parts:
            return "Other", "Other"
            
        category = rel_parts[0]
        
        # If we have [Category, Subcategory, Filename] (len >= 3)
        if len(rel_parts) >= 3:
            subcategory = rel_parts[1]
        else:
            # [Category, Filename]
            subcategory = "General"
            
        return category, subcategory
        
    except Exception as e:
        logger.error(f"Error inferring metadata for {question_path}: {e}")
        return "Other", "Other"

def discover_question_codes(specific_questions: Optional[List[str]] = None) -> List[str]:
    """
    Determines the list of question codes to run.
    If specific_questions is provided, uses that.
    Otherwise, reads from config/questions.txt.
    Handles 'ALL' keyword and subfolder expansions.
    """
    if specific_questions:
        logger.info(f"Running benchmark on {len(specific_questions)} specific questions provided by system.")
        question_codes = specific_questions
    else:
        questions_file = os.path.join(PROJECT_ROOT, "config", "questions.txt")
        if not os.path.exists(questions_file):
            logger.error("questions.txt not found.")
            return []

        with open(questions_file, "r") as f:
            raw_lines = [line.strip() for line in f if line.strip()]

        if not raw_lines:
            logger.error("No question codes found in questions.txt")
            return []

        # Expand subfolder entries (lines ending with "/") to all questions in that subfolder
        question_codes = []
        for line in raw_lines:
            if line.endswith("/"):
                # Treat as subfolder path relative to questions directory
                questions_dir = os.path.join(PROJECT_ROOT, "questions")
                subfolder_path = os.path.join(questions_dir, line.rstrip("/"))
                if os.path.isdir(subfolder_path):
                    subfolder_files = sorted(glob.glob(os.path.join(subfolder_path, "**", "*.txt"), recursive=True))
                    for fpath in subfolder_files:
                        fname = os.path.basename(fpath)
                        # Exclude known non-question files
                        if fname in ["readme.txt", "README.txt"]:
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
        return []

    # Handle "ALL" keyword
    if any(line.upper() == "ALL" for line in question_codes):
        logger.info("Found 'ALL' in questions.txt. Loading all available questions...")
        questions_dir = os.path.join(PROJECT_ROOT, "questions")
        all_files = glob.glob(f"{questions_dir}/**/*.txt", recursive=True)
        question_codes = []
        for fpath in all_files:
            fname = os.path.basename(fpath)
            # Filter out exclude files
            if fname in ["readme.txt", "README.txt"]:
                continue
            
            # Use filename without extension as the code
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
            return []

    return question_codes

def load_questions_data(question_codes: List[str]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """
    Loads question data, content, and metadata for the provided codes.
    Returns:
        questions_data: Dict containing parsed details for each question.
        valid_question_codes: List of codes that were successfully loaded.
    """
    questions_data: Dict[str, Dict[str, Any]] = {}
    valid_question_codes = []

    logger.info("Loading questions...")

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

        # Determine evaluation type from filename markers
        question_basename = os.path.basename(question_path)
        is_verifier_eval = "-V-" in question_basename

        # Determine Evaluation Type
        if is_verifier_eval:
            eval_type = "eval by verifier scripts"
        elif "-J-" in question_basename:
            eval_type = "eval by Judge LLM"
        else:
            # Fallback for questions not yet renamed or special cases
            eval_type = "eval by Judge LLM"

        # Determine Category and Subcategory
        category, subcategory = get_question_metadata(question_path)

        questions_data[code] = {
            "question": question,
            "ground_truth": ground_truth if ground_truth else "N/A",
            "points": points,
            "eval_type": eval_type,
            "is_verifier_eval": is_verifier_eval,
            "category": category,
            "subcategory": subcategory
        }
        valid_question_codes.append(code)

    return questions_data, valid_question_codes
