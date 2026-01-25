"""
Benchmarking script to evaluate multiple models on specific questions.
Parses question files, calls models via ModelAPI, and generates reports.
"""

import os
import sys
import argparse
import asyncio
from dotenv import load_dotenv

# Local Utils
from utils.utils import setup_logging, clear_history
from utils.benchmark_runner import BenchmarkRunner
from utils.question_renamer import ensure_question_naming_convention

# Load environment variables
load_dotenv()

logger = setup_logging(__name__)

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
    
    # Check for "!!!" trigger in questions.txt
    questions_file = "config/questions.txt"
    run_only_renamed = False
    
    if os.path.exists(questions_file):
        with open(questions_file, "r") as f:
            content = f.read().strip()
            if content == "!!!":
                run_only_renamed = True

    questions_to_run = None

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
        questions_to_run = new_codes

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

    # Initialize and run benchmark
    try:
        runner = BenchmarkRunner()
        asyncio.run(runner.run(specific_questions=questions_to_run))
    except Exception as e:
        logger.error("Benchmark failed: %s", e)
        sys.exit(1)

    logger.info("Benchmark run complete.")
