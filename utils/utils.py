"""
Shared utilities for the Kubis-Benchmark project.
Provides standardized logging configuration.
"""

import logging
import sys
import re
import subprocess
import os
import shutil
from typing import Tuple

def setup_logging(name: str | None = None) -> logging.Logger:
    """
    Sets up a standardized logger with a stream handler and formatting.
    
    Args:
        name: The name of the logger.
        
    Returns:
        A configured logging.Logger instance.
    """
    logger = logging.getLogger(name or "kubis-benchmark")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

def parse_question_file(content: str) -> Tuple[str, str, int]:
    """
    Parses the question file content to separate the question, ground truth, and points.
    Case-insensitive search for 'Ground Truth:' and 'Point:' markers.
    
    Args:
        content: The raw text content of the question file.
        
    Returns:
        A tuple of (question_text, ground_truth, points).
        Points defaults to 1 if not specified.
    """
    marker_pattern = re.compile(r"^----+\s*$", re.MULTILINE)
    match = marker_pattern.search(content)
    
    ground_truth = ""
    points = 1
    
    if match:
        # Split by the separator
        question = content[:match.start()].strip()
        metadata_section = content[match.end():].strip()
        
        # Parse metadata - extract Ground Truth
        # Find "Ground Truth:"
        gt_start = re.search(r"Ground\s+Truth\s*:\s*", metadata_section, re.IGNORECASE)
        if gt_start:
            # Find the next key or end of string
            # We assume "Point:" is another key.
            # Let's search for "Point:" after GT
            gt_content_start = gt_start.end()
            point_in_meta = re.search(r"\n\s*Point\s*:", metadata_section[gt_content_start:], re.IGNORECASE)
            
            if point_in_meta:
                 ground_truth = metadata_section[gt_content_start : gt_content_start + point_in_meta.start()].strip()
            else:
                 ground_truth = metadata_section[gt_content_start:].strip()
        
        # Parse Points
        point_match = re.search(r"Point\s*:\s*(\d+)", metadata_section, re.IGNORECASE)
        if point_match:
            try:
                points = int(point_match.group(1))
            except ValueError:
                points = 1
                
    else:
        # Fallback to old logic or return as is (assuming whole file is question if no separator?)
        # For backward compatibility during migration, we can keep the old logic or Assume failure.
        # The user said "go over all questions... set a clear border". 
        # If I migrated everything, I should expect the separator.
        # But for robustness, I'll keep the old logic as fallback or just log warning?
        # Let's keep the old logic as fallback for now, just in case.
        
        marker_pattern_old = re.compile(r"\n\s*Ground\s+Truth\s*:\s*", re.IGNORECASE)
        match_old = marker_pattern_old.search(content)
        
        if match_old:
            start_idx = match_old.start()
            end_idx = match_old.end()
            question = content[:start_idx].strip()
            remaining_content = content[end_idx:].strip()
            
            point_pattern = re.compile(r"\n\s*Point\s*:\s*(\d+)", re.IGNORECASE)
            point_match = point_pattern.search(remaining_content)
            
            if point_match:
                ground_truth = remaining_content[:point_match.start()].strip()
                points_str = point_match.group(1)
                try:
                    points = int(points_str)
                except ValueError:
                    points = 1
            else:
                ground_truth = remaining_content
                points = 1
        else:
            # Check for Point only
            point_pattern = re.compile(r"\n\s*Point\s*:\s*(\d+)", re.IGNORECASE)
            point_match = point_pattern.search(content)
            
            if point_match:
                question = content[:point_match.start()].strip()
                points_str = point_match.group(1)
                try:
                    points = int(points_str)
                except ValueError:
                    points = 1
            else:
                question = content.strip()
                points = 1
        
    return question, ground_truth, points


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


def kill_process_on_port(port: int) -> None:
    """
    Kills any process listening on the specified port.
    Uses 'fuser -k' command.
    """
    try:
        # Check if fuser is available
        subprocess.run(["which", "fuser"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Kill process on port
        subprocess.run(
            ["fuser", "-k", f"{port}/tcp"], 
            check=False,  # Don't raise error if no process found (exit code 1)
            stdout=subprocess.DEVNULL, 
            stderr=subprocess.DEVNULL
        )
        # Give it a moment to release the port
        import time
        time.sleep(0.5)
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        # fuser might not be installed or failed, try lsof as backup
        try:
            # lsof -t -i:8765 returns pid
            result = subprocess.run(
                ["lsof", "-t", f"-i:{port}"],
                capture_output=True,
                text=True
            )
            if result.stdout.strip():
                pid = result.stdout.strip()
                subprocess.run(["kill", "-9", pid], check=False)
                import time
                time.sleep(0.5)
        except Exception:
            pass  # Best effort, suppress errors


def clear_history() -> None:
    """
    Clears all benchmark history by removing contents of results/,
    results_advanced/, and manual_run_codes/ directories.
    """
    logger = setup_logging(__name__)

    directories_to_clear = [
        "results",
        "results_advanced",
        "manual_run_codes"
    ]

    for dir_name in directories_to_clear:
        if os.path.exists(dir_name):
            try:
                shutil.rmtree(dir_name)
                os.makedirs(dir_name)
                logger.info("Cleared directory: %s", dir_name)
            except Exception as e:
                logger.error("Failed to clear directory %s: %s", dir_name, e)
        else:
            os.makedirs(dir_name)
            logger.info("Created directory: %s", dir_name)
