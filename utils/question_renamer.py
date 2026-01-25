"""
Question file naming convention enforcer for the Kubis-Benchmark framework.
Scans questions directory and renames non-conforming files to A{number}-... format.
"""

import os
import re
from typing import Set, List

def ensure_question_naming_convention(questions_dir: str = "questions") -> List[str]:
    """
    Scans the questions directory for files that do not conform to the 'A{number}-...' naming convention.
    Renames non-conforming files by assigning them the lowest available question ID.
    
    Args:
        questions_dir: Path to the questions directory (relative to project root).
        
    Returns:
        List[str]: A list of the new question codes (e.g., ['A61', 'A62']) required for the renamed files.
    """
    print(f"Checking question naming conventions in '{questions_dir}'...")
    
    renamed_codes: List[str] = []

    if not os.path.exists(questions_dir):
        print(f"Warning: Directory '{questions_dir}' not found. Skipping naming check.")
        return []

    # Files to ignore
    IGNORED_FILES = {
        "html_css_js_questions_prefix.txt",
        "readme.txt",
        "README.txt",
        "README.md",
        ".DS_Store"
    }

    # 1. Identify all currently used IDs and collect non-conforming files
    used_ids: Set[int] = set()
    non_conforming_files = []

    # Walk through all files
    for root, _, files in os.walk(questions_dir):
        for filename in files:
            if filename in IGNORED_FILES:
                continue
            
            # Check for pattern A{number}- or A{number}.
            match = re.match(r'^A(\d+)', filename)
            
            if match:
                used_ids.add(int(match.group(1)))
            else:
                non_conforming_files.append(os.path.join(root, filename))

    if not non_conforming_files:
        print("All question files conform to the naming convention.")
        return []

    print(f"Found {len(non_conforming_files)} non-conforming files. Renaming...")

    # 2. Rename non-conforming files
    # Sort to ensure deterministic order if multiple files need renaming
    non_conforming_files.sort()

    for old_path in non_conforming_files:
        dirname = os.path.dirname(old_path)
        basename = os.path.basename(old_path)
        
        # prohibited characters for safety
        safe_name = basename.replace(" ", "-")
        
        # Find lowest available ID
        next_id = 1
        while next_id in used_ids:
            next_id += 1
            
        new_filename = f"A{next_id}-{safe_name}"
        new_path = os.path.join(dirname, new_filename)
        
        try:
            os.rename(old_path, new_path)
            print(f"Renamed: {basename} -> {new_filename}")
            
            # Record the new code
            renamed_codes.append(f"A{next_id}")
            
            # Mark this ID as used so next file gets next ID
            used_ids.add(next_id)
            
        except OSError as e:
            print(f"Error renaming {old_path}: {e}")

    print("Renaming complete.")
    return renamed_codes
