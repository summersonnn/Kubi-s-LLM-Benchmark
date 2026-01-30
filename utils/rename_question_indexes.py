"""
Renames question indexes from non-continuous to continuous numbering.

This script applies the following mapping to make question indexes continuous:
- A1-A6: unchanged
- A27-A47: mapped to A7-A25
- A48.1-A48.17: mapped to A26.1-A26.17
- A49-A62: mapped to A27-A37

Final result: A1-A37 with A26.1-A26.17 for visual puzzles.
"""

import re
from pathlib import Path


# Mapping from old index to new index (explicit for clarity and correctness)
INDEX_MAPPING: dict[str, str] = {
    # Keep A1-A6 unchanged
    "A1": "A1",
    "A2": "A2",
    "A3": "A3",
    "A4": "A4",
    "A5": "A5",
    "A6": "A6",
    # Map A27-A30, A33 to A7-A11
    "A27": "A7",
    "A28": "A8",
    "A29": "A9",
    "A30": "A10",
    "A33": "A11",
    # Map A34-A47 to A12-A25
    "A34": "A12",
    "A35": "A13",
    "A36": "A14",
    "A37": "A15",
    "A38": "A16",
    "A39": "A17",
    "A40": "A18",
    "A41": "A19",
    "A42": "A20",
    "A43": "A21",
    "A44": "A22",
    "A45": "A23",
    "A46": "A24",
    "A47": "A25",
    # Map A48.X to A26.X (visual puzzles)
    "A48.1": "A26.1",
    "A48.2": "A26.2",
    "A48.3": "A26.3",
    "A48.4": "A26.4",
    "A48.5": "A26.5",
    "A48.6": "A26.6",
    "A48.7": "A26.7",
    "A48.8": "A26.8",
    "A48.9": "A26.9",
    "A48.10": "A26.10",
    "A48.11": "A26.11",
    "A48.12": "A26.12",
    "A48.13": "A26.13",
    "A48.14": "A26.14",
    "A48.15": "A26.15",
    "A48.16": "A26.16",
    "A48.17": "A26.17",
    # Map A49-A50, A52-A58, A61-A62
    "A49": "A27",
    "A50": "A28",
    "A52": "A29",
    "A53": "A30",
    "A54": "A31",
    "A55": "A32",
    "A56": "A33",
    "A57": "A34",
    "A58": "A35",
    "A61": "A36",
    "A62": "A37",
}


def replace_indexes(content: str) -> str:
    """
    Replace old question indexes with new ones.

    Uses word boundaries to avoid partial matches. Processes longer patterns
    first to prevent A48.1 being partially matched by A48.
    """
    # Sort keys by length descending so longer patterns are replaced first
    sorted_keys = sorted(INDEX_MAPPING.keys(), key=len, reverse=True)

    for old_idx in sorted_keys:
        new_idx = INDEX_MAPPING[old_idx]
        if old_idx == new_idx:
            continue  # Skip unchanged indexes

        # Use word boundary regex to avoid partial matches
        # Match A48.10 but not A48.100 (use negative lookahead for digits)
        pattern = rf"\b{re.escape(old_idx)}(?![0-9])"
        content = re.sub(pattern, new_idx, content)

    return content


def process_file(filepath: Path) -> None:
    """Read, transform, and overwrite a file with new indexes."""
    print(f"Processing: {filepath}")
    content = filepath.read_text(encoding="utf-8")
    new_content = replace_indexes(content)
    filepath.write_text(new_content, encoding="utf-8")
    print(f"  Done: {filepath.name}")


def main() -> None:
    project_root = Path(__file__).parent.parent

    # Files to process
    files_to_process = [
        project_root / "results_advanced" / "benchmark_results_advanced_20260130_172227_final.txt",
        project_root / "results" / "benchmark_results_20260130_172227_final.txt",
        project_root / "results" / "performance_table_20260130_172227_final.html",
    ]

    for filepath in files_to_process:
        if filepath.exists():
            process_file(filepath)
        else:
            print(f"Skipping (not found): {filepath}")

    print("\nRenaming complete!")


if __name__ == "__main__":
    main()
