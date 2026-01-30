"""
Reorders questions in log files and HTML to be sorted from A1 to A37.

Handles:
1. The "Questions in the run" list in log files
2. The table rows in HTML files
"""

import re
from pathlib import Path


def parse_question_index(q_id: str) -> tuple[int, int]:
    """
    Parse a question ID into a sortable tuple.

    A26.10 -> (26, 10)
    A5 -> (5, 0)
    """
    match = re.match(r"A(\d+)(?:\.(\d+))?", q_id)
    if match:
        main = int(match.group(1))
        sub = int(match.group(2)) if match.group(2) else 0
        return (main, sub)
    return (999, 999)  # Fallback for unexpected formats


def reorder_questions_list(content: str) -> str:
    """Reorder the 'Questions in the run' list in log files."""
    # Pattern to match the questions list section
    pattern = r"(Questions in the run \(\d+\):)\n((?:- A\d+(?:\.\d+)? \([^)]+\)\n)+)"

    def reorder_match(match: re.Match) -> str:
        header = match.group(1)
        lines = match.group(2).strip().split("\n")

        # Parse and sort
        def extract_id(line: str) -> str:
            m = re.match(r"- (A\d+(?:\.\d+)?)", line)
            return m.group(1) if m else ""

        sorted_lines = sorted(lines, key=lambda x: parse_question_index(extract_id(x)))
        return header + "\n" + "\n".join(sorted_lines) + "\n"

    return re.sub(pattern, reorder_match, content)


def reorder_html_rows(content: str) -> str:
    """Reorder table rows in HTML by question index."""
    # Find all question rows (tbody content between <tbody> and </tbody>)
    tbody_pattern = r"(<tbody>)(.*?)(</tbody>)"

    def reorder_tbody(match: re.Match) -> str:
        start_tag = match.group(1)
        tbody_content = match.group(2)
        end_tag = match.group(3)

        # Extract all <tr> elements
        row_pattern = r"(<tr[^>]*>.*?</tr>)"
        rows = re.findall(row_pattern, tbody_content, re.DOTALL)

        def extract_question_id(row: str) -> str:
            # Match q-col cell content like <td class='q-col'>A26.10</td>
            m = re.search(r"<td class='q-col'>(A\d+(?:\.\d+)?)</td>", row)
            return m.group(1) if m else "A999"

        sorted_rows = sorted(rows, key=lambda r: parse_question_index(extract_question_id(r)))

        # Preserve indentation
        formatted_rows = "\n                    ".join(sorted_rows)
        return f"{start_tag}\n                    {formatted_rows}\n                {end_tag}"

    return re.sub(tbody_pattern, reorder_tbody, content, flags=re.DOTALL)


def process_log_file(filepath: Path) -> None:
    """Process a log file to reorder questions list."""
    print(f"Processing log: {filepath}")
    content = filepath.read_text(encoding="utf-8")
    new_content = reorder_questions_list(content)
    filepath.write_text(new_content, encoding="utf-8")
    print(f"  Done: {filepath.name}")


def process_html_file(filepath: Path) -> None:
    """Process an HTML file to reorder table rows."""
    print(f"Processing HTML: {filepath}")
    content = filepath.read_text(encoding="utf-8")
    new_content = reorder_html_rows(content)
    filepath.write_text(new_content, encoding="utf-8")
    print(f"  Done: {filepath.name}")


def main() -> None:
    project_root = Path(__file__).parent.parent

    # Log files to process
    log_files = [
        project_root / "results_advanced" / "benchmark_results_advanced_20260130_172227_final.txt",
        project_root / "results" / "benchmark_results_20260130_172227_final.txt",
    ]

    # HTML files to process
    html_files = [
        project_root / "results" / "performance_table_20260130_172227_final.html",
    ]

    for filepath in log_files:
        if filepath.exists():
            process_log_file(filepath)
        else:
            print(f"Skipping (not found): {filepath}")

    for filepath in html_files:
        if filepath.exists():
            process_html_file(filepath)
        else:
            print(f"Skipping (not found): {filepath}")

    print("\nReordering complete!")


if __name__ == "__main__":
    main()
