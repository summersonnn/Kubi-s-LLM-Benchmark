"""
Removes "RUN RESULT" lines from benchmark result logs.
"""

from pathlib import Path

def remove_run_result_lines(filepath: Path) -> None:
    if not filepath.exists():
        return
    
    print(f"Processing: {filepath}")
    lines = filepath.read_text(encoding="utf-8").splitlines()
    
    # Filter out lines containing "RUN RESULT"
    filtered_lines = [line for line in lines if "RUN RESULT" not in line]
    
    # Compress multiple consecutive empty lines into one
    import re
    content = "\n".join(filtered_lines)
    new_content = re.sub(r'\n{3,}', '\n\n', content)
    
    if content != new_content or len(lines) != len(filtered_lines):
        filepath.write_text(new_content.strip() + "\n", encoding="utf-8")
        print(f"  Cleaned up file: {filepath.name}")

def main() -> None:
    project_root = Path(__file__).parent.parent
    
    files_to_clean = [
        project_root / "results_advanced" / "benchmark_results_advanced_20260130_172227_final.txt",
        project_root / "results" / "benchmark_results_20260130_172227_final.txt",
    ]
    
    for filepath in files_to_clean:
        remove_run_result_lines(filepath)

if __name__ == "__main__":
    main()
