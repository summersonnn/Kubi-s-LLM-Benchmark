"""
Reorders question detailed logs in both advanced and basic result files.

1. In advanced report: Reorders QUESTION blocks from A1 to A37.
2. In basic report: Reorders Question entries within each model section from A1 to A37.
"""

import re
from pathlib import Path


def parse_question_index(q_id: str) -> tuple[int, int | float]:
    """Parse A26.10 -> (26, 10), A5 -> (5, 0)."""
    match = re.match(r"A(\d+)(?:\.(\d+))?", q_id)
    if match:
        main = int(match.group(1))
        sub = int(match.group(2)) if match.group(2) else 0
        return (main, sub)
    return (999, 999)


def reorder_advanced_log(content: str) -> str:
    """Reorder the detailed QUESTION blocks in the advanced log file."""
    # Split the file into three parts: Intro, Questions, Footer
    # First question starts at ####################
    #                               QUESTION 1: A
    start_marker = r"####################################################################################################\nQUESTION \d+: A"
    match_start = re.search(start_marker, content)
    if not match_start:
        return content

    intro_end_idx = match_start.start()
    intro = content[:intro_end_idx]

    # Footer starts at MODEL RANKINGS
    footer_marker = r"####################################################################################################\nMODEL RANKINGS"
    match_footer = re.search(footer_marker, content)
    if not match_footer:
        # Maybe it's just questions until the end
        questions_part = content[intro_end_idx:]
        footer = ""
    else:
        questions_part = content[intro_end_idx : match_footer.start()]
        footer = content[match_footer.start() :]

    # Split questions_part into blocks
    # Each block starts with the #################### marker
    block_marker = r"(####################################################################################################\nQUESTION \d+: (A\d+(?:\.\d+)?)\n####################################################################################################)"
    
    # We use finditer to find blocks and their content between them
    blocks = []
    matches = list(re.finditer(block_marker, questions_part))
    
    for i, match in enumerate(matches):
        header_full = match.group(1)
        q_id = match.group(2)
        start_idx = match.start()
        end_idx = matches[i+1].start() if i + 1 < len(matches) else len(questions_part)
        
        block_content = questions_part[match.end() : end_idx]
        blocks.append({
            'id': q_id,
            'header': header_full,
            'content': block_content
        })

    # Sort blocks
    blocks.sort(key=lambda x: parse_question_index(x['id']))

    # Reconstruct with new Question numbers
    reordered_questions = []
    for i, block in enumerate(blocks):
        new_header = f"####################################################################################################\nQUESTION {i+1}: {block['id']}\n####################################################################################################"
        reordered_questions.append(new_header + block['content'])

    return intro + "".join(reordered_questions) + footer


def reorder_basic_log(content: str) -> str:
    """Reorder Question sections within each model section in the basic log file."""
    # Model sections start with NAME RESULTS:
    model_section_marker = r"([A-Z0-9_/.-]+ RESULTS:)\n(--------------------------------------------------------------------------------)"
    
    parts = []
    last_idx = 0
    matches = list(re.finditer(model_section_marker, content))
    
    if not matches:
        return content

    # Add the text before the first model section
    parts.append(content[:matches[0].start()])

    for i, match in enumerate(matches):
        header = match.group(0)
        start_idx = match.end()
        end_idx = matches[i+1].start() if i + 1 < len(matches) else len(content)
        
        section_content = content[start_idx:end_idx]
        
        # Split section_content into Question sub-blocks
        # Question 1 (A9):
        q_subblock_marker = r"(Question \d+ \((A\d+(?:\.\d+)?)\):)"
        sub_matches = list(re.finditer(q_subblock_marker, section_content))
        
        if sub_matches:
            sub_intro = section_content[:sub_matches[0].start()]
            sub_blocks = []
            for j, sub_match in enumerate(sub_matches):
                sub_header_full = sub_match.group(1)
                q_id = sub_match.group(2)
                sub_start_idx = sub_match.start()
                sub_end_idx = sub_matches[j+1].start() if j + 1 < len(sub_matches) else len(section_content)
                
                sub_block_content = section_content[sub_match.end() : sub_end_idx]
                sub_blocks.append({
                    'id': q_id,
                    'content': sub_block_content
                })
            
            # Sort sub-blocks
            sub_blocks.sort(key=lambda x: parse_question_index(x['id']))
            
            # Reconstruct sub-blocks with new indices
            reordered_sub = []
            for j, sub_block in enumerate(sub_blocks):
                new_sub_header = f"Question {j+1} ({sub_block['id']}):"
                reordered_sub.append(new_sub_header + sub_block['content'])
            
            parts.append(header + sub_intro + "".join(reordered_sub))
        else:
            parts.append(header + section_content)

    return "".join(parts)


def main() -> None:
    project_root = Path(__file__).parent.parent
    
    advanced_log = project_root / "results_advanced" / "benchmark_results_advanced_20260130_172227_final.txt"
    basic_log = project_root / "results" / "benchmark_results_20260130_172227_final.txt"

    if advanced_log.exists():
        print(f"Processing advanced log: {advanced_log.name}")
        content = advanced_log.read_text(encoding="utf-8")
        new_content = reorder_advanced_log(content)
        advanced_log.write_text(new_content, encoding="utf-8")
        print("  Done.")

    if basic_log.exists():
        print(f"Processing basic log: {basic_log.name}")
        content = basic_log.read_text(encoding="utf-8")
        new_content = reorder_basic_log(content)
        basic_log.write_text(new_content, encoding="utf-8")
        print("  Done.")

    print("\nDetailed log reordering complete!")


if __name__ == "__main__":
    main()
