"""
Script to calculate total tokens and costs from advanced benchmark result reports.
Parses detailed logs to aggregate model performance data and update rankings.
"""

import sys
import re
import os
from typing import Dict, List, Tuple

def parse_benchmark_results(file_path: str) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, float], float]:
    """
    Parses the result file to extract scores, tokens, and costs per model.
    Returns: (scores, tokens, costs, total_possible_points)
    """
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        sys.exit(1)

    scores: Dict[str, float] = {}
    tokens: Dict[str, int] = {}
    costs: Dict[str, float] = {}
    total_possible_points = 0.0

    current_model = None
    
    with open(file_path, "r") as f:
        content = f.read()

    # Find all question blocks to calculate total points
    # Format: POINTS: 1
    points_matches = re.findall(r"POINTS:\s*([\d.]+)", content)
    # This might double count if it's in multiple places, but we really want the points per question
    # Let's look for QUESTION blocks
    question_sections = re.split(r"#{100,}\s+QUESTION \d+:", content)
    
    # The first section is the header, skip it
    if len(question_sections) > 1:
        for section in question_sections[1:]:
            points_match = re.search(r"POINTS:\s*([\d.]+)", section)
            if points_match:
                total_possible_points += float(points_match.group(1))

    # Pattern for MODEL data blocks
    # MODEL: moonshotai/kimi-k2.5
    # SCORE: 1.00/1.00
    # TOKENS USED: 1182
    # COST INCURRED: $0.003534
    
    model_blocks = re.finditer(
        r"MODEL:\s*(.+?)\nSCORE:\s*([\d.]+)/[\d.]+\nTOKENS USED:\s*(\d+)\nCOST INCURRED:\s*\$([\d.]+)",
        content
    )

    for match in model_blocks:
        model_name = match.group(1).strip()
        score = float(match.group(2))
        token_count = int(match.group(3))
        cost_val = float(match.group(4))

        scores[model_name] = scores.get(model_name, 0.0) + score
        tokens[model_name] = tokens.get(model_name, 0) + token_count
        costs[model_name] = costs.get(model_name, 0.0) + cost_val

    return scores, tokens, costs, total_possible_points

def main():
    if len(sys.argv) < 2:
        print("Usage: python utils/calculate_costs.py <path_to_result_file>")
        return

    result_file = sys.argv[1]
    scores, tokens, costs, total_points = parse_benchmark_results(result_file)

    if not scores:
        print("No data found in the provided file.")
        return

    # Sort models by score (descending)
    ranked_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)

    print("#" * 100)
    print("MODEL RANKINGS (Recalculated Tokens & Costs)")
    print("#" * 100)
    print()

    for i, (model, score) in enumerate(ranked_models, 1):
        percentage = (score / total_points * 100) if total_points > 0 else 0
        token_sum = tokens.get(model, 0)
        cost_sum = costs.get(model, 0.0)
        
        # Format: 1. model: score/total points (percentage%) - tokens tokens - $cost
        print(f"{i}. {model}: {score:.2f}/{total_points:.2f} points ({percentage:.1f}%) - {token_sum} tokens - ${cost_sum:.3f}")

    print()
    print("=" * 100)

if __name__ == "__main__":
    main()
