from bs4 import BeautifulSoup
import re

def calculate_multiplier(success_rate):
    """
    Calculates the point multiplier based on the success rate (0.0 to 1.0).
    """
    if success_rate < 0.45:
        # Hard Zone: Scale linearly from 2.0 (at 0) to 1.0 (at 0.45)
        # Slope = (1.0 - 2.0) / 0.45 = -2.222...
        return 2.0 - (success_rate / 0.45)
    
    elif 0.45 <= success_rate <= 0.55:
        # Fair Zone: No change
        return 1.0
    
    else: # success_rate > 0.55
        # Easy Zone: Scale linearly from 1.0 (at 0.55) to 0.5 (at 1.0)
        # Slope = (0.5 - 1.0) / 0.45 = -1.111...
        # M = 1.0 + slope * (rate - 0.55)
        slope = (0.5 - 1.0) / 0.45
        return 1.0 + slope * (success_rate - 0.55)

def parse_score(cell_text, max_points):
    """
    Parses a score cell which can be 'PASS', 'FAIL', or a number.
    Returns the normalized score (0.0 to 1.0) and the raw numeric value.
    """
    clean_text = cell_text.strip().upper()
    
    if clean_text == 'PASS':
        return 1.0, float(max_points)
    elif clean_text == 'FAIL':
        return 0.0, 0.0
    else:
        try:
            score = float(clean_text)
            # Avoid division by zero if max_points is somehow 0
            normalized = score / max_points if max_points > 0 else 0
            return normalized, score
        except ValueError:
            return 0.0, 0.0

def process_leaderboard(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    # Find the table body
    tbody = soup.find('tbody')
    rows = tbody.find_all('tr')
    
    # Track totals for the footer [Total Possible, Model 1 Total, Model 2 Total, Model 3 Total]
    # We assume 3 models based on the input structure
    model_totals = [0.0, 0.0, 0.0] 
    new_total_points = 0.0
    
    # Iterate through question rows
    for row in rows:
        cells = row.find_all('td')
        
        # Assumption: Index 0=QID, 1=Points. 
        # Model 1 Score is at Index 2, Model 2 at 5, Model 3 at 8 (skipping Tokens/Cost)
        # Check standard layout: [ID, Pts, S1, T1, C1, S2, T2, C2, S3, T3, C3]
        
        try:
            old_points_cell = cells[1]
            old_points = float(old_points_cell.text.strip())
        except (ValueError, IndexError):
            continue # Skip malformed rows

        # Indices for score cells for the 3 models
        score_indices = [2, 5, 8]
        normalized_scores = []
        raw_scores = []

        # 1. Calculate Success Rate
        for idx in score_indices:
            norm, raw = parse_score(cells[idx].text, old_points)
            normalized_scores.append(norm)
            raw_scores.append(raw)

        avg_success_rate = sum(normalized_scores) / len(normalized_scores)

        # 2. Determine Multiplier
        multiplier = calculate_multiplier(avg_success_rate)
        new_points = round(old_points * multiplier, 2)
        
        # 3. Update the Points Cell
        old_points_cell.string = f"{new_points:.2f}"
        
        # 4. Update Score Cells and Accumulate Totals
        new_total_points += new_points
        
        for i, idx in enumerate(score_indices):
            cell = cells[idx]
            old_norm = normalized_scores[i]
            
            # Calculate new score based on the new max points
            new_score = round(old_norm * new_points, 2)
            model_totals[i] += new_score
            
            # Update cell text
            if cell.text.strip().upper() == 'PASS':
                # Leave as PASS? Or update to new max? 
                # Usually leaderboard convention keeps "PASS" for max, but let's be explicit if desired.
                # Use "PASS" text but treat as full points for logic.
                pass 
            elif cell.text.strip().upper() == 'FAIL':
                pass
            else:
                # Update numeric score
                cell.string = f"{new_score:.2f}"

    # 5. Update Footer
    tfoot = soup.find('tfoot')
    if tfoot:
        footer_row = tfoot.find('tr')
        footer_cells = footer_row.find_all('td')
        
        # Total Available Points usually shown in the "Score" column format "X/Y"
        # Footer structure in provided HTML:
        # [Label(colspan2), M1_Score, M1_Tokens, M1_Cost, M2_Score, ...]
        
        # The first cell is the label "TOTAL" (colspan=2)
        # Index 1 is Model 1 Score
        # Index 4 is Model 2 Score
        # Index 7 is Model 3 Score
        
        score_indices_footer = [1, 4, 7]
        
        for i, idx in enumerate(score_indices_footer):
            if idx < len(footer_cells):
                cell = footer_cells[idx]
                cell.string = f"{model_totals[i]:.2f}/{new_total_points:.2f}"

    # 6. Save
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(str(soup))
    
    print(f"Leaderboard V2 generated: {output_file}")
    print(f"Total New Points Available: {new_total_points:.2f}")

# --- Execution ---
input_filename = 'original.html'
output_filename = 'performance_table_v2.html'

process_leaderboard(input_filename, output_filename)