import re
import os

def check_consistency(adv_path):
    with open(adv_path, 'r') as f:
        content = f.read()

    q_blocks = re.split(r'#{100}\nQUESTION \d+: ', content)
    q_blocks = q_blocks[1:]
    
    anomalies = 0
    for q_block in q_blocks:
        lines = q_block.split('\n')
        q_code = lines[0].strip()
        pts_match = re.search(r'POINTS: ([\d.]+)', q_block)
        points = float(pts_match.group(1)) if pts_match else 1.0
        
        m_blocks = re.split(r'\nMODEL: ', q_block)
        m_blocks = m_blocks[1:]
        
        for m_block in m_blocks:
            m_lines = m_block.split('\n')
            m_name = m_lines[0].strip()
            
            score_match = re.search(r'SCORE: ([\d.]+)/([\d.]+)', m_block)
            agg_score = float(score_match.group(1)) if score_match else 0.0
            
            # Find all run results
            run_results = []
            
            # Split by runs
            r_blocks = re.split(r'--- RUN #\d+ ---', m_block)
            total_runs = len(r_blocks) - 1
            r_blocks = r_blocks[1:]
            
            for rb in r_blocks:
                # 1. Granular SCORE:X/Y
                sm = re.search(r'SCORE:\s*([\d.]+)/([\d.]+)', rb)
                if sm:
                    s = float(sm.group(1))
                    m = float(sm.group(2))
                    run_results.append((s / m) * points)
                # 2. X/Y pts
                elif re.search(r'RUN RESULT: ([\d.]+)/([\d.]+) pts', rb):
                    m = re.search(r'RUN RESULT: ([\d.]+)/([\d.]+) pts', rb)
                    run_results.append(float(m.group(1)))
                # 3. PASS/FAIL
                elif "RUN RESULT: PASS" in rb:
                    run_results.append(points)
                elif "RUN RESULT: FAIL" in rb:
                    run_results.append(0.0)
                else:
                    # No result found? Assume 0
                    run_results.append(0.0)
            
            if total_runs > 0:
                expected = sum(run_results) / total_runs
                if abs(agg_score - expected) > 0.01:
                    print(f"ANOMALY: {q_code} | {m_name}")
                    print(f"  Header: {agg_score}/{points}, Calculated: {expected:.2f} (Runs: {run_results})")
                    anomalies += 1
    return anomalies

if __name__ == "__main__":
    count = check_consistency("results_advanced/benchmark_results_advanced_20260129_200141.txt")
    print(f"TOTAL ANOMALIES: {count}")
