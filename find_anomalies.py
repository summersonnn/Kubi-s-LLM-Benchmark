import re
import os

def find_anomalies(adv_path):
    with open(adv_path, 'r') as f:
        content = f.read()

    q_blocks = re.split(r'#{100}\nQUESTION \d+: ', content)
    q_blocks = q_blocks[1:]
    
    for q_block in q_blocks:
        q_code = q_block.split('\n')[0].strip()
        pts_match = re.search(r'POINTS: ([\d.]+)', q_block)
        points = float(pts_match.group(1)) if pts_match else 1.0
        
        m_blocks = re.split(r'\nMODEL: ', q_block)
        m_blocks = m_blocks[1:]
        
        for m_block in m_blocks:
            m_lines = m_block.split('\n')
            m_name = m_lines[0].strip()
            
            score_match = re.search(r'SCORE: ([\d.]+)/([\d.]+)', m_block)
            agg_score = float(score_match.group(1)) if score_match else 0.0
            
            passes = len(re.findall(r'RUN RESULT: PASS', m_block))
            fails = len(re.findall(r'RUN RESULT: FAIL', m_block))
            total_runs = len(re.findall(r'--- RUN #\d+ ---', m_block))
            
            # If all are binary PASS/FAIL, expected score is (passes / total_runs) * points
            if total_runs > 0:
                expected_if_binary = (passes / total_runs) * points
                if abs(agg_score - expected_if_binary) > 0.01:
                    # Check if there are granular scores
                    granulars = re.findall(r'SCORE:(\d+\.?\d*)/(\d+\.?\d*)', m_block)
                    # Exclude the aggregate score one
                    granulars = [g for g in granulars if float(g[1]) != points or (float(g[0])/float(g[1]) * points) != agg_score]
                    
                    if not granulars:
                        print(f"ANOMALY: {q_code} | {m_name}")
                        print(f"  Agg: {agg_score}, Expected: {expected_if_binary} ({passes} passes, {total_runs} runs)")

if __name__ == "__main__":
    find_anomalies("results_advanced/benchmark_results_advanced_20260129_200141.txt")
