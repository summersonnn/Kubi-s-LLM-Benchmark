import re
import os

def fix_results(adv_path, std_path, html_path):
    print(f"Fixing {adv_path}...")
    
    with open(adv_path, 'r') as f:
        adv_content = f.read()

    # Split by questions - use a more flexible question split
    q_pattern = r'#{100}\nQUESTION \d+: .+\n#{100}'
    q_matches = list(re.finditer(q_pattern, adv_content))
    
    if not q_matches:
        print("No questions found!")
        return 0

    header = adv_content[:q_matches[0].start()]
    
    # Map to store {question_code: {model_name: (new_score, points)}}
    scores_map = {}
    total_fixes = 0
    
    # We will reconstruct the content
    new_content_parts = [header]
    
    for i, q_match in enumerate(q_matches):
        start = q_match.start()
        end = q_matches[i+1].start() if i+1 < len(q_matches) else adv_content.find("MODEL RANKINGS (Automated Evaluation Only)")
        if end == -1: end = len(adv_content)
        
        q_block = adv_content[start:end]
        
        # Get question info
        header_lines = q_block.split('\n')
        q_line = header_lines[1] # QUESTION X: CODE
        q_code = q_line.split(': ', 1)[1].strip()
        
        pts_match = re.search(r'POINTS: ([\d.]+)', q_block)
        points = float(pts_match.group(1)) if pts_match else 1.0
        
        print(f"  Question: {q_code} ({points} pts)")
        
        # Split by models
        m_blocks = re.split(r'\nMODEL: ', q_block)
        q_header = m_blocks[0]
        m_blocks = m_blocks[1:]
        
        new_m_blocks = []
        for m_block in m_blocks:
            m_lines = m_block.split('\n')
            m_name = m_lines[0].strip()
            
            # Find the aggregate score line
            agg_score_match = re.search(r'SCORE: ([\d.]+)/([\d.]+)', m_block)
            current_agg = float(agg_score_match.group(1)) if agg_score_match else 0.0
            
            # Split by runs to find granular scores
            run_blocks = re.split(r'--- RUN #\d+ ---', m_block)
            run_blocks = run_blocks[1:] # First part is model header
            
            run_scores = []
            for rb in run_blocks:
                # Find SCORE:X/Y after JUDGE EVALUATION
                eval_pos = rb.find("JUDGE EVALUATION:")
                if eval_pos != -1:
                    eval_text = rb[eval_pos:]
                    # Flexible regex for SCORE:X/Y
                    score_match = re.search(r'SCORE:\s*([\d.]+)/([\d.]+)', eval_text)
                    if score_match:
                        s = float(score_match.group(1))
                        m = float(score_match.group(2))
                        run_scores.append((s / m) * points)
                    else:
                        # Fallback to PASS/FAIL
                        if "RUN RESULT: PASS" in rb:
                            run_scores.append(points)
                        elif "RUN RESULT: FAIL" in rb:
                            run_scores.append(0.0)
                        elif "JUDGE VERDICT: Pass" in rb:
                            run_scores.append(points)
                        elif "JUDGE VERDICT: Fail" in rb:
                            run_scores.append(0.0)
            
            if run_scores:
                avg_score = sum(run_scores) / len(run_scores)
            else:
                avg_score = current_agg
            
            if abs(avg_score - current_agg) > 0.001:
                print(f"    FIX: {m_name} {current_agg:.2f} -> {avg_score:.2f}")
                total_fixes += 1
            
            if q_code not in scores_map:
                scores_map[q_code] = {}
            scores_map[q_code][m_name] = (avg_score, points)
            
            # Update SCORE line
            updated_m_block = re.sub(
                r'SCORE: [\d.]+/[\d.]+', 
                f'SCORE: {avg_score:.2f}/{points:.2f}', 
                m_block, 
                count=1
            )
            new_m_blocks.append(updated_m_block)
            
        new_q_block = q_header + '\nMODEL: ' + '\nMODEL: '.join(new_m_blocks)
        new_content_parts.append(new_q_block)

    # Recalculate rankings
    all_models = set()
    for q_scores in scores_map.values():
        all_models.update(q_scores.keys())
    
    total_scores = {m: 0.0 for m in all_models}
    total_possible = 0.0
    for q_code, q_scores in scores_map.items():
        m0 = next(iter(q_scores))
        pts = q_scores[m0][1]
        total_possible += pts
        for m, (s, _) in q_scores.items():
            total_scores[m] += s

    ranked = sorted(total_scores.items(), key=lambda x: x[1], reverse=True)
    
    rankings_text = "\n" + "#" * 100 + "\n"
    rankings_text += "MODEL RANKINGS (Automated Evaluation Only)\n"
    rankings_text += "#" * 100 + "\n\n"
    for i, (m, s) in enumerate(ranked, 1):
        pct = (s / total_possible * 100) if total_possible > 0 else 0
        rankings_text += f"{i}. {m}: {s:.2f}/{total_possible:.2f} points ({pct:.1f}%) - [Data Omitted]\n"

    new_adv_content = "".join(new_content_parts) + rankings_text + "\n" + "=" * 100 + "\n"

    with open(adv_path, 'w') as f:
        f.write(new_adv_content)

    # Fixed Standard/HTML logic (reusing and slightly improving from before)
    # ... (same as before but using all_models and scores_map correctly)
    
    # Standard Results update
    if os.path.exists(std_path):
        with open(std_path, 'r') as f:
            std_content = f.read()
        
        for m_name in all_models:
            m_pattern = rf'{re.escape(m_name.upper())} RESULTS:\n-+\n\n'
            m_section_match = re.search(m_pattern, std_content)
            if not m_section_match: continue
                
            start = m_section_match.end()
            next_m = re.search(r'\n[A-Z/_-]+ RESULTS:', std_content[start:])
            end = (start + next_m.start()) if next_m else std_content.find("=", start)
            
            m_section = std_content[start:end]
            for q_code, q_data in scores_map.items():
                if m_name in q_data:
                    score, pts = q_data[m_name]
                    # Robust question match
                    m_section = re.sub(
                        rf'(Question \d+ \({re.escape(q_code)}\):\n)(  Expected: .+\n)  Score: [\d.]+/{int(pts) if pts.is_integer() else pts}',
                        rf'\1\2  Score: {score:.2f}/{int(pts) if pts.is_integer() else pts}',
                        m_section
                    )
            std_content = std_content[:start] + m_section + std_content[end:]

        std_rank_start = std_content.find("MODEL RANKINGS (Automated Evaluation Only)")
        if std_rank_start != -1:
            h_start = std_content.rfind("=", 0, std_rank_start)
            std_content = std_content[:h_start]
            std_rt = "=" * 80 + "\nMODEL RANKINGS (Automated Evaluation Only)\n" + "=" * 80 + "\n\n"
            for i, (m, s) in enumerate(ranked, 1):
                pct = (s/total_possible*100) if total_possible else 0
                std_rt += f"{i}. {m}: {s:.2f}/{total_possible:.2f} points ({pct:.1f}%)\n"
            std_content += std_rt + "\n" + "=" * 80 + "\n"
        
        with open(std_path, 'w') as f:
            f.write(std_content)

    # HTML update
    if os.path.exists(html_path):
        with open(html_path, 'r') as f:
            html = f.read()
        m_hs = re.findall(r"<th colspan='3' class='model-header'>(.+?)</th>", html)
        for q_code, q_data in scores_map.items():
            disp = q_code.split('-')[0]
            mkr = f"<td class='q-col'>{disp}</td>"
            pos = html.find(mkr)
            if pos != -1:
                r_end = html.find("</tr>", pos)
                r_html = html[pos:r_end]
                new_r = r_html
                for i, mh in enumerate(m_hs):
                    fn = next((n for n in all_models if n.split("@")[0].split("/")[-1] == mh), None)
                    if fn and fn in q_data:
                        ns, pt = q_data[fn]
                        cls = "score"
                        txt = f"{ns:.2f}".rstrip('0').rstrip('.') or "0"
                        if abs(ns - pt) < 0.001: txt, cls = "PASS", "pass"
                        elif ns == 0: txt, cls = "FAIL", "fail"
                        pat = r"(<td class='(?:score|pass|fail)'>)(.+?)(</td>)"
                        idx = [0]
                        def repl(m):
                            if idx[0] == i:
                                idx[0] += 1
                                return f"<td class='{cls}'>{txt}</td>"
                            idx[0] += 1
                            return m.group(0)
                        new_r = re.sub(pat, repl, new_r)
                html = html[:pos] + new_r + html[r_end:]
        
        f_pos = html.find("<tfoot")
        if f_pos != -1:
            f_end = html.find("</tfoot>", f_pos)
            f_html = html[f_pos:f_end]
            tp_s = str(int(total_possible)) if total_possible.is_integer() else f"{total_possible:.2f}"
            for i, mh in enumerate(m_hs):
                fn = next((n for n in all_models if n.split("@")[0].split("/")[-1] == mh), None)
                if fn:
                    ts = total_scores[fn]
                    s_s = f"{ts:.2f}".rstrip('0').rstrip('.') or "0"
                    idx = [0]
                    f_html = re.sub(r"(<td class='score'>)(.+?)(</td>)", lambda m: (f"<td class='score'>{s_s}/{tp_s}</td>" if idx[0] == i else (idx.__setitem__(0, idx[0]+1) or m.group(0))), f_html)
                    # wait, lambda __setitem__ is ugly, better use nested function
            # Let's use simple repl function
            def f_repl(match, cur_idx=[0]):
                res = f"<td class='score'>{total_scores_str[cur_idx[0]]}/{tp_s}</td>" if cur_idx[0] < len(total_scores_list) else match.group(0)
                cur_idx[0] += 1
                return res
            # Actually just do it properly
            total_scores_list = []
            for mh in m_hs:
                fn = next((n for n in all_models if n.split("@")[0].split("/")[-1] == mh), None)
                if fn:
                    ts = total_scores[fn]
                    s_s = f"{ts:.2f}".rstrip('0').rstrip('.') or "0"
                    total_scores_list.append(s_s)
                else: total_scores_list.append("0")
            
            idx = [0]
            def real_f_repl(m):
                res = f"<td class='score'>{total_scores_list[idx[0]]}/{tp_s}</td>"
                idx[0] += 1
                return res
            f_html = re.sub(r"<td class='score'>.+?</td>", real_f_repl, f_html)
            html = html[:f_pos] + f_html + html[f_end:]
        with open(html_path, 'w') as f:
            f.write(html)
            
    return total_fixes

if __name__ == "__main__":
    TS = "20260129_200141"
    ADV = f"results_advanced/benchmark_results_advanced_{TS}.txt"
    STD = f"results/benchmark_results_{TS}.txt"
    HTML = f"results/performance_table_{TS}.html"
    count = fix_results(ADV, STD, HTML)
    print(f"FIX_COUNT:{count}")
