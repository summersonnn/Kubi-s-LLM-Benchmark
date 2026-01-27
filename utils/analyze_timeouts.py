
import re
import sys

def analyze_timeouts(filepath):
    timeouts = []
    current_question = None
    current_model = None

    # Patterns
    question_pattern = re.compile(r"^QUESTION \d+: (.+)$")
    model_pattern = re.compile(r"^MODEL: (.+)$")
    timeout_pattern = re.compile(r"Model call timed out")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                
                # Check for Question Header
                q_match = question_pattern.match(line)
                if q_match:
                    current_question = q_match.group(1)
                    current_model = None # Reset model when new question starts
                    continue

                # Check for Model Header
                m_match = model_pattern.match(line)
                if m_match:
                    current_model = m_match.group(1)
                    continue

                # Check for Timeout
                if timeout_pattern.search(line):
                    if current_question and current_model:
                        timeouts.append({
                            "question": current_question,
                            "model": current_model,
                            "line": line
                        })
                    else:
                        print(f"Warning: Found timeout but context missing. Line: {line}, Q: {current_question}, M: {current_model}")

        # Deduplicate results if needed (e.g. if multiple runs fail for same model/question)
        # We can just list them all or aggregate. The user asked "which models timed out at which questions".
        # A list of unique (model, question) pairs might be best, or a count.
        
        # Let's aggregate by model
        model_timeouts = {}
        for t in timeouts:
            m = t['model']
            q = t['question']
            if m not in model_timeouts:
                model_timeouts[m] = set()
            model_timeouts[m].add(q)

        print(f"Found {len(timeouts)} timeout occurrences.")
        print("-" * 40)
        
        for model, questions in model_timeouts.items():
            print(f"Model: {model}")
            for q in sorted(list(questions)):
                print(f"  - Question: {q}")
            print()

    except FileNotFoundError:
        print(f"File not found: {filepath}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_timeouts.py <logfile>")
        sys.exit(1)
        
    analyze_timeouts(sys.argv[1])
