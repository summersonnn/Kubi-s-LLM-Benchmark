# Kubis-Benchmark

A flexible, automated framework for benchmarking Large Language Models (LLMs) across diverse question sets using two distinct evaluation methodologies: **Judge LLM** and **Deterministic Verifiers**.

To see the leaderboard and the published questions for my personal benchmark, visit [here](https://summersonnn.github.io/Kubis-Benchmark-WebApp/).

## 🚀 Overview

This benchmark evaluates models by running them against a set of queries defined in the `questions/` directory. It supports parallel execution, detailed logging, and generates both raw and aggregated performance reports.

Key Features:
-   **Dual Evaluation Types**:
    -   **Judge LLM**: Uses a stronger model (e.g., standard API model) to qualitatively evaluate answers against a ground truth.
    -   **Verifier Scripts**: Uses Python scripts to deterministically validate answers (e.g., "contains letter 'a' 5 times", "solves this math equation").
-   **Automated Scoring**: Pass/Fail or granular scoring (0.0 - 1.0).
-   **Parallel Execution**: High-throughput benchmarking using `asyncio` and worker pools.

## 🛠️ Installation & Setup

### Prerequisites
-   Python 3.10+
-   [uv](https://github.com/astral-sh/uv) (Recommended for package management)

### 1. Install Dependencies
Using `uv`:
```bash
uv sync
```
Or standard pip:
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the project root:

```ini
# Required: API Key for OpenRouter (or compatible service)
MODEL_API_KEY=sk-or-v1-...

# Optional: API Base URL (Default: https://openrouter.ai/api/v1)
MODEL_API_BASE_URL=https://openrouter.ai/api/v1

# Benchmark Configuration
NUM_RUNS=4          # Number of times to ask each question per model
MAX_WORKERS=28      # Parallel workers for maximum throughput
MODEL_MAX_TOKENS=8196
MODEL_TEMPERATURE=0.7
```

### 3. Configure Models
Edit `config/models.txt` to list the models you want to valid. Each line is a model ID.
```text
openai/gpt-4o
google/gemini-pro-1.5
anthropic/claude-3-opus
```

To configure the **Judge LLM**, edit `config/judge_model.txt`:
```text
google/gemini-3-flash-preview
```

## 🏃 parameters

### Basic Run
Run the benchmark with default settings (reads config from `config/questions.txt`):
```bash
uv run main.py
```

### Run specific questions
To run only specific question sets, modify `config/questions.txt`. You can list:
-   Specific codes: `A1`, `A5`
-   Subdirectories: `Basic Mix/`
-   All: `ALL`

### Clear History
To delete previous results before running:
```bash
uv run main.py --delete_history
```

## 🧠 Evaluation Methods

The benchmark automatically determines the evaluation method based on the **Question File Name**.

### 1. ID Naming Convention
Every question file must start with a unique ID in the format: **`A{ID}-`**.
-   **`A{ID}-J-...`**: **Judge Evaluation**. An LLM compares the model's output to the Ground Truth.
-   **`A{ID}-V-...`**: **Verifier Evaluation**. A Python script validates the output programmatically.

### 2. File Format
Questions are text files located in `questions/`.
**Format:**
```text
<Your Question Here>
Can span multiple lines.

----
Ground Truth: <The expected answer or key facts>
Point: 1
```

For **Judge** questions (`-J-`), `Ground Truth` is the reference text for the judge.
For **Verifier** questions (`-V-`), `Ground Truth` is **optional/informational** (often set to "VERIFIER"). The verifier script handles all validation logic, so the system ignores this field.

## ➕ How to Add New Information

### Adding a "Judge" Question
1.  Create a file in `questions/Category/` (e.g., `questions/General/A100-J-capital-of-france.txt`).
2.  Content:
    ```text
    What is the capital of France?
    ----
    Ground Truth: Paris
    Point: 1
    ```
3.  Add `Questions/General/` or `A100` to `config/questions.txt`.

### Adding a "Verifier" Question
This requires two parts: the question file and the verification logic.

1.  **Create Question File**: `questions/Math/A101-V-math-proof.txt`
    ```text
    Calculate 2 + 2. Return answer in \boxed{}.
    ----
    Ground Truth: 4 (This is optional for verifier questions. However, if you plan to generate the verifier using an LLM, it’s best to include this in the question file so the model knows the expected answer.)
    Point: 1
    ```
    *Note: The `-V-` in the filename triggers the verifier logic.*

2.  **Create Verifier Script**:
    Create a python file in `verifier_scripts/` named exactly to match the question ID base: `A101_math_proof_verifier.py`.
    
    > **Tip**: Check `verifier_scripts/A9_longest_word_verifier_example.py` for a complete, well-commented example.
    
    *Structure:*
    ```python
    def verify_answer(answer: str) -> tuple[bool, str]:
        """
        Validates the answer.
        Returns:
            (is_success, reasoning_log)
        """
        # 1. Parse answer
        # 2. Check logic
        if "4" in answer:
            return True, "Correct answer found."
        
        return False, f"Expected 4, got {answer}"
    ```

    The script **MUST** correspond to the ID (`A101`) of the question.

## 📂 Directory Structure

-   `questions/`: Source text files for questions. Organized by category.
-   `verifier_scripts/`: Python logic for `-V-` questions.
-   `utils/`: Core logic (Runner, API, Evaluators).
-   `results/`: JSON/HTML output of benchmark runs.
-   `config/`: Configuration files (`models.txt`, `questions.txt`).

## ⚠️ Important Notes
-   **Naming Collision**: If you add a question without following the convention, the system might try to rename it or fail to evaluate it correctly. Always use `A{ID}-[TYPE]-description`.
-   **Verifier Matching**: The `VerifierEvaluator` looks for scripts in `verifier_scripts/` that start with the question ID (e.g., `A5_...` for question `A5-V-...`).
