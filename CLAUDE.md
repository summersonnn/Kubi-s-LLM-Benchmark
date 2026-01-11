# CLAUDE.md - Kubis-Benchmark

## Project Overview

Kubis-Benchmark is a framework for evaluating and benchmarking Large Language Models (LLMs) across diverse question sets. It supports multiple evaluation strategies (automated LLM judge, hardcoded validators, blind human evaluation), parallel execution, and comprehensive result reporting.

## Quick Start

```bash
# Install dependencies
uv sync

# Configure environment
cp .env.example .env  # Add your MODEL_API_KEY

# Run benchmark
python main.py
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         main.py                              │
│  (Orchestrator: loads questions, runs models, collects scores)│
└─────────────────────────────────────────────────────────────┘
         │                    │                    │
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   model_api.py  │  │  evaluators.py  │  │    utils.py     │
│  (OpenRouter    │  │  (Judge LLM,    │  │  (Logging,      │
│   API client)   │  │   Validity,     │  │   file parsing) │
│                 │  │   Human eval)   │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                            │
         ┌──────────────────┼──────────────────┐
         ▼                  ▼                  ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ JudgeLLMEval    │  │ ValidityEval    │  │ HumanEvaluator  │
│ (LLM compares   │  │ (Custom checker │  │ (Blind web UI   │
│  answer to GT)  │  │  modules)       │  │  scoring 1-10)  │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Key Files

| File | Purpose |
|------|---------|
| `main.py` | Entry point, orchestrates benchmarking pipeline |
| `model_api.py` | OpenRouter API wrapper, model configuration |
| `evaluators.py` | Three evaluation strategies (Judge LLM, Validity, Human) |
| `utils.py` | Shared utilities (logging, question file parsing) |
| `human_eval_server.py` | HTTP server for blind human evaluation |
| `integrate_human_scores.py` | Post-benchmark score integration |

## Configuration Files

| File | Purpose |
|------|---------|
| `models.txt` | List of models to benchmark (prefix with `!` to disable) |
| `questions.txt` | Question codes to include in benchmark |
| `judge_model.txt` | LLM used for automated judging (default: `google/gemini-3-pro-preview`) |
| `.env` | API credentials (`MODEL_API_KEY`, `MODEL_API_BASE_URL`) |

## Directory Structure

```
questions/
├── basic/              # Simple reasoning & math
├── general-knowledge/  # Factual & linguistic questions
├── math/               # Mathematical problems
├── reasoning/          # Complex logic & analysis
├── science/            # Biology, chemistry, physics
└── manual_checks/      # HTML/CSS/JS implementations (human eval)

validity_checkers/      # Custom validation modules for VALIDITY CHECK questions
results/                # Standard benchmark results
results_advanced/       # Detailed results with full model outputs
manual_run_codes/       # Human evaluation session data
```

## Question File Format

```
[Question text]

----
Ground Truth: [answer | "VALIDITY CHECK" | empty for human eval]
Point: [integer, default 1]
```

**Evaluation type is determined by Ground Truth:**
- Specific answer → JudgeLLMEvaluator
- `VALIDITY CHECK` → ValidityEvaluator (uses `validity_checkers/<code>_checker.py`)
- Empty/missing → HumanEvaluator (blind scoring 1-10)

## Benchmarking Flow

1. Load models from `models.txt` and questions from `questions.txt`
2. For each question, for each model: run 4 parallel evaluations
3. Evaluate responses using appropriate evaluator
4. Calculate scores: `points * (passes / total_runs)`
5. Generate timestamped results files
6. If human eval needed: spawn web server at `localhost:8765`
7. Run `python integrate_human_scores.py <session_dir>` after scoring

## Common Commands

```bash
# Run full benchmark
python main.py

# Integrate human scores after evaluation
python integrate_human_scores.py manual_run_codes/<session_dir>

# Type checking
uv run mypy .

# Linting
uv run ruff check .
```

## Adding New Questions

1. Create `questions/<category>/A<N>-<name>.txt` with question format
2. Add question code to `questions.txt`
3. For VALIDITY CHECK questions, create `validity_checkers/A<N>_<name>_checker.py`:

```python
def check_validity(model_answer: str) -> tuple[bool, str]:
    """Returns (is_valid, failure_reason)"""
    # Validation logic
    return True, ""  # or False, "reason"
```

## Adding New Models

Edit `models.txt` (one model per line, OpenRouter format):
```
openai/gpt-4o
anthropic/claude-3.5-sonnet
google/gemini-2.0-flash
```

## Key Implementation Details

### Parallel Execution
- Uses `ThreadPoolExecutor` with max 16 workers
- 4 runs per model per question (configurable via `NUM_RUNS`)
- Results collected via `as_completed()` for progressive reporting

### Human Evaluation
- Implementations saved anonymously as `impl_XXX.html`
- Shuffled order prevents evaluator bias
- Web UI supports keyboard shortcuts (Space+digit to score)
- Multi-window support with shared work queue

### Result Files
- `results/benchmark_results_*.txt`: Summary with scores
- `results_advanced/benchmark_results_advanced_*.txt`: Full model outputs and reasoning
- `manual_run_codes/<session>/manifest.json`: Human eval metadata

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `MODEL_API_KEY` | OpenRouter API key | (required) |
| `MODEL_API_BASE_URL` | API endpoint | `https://openrouter.ai/api/v1` |

## Dependencies

- Python 3.12+
- `openai` - API client
- `httpx` - HTTP client
- `python-dotenv` - Environment loading

Dev dependencies: `mypy`, `ruff`, `pytest`

## Code Conventions

- Use `utils.setup_logging()` for consistent log formatting
- Evaluators return `{success: bool, reasoning: str, verdict: str}`
- Validity checkers return `tuple[bool, str]` (is_valid, failure_reason)
- Question codes use format `A<N>` (e.g., A15, A48)

## Troubleshooting

**API errors**: Check `.env` has valid `MODEL_API_KEY`

**Question not found**: Ensure code in `questions.txt` matches filename pattern `A<N>-*.txt`

**Human eval server issues**: Check port 8765 is available, or kill existing process

**Score integration fails**: Ensure all implementations are scored in web UI before running integration
