"""
Battleship Match Runner: Orchestrates head-to-head matches between two AI models.

Prompts two models to implement BattleshipAgent, extracts their code, renames
them to BattleshipAgent_1 and BattleshipAgent_2, runs games, and reports
win/loss statistics.

Supports parallel execution: fires all prompts concurrently and starts matches
as soon as response pairs become available.
"""

import asyncio
from datetime import datetime
import os
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

from dotenv import load_dotenv

# Add project root to path for direct execution
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.model_api import ModelAPI
from utils.utils import setup_logging

logger = setup_logging(__name__)

# Load environment variables
load_dotenv()

# Configuration
NUM_RUNS = int(os.getenv("NUM_RUNS", "4"))
NUM_ROUNDS_PER_BATTLESHIP_MATCH = 100
BOARD_SIZE = 8
SHIPS = [6, 5, 4, 3]

# Results directories
BATTLESHIP_RESULTS_DIR = Path(__file__).parent.parent / "results" / "matches" / "battleship"
GAME_LOGS_DIR = BATTLESHIP_RESULTS_DIR / "game_logs"
MODEL_RESPONSES_DIR = BATTLESHIP_RESULTS_DIR / "model_responses"

# The game code template with placeholders for agent implementations
GAME_CODE_TEMPLATE = '''
import sys
import random
from collections import deque
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# Move timeout in seconds
MOVE_TIMEOUT = 1.0

# --- Game Configuration ---
BOARD_SIZE = 8
SHIPS = [5, 4, 3]
NUM_GAMES = {num_games}

# --- Board Representations ---
EMPTY = 'O'
SHIP = 'S'
HIT = 'X'
SUNK = '#'
MISS = 'M'

{extra_imports}

{agent1_code}

{agent2_code}


class BattleshipGame:
    """Manages the state and rules of the game."""
    def __init__(self, size, ships_config):
        self.size = size
        self.ships_config = ships_config
        self.player1_ships_board = self._create_ship_board()
        self.player2_ships_board = self._create_ship_board()

    def _create_empty_board(self):
        return [[EMPTY for _ in range(self.size)] for _ in range(self.size)]

    def _create_ship_board(self):
        """Creates a board and places ships on it."""
        board = self._create_empty_board()
        for length in self.ships_config:
            placed = False
            while not placed:
                orientation = random.choice(['horizontal', 'vertical'])
                r = random.randint(0, self.size - (length if orientation == 'vertical' else 1))
                c = random.randint(0, self.size - (length if orientation == 'horizontal' else 1))
                
                if orientation == 'horizontal':
                    if all(board[r][c+i] == EMPTY for i in range(length)):
                        for i in range(length): board[r][c+i] = SHIP
                        placed = True
                else:
                    if all(board[r+i][c] == EMPTY for i in range(length)):
                        for i in range(length): board[r+i][c] = SHIP
                        placed = True
        return board

    def is_game_over(self, ships_board):
        """Checks if all ships on a given board have been sunk."""
        return not any(SHIP in row for row in ships_board)


# --- Stats ---
stats = {{
    "normal": 0,
    "draw": 0,
    "c1": 0,
    "c2": 0,
    "r1_timeout": 0,
    "r1_crash": 0,
    "r1_invalid": 0,
    "r2_timeout": 0,
    "r2_crash": 0,
    "r2_invalid": 0,
}}

def play_game(game_num, scores):
    """Plays a single game of Battleship and returns the winner's name or crash info."""
    game = BattleshipGame(BOARD_SIZE, SHIPS)
    
    # Try to initialize agents - if one fails, the other wins
    try:
        agent1 = BattleshipAgent_1("Agent-1", BOARD_SIZE, SHIPS)
    except Exception as e:
        return ("Agent-2", "Crash during init (Agent-1): " + str(e)[:100])
    
    try:
        agent2 = BattleshipAgent_2("Agent-2", BOARD_SIZE, SHIPS)
    except Exception as e:
        return ("Agent-1", "Crash during init (Agent-2): " + str(e)[:100])
    
    p1_active_board = [row[:] for row in game.player1_ships_board]
    p2_active_board = [row[:] for row in game.player2_ships_board]
    p1_guess_board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]
    p2_guess_board = [[EMPTY for _ in range(BOARD_SIZE)] for _ in range(BOARD_SIZE)]

    players = {{
        agent1: {{'opponent_ships_board': p2_active_board, 'guess_board': p1_guess_board}},
        agent2: {{'opponent_ships_board': p1_active_board, 'guess_board': p2_guess_board}}
    }}
    current_agent, opponent_agent = agent1, agent2
    
    last_shot_coord, last_shot_result = None, None
    turn_continues = False
    turn_count = 0
    max_turns = BOARD_SIZE * BOARD_SIZE * 2  # Max turns before draw

    while True:
        turn_count += 1
        if turn_count > max_turns:
            # Game exceeded max turns - return draw
            stats["draw"] += 1
            return "DRAW"
        
        # Try to get move with timeout - if timeout or crash, use random move
        move = None
        sunk_coords = []
        
        try:
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(
                    current_agent.make_move, last_shot_result, last_shot_coord
                )
                try:
                    move_data = future.result(timeout=MOVE_TIMEOUT)
                    if isinstance(move_data, tuple) and len(move_data) == 2:
                        move, sunk_coords = move_data
                    else:
                        move, sunk_coords = move_data, []
                except FuturesTimeoutError:
                    # Agent took too long - use random move (any cell)
                    move = (random.randint(0, BOARD_SIZE - 1), random.randint(0, BOARD_SIZE - 1))
                    sunk_coords = []
                    if current_agent.name == "Agent-1": stats["r1_timeout"] += 1
                    else: stats["r2_timeout"] += 1
        except Exception:
            # Agent crashed - use random move
            move = (random.randint(0, BOARD_SIZE - 1), random.randint(0, BOARD_SIZE - 1))
            sunk_coords = []
            if current_agent.name == "Agent-1": stats["r1_crash"] += 1
            else: stats["r2_crash"] += 1
        
        if move is None:
            # Agent returned None - use random move
            move = (random.randint(0, BOARD_SIZE - 1), random.randint(0, BOARD_SIZE - 1))
            if current_agent.name == "Agent-1": stats["r1_invalid"] += 1
            else: stats["r2_invalid"] += 1
        
        # Validate move coordinates
        try:
            row, col = move
            if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE):
                # Invalid coordinates - use random move
                row, col = random.randint(0, BOARD_SIZE - 1), random.randint(0, BOARD_SIZE - 1)
                if current_agent.name == "Agent-1": stats["r1_invalid"] += 1
                else: stats["r2_invalid"] += 1
        except Exception:
            row, col = random.randint(0, BOARD_SIZE - 1), random.randint(0, BOARD_SIZE - 1)
            if current_agent.name == "Agent-1": stats["r1_invalid"] += 1
            else: stats["r2_invalid"] += 1
            
        p_data = players[current_agent]
        opponent_ships_board, guess_board = p_data['opponent_ships_board'], p_data['guess_board']
        
        result = opponent_ships_board[row][col]
        
        if result == SHIP:
            opponent_ships_board[row][col] = HIT
            guess_board[row][col] = HIT
            last_shot_result = 'HIT'
            turn_continues = True
        else:
            guess_board[row][col] = MISS
            last_shot_result = 'MISS'
            turn_continues = False

        last_shot_coord = move

        if sunk_coords:
            for r, c in sunk_coords:
                guess_board[r][c] = SUNK

        if game.is_game_over(opponent_ships_board):
            stats["normal"] += 1
            return current_agent.name
        
        if not turn_continues:
            current_agent, opponent_agent = opponent_agent, current_agent
            last_shot_coord, last_shot_result = None, None


def main():
    """Main function to run the Battleship simulation."""
    scores = {{"Agent-1": 0, "Agent-2": 0}}
    crash_detected = None

    for i in range(NUM_GAMES):
        result = play_game(i + 1, scores)
        
        # Check if result is a crash tuple, draw, or just winner name
        if isinstance(result, tuple):
            winner, crash_msg = result
            crash_detected = crash_msg
            # Award ALL remaining games to the winner
            remaining = NUM_GAMES - i
            scores[winner] += remaining
            
            # Count crash for the loser
            if winner == "Agent-1": stats["c2"] += 1
            else: stats["c1"] += 1
            
            # Print intermediate progress before breaking
            print(f"PROGRESS:Agent-1={{scores['Agent-1']}},Agent-2={{scores['Agent-2']}},N={{stats['normal']}},D={{stats['draw']}},C1={{stats['c1']}},C2={{stats['c2']}},R1T={{stats['r1_timeout']}},R1C={{stats['r1_crash']}},R1I={{stats['r1_invalid']}},R2T={{stats['r2_timeout']}},R2C={{stats['r2_crash']}},R2I={{stats['r2_invalid']}}")
            sys.stdout.flush()
            break
        elif result == "DRAW":
            # Draw - both get 0.5 points
            scores["Agent-1"] += 0.5
            scores["Agent-2"] += 0.5
        else:
            winner = result
            if winner in scores:
                scores[winner] += 1
        
        # Print intermediate progress for partial result parsing on timeout
        print(f"PROGRESS:Agent-1={{scores['Agent-1']}},Agent-2={{scores['Agent-2']}},N={{stats['normal']}},D={{stats['draw']}},C1={{stats['c1']}},C2={{stats['c2']}},R1T={{stats['r1_timeout']}},R1C={{stats['r1_crash']}},R1I={{stats['r1_invalid']}},R2T={{stats['r2_timeout']}},R2C={{stats['r2_crash']}},R2I={{stats['r2_invalid']}}")
        sys.stdout.flush()
    
    print(f"RESULT:Agent-1={{scores['Agent-1']}},Agent-2={{scores['Agent-2']}}")
    print(f"STATS:Normal={{stats['normal']}},Draw={{stats['draw']}},C1={{stats['c1']}},C2={{stats['c2']}},R1T={{stats['r1_timeout']}},R1C={{stats['r1_crash']}},R1I={{stats['r1_invalid']}},R2T={{stats['r2_timeout']}},R2C={{stats['r2_crash']}},R2I={{stats['r2_invalid']}}")
    if crash_detected:
        print(f"CRASH:{{crash_detected}}")



if __name__ == "__main__":
    main()
'''


def load_prompt() -> str:
    """Load the Battleship prompt from the questions directory."""
    prompt_path = (
        Path(__file__).parent.parent
        / "questions"
        / "Coding"
        / "Competitive Game Algo"
        / "A19-V-BattleShip.txt"
    )
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    return prompt_path.read_text()


def select_models(api: ModelAPI) -> tuple[str, str]:
    """Interactive model selection for the two competing models."""
    print("\n" + "=" * 60)
    print("BATTLESHIP MATCH - MODEL SELECTION")
    print("=" * 60)
    print("\nAvailable models:")
    for i, model in enumerate(api.models):
        print(f"  [{i}] {model}")

    print()
    while True:
        try:
            idx1 = int(input("Select Model 1 (index): ").strip())
            if 0 <= idx1 < len(api.models):
                break
            print(f"Invalid index. Must be 0-{len(api.models) - 1}")
        except ValueError:
            print("Please enter a number.")

    while True:
        try:
            idx2 = int(input("Select Model 2 (index): ").strip())
            if 0 <= idx2 < len(api.models):
                break
            print(f"Invalid index. Must be 0-{len(api.models) - 1}")
        except ValueError:
            print("Please enter a number.")

    model1 = api.models[idx1]
    model2 = api.models[idx2]

    print("\nMatch setup:")
    print(f"  Agent-1: {model1}")
    print(f"  Agent-2: {model2}")
    print(f"  Runs: {NUM_RUNS}")
    print(f"  Games per run: {NUM_ROUNDS_PER_BATTLESHIP_MATCH}")

    return model1, model2


async def prompt_model(
    api: ModelAPI, model_name: str, prompt: str, run_id: int, max_retries: int = 3
) -> tuple[int, str, str]:
    """
    Call a model with the Battleship prompt and return its response.
    Retries up to max_retries times on failure or empty response.

    Returns:
        Tuple of (run_id, model_name, response_content)
    """
    for attempt in range(max_retries + 1):
        if attempt > 0:
            logger.info("Retrying model: %s (run %d) - attempt %d/%d", model_name, run_id, attempt, max_retries)
        else:
            logger.info("Prompting model: %s (run %d)", model_name, run_id)
            
        try:
            # Use 4× max_tokens for Battleship to allow for more complex implementations
            battleship_max_tokens = api.max_tokens * 4
            response = await api.call(
                prompt, model_name=model_name, reasoning=True, max_tokens=battleship_max_tokens
            )
            content = response.choices[0].message.content
            if not content:
                logger.error("Empty response from model: %s (run %d) on attempt %d", model_name, run_id, attempt)
                if attempt < max_retries:
                    continue
                return (run_id, model_name, "")
                
            logger.info(
                "Received response from %s (run %d): %d chars",
                model_name,
                run_id,
                len(content),
            )
            return (run_id, model_name, content)
        except Exception as e:
            logger.error("Error calling model %s (run %d) on attempt %d: %s", model_name, run_id, attempt, e)
            if attempt < max_retries:
                # Add a small delay before retrying
                await asyncio.sleep(2 ** attempt) 
                continue
            return (run_id, model_name, "")
    
    return (run_id, model_name, "")


def validate_agent_syntax(agent_code: str, extra_imports: str = "") -> tuple[bool, str]:
    """
    Validate that agent code has valid Python syntax.
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    import ast
    
    # Create a minimal test script with the agent code
    test_code = f"""
import random
from collections import deque
{extra_imports}

{agent_code}
"""
    try:
        ast.parse(test_code)
        return True, ""
    except SyntaxError as e:
        return False, f"Syntax error at line {e.lineno}: {e.msg}"


def extract_agent_code(model_response: str, target_class_name: str) -> tuple[str, str]:
    """
    Extract BattleshipAgent class from model response and rename it.

    Args:
        model_response: The model's full response text
        target_class_name: The new class name (e.g., 'BattleshipAgent_1')

    Returns:
        Tuple of (renamed_agent_code, extra_imports)
    """
    # Find code blocks
    code_blocks = re.findall(
        r"```(?:python)?\s*(.*?)```", model_response, re.DOTALL
    )
    agent_code = ""

    # Look for BattleshipAgent class
    for block in code_blocks:
        if "class BattleshipAgent" in block:
            agent_code = block
            break

    # Fallback: search in raw text
    if not agent_code and "class BattleshipAgent" in model_response:
        match = re.search(
            r"(class BattleshipAgent.*?)(?=\nclass\s|\ndef\s+(?!__)|if\s+__name__|$)",
            model_response,
            re.DOTALL,
        )
        if match:
            agent_code = match.group(1)

    if not agent_code:
        return "", ""

    # Rename the class
    agent_code = re.sub(
        r"class\s+BattleshipAgent\b", f"class {target_class_name}", agent_code
    )

    # Extract imports (excluding standard ones already in template)
    source_for_imports = code_blocks[0] if code_blocks else model_response
    import_lines = []
    for line in source_for_imports.split("\n"):
        line = line.strip()
        if line.startswith("import ") or line.startswith("from "):
            # Skip imports already in template
            if "random" not in line and "collections" not in line:
                import_lines.append(line)

    return agent_code.strip(), "\n".join(import_lines)


def build_game_code(
    agent1_code: str,
    agent2_code: str,
    extra_imports: str,
    num_games: int = NUM_ROUNDS_PER_BATTLESHIP_MATCH,
) -> str:
    """Build the complete game code with both agent implementations."""
    return GAME_CODE_TEMPLATE.format(
        num_games=num_games,
        extra_imports=extra_imports,
        agent1_code=agent1_code,
        agent2_code=agent2_code,
    )


def run_match(game_code: str, match_id: int, run_ids: tuple[int, int], timeout: int = 900) -> dict:
    """
    Execute the match and parse results.

    Returns:
        Dict with keys: success, agent1_wins, agent2_wins, error, match_id, agent1_run_id, agent2_run_id
    """
    temp_id = uuid.uuid4().hex[:8]
    temp_file = os.path.join(
        tempfile.gettempdir(), f"battleship_match_{match_id}_{temp_id}.py"
    )

    try:
        with open(temp_file, "w") as f:
            f.write(game_code)

        result = subprocess.run(
            ["python", temp_file], capture_output=True, text=True, timeout=timeout
        )

        if result.returncode != 0:
            return {
                "match_id": match_id,
                "agent1_run_id": run_ids[0],
                "agent2_run_id": run_ids[1],
                "success": False,
                "agent1_wins": 0,
                "agent2_wins": 0,
                "error": result.stderr[:500],
            }

        # Parse results
        match = re.search(r"RESULT:Agent-1=([\d.]+),Agent-2=([\d.]+)", result.stdout)
        if match:
            # Check for crash info
            crash_match = re.search(r"CRASH:(.+)", result.stdout)
            crash_info = crash_match.group(1) if crash_match else None
            
            # Parse detailed stats
            # ...
            stats = {
                "normal": 0, "draw": 0, "c1": 0, "c2": 0,
                "r1_timeout": 0, "r1_crash": 0, "r1_invalid": 0,
                "r2_timeout": 0, "r2_crash": 0, "r2_invalid": 0
            }
            stats_match = re.search(r"STATS:Normal=(\d+),Draw=(\d+),C1=(\d+),C2=(\d+),R1T=(\d+),R1C=(\d+),R1I=(\d+),R2T=(\d+),R2C=(\d+),R2I=(\d+)", result.stdout)
            if stats_match:
                stats = {
                    "normal": int(stats_match.group(1)),
                    "draw": int(stats_match.group(2)),
                    "c1": int(stats_match.group(3)),
                    "c2": int(stats_match.group(4)),
                    "r1_timeout": int(stats_match.group(5)),
                    "r1_crash": int(stats_match.group(6)),
                    "r1_invalid": int(stats_match.group(7)),
                    "r2_timeout": int(stats_match.group(8)),
                    "r2_crash": int(stats_match.group(9)),
                    "r2_invalid": int(stats_match.group(10)),
                }

            return {
                "match_id": match_id,
                "agent1_run_id": run_ids[0],
                "agent2_run_id": run_ids[1],
                "success": True,
                "agent1_wins": float(match.group(1)),
                "agent2_wins": float(match.group(2)),
                "error": None,
                "crash_info": crash_info,
                "stats": stats,
            }

        return {
            "match_id": match_id,
            "agent1_run_id": run_ids[0],
            "agent2_run_id": run_ids[1],
            "success": False,
            "agent1_wins": 0,
            "agent2_wins": 0,
            "error": "Could not parse results from output",
            "crash_info": None,
        }

    except subprocess.TimeoutExpired as e:
        # On timeout, try to parse partial results from e.stdout
        stdout = e.stdout if isinstance(e.stdout, str) else (e.stdout.decode() if e.stdout else "")
        # ... (keeping implementation-specific parsing logic)
        
        # Find PROGRESS matches with granular random move reasons
        progress_pattern = r"PROGRESS:Agent-1=([\d.]+),Agent-2=([\d.]+),N=(\d+),D=(\d+),C1=(\d+),C2=(\d+),R1T=(\d+),R1C=(\d+),R1I=(\d+),R2T=(\d+),R2C=(\d+),R2I=(\d+)"
        progress_matches = re.findall(progress_pattern, stdout)
        
        a1_wins, a2_wins = 0.0, 0.0
        stats = {
            "normal": 0, "draw": 0, "c1": 0, "c2": 0,
            "r1_timeout": 0, "r1_crash": 0, "r1_invalid": 0,
            "r2_timeout": 0, "r2_crash": 0, "r2_invalid": 0
        }
        
        if progress_matches:
            last = progress_matches[-1]
            a1_wins, a2_wins = float(last[0]), float(last[1])
            stats = {
                "normal": int(last[2]), "draw": int(last[3]), "c1": int(last[4]), "c2": int(last[5]),
                "r1_timeout": int(last[6]), "r1_crash": int(last[7]), "r1_invalid": int(last[8]),
                "r2_timeout": int(last[9]), "r2_crash": int(last[10]), "r2_invalid": int(last[11]),
            }
            
        return {
            "match_id": match_id,
            "agent1_run_id": run_ids[0],
            "agent2_run_id": run_ids[1],
            "success": True,
            "agent1_wins": a1_wins,
            "agent2_wins": a2_wins,
            "error": None,
            "crash_info": f"Match timed out after {timeout}s (Partial)",
            "stats": stats,
        }
    except Exception as e:
        return {
            "match_id": match_id,
            "agent1_run_id": run_ids[0],
            "agent2_run_id": run_ids[1],
            "success": False,
            "agent1_wins": 0,
            "agent2_wins": 0,
            "error": str(e),
        }
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)


async def run_match_async(game_code: str, match_id: int, run_ids: tuple[int, int]) -> dict:
    """Run a match in a thread pool to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, run_match, game_code, match_id, run_ids)


async def match_coordinator(
    model1_queue: asyncio.Queue,
    model2_queue: asyncio.Queue,
    results_list: list,
    num_runs: int,
    log_file: Path,
    responses_file: Path,
    model1_name: str,
    model2_name: str,
) -> None:
    """
    Coordinates match execution by pairing responses as they arrive.

    Waits for one response from each model queue, then starts a match.
    Matches run in parallel as pairs become available.
    Results are logged to file as they complete.
    """
    pending_tasks: dict[asyncio.Task, int] = {}  # task -> match_id
    matches_started = 0
    matches_completed = 0

    async def handle_completed_match(task: asyncio.Task, run_ids: tuple[int, int]) -> None:
        """Process a completed match result."""
        nonlocal matches_completed
        result = task.result()
        results_list.append(result)
        matches_completed += 1
        run_id1, run_id2 = run_ids

        # Print progress
        if result["success"]:
            a1 = result["agent1_wins"]
            a2 = result["agent2_wins"]
            crash_note = ""
            if result.get("crash_info"):
                crash_note = f" [{result['crash_info'][:50]}]"
            print(f"  Match {matches_completed} complete: Agent-1: {a1:g} | Agent-2: {a2:g}{crash_note}")
            
            # Log to file with detailed breakdown
            with open(log_file, "a") as f:
                f.write(f"Match {matches_completed}:\n")
                f.write(f"Agent-1 (Run {run_id1}): {a1:g} wins | Agent-2 (Run {run_id2}): {a2:g} wins{crash_note}\n")
                if "stats" in result:
                    s = result["stats"]
                    if s['normal'] > 0:
                        f.write(f"- {s['normal']} games finish with normal reasons\n")
                    if s['draw'] > 0:
                        f.write(f"- {s['draw']} games finish with draw due to hitting game turn limit\n")
                    if s['c1'] > 0:
                        f.write(f"- {s['c1']} games finish because of Crash during init (Agent-1)\n")
                    if s['c2'] > 0:
                        f.write(f"- {s['c2']} games finish because of Crash during init (Agent-2)\n")
                    
                    # Log random moves if any occurred
                    r1_total = s['r1_timeout'] + s['r1_crash'] + s['r1_invalid']
                    r2_total = s['r2_timeout'] + s['r2_crash'] + s['r2_invalid']
                    
                    if r1_total > 0 or r2_total > 0:
                        f.write("\nAs well as:\n")
                        if r1_total > 0:
                            f.write(f"- {r1_total} times random move selected for Agent-1\n")
                            f.write(f"  (Timeout: {s['r1_timeout']}, Crash during make_move: {s['r1_crash']}, Invalid: {s['r1_invalid']})\n")
                        if r2_total > 0:
                            f.write(f"- {r2_total} times random move selected for Agent-2\n")
                            f.write(f"  (Timeout: {s['r2_timeout']}, Crash during make_move: {s['r2_crash']}, Invalid: {s['r2_invalid']})\n")
                f.write("-" * 40 + "\n\n")
        else:
            print(f"  Match {matches_completed} FAILED: {result['error'][:50]}")
            with open(log_file, "a") as f:
                f.write(f"Match {matches_completed}:\n")
                f.write(f"Agent-1 (Run {run_id1}) vs Agent-2 (Run {run_id2}): FAILED - {result['error'][:80]}\n")
                f.write("-" * 40 + "\n\n")

    while matches_started < num_runs or pending_tasks:
        # Start new matches if we have pairs available (both queues must have items)
        while matches_started < num_runs:
            # Check both queues have items before popping (to avoid losing items)
            if model1_queue.empty() or model2_queue.empty():
                break
            
            response1 = model1_queue.get_nowait()
            response2 = model2_queue.get_nowait()

            run_id1, model1, content1 = response1
            run_id2, model2, content2 = response2

            match_id = min(run_id1, run_id2)
            matches_started += 1

            # Get short model names for display (split @ first, then take model name after /)
            m1_short = model1.split("@")[0].split("/")[-1]
            m2_short = model2.split("@")[0].split("/")[-1]
            print(
                f"\n  >>> MATCH {matches_started}/{num_runs} STARTED: "
                f"{m1_short} (run {run_id1}) vs {m2_short} (run {run_id2})"
            )

            # Log raw responses for this match
            with open(responses_file, "a") as f:
                f.write(f"--- MATCH {matches_started}: {m1_short} (run {run_id1}) vs {m2_short} (run {run_id2}) ---\n")
                f.write(f"PROMPT RESPONSE FROM {model1} (Run {run_id1}):\n")
                f.write(f"{content1}\n")
                f.write("-" * 40 + "\n")
                f.write(f"PROMPT RESPONSE FROM {model2} (Run {run_id2}):\n")
                f.write(f"{content2}\n")
                f.write("=" * 80 + "\n\n")

            # Extract agent code
            agent1_code, imports1 = extract_agent_code(content1, "BattleshipAgent_1")
            agent2_code, imports2 = extract_agent_code(content2, "BattleshipAgent_2")

            if not agent1_code:
                logger.error("Failed to extract agent from model1 (run %d)", run_id1)
                result = {
                    "match_id": matches_started + 1,
                    "agent1_run_id": run_id1,
                    "agent2_run_id": run_id2,
                    "success": True,
                    "agent1_wins": 0,
                    "agent2_wins": NUM_ROUNDS_PER_BATTLESHIP_MATCH,
                    "error": None,
                    "crash_info": "Agent-1 failed extraction",
                }
                results_list.append(result)
                matches_completed += 1
                msg = f"Agent-1: 0 | Agent-2: {NUM_ROUNDS_PER_BATTLESHIP_MATCH:g} [Agent-1 code extraction failed]"
                print(f"  Match {matches_completed} complete: {msg}")
                with open(log_file, "a") as f:
                    f.write(f"Match {matches_completed}:\n")
                    f.write(f"Agent-1 (Run {run_id1}): 0 wins | Agent-2 (Run {run_id2}): {NUM_ROUNDS_PER_BATTLESHIP_MATCH:g} wins [Agent-1 code extraction failed]\n")
                    f.write("-" * 40 + "\n\n")
                continue

            if not agent2_code:
                logger.error("Failed to extract agent from model2 (run %d)", run_id2)
                result = {
                    "match_id": matches_started + 1,
                    "agent1_run_id": run_id1,
                    "agent2_run_id": run_id2,
                    "success": True,
                    "agent1_wins": NUM_ROUNDS_PER_BATTLESHIP_MATCH,
                    "agent2_wins": 0,
                    "error": None,
                    "crash_info": "Agent-2 failed extraction",
                }
                results_list.append(result)
                matches_completed += 1
                msg = f"Agent-1: {NUM_ROUNDS_PER_BATTLESHIP_MATCH:g} | Agent-2: 0 [Agent-2 code extraction failed]"
                print(f"  Match {matches_completed} complete: {msg}")
                with open(log_file, "a") as f:
                    f.write(f"Match {matches_completed}:\n")
                    f.write(f"Agent-1 (Run {run_id1}): {NUM_ROUNDS_PER_BATTLESHIP_MATCH:g} wins | Agent-2 (Run {run_id2}): 0 wins [Agent-2 code extraction failed]\n")
                    f.write("-" * 40 + "\n\n")
                continue

            # Combine imports
            all_imports = set(imports1.split("\n") + imports2.split("\n"))
            extra_imports = "\n".join(imp for imp in all_imports if imp.strip())

            # Validate syntax for both agents
            valid1, err1 = validate_agent_syntax(agent1_code, imports1)
            valid2, err2 = validate_agent_syntax(agent2_code, imports2)
            
            if not valid1 and not valid2:
                # Both invalid - mark as tie (0-0) with error
                result = {
                    "match_id": matches_started + 1,
                    "agent1_run_id": run_id1,
                    "agent2_run_id": run_id2,
                    "success": False,
                    "agent1_wins": 0,
                    "agent2_wins": 0,
                    "error": "Both agents have syntax errors",
                }
                results_list.append(result)
                matches_completed += 1
                print(f"  Match {matches_completed} FAILED: Both agents have syntax errors")
                with open(log_file, "a") as f:
                    f.write(f"Match {matches_completed}:\n")
                    f.write(f"Agent-1 (Run {run_id1}) vs Agent-2 (Run {run_id2}): FAILED - Both agents have syntax errors\n")
                    f.write("-" * 40 + "\n\n")
                continue
            elif not valid1:
                # Agent 1 has syntax error - Agent 2 wins all games
                result = {
                    "match_id": matches_started + 1,
                    "agent1_run_id": run_id1,
                    "agent2_run_id": run_id2,
                    "success": True,
                    "agent1_wins": 0,
                    "agent2_wins": NUM_ROUNDS_PER_BATTLESHIP_MATCH,
                    "error": None,
                    "crash_info": f"Agent-1 syntax error: {err1[:50]}",
                }
                results_list.append(result)
                matches_completed += 1
                print(f"  Match {matches_completed} complete: Agent-1: 0 | Agent-2: {NUM_ROUNDS_PER_BATTLESHIP_MATCH:g} [Agent-1 syntax error]")
                with open(log_file, "a") as f:
                    f.write(f"Match {matches_completed}:\n")
                    f.write(f"Agent-1 (Run {run_id1}): 0 wins | Agent-2 (Run {run_id2}): {NUM_ROUNDS_PER_BATTLESHIP_MATCH:g} wins [Agent-1 syntax error]\n")
                    f.write("-" * 40 + "\n\n")
                continue
            elif not valid2:
                # Agent 2 has syntax error - Agent 1 wins all games
                result = {
                    "match_id": matches_started + 1,
                    "agent1_run_id": run_id1,
                    "agent2_run_id": run_id2,
                    "success": True,
                    "agent1_wins": NUM_ROUNDS_PER_BATTLESHIP_MATCH,
                    "agent2_wins": 0,
                    "error": None,
                    "crash_info": f"Agent-2 syntax error: {err2[:50]}",
                }
                results_list.append(result)
                matches_completed += 1
                print(f"  Match {matches_completed} complete: Agent-1: {NUM_ROUNDS_PER_BATTLESHIP_MATCH:g} | Agent-2: 0 [Agent-2 syntax error]")
                with open(log_file, "a") as f:
                    f.write(f"Match {matches_completed}:\n")
                    f.write(f"Agent-1 (Run {run_id1}): {NUM_ROUNDS_PER_BATTLESHIP_MATCH:g} wins | Agent-2 (Run {run_id2}): 0 wins [Agent-2 syntax error]\n")
                    f.write("-" * 40 + "\n\n")
                continue

            # Build and run match
            game_code = build_game_code(agent1_code, agent2_code, extra_imports)
            task = asyncio.create_task(run_match_async(game_code, matches_started + 1, (run_id1, run_id2)))
            pending_tasks[task] = (run_id1, run_id2)

        # Wait for at least one task or queue item
        if pending_tasks:
            done, _ = await asyncio.wait(
                pending_tasks.keys(),
                timeout=0.1,
                return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                run_ids = pending_tasks.pop(task)
                await handle_completed_match(task, run_ids)
        else:
            # No pending tasks, wait for queue items
            await asyncio.sleep(0.1)


async def prompt_and_queue(
    api: ModelAPI,
    model_name: str,
    prompt: str,
    run_id: int,
    queue: asyncio.Queue,
) -> None:
    """Prompt a model and put the response in the queue."""
    result = await prompt_model(api, model_name, prompt, run_id)
    await queue.put(result)


def print_results(model1: str, model2: str, all_results: list[dict]) -> None:
    """Print the match results in a formatted way."""
    print("\n" + "=" * 60)
    print("BATTLESHIP MATCH RESULTS")
    print("=" * 60)

    total_agent1 = 0
    total_agent2 = 0
    matches_agent1 = 0.0
    matches_agent2 = 0.0
    matches_tie = 0
    successful_runs = 0

    # Sort by completion order
    sorted_results = sorted(all_results, key=lambda x: x.get("match_id", 0))

    for result in sorted_results:
        m_id = result.get("match_id", "?")
        r1 = result.get("agent1_run_id", "?")
        r2 = result.get("agent2_run_id", "?")
        
        if result["success"]:
            a1 = result["agent1_wins"]
            a2 = result["agent2_wins"]
            total_agent1 += a1
            total_agent2 += a2
            successful_runs += 1
            
            if a1 > a2:
                matches_agent1 += 1
            elif a2 > a1:
                matches_agent2 += 1
            else:
                matches_agent1 += 0.5
                matches_agent2 += 0.5
                matches_tie += 1

            crash_note = ""
            if result.get("crash_info"):
                crash_note = f" [{result['crash_info'][:40]}]"
            print(f"  Match {m_id}: Agent-1 (Run {r1}): {a1:g} | Agent-2 (Run {r2}): {a2:g}{crash_note}")
        else:
            print(f"  Match {m_id}: Agent-1 (Run {r1}) vs Agent-2 (Run {r2}): FAILED - {result['error'][:50]}")

    print("-" * 60)

    if successful_runs > 0:
        total_games = total_agent1 + total_agent2
        print(f"\nTOTAL ({total_games:g} games across {successful_runs} runs):")
        tie_str = f" (Ties: {matches_tie})" if matches_tie > 0 else ""
        print(f"  Match Score: Agent-1: {matches_agent1:g} | Agent-2: {matches_agent2:g}{tie_str}")
        print(f"  Points: Agent-1 ({model1}): {total_agent1:g} ({total_agent1/total_games*100:.1f}%)")
        print(f"  Points: Agent-2 ({model2}): {total_agent2:g} ({total_agent2/total_games*100:.1f}%)")
        print("-" * 40)

        if total_agent1 > total_agent2:
            print(f"\nOVERALL WINNER: Agent-1 ({model1})")
        elif total_agent2 > total_agent1:
            print(f"\nOVERALL WINNER: Agent-2 ({model2})")
        else:
            print("\nOVERALL RESULT: TIE!")
    else:
        print("\nNo successful runs to report.")

    print("=" * 60)


async def run_battleship_match() -> None:
    """Main entry point for the Battleship match runner."""
    # Initialize API
    api = ModelAPI()

    # Select models
    model1, model2 = select_models(api)

    # Load prompt
    prompt = load_prompt()

    print("\n" + "-" * 60)
    print(f"Firing {NUM_RUNS * 2} parallel requests ({NUM_RUNS} per model)...")
    print("-" * 60)

    # Create queues for each model's responses
    model1_queue: asyncio.Queue = asyncio.Queue()
    model2_queue: asyncio.Queue = asyncio.Queue()

    # Results collector
    results_list: list[dict] = []

    # Create results directories and log files
    GAME_LOGS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_RESPONSES_DIR.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Clean model names for filename
    def get_short_name(m: str) -> str:
        # If it has @preset/fp8 etc., take the part before @
        base = m.split("@")[0]
        # Then take the last part of the path
        return base.split("/")[-1].replace(":", "_")

    model1_short = get_short_name(model1)
    model2_short = get_short_name(model2)
    
    log_filename = f"{timestamp}_{model1_short}_vs_{model2_short}.txt"
    log_file = GAME_LOGS_DIR / log_filename
    responses_file = MODEL_RESPONSES_DIR / log_filename

    # Write header to log file
    with open(log_file, "w") as f:
        f.write("Battleship Match Log\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Agent-1: {model1}\n")
        f.write(f"Agent-2: {model2}\n")
        f.write(f"Runs: {NUM_RUNS}\n")
        f.write(f"Games per run: {NUM_ROUNDS_PER_BATTLESHIP_MATCH}\n")
        f.write("-" * 40 + "\n")

    # Write header to responses log
    with open(responses_file, "w") as f:
        f.write("Battleship Model Responses Log\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Agent-1: {model1}\n")
        f.write(f"Agent-2: {model2}\n")
        f.write("-" * 40 + "\n\n")

    print(f"Game logs: {log_file}")
    print(f"Model responses: {responses_file}")

    # Create all prompt tasks (fire in parallel)
    prompt_tasks = []
    for run_id in range(1, NUM_RUNS + 1):
        task1 = asyncio.create_task(
            prompt_and_queue(api, model1, prompt, run_id, model1_queue)
        )
        task2 = asyncio.create_task(
            prompt_and_queue(api, model2, prompt, run_id, model2_queue)
        )
        prompt_tasks.extend([task1, task2])

    # Create coordinator task
    coordinator_task = asyncio.create_task(
        match_coordinator(
            model1_queue, model2_queue, results_list, NUM_RUNS,
            log_file, responses_file, model1, model2
        )
    )

    # Wait for all prompts to complete
    await asyncio.gather(*prompt_tasks)

    # Wait for coordinator to finish
    await coordinator_task

    # Write summary to log file
    with open(log_file, "a") as f:
        f.write("-" * 40 + "\n")
        successful = [r for r in results_list if r["success"]]
        total_a1 = sum(r["agent1_wins"] for r in successful)
        total_a2 = sum(r["agent2_wins"] for r in successful)
        
        m1 = 0.0
        m2 = 0.0
        mt = 0
        for r in successful:
            if r["agent1_wins"] > r["agent2_wins"]:
                m1 += 1
            elif r["agent2_wins"] > r["agent1_wins"]:
                m2 += 1
            else:
                m1 += 0.5
                m2 += 0.5
                mt += 1
        
        tie_str = f" (Ties: {mt})" if mt > 0 else ""
        f.write(f"MATCH SCORE: Agent-1: {m1:g} | Agent-2: {m2:g}{tie_str}\n")
        f.write(f"TOTAL POINTS: Agent-1={total_a1:g}, Agent-2={total_a2:g}\n")
        if total_a1 > total_a2:
            f.write(f"WINNER: Agent-1 ({model1})\n")
        elif total_a2 > total_a1:
            f.write(f"WINNER: Agent-2 ({model2})\n")
        else:
            f.write("RESULT: TIE\n")

    # Display results
    print_results(model1, model2, results_list)


def main() -> None:
    """Entry point."""
    print("\nConfiguration:")
    print(f"  NUM_RUNS: {NUM_RUNS}")
    print(f"  NUM_ROUNDS_PER_BATTLESHIP_MATCH: {NUM_ROUNDS_PER_BATTLESHIP_MATCH}")
    asyncio.run(run_battleship_match())


if __name__ == "__main__":
    main()
