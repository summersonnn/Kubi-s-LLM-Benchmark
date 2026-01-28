"""
HTTP server for blind human evaluation of HTML/CSS/JS implementations.
Serves shuffled implementations via iframe and collects scores without revealing model origins.
"""

import argparse
import json
import os
import sys
import threading
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# Load HTML template
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "templates", "human_eval.html")
with open(TEMPLATE_PATH, "r", encoding="utf-8") as f:
    EVALUATION_HTML = f.read()


class EvaluationHandler(SimpleHTTPRequestHandler):
    """Custom HTTP handler for the evaluation interface."""

    # Shared state for work queue (class-level, thread-safe)
    queue_lock = threading.Lock()
    queue_index = 0  # Next implementation index to serve
    session_dir = None
    total_implementations = 0
    auto_integrate = True  # Whether to auto-integrate scores when complete
    integration_done = False  # Prevent duplicate integration calls

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def init_queue(cls, session_dir: str):
        """Initialize the shared queue state, resuming from first unscored implementation."""
        cls.session_dir = session_dir
        manifest_path = os.path.join(session_dir, "manifest.json")
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        shuffled_order = manifest["shuffled_order"]
        implementations = {impl["id"]: impl for impl in manifest["implementations"]}
        
        cls.total_implementations = len(shuffled_order)
        
        # Find first unscored implementation in shuffled order
        cls.queue_index = 0
        for i, impl_id in enumerate(shuffled_order):
            if implementations[impl_id].get("score") is None:
                cls.queue_index = i
                break
        else:
            # All scored
            cls.queue_index = len(shuffled_order)
        
        scored_count = sum(1 for impl in manifest["implementations"] if impl.get("score") is not None)
        if scored_count > 0:
            print(f"Resuming evaluation: {scored_count}/{cls.total_implementations} already scored")

    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path == "/" or parsed.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            self.wfile.write(EVALUATION_HTML.encode())

        elif parsed.path == "/api/next":
            # Atomically claim the next unscored implementation
            with EvaluationHandler.queue_lock:
                manifest_path = os.path.join(EvaluationHandler.session_dir, "manifest.json")
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                
                shuffled_order = manifest["shuffled_order"]
                implementations = {impl["id"]: impl for impl in manifest["implementations"]}
                total = len(shuffled_order)
                
                # Find next unscored implementation starting from queue_index
                impl = None
                current_idx = None
                while EvaluationHandler.queue_index < total:
                    impl_id = shuffled_order[EvaluationHandler.queue_index]
                    candidate = implementations[impl_id]
                    if candidate.get("score") is None:
                        # Found an unscored implementation
                        impl = candidate
                        current_idx = EvaluationHandler.queue_index
                        EvaluationHandler.queue_index += 1
                        break
                    # Skip already-scored implementations
                    EvaluationHandler.queue_index += 1
                
                if impl is None:
                    # No more unscored implementations
                    scored = sum(1 for i in manifest["implementations"] if i.get("score") is not None)
                    response = {"done": True, "total": total, "scored": scored}
                else:
                    # Count scored so far
                    scored = sum(1 for i in manifest["implementations"] if i.get("score") is not None)

                    # Detect LeetCode by filename extension as fallback for old manifests
                    is_leetcode = impl.get("is_leetcode", impl["filename"].endswith(".txt"))

                    question_code = impl.get("question_code", "Unknown")
                    
                    # Log to terminal which question is being evaluated
                    print(f"User is evaluating the question: {question_code} (Implementation: {impl['id']})")

                    response = {
                        "done": False,
                        "impl_id": impl["id"],
                        "filename": impl["filename"],
                        "index": current_idx,
                        "total": total,
                        "scored": scored,
                        "is_leetcode": is_leetcode,
                        "max_points": impl.get("max_points", 1),
                        "question_code": question_code
                    }
            
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        elif parsed.path == "/api/status":
            # Get current progress
            manifest_path = os.path.join(EvaluationHandler.session_dir, "manifest.json")
            with open(manifest_path, "r") as f:
                manifest = json.load(f)
            
            total = len(manifest["shuffled_order"])
            scored = sum(1 for i in manifest["implementations"] if i.get("score") is not None)
            
            response = {"total": total, "scored": scored, "done": scored >= total}
            
            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(response).encode())

        elif parsed.path.startswith("/implementations/"):
            filename = os.path.basename(parsed.path)
            filepath = os.path.join(EvaluationHandler.session_dir, filename)
            if os.path.exists(filepath):
                with open(filepath, "r") as f:
                    content = f.read()
                
                # Extract last code block if file contains markdown fences
                import re
                code_block_pattern = r'```(\w*)\s*\n?([\s\S]*?)```'
                matches = list(re.finditer(code_block_pattern, content))
                
                if matches:
                    # Use the last code block found
                    last_match = matches[-1]
                    extracted_code = last_match.group(2).strip()
                    content = extracted_code
                
                self.send_response(200)
                # Determine content type based on file extension
                if filename.endswith(".txt"):
                    self.send_header("Content-type", "text/plain; charset=utf-8")
                else:
                    self.send_header("Content-type", "text/html")
                self.end_headers()
                self.wfile.write(content.encode())
            else:
                self.send_error(404, "Implementation not found")

        else:
            self.send_error(404, "Not found")


    def do_POST(self):
        parsed = urlparse(self.path)

        if parsed.path == "/api/score":
            content_length = int(self.headers["Content-Length"])
            post_data = self.rfile.read(content_length)
            data = json.loads(post_data.decode())
            
            impl_id = data["impl_id"]
            score = data["score"]

            with EvaluationHandler.queue_lock:
                manifest_path = os.path.join(EvaluationHandler.session_dir, "manifest.json")
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)

                # Update score for this implementation
                for impl in manifest["implementations"]:
                    if impl["id"] == impl_id:
                        impl["score"] = score
                        break

                # Check if all scored
                all_scored = all(impl.get("score") is not None 
                               for impl in manifest["implementations"])
                
                if all_scored:
                    manifest["scores_collected"] = True

                with open(manifest_path, "w") as f:
                    json.dump(manifest, f, indent=2)

                # Trigger auto-integration if enabled and not already done
                # Only integrate if results_file is present (meaning main benchmark finished writing results)
                should_integrate = (all_scored 
                                    and EvaluationHandler.auto_integrate 
                                    and not EvaluationHandler.integration_done
                                    and manifest.get("results_file"))
                if should_integrate:
                    EvaluationHandler.integration_done = True

            # Run integration outside the lock to avoid blocking
            if should_integrate:
                self._run_integration()

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')

        else:
            self.send_error(404, "Not found")

    def log_message(self, format, *args):
        """Suppress default logging."""
        pass

    def _run_integration(self):
        """Runs the score integration after all implementations are scored."""
        try:
            # Add parent directory to path for proper imports when running from utils/
            import sys
            script_dir = os.path.dirname(os.path.abspath(__file__))
            parent_dir = os.path.dirname(script_dir)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)
            
            from utils.integrate_human_scores import integrate_scores
            print("\n" + "=" * 60)
            print("ALL IMPLEMENTATIONS SCORED - INTEGRATING RESULTS")
            print("=" * 60)
            print(f"Session directory: {EvaluationHandler.session_dir}")
            integrate_scores(EvaluationHandler.session_dir)
            print("\nScore integration complete!")
            print("You may now close this browser tab and stop the server (Ctrl+C).")
            print("=" * 60 + "\n")
        except Exception as e:
            import traceback
            print(f"\nError during integration: {e}")
            traceback.print_exc()
            print(f"Run manually: uv run python -m utils.integrate_human_scores {EvaluationHandler.session_dir}")


def run_server(session_dir: str, port: int = 8765, num_parallel: int = 1, auto_integrate: bool = True) -> None:
    """
    Starts the evaluation HTTP server.
    If num_parallel > 1, opens multiple browser windows that share a work queue.
    """
    
    # Initialize the shared queue and integration settings
    EvaluationHandler.init_queue(session_dir)
    EvaluationHandler.auto_integrate = auto_integrate
    EvaluationHandler.integration_done = False

    # Custom server class with port reuse enabled
    class ReusableHTTPServer(HTTPServer):
        allow_reuse_address = True
    
    server = ReusableHTTPServer(("localhost", port), EvaluationHandler)
    
    print(f"\n{'=' * 60}")
    print("HUMAN EVALUATION SERVER STARTED")
    print(f"{'=' * 60}")
    
    if num_parallel == 1:
        print(f"Open in browser: http://localhost:{port}")
    else:
        print(f"Parallel evaluation mode: {num_parallel} windows")
        print(f"All windows share a common work queue")
    
    print("Press Ctrl+C to stop the server after evaluation is complete.")
    print(f"{'=' * 60}\n")

    # Start server in background thread
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    
    # Give server time to start
    time.sleep(0.5)
    
    # Open browser windows
    base_url = f"http://localhost:{port}"
    for i in range(num_parallel):
        print(f"Opening evaluation window {i + 1}/{num_parallel}")
        webbrowser.open(base_url)
        time.sleep(0.3)

    # Check if all scores are already collected (resume with completed evaluation)
    manifest_path = os.path.join(session_dir, "manifest.json")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    all_scored = all(impl.get("score") is not None for impl in manifest["implementations"])
    if all_scored and auto_integrate and not EvaluationHandler.integration_done:
        print("\nAll implementations already scored - triggering integration...")
        EvaluationHandler.integration_done = True
        # Create a dummy handler to call integration
        dummy = EvaluationHandler.__new__(EvaluationHandler)
        dummy._run_integration()

    try:
        # Keep main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Human evaluation server for blind HTML/CSS/JS implementation scoring"
    )
    parser.add_argument(
        "session_dir",
        help="Directory containing manifest.json and implementation files"
    )
    parser.add_argument(
        "--parallel",
        type=int,
        metavar="X",
        default=1,
        help="Number of parallel evaluation windows (default: 1)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Server port (default: 8765)"
    )
    parser.add_argument(
        "--no_int",
        action="store_true",
        help="Disable automatic score integration after evaluation completes"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.session_dir):
        print(f"Error: Session directory not found: {args.session_dir}")
        sys.exit(1)
    
    if args.parallel < 1:
        print("Error: --parallel must be at least 1")
        sys.exit(1)

    run_server(args.session_dir, port=args.port, num_parallel=args.parallel, auto_integrate=not args.no_int)
