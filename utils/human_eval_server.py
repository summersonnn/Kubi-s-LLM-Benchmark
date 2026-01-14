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

EVALUATION_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Human Evaluation Interface</title>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/vs2015.min.css">
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/highlight.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/languages/python.min.js"></script>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #0f0f0f;
            color: #fff;
            height: 100vh;
            overflow: hidden;
        }
        
        .header {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            height: 50px;
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            display: flex;
            align-items: center;
            justify-content: space-between;
            padding: 0 20px;
            z-index: 1000;
            border-bottom: 1px solid #333;
        }
        
        .progress {
            font-size: 14px;
            color: #888;
        }
        
        .progress-bar {
            width: 200px;
            height: 6px;
            background: #333;
            border-radius: 3px;
            overflow: hidden;
            margin-left: 15px;
        }
        
        .progress-fill {
            height: 100%;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            transition: width 0.3s ease;
        }
        
        .hint {
            font-size: 12px;
            color: #666;
        }
        
        .iframe-container {
            position: fixed;
            top: 50px;
            left: 0;
            right: 0;
            bottom: 0;
        }

        iframe {
            width: 100%;
            height: 100%;
            border: none;
            background: #fff;
        }

        .code-viewer-container {
            position: fixed;
            top: 50px;
            left: 0;
            right: 0;
            bottom: 0;
            display: none;
            background: #1e1e1e;
            flex-direction: column;
        }

        .code-viewer-container.active {
            display: flex;
        }

        .code-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            padding: 12px 20px;
            background: #252526;
            border-bottom: 1px solid #3c3c3c;
            flex-shrink: 0;
        }

        .code-title {
            font-size: 14px;
            color: #cccccc;
            font-weight: 500;
        }

        .copy-btn {
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 10px 20px;
            background: #0e639c;
            border: none;
            border-radius: 4px;
            color: #fff;
            cursor: pointer;
            font-size: 14px;
            font-weight: 500;
            transition: background 0.15s;
        }

        .copy-btn:hover {
            background: #1177bb;
        }

        .copy-btn.copied {
            background: #16825d;
        }

        .copy-btn::before {
            content: '';
            display: inline-block;
            width: 16px;
            height: 16px;
            background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='white' stroke-width='2'%3E%3Crect x='9' y='9' width='13' height='13' rx='2' ry='2'%3E%3C/rect%3E%3Cpath d='M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1'%3E%3C/path%3E%3C/svg%3E");
            background-size: contain;
        }

        .code-wrapper {
            flex: 1;
            overflow: auto;
            display: flex;
        }

        .line-numbers {
            padding: 16px 0;
            background: #1e1e1e;
            border-right: 1px solid #3c3c3c;
            text-align: right;
            user-select: none;
            flex-shrink: 0;
            min-width: 50px;
        }

        .line-numbers span {
            display: block;
            padding: 0 12px;
            font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', 'Monaco', monospace;
            font-size: 14px;
            line-height: 1.5;
            color: #858585;
        }

        .code-content {
            flex: 1;
            margin: 0;
            padding: 16px 20px;
            background: #1e1e1e;
            font-family: 'JetBrains Mono', 'Fira Code', 'Consolas', 'Monaco', monospace;
            font-size: 14px;
            line-height: 1.5;
            color: #d4d4d4;
            white-space: pre;
            overflow-x: auto;
            tab-size: 4;
        }

        .code-content.hljs {
            background: #1e1e1e;
            padding: 16px 20px;
        }
        
        .modal-overlay {
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0, 0, 0, 0.85);
            display: none;
            align-items: center;
            justify-content: center;
            z-index: 2000;
        }
        
        .modal-overlay.active {
            display: flex;
        }
        
        .modal {
            background: linear-gradient(135deg, #1e1e2e 0%, #2d2d44 100%);
            padding: 40px 50px;
            border-radius: 16px;
            text-align: center;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.5);
            border: 1px solid #444;
        }
        
        .modal h2 {
            margin-bottom: 25px;
            font-weight: 400;
            font-size: 24px;
        }
        
        .score-input {
            width: 120px;
            padding: 15px;
            font-size: 32px;
            text-align: center;
            background: #0f0f1a;
            border: 2px solid #444;
            border-radius: 10px;
            color: #fff;
            outline: none;
            transition: border-color 0.2s;
        }
        
        .score-input:focus {
            border-color: #3a7bd5;
        }
        
        .score-hint {
            margin-top: 15px;
            font-size: 13px;
            color: #666;
        }
        
        .btn {
            margin-top: 25px;
            padding: 12px 40px;
            font-size: 16px;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            border: none;
            border-radius: 8px;
            color: #fff;
            cursor: pointer;
            transition: transform 0.1s, box-shadow 0.2s;
        }
        
        .btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 20px rgba(58, 123, 213, 0.4);
        }
        
        .complete-modal {
            max-width: 500px;
        }
        
        .complete-modal h2 {
            font-size: 28px;
            margin-bottom: 20px;
        }
        
        .complete-modal p {
            color: #aaa;
            line-height: 1.6;
        }
    </style>
</head>
<body>
    <div class="header">
        <div style="display: flex; align-items: center;">
            <span class="progress" id="progressText">0 / 0</span>
            <div class="progress-bar">
                <div class="progress-fill" id="progressFill" style="width: 0%"></div>
            </div>
        </div>
        <span class="hint">Press <kbd>Space</kbd> + <kbd>0</kbd> to score</span>
    </div>
    
    <div class="iframe-container" id="iframeContainer">
        <iframe id="implFrame" src="about:blank"></iframe>
    </div>

    <div class="code-viewer-container" id="codeViewerContainer">
        <div class="code-header">
            <div class="code-title">Solution - Copy and test in your environment</div>
            <button class="copy-btn" id="copyBtn">Copy to Clipboard</button>
        </div>
        <div class="code-wrapper">
            <div class="line-numbers" id="lineNumbers"></div>
            <pre class="code-content" id="codeContent"></pre>
        </div>
    </div>

    <div class="modal-overlay" id="scoreModal">
        <div class="modal">
            <h2>Rate This Implementation</h2>
            <input type="number" class="score-input" id="scoreInput" min="0" max="10" step="any" placeholder="0-10">
            <p class="score-hint" id="scoreHint">Enter a score from 0 to max points</p>
            <button class="btn" id="nextBtn">Next</button>
        </div>
    </div>
    
    <div class="modal-overlay" id="completeModal">
        <div class="modal complete-modal">
            <h2>Evaluation Complete</h2>
            <p>All implementations have been scored. The results have been saved.</p>
            <p style="margin-top: 15px;">You may now close this browser tab.</p>
            <p style="margin-top: 10px; font-size: 12px; color: #666;">(Or press Ctrl+C in the terminal to stop the server)</p>
        </div>
    </div>

    <script>
        let currentImpl = null;  // Current implementation being evaluated
        let totalImpls = 0;
        let scoredCount = 0;
        let spacePressed = false;

        async function init() {
            await loadNext();
            window.focus();
        }

        async function loadNext() {
            console.log('loadNext called');
            const response = await fetch('/api/next');
            const data = await response.json();
            console.log('API response:', data);

            if (data.done) {
                // All implementations have been claimed
                showComplete();
                return;
            }

            // Use filename extension as fallback for old manifests without is_leetcode field
            var isLeetcode = data.is_leetcode || data.filename.endsWith('.txt');
            console.log('isLeetcode:', isLeetcode, 'filename:', data.filename);

            currentImpl = {
                id: data.impl_id,
                filename: data.filename,
                is_leetcode: isLeetcode,
                max_points: data.max_points || 1
            };
            
            // Update score input constraints based on max_points
            const scoreInput = document.getElementById('scoreInput');
            scoreInput.min = 0;
            scoreInput.max = currentImpl.max_points;
            scoreInput.placeholder = '0-' + currentImpl.max_points;
            document.getElementById('scoreHint').textContent = 
                'Enter a score from 0 to ' + currentImpl.max_points + ' (absolute points)';
            totalImpls = data.total;
            scoredCount = data.scored;

            // Fetch content first to detect type
            const codeResponse = await fetch(`/implementations/${data.filename}`);
            const content = await codeResponse.text();

            // Detect LeetCode by content: look for code patterns, not HTML
            const looksLikeCode = (
                content.includes('class Solution') ||
                content.includes('def ') ||
                content.includes('```python') ||
                content.includes('```java') ||
                content.includes('```cpp') ||
                (content.includes('```') && !content.includes('<html') && !content.includes('<!DOCTYPE'))
            );

            // Use code viewer if flagged as leetcode OR if content looks like code
            const useCodeViewer = isLeetcode || looksLikeCode;
            console.log('looksLikeCode:', looksLikeCode, 'useCodeViewer:', useCodeViewer);

            if (useCodeViewer) {
                console.log('Using code viewer');
                // Show code viewer, hide iframe
                document.getElementById('iframeContainer').style.display = 'none';
                document.getElementById('codeViewerContainer').classList.add('active');
                displayCode(content);
            } else {
                console.log('Using iframe');
                // Show iframe, hide code viewer
                document.getElementById('iframeContainer').style.display = 'block';
                document.getElementById('codeViewerContainer').classList.remove('active');

                // Load HTML in iframe
                document.getElementById('implFrame').src = `/implementations/${data.filename}`;
            }

            updateProgress(data.index + 1, data.total);
        }
        
        function updateProgress(current, total) {
            document.getElementById('progressText').textContent = `${current} / ${total}`;
            document.getElementById('progressFill').style.width = `${((current - 1) / total) * 100}%`;
        }

        function displayCode(rawCode) {
            console.log('displayCode called, rawCode length:', rawCode.length);
            console.log('First 300 chars:', rawCode.substring(0, 300));
            console.log('Has triple backticks:', rawCode.includes('```'));
            console.log('Has newlines:', rawCode.includes('\\n'));
            console.log('Has literal backslash-n:', rawCode.includes('\\\\n'));

            // Normalize line endings: convert \\r\\n and \\r to \\n
            let text = rawCode.replace(/\\r\\n/g, '\\n').replace(/\\r/g, '\\n');

            // Also handle literal backslash-n sequences (escaped newlines from JSON)
            text = text.replace(/\\\\n/g, '\\n');

            console.log('After normalization, has newlines:', text.includes('\\n'));

            let code = text;
            let detectedLang = 'python';  // Default for LeetCode

            // Try to extract code from markdown code blocks
            // Pattern: ```language followed by code and closing ```
            const codeBlockPattern = /```(\\w*)\\s*\\n?([\\s\\S]*?)```/g;
            const blocks = [];
            let match;

            while ((match = codeBlockPattern.exec(text)) !== null) {
                blocks.push({
                    lang: match[1] || 'python',
                    code: match[2]
                });
            }

            console.log('Found', blocks.length, 'code blocks');

            // Use the largest code block found
            if (blocks.length > 0) {
                let largest = blocks[0];
                for (const block of blocks) {
                    if (block.code.length > largest.code.length) {
                        largest = block;
                    }
                }
                code = largest.code.trim();
                detectedLang = largest.lang.toLowerCase() || 'python';
                console.log('Using largest block, lang:', detectedLang, 'length:', code.length);
            } else {
                // Fallback: if no code blocks found, try to find class/def pattern
                const classMatch = text.match(/^(class\\s+\\w+[\\s\\S]*)/m);
                const defMatch = text.match(/^(def\\s+\\w+[\\s\\S]*)/m);
                if (classMatch) {
                    code = classMatch[1].trim();
                    console.log('Fallback: found class pattern');
                } else if (defMatch) {
                    code = defMatch[1].trim();
                    console.log('Fallback: found def pattern');
                } else {
                    console.log('No code blocks or patterns found, showing raw text');
                }
            }

            // Update header with detected language
            const titleEl = document.querySelector('.code-title');
            const langDisplay = detectedLang.charAt(0).toUpperCase() + detectedLang.slice(1);
            titleEl.textContent = langDisplay + ' Solution - Copy and test in your environment';

            // Set code content
            const codeEl = document.getElementById('codeContent');
            codeEl.textContent = code;
            codeEl.removeAttribute('data-highlighted');
            codeEl.className = 'code-content language-' + detectedLang;

            // Apply syntax highlighting if highlight.js is available
            console.log('hljs available:', typeof hljs !== 'undefined');
            if (typeof hljs !== 'undefined') {
                try {
                    hljs.highlightElement(codeEl);
                    console.log('Syntax highlighting applied');
                } catch (e) {
                    console.error('Highlight.js error:', e);
                }
            }

            // Generate line numbers
            const lines = code.split('\\n');
            const lineNumbersEl = document.getElementById('lineNumbers');
            lineNumbersEl.innerHTML = lines.map(function(_, i) {
                return '<span>' + (i + 1) + '</span>';
            }).join('');
            console.log('Line numbers generated:', lines.length, 'lines');
        }
        
        function openScoreModal() {
            document.getElementById('scoreModal').classList.add('active');
            document.getElementById('scoreInput').value = '';
            document.getElementById('scoreInput').focus();
        }
        
        function closeScoreModal() {
            document.getElementById('scoreModal').classList.remove('active');
        }
        
        async function submitScore() {
            const input = document.getElementById('scoreInput');
            const score = parseFloat(input.value);
            
            const maxPoints = currentImpl ? currentImpl.max_points : 10;
            if (isNaN(score) || score < 0 || score > maxPoints) {
                input.style.borderColor = '#ff4444';
                return;
            }
            
            // Submit score for current implementation
            await fetch('/api/score', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    impl_id: currentImpl.id,
                    score: score
                })
            });
            
            closeScoreModal();
            
            // Get next implementation
            await loadNext();
        }
        
        function showComplete() {
            document.getElementById('completeModal').classList.add('active');
        }
        
        async function reloadCurrentImpl() {
            // Reload current implementation
            if (currentImpl) {
                if (currentImpl.is_leetcode) {
                    // Reload code content
                    const codeResponse = await fetch(`/implementations/${currentImpl.filename}`);
                    const code = await codeResponse.text();
                    displayCode(code);
                } else {
                    // Reload iframe
                    document.getElementById('implFrame').src = `/implementations/${currentImpl.filename}`;
                }
            }
        }

        function copyCode() {
            const codeContent = document.getElementById('codeContent').textContent;
            const copyBtn = document.getElementById('copyBtn');

            navigator.clipboard.writeText(codeContent).then(() => {
                // Visual feedback
                copyBtn.textContent = 'Copied!';
                copyBtn.classList.add('copied');

                setTimeout(() => {
                    copyBtn.textContent = 'Copy to Clipboard';
                    copyBtn.classList.remove('copied');
                }, 2000);
            }).catch(err => {
                // Fallback for older browsers or when clipboard API fails
                const textarea = document.createElement('textarea');
                textarea.value = codeContent;
                textarea.style.position = 'fixed';
                textarea.style.opacity = '0';
                document.body.appendChild(textarea);
                textarea.select();
                try {
                    document.execCommand('copy');
                    copyBtn.textContent = 'Copied!';
                    copyBtn.classList.add('copied');
                    setTimeout(() => {
                        copyBtn.textContent = 'Copy to Clipboard';
                        copyBtn.classList.remove('copied');
                    }, 2000);
                } catch (e) {
                    copyBtn.textContent = 'Copy failed';
                }
                document.body.removeChild(textarea);
            });
        }
        
        // Key event handling - using window level to capture even when iframe has focus
        window.addEventListener('keydown', (e) => {
            if (e.code === 'Space') {
                spacePressed = true;
            }
            
            // Support both regular 0 and Numpad 0
            if (spacePressed && (e.code === 'Digit0' || e.code === 'Numpad0')) {
                e.preventDefault();
                openScoreModal();
            }
            
            // R to reset/reload current implementation (both regular and when Space not pressed)
            if ((e.code === 'KeyR') && !document.getElementById('scoreModal').classList.contains('active')) {
                e.preventDefault();
                reloadCurrentImpl();
            }
            
            // Enter to submit in modal
            if ((e.code === 'Enter' || e.code === 'NumpadEnter') && document.getElementById('scoreModal').classList.contains('active')) {
                e.preventDefault();
                submitScore();
            }
            
            // Escape to close score modal without submitting
            if (e.code === 'Escape' && document.getElementById('scoreModal').classList.contains('active')) {
                e.preventDefault();
                closeScoreModal();
            }
        }, true); // Use capture phase to get events before iframe
        
        window.addEventListener('keyup', (e) => {
            if (e.code === 'Space') {
                spacePressed = false;
            }
        }, true);
        
        // Also listen on iframe for focus restoration
        document.getElementById('implFrame').addEventListener('load', () => {
            try {
                const iframeDoc = document.getElementById('implFrame').contentWindow;
                iframeDoc.addEventListener('keydown', (e) => {
                    // Forward R and Space+0 to parent
                    if (e.code === 'Space') {
                        spacePressed = true;
                    }
                    if (spacePressed && (e.code === 'Digit0' || e.code === 'Numpad0')) {
                        e.preventDefault();
                        openScoreModal();
                    }
                    if (e.code === 'KeyR') {
                        e.preventDefault();
                        reloadCurrentImpl();
                    }
                }, true);
                iframeDoc.addEventListener('keyup', (e) => {
                    if (e.code === 'Space') {
                        spacePressed = false;
                    }
                }, true);
            } catch (err) {
                // Cross-origin iframe, can't add listeners
            }
        });
        
        document.getElementById('nextBtn').addEventListener('click', submitScore);
        document.getElementById('copyBtn').addEventListener('click', copyCode);

        init();
    </script>
</body>
</html>
"""


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
            # Atomically claim the next implementation
            with EvaluationHandler.queue_lock:
                manifest_path = os.path.join(EvaluationHandler.session_dir, "manifest.json")
                with open(manifest_path, "r") as f:
                    manifest = json.load(f)
                
                shuffled_order = manifest["shuffled_order"]
                total = len(shuffled_order)
                
                if EvaluationHandler.queue_index >= total:
                    # No more implementations
                    response = {"done": True, "total": total, "scored": total}
                else:
                    impl_id = shuffled_order[EvaluationHandler.queue_index]
                    impl = next(i for i in manifest["implementations"] if i["id"] == impl_id)
                    current_idx = EvaluationHandler.queue_index
                    EvaluationHandler.queue_index += 1

                    # Count scored so far
                    scored = sum(1 for i in manifest["implementations"] if i.get("score") is not None)

                    # Detect LeetCode by filename extension as fallback for old manifests
                    is_leetcode = impl.get("is_leetcode", impl["filename"].endswith(".txt"))

                    response = {
                        "done": False,
                        "impl_id": impl_id,
                        "filename": impl["filename"],
                        "index": current_idx,
                        "total": total,
                        "scored": scored,
                        "is_leetcode": is_leetcode,
                        "max_points": impl.get("max_points", 1)
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
                should_integrate = (all_scored 
                                    and EvaluationHandler.auto_integrate 
                                    and not EvaluationHandler.integration_done)
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
            print(f"Run manually: uv run integrate_human_scores.py {EvaluationHandler.session_dir}")


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
