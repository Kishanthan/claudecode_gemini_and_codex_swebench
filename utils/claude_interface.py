import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv

load_dotenv()

class ClaudeCodeInterface:
    """Interface for interacting with Claude Code CLI."""

    def __init__(self):
        """Ensure the Claude CLI is available on the system."""
        try:
            self.use_trace = os.environ.get("CLAUDE_TRACE", "true").strip().lower() in {"1", "true", "yes"}
            cli = "claude-trace" if self.use_trace else "claude"
            result = subprocess.run([
                cli, "--version"
            ], capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"Claude CLI not found. Please ensure '{cli}' is installed and in PATH"
                )
        except FileNotFoundError:
            raise RuntimeError(
                "Claude CLI not found. Please ensure 'claude' or 'claude-trace' is installed and in PATH"
            )

    def execute_code_cli(
        self,
        prompt: str,
        cwd: str,
        model: str = None,
        trajectory_name: Optional[str] = None,
    ) -> Dict[str, any]:
        """Execute Claude Code via CLI and capture the response.

        Args:
            prompt: The prompt to send to Claude.
            cwd: Working directory to execute in.
            model: Optional model to use (e.g., 'opus-4.1', 'sonnet-3.7').
        """
        try:
            # Save the current directory
            original_cwd = os.getcwd()

            # Change to the working directory
            os.chdir(cwd)

            instance_trace_dir = None
            if self.use_trace:
                trace_dir = os.environ.get(
                    "CLAUDE_TRACE_DIR",
                    str(Path.home() / ".claude-trace"),
                )
                if trace_dir:
                    trace_path = Path(trace_dir).resolve()
                    if trajectory_name:
                        instance_trace_dir = trace_path / trajectory_name
                        instance_trace_dir.mkdir(parents=True, exist_ok=True)
                        trace_path = instance_trace_dir
                    else:
                        trace_path.mkdir(parents=True, exist_ok=True)

            # Build command with optional model parameter
            if self.use_trace:
                cmd = [
                    "claude-trace",
                    "--include-all-requests",
                    "--run-with",
                    "--dangerously-skip-permissions",
                    "--print",
                    prompt,
                ]
                if model:
                    cmd.extend(["--model", model])
            else:
                cmd = ["claude", "--dangerously-skip-permissions", "--print", prompt]
                if model:
                    cmd.extend(["--model", model])

            # Execute claude command
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            # Restore original directory
            os.chdir(original_cwd)

            if instance_trace_dir and trajectory_name:
                self._rename_latest_trace(instance_trace_dir, trajectory_name)

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

        except subprocess.TimeoutExpired:
            os.chdir(original_cwd)
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out after 10 minutes",
                "returncode": -1,
            }
        except Exception as e:
            os.chdir(original_cwd)
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
            }

    def _rename_latest_trace(self, trace_dir: Path, trajectory_name: str) -> None:
        """Rename the latest claude-trace log files to include the instance id."""
        try:
            candidates = sorted(trace_dir.glob("log-*.jsonl"))
            if candidates:
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                target = trace_dir / f"{trajectory_name}.jsonl"
                if latest != target:
                    latest.rename(target)
            html_candidates = sorted(trace_dir.glob("log-*.html"))
            if html_candidates:
                latest_html = max(html_candidates, key=lambda p: p.stat().st_mtime)
                html_target = trace_dir / f"{trajectory_name}.html"
                if latest_html != html_target:
                    latest_html.rename(html_target)
        except Exception:
            return

    def extract_file_changes(self, response: str) -> List[Dict[str, str]]:
        """Extract file changes from Claude's response."""
        # This will be implemented by patch_extractor.py
        # For now, return empty list
        return []
