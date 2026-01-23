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
            self.use_trace = os.environ.get("ENABLE_CLAUDE_TRACE", "true").strip().lower() in {"1", "true", "yes"}
            self.enable_stream_output = os.environ.get("ENABLE_CLAUDE_STREAM_OUTPUT", "true").strip().lower() in {"1", "true", "yes"}
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

            # Build command with optional model parameter
            prompt_mode = os.environ.get("CODE_SWE_PROMPT_MODE", "").strip().lower()
            permission_mode_flag = (
                ["--permission-mode", "plan"] if prompt_mode == "plan" else []
            )
            if self.use_trace:
                trace_log_name = trajectory_name or "trace"
                cmd = [
                    "claude-trace",
                    "--log",
                    trace_log_name,
                    "--include-all-requests",
                    "--run-with",
                ]
                if self.enable_stream_output:
                    cmd.extend(
                        [
                            "--output-format",
                            "stream-json",
                            "--include-partial-messages",
                            "--verbose",
                        ]
                    )
                if model:
                    cmd.extend(["--model", model])
                cmd.extend(
                    [
                        "--dangerously-skip-permissions",
                        *permission_mode_flag,
                        "--print",
                        prompt,
                    ]
                )
            else:
                cmd = [
                    "claude",
                    "--dangerously-skip-permissions",
                    *permission_mode_flag,
                    "--print",
                    prompt,
                ]
                if model:
                    cmd.extend(["--model", model])

            print(f"Running claude command (cwd={cwd}): {' '.join(cmd)}")

            # Execute claude command
            result = subprocess.run(
                cmd,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            if result.stdout:
                print(result.stdout.strip())

            # Restore original directory
            os.chdir(original_cwd)

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

    def extract_file_changes(self, response: str) -> List[Dict[str, str]]:
        """Extract file changes from Claude's response."""
        # This will be implemented by patch_extractor.py
        # For now, return empty list
        return []
