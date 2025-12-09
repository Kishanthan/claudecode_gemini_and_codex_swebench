import os
import subprocess
import shutil
import shlex
from typing import Dict, List, Optional


class RouterCodeInterface:
    """Interface for invoking Claude Code through claude-code-router (ccr)."""

    def __init__(self):
        """Ensure the router CLI is available."""
        ccr_path = shutil.which("ccr")
        if not ccr_path:
            raise RuntimeError(
                "Claude Code Router CLI not found. Please ensure 'ccr' is installed and in PATH."
            )
        result = subprocess.run([ccr_path, "-v"], capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"Claude Code Router CLI check failed (rc={result.returncode}). Stderr: {result.stderr}"
            )

    def execute_code_cli(self, prompt: str, cwd: str, model: Optional[str] = None) -> Dict[str, any]:
        """Execute Claude Code via the router (`ccr code`) and capture the response.

        A `/model provider,model` line is prepended when `model` is provided so
        the router can switch targets.
        """
        original_cwd = os.getcwd()
        try:
            os.chdir(cwd)

            routed_prompt = prompt
            if model:
                routed_prompt = f"/model {model}\n\n{prompt}"

            # Run through router-managed Claude Code; --print for non-interactive use.
            # Default to bypass permissions; override via ROUTER_CODE_FLAGS if needed.
            extra_flags = ["--dangerously-skip-permissions"]
            flags_str = os.environ.get("ROUTER_CODE_FLAGS")
            if flags_str is not None:
                extra_flags = shlex.split(flags_str) if flags_str else []

            cmd = ["ccr", "code", "--print", *extra_flags]

            result = subprocess.run(
                cmd,
                input=routed_prompt,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )

            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode,
            }

        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "stdout": "",
                "stderr": "Command timed out after 10 minutes",
                "returncode": -1,
            }
        except Exception as e:
            return {
                "success": False,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1,
            }
        finally:
            os.chdir(original_cwd)

    def extract_file_changes(self, response: str) -> List[Dict[str, str]]:
        """Placeholder to mirror other interfaces; patch extraction handled elsewhere."""
        return []
