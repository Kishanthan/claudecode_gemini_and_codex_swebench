import os
import subprocess
import shutil
import shlex
import threading
import time
from pathlib import Path
import json
from typing import Dict, List, Optional


DEFAULT_ROUTER_TIMEOUT_SECONDS = 1800
DEFAULT_ROUTER_MAX_ITERATIONS: Optional[int] = None  # Disabled by default
DEFAULT_ROUTER_MAX_RETRIES: int = 3
DEFAULT_ROUTER_TRAJECTORIES_DIR = Path.home() / ".claude-code-router" / "logs" / "trajectories"


def _get_int_env(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value)
    except ValueError:
        return default


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

    def execute_code_cli(
        self,
        prompt: str,
        cwd: str,
        model: Optional[str] = None,
        trajectory_name: Optional[str] = None,
    ) -> Dict[str, any]:
        """Execute Claude Code via the router (`ccr code`) and capture the response.

        A `/model provider,model` line is prepended when `model` is provided so
        the router can switch targets.

        Controls (via env vars):
        - `ROUTER_TIMEOUT_SECONDS`: hard wall-time timeout for the `ccr code` process
          (default {DEFAULT_ROUTER_TIMEOUT_SECONDS}).
        - `ROUTER_MAX_ITERATIONS`: if set, stop after N router->model iterations (counts `stage=="request"`
          lines in the trajectory log) by terminating the process (default: disabled).
        - `ROUTER_TRAJECTORIES_DIR`: where to watch for `session-*.jsonl`
          (default `{DEFAULT_ROUTER_TRAJECTORIES_DIR}`).
        - `ROUTER_CODE_FLAGS`: extra flags for `ccr code` (default `--dangerously-skip-permissions`).
        - `ROUTER_TRAJECTORY_NAME`: optional trajectory name to help select the right session log (prefix match)
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

            traj_name = trajectory_name or os.environ.get("ROUTER_TRAJECTORY_NAME", "").strip() or None

            cmd = ["ccr", "code", "--print", *extra_flags]

            # Default time/iteration limits. Add retries on timeout.
            timeout_s = _get_int_env("ROUTER_TIMEOUT_SECONDS", DEFAULT_ROUTER_TIMEOUT_SECONDS)
            max_iterations_str = os.environ.get("ROUTER_MAX_ITERATIONS", "").strip()
            max_iterations = int(max_iterations_str) if max_iterations_str else DEFAULT_ROUTER_MAX_ITERATIONS
            max_retries = _get_int_env("ROUTER_MAX_RETRIES", DEFAULT_ROUTER_MAX_RETRIES)
            if max_retries < 1:
                max_retries = 1

            trajectories_dir = Path(
                os.environ.get(
                    "ROUTER_TRAJECTORIES_DIR",
                    str(DEFAULT_ROUTER_TRAJECTORIES_DIR),
                )
            )

            def run_once() -> Dict[str, any]:
                start_ts = time.time()
                existing_sessions: set[str] = set()
                glob_pattern = f"{traj_name}*session-*.jsonl" if traj_name else "*session-*.jsonl"
                if trajectories_dir.exists():
                    existing_sessions = {p.name for p in trajectories_dir.glob(glob_pattern)}

                proc = subprocess.Popen(
                    cmd,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )

                assert proc.stdin is not None
                proc.stdin.write(routed_prompt)
                proc.stdin.close()

                stdout_chunks: List[str] = []
                stderr_chunks: List[str] = []

                def _drain(stream, sink: List[str]):
                    try:
                        for line in stream:
                            sink.append(line)
                    finally:
                        try:
                            stream.close()
                        except Exception:
                            pass

                assert proc.stdout is not None and proc.stderr is not None
                t_out = threading.Thread(target=_drain, args=(proc.stdout, stdout_chunks), daemon=True)
                t_err = threading.Thread(target=_drain, args=(proc.stderr, stderr_chunks), daemon=True)
                t_out.start()
                t_err.start()

                session_file: Optional[Path] = None
                file_offset = 0
                request_count = 0

                def _discover_session_file() -> Optional[Path]:
                    if not trajectories_dir.exists():
                        return None
                    candidates = [p for p in trajectories_dir.glob(glob_pattern) if p.name not in existing_sessions]
                    if not candidates:
                        all_sessions = list(trajectories_dir.glob(glob_pattern))
                        all_sessions.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                        for p in all_sessions:
                            if p.stat().st_mtime >= start_ts - 1:
                                return p
                        return None
                    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    return candidates[0]

                def _update_request_count(path: Path) -> None:
                    nonlocal file_offset, request_count
                    try:
                        with path.open("r", encoding="utf-8") as f:
                            f.seek(file_offset)
                            for line in f:
                                try:
                                    obj = json.loads(line)
                                except Exception:
                                    continue
                                if obj.get("stage") == "request":
                                    request_count += 1
                            file_offset = f.tell()
                    except FileNotFoundError:
                        return

                stop_reason: Optional[str] = None
                deadline = time.monotonic() + timeout_s
                while proc.poll() is None:
                    now = time.monotonic()
                    if now >= deadline:
                        stop_reason = "timeout"
                        proc.kill()
                        break

                    if max_iterations is not None and max_iterations > 0:
                        if session_file is None:
                            session_file = _discover_session_file()
                        if session_file is not None:
                            _update_request_count(session_file)
                            if request_count >= max_iterations:
                                stop_reason = f"max_iterations({max_iterations})"
                                proc.terminate()
                                break

                    time.sleep(0.5)

                if stop_reason and stop_reason.startswith("max_iterations"):
                    try:
                        proc.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        proc.kill()

                returncode = proc.wait(timeout=10) if proc.poll() is None else proc.returncode
                t_out.join(timeout=5)
                t_err.join(timeout=5)

                stdout = "".join(stdout_chunks)
                stderr = "".join(stderr_chunks)
                result = {
                    "success": returncode == 0,
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": returncode or 0,
                    "stop_reason": stop_reason,
                }
                if stop_reason == "timeout":
                    result["stderr"] = (stderr + "\nCommand timed out").strip()
                    result["success"] = False
                    result["returncode"] = -1
                if stop_reason and stop_reason.startswith("max_iterations"):
                    result["stderr"] = (stderr + f"\nStopped after {stop_reason}").strip()
                    result["success"] = False
                    result["returncode"] = -1
                return result

            last_result: Dict[str, any] = {}
            for attempt in range(max_retries):
                last_result = run_once()
                last_result["attempt"] = attempt + 1
                stop_reason = last_result.get("stop_reason")
                if stop_reason == "timeout" and attempt + 1 < max_retries:
                    continue
                return last_result

            return last_result

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
