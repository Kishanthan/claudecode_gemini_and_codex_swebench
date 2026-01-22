#!/usr/bin/env python3
"""
SWE-bench agent capable of using Claude Code or Codex backends.
"""

import argparse
import json
import os
import sys
import subprocess
import tempfile
import shutil
from datetime import datetime
from typing import List, Dict, Optional
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm
import jsonlines

from utils.claude_interface import ClaudeCodeInterface
from utils.codex_interface import CodexCodeInterface
from utils.gemini_interface import GeminiCodeInterface
from utils.router_interface import RouterCodeInterface
from utils.prompt_formatter import PromptFormatter
from utils.patch_extractor import PatchExtractor
from utils.model_registry import get_model_name


DEFAULT_BACKEND = os.environ.get("CODE_SWE_BACKEND", "claude")


class CodeSWEAgent:
    """Main agent for running SWE-bench using different code models."""

    def __init__(
        self,
        prompt_template: Optional[str] = None,
        model: Optional[str] = None,
        backend: str = DEFAULT_BACKEND,
    ):
        self.backend = (backend or DEFAULT_BACKEND).lower()
        if self.backend == "codex":
            self.interface = CodexCodeInterface()
        elif self.backend == "gemini":
            self.interface = GeminiCodeInterface()
        elif self.backend == "router":
            self.interface = RouterCodeInterface()
        else:
            self.backend = "claude"
            self.interface = ClaudeCodeInterface()

        plan_template = os.environ.get("CODE_SWE_PLAN_TEMPLATE")
        self.prompt_mode = os.environ.get("CODE_SWE_PROMPT_MODE", "fix").strip().lower()
        self.prompt_formatter = PromptFormatter(
            prompt_template_path=prompt_template,
            plan_template_path=plan_template,
        )
        self.patch_extractor = PatchExtractor()
        self.base_dir = Path.cwd()
        output_root = Path(
            os.environ.get("CODE_SWE_OUTPUT_DIR", "/opt/swebench_outputs")
        )
        self.results_dir = output_root / "results"
        self.predictions_dir = output_root / "predictions"
        self.plan_dir = Path(
            os.environ.get("CODE_SWE_PLAN_DIR", str(output_root / "claude_plans"))
        )

        # Resolve model name from alias
        self.model = get_model_name(model, self.backend) if model else None
        self.model_alias = model  # Keep original alias for logging

        # Create directories if they don't exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.predictions_dir.mkdir(parents=True, exist_ok=True)
        self.plan_dir.mkdir(parents=True, exist_ok=True)
        self.pred_timestamp: Optional[str] = None
        self.pred_file: Optional[Path] = None

    def setup_repository(self, instance: Dict) -> Optional[str]:
        """Set up a repository for testing."""
        instance_id = instance["instance_id"]
        # Try to materialize prebuilt SWE-bench testbed into a writable staging dir (host-mounted).
        prebuilt_path = Path(os.environ.get("SWEB_TESTBED_DIR", "/tmp/testbed_host"))
        try:
            prebuilt_path.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            print(f"Failed to prepare testbed directory {prebuilt_path}: {e}")
            return None

        iid = instance.get("instance_id", "")
        if "__" in iid:
            org, rest = iid.split("__", 1)
            swe_image = f"swebench/sweb.eval.x86_64.{org}_1776_{rest}"
        else:
            print(f"Could not parse instance_id '{iid}' for SWE image name.")
            return None

        # Use docker binary from PATH (expected to be bind-mounted into the container).
        docker_bin = "docker"

        def copy_from_image(src: str, dest: Path, label: str) -> bool:
            try:
                dest.mkdir(parents=True, exist_ok=True)
                timeout_s = int(os.environ.get("SWEB_COPY_TIMEOUT", "300"))
                print(f"Copying {label} from {swe_image} into {dest}...")
                create_cmd = [docker_bin, "create", "--network=none", swe_image]
                try:
                    create = subprocess.run(
                        create_cmd,
                        capture_output=True,
                        text=True,
                        timeout=timeout_s,
                    )
                except subprocess.TimeoutExpired:
                    print(
                        f"Timed out creating container for {swe_image} after {timeout_s}s"
                    )
                    return False
                if create.returncode != 0:
                    print(
                        f"Failed to create container for {swe_image}: {create.stderr.strip()}"
                    )
                    return False
                container_id = create.stdout.strip()
                try:
                    cp_cmd = [docker_bin, "cp", f"{container_id}:{src}/.", str(dest)]
                    result = subprocess.run(
                        cp_cmd,
                        capture_output=True,
                        text=True,
                        timeout=timeout_s,
                    )
                    if result.returncode != 0:
                        print(
                            f"Failed to copy {label} from {swe_image}: {result.stderr.strip()}"
                        )
                        return False
                    return True
                except subprocess.TimeoutExpired:
                    print(
                        f"Timed out copying {label} from {swe_image} after {timeout_s}s"
                    )
                    return False
                finally:
                    subprocess.run(
                        [docker_bin, "rm", "-f", container_id],
                        capture_output=True,
                        text=True,
                    )
            except Exception as e:
                print(f"Error running docker cmd to copy {label}: {e}")
                return False

        if not copy_from_image("/testbed", prebuilt_path, "testbed"):
            return None

        env_root = Path("/tmp/conda_envs")
        env_path = env_root / instance_id
        if not copy_from_image("/opt/miniconda3/envs/testbed", env_path, "conda env"):
            return None

        # Confirm contents exist
        if not any(prebuilt_path.iterdir()):
            print(f"{prebuilt_path} is empty after copying from {swe_image}")
            return None
        if not (env_path / "bin").exists():
            print(f"{env_path} does not contain a usable conda env from {swe_image}")
            return None
        try:
            subprocess.run(
                [
                    "git",
                    "config",
                    "--global",
                    "--add",
                    "safe.directory",
                    str(prebuilt_path),
                ],
                capture_output=True,
                text=True,
            )
        except Exception as e:
            print(f"Warning: could not mark {prebuilt_path} as safe for git: {e}")

        instance["repo_path"] = str(prebuilt_path)
        return str(prebuilt_path)

    def process_instance(self, instance: Dict) -> Dict:
        """Process a single SWE-bench instance."""
        instance_id = instance["instance_id"]
        print(f"\nProcessing {instance_id}")

        original_dir = os.getcwd()

        repo_path = self.setup_repository(instance)
        if not repo_path:
            return {
                "instance_id": instance_id,
                "model": f"{self.backend}-code",
                "prediction": "",
                "error": "Failed to set up repository",
            }

        try:
            plan_input_dir = os.environ.get("CODE_SWE_PLAN_INPUT_DIR", "").strip()

            if self.prompt_mode == "plan":
                prompt = self.prompt_formatter.format_plan(instance)
            else:
                task_block_text = None
                plan_path = (
                    Path(plan_input_dir) / f"{instance_id}.txt"
                    if plan_input_dir
                    else None
                )
                has_plan = bool(plan_path and plan_path.exists())
                if has_plan and plan_path:
                    try:
                        plan_text = plan_path.read_text(encoding="utf-8").strip()
                        if plan_text:
                            task_block_text = (
                                "Use the plan below to create a Todo list and follow it step-by-step. "
                                "Do not ignore the plan.\n\n"
                                f"{plan_text}\n"
                            )
                    except Exception as e:
                        print(f"Warning: could not read plan file {plan_path}: {e}")

                if task_block_text:
                    prompt = self.prompt_formatter.format_issue(
                        instance,
                        task_block_text=task_block_text,
                    )
                else:
                    prompt = self.prompt_formatter.format_issue(
                        instance,
                    )

            os.chdir(repo_path)
            using_prebuilt = Path(repo_path).resolve() == Path("/tmp/testbed_host")
            if not using_prebuilt:
                subprocess.run(["git", "add", "-A"], capture_output=True)
                subprocess.run(["git", "stash"], capture_output=True)
            else:
                conda_bin = Path("/tmp/conda_envs") / instance_id / "bin"
                if conda_bin.exists():
                    os.environ["PATH"] = f"{conda_bin}:{os.environ.get('PATH','')}"
                    os.environ["CONDA_PREFIX"] = str(conda_bin.parent)
                    os.environ["CONDA_DEFAULT_ENV"] = "testbed"
                    os.environ["PYTHONPATH"] = (
                        f"{repo_path}:{os.environ.get('PYTHONPATH','')}"
                    )
                    base_activate = Path("/opt/miniconda3/bin/activate")
                    if base_activate.exists():
                        os.environ["PATH"] = (
                            f"{base_activate.parent}:{os.environ.get('PATH','')}"
                        )
                        result = subprocess.run(
                            [
                                "bash",
                                "-lc",
                                f"source '{base_activate}' '{conda_bin.parent}' && env",
                            ],
                            capture_output=True,
                            text=True,
                        )
                        if result.returncode == 0:
                            for line in result.stdout.splitlines():
                                if "=" in line:
                                    key, val = line.split("=", 1)
                                    os.environ[key] = val
            os.chdir(repo_path)
            model_info = f" with model {self.model_alias}" if self.model else ""
            print(f"Running {self.backend.title()} Code{model_info}...")
            trajectory_name = instance_id
            result = self.interface.execute_code_cli(
                prompt, repo_path, self.model, trajectory_name=trajectory_name
            )

            if self.prompt_mode == "plan":
                plan_path = self.plan_dir / f"{instance_id}.txt"
                plan_text = (result.get("stdout") or "").strip()
                try:
                    plan_path.write_text(
                        plan_text + ("\n" if plan_text else ""), encoding="utf-8"
                    )
                    print(f"Wrote plan for {instance_id} to {plan_path}")
                except Exception as e:
                    print(f"Warning: could not write plan for {instance_id}: {e}")

            if not result["success"]:
                print(
                    f"{self.backend.title()} Code execution failed: {result['stderr']}"
                )
                # return {
                #     "instance_id": instance_id,
                #     "model": self.model_alias or f"{self.backend}-code",
                #     "prediction": "",
                #     "error": f"Execution failed: {result['stderr']}",
                # }

            patch = self.patch_extractor.extract_from_cli_output(
                result["stdout"], repo_path
            )

            is_valid, error = self.patch_extractor.validate_patch(patch)
            if not is_valid:
                print(f"Invalid patch: {error}")
                patch = ""

            prediction = self.patch_extractor.format_for_swebench(
                patch, instance_id, self.model_alias or f"{self.backend}-code"
            )

            self._save_result(instance_id, result, patch)

            return prediction

        except Exception as e:
            import traceback

            print(f"Error processing instance: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            return {
                "instance_id": instance_id,
                "model": self.model_alias or f"{self.backend}-code",
                "prediction": "",
                "error": str(e),
            }
        finally:
            try:
                os.chdir(original_dir)
            except Exception as e:
                print(f"Warning: Could not restore directory: {e}")

            using_prebuilt = Path(repo_path).resolve() == Path("/tmp/testbed_host")
            if repo_path and os.path.exists(repo_path) and not using_prebuilt:
                shutil.rmtree(repo_path)

    def _save_result(self, instance_id: str, result: Dict, patch: str):
        """Save detailed results for debugging."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = self.results_dir / f"{instance_id}_{timestamp}.json"

        with open(result_file, "w") as f:
            json.dump(
                {
                    "instance_id": instance_id,
                    "timestamp": timestamp,
                    "claude_output": result,
                    "extracted_patch": patch,
                },
                f,
                indent=2,
            )

    def run_on_dataset(
        self, dataset_name: str, split: str = "test", limit: Optional[int] = None
    ) -> List[Dict]:
        """Run on a full dataset."""
        print(f"Loading dataset: {dataset_name}")
        dataset = load_dataset(dataset_name, split=split)

        if limit:
            dataset = dataset.select(range(min(limit, len(dataset))))

        self.pred_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.pred_file = (
            self.predictions_dir / f"predictions_{self.pred_timestamp}.jsonl"
        )
        if self.pred_file.exists():
            self.pred_file.unlink()
        json_file = self.predictions_dir / f"predictions_{self.pred_timestamp}.json"
        if json_file.exists():
            json_file.unlink()

        predictions: List[Dict] = []

        for instance in tqdm(dataset, desc="Processing instances"):
            prediction = self.process_instance(instance)
            predictions.append(prediction)

            # Save prediction incrementally
            self._save_predictions(prediction)

        with open(json_file, "w") as f:
            json.dump(predictions, f, indent=2)

        print(f"Saved predictions to {self.pred_file}")
        return predictions

    def run_on_instance(
        self, instance_id: str, dataset_name: str = "princeton-nlp/SWE-bench_Lite"
    ) -> Dict:
        """Run on a single instance by ID."""
        dataset = load_dataset(dataset_name, split="test")

        # Find the instance
        instance = None
        for item in dataset:
            if item["instance_id"] == instance_id:
                instance = item
                break

        if not instance:
            raise ValueError(f"Instance {instance_id} not found in dataset")

        return self.process_instance(instance)

    def _save_predictions(self, prediction: Dict):
        """Append a single prediction to the jsonl file."""
        if not self.pred_file:
            raise ValueError(
                "Prediction timestamp not initialized. Call run_on_dataset first."
            )

        with jsonlines.open(self.pred_file, mode="a") as writer:
            writer.write(prediction)


def main():
    parser = argparse.ArgumentParser(description="Run code models on SWE-bench")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="princeton-nlp/SWE-bench_Lite",
        help="Dataset to use",
    )
    parser.add_argument("--instance_id", type=str, help="Run on a specific instance ID")
    parser.add_argument(
        "--limit", type=int, help="Limit number of instances to process"
    )
    parser.add_argument(
        "--prompt_template", type=str, help="Path to custom prompt template"
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Model to use (e.g., opus-4.1, codex-4.2, or any name)",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["claude", "codex", "gemini", "router"],
        help="Code model backend to use",
    )

    args = parser.parse_args()

    backend = args.backend or DEFAULT_BACKEND

    # Check if selected CLI is available
    if backend == "codex":
        cli_cmd = "codex"
    elif backend == "gemini":
        cli_cmd = "gemini"
    elif backend == "router":
        cli_cmd = "ccr"
    else:
        cli_cmd = "claude"

    try:
        if backend == "router":
            result = subprocess.run([cli_cmd, "-v"], capture_output=True, text=True)
        else:
            result = subprocess.run(
                [cli_cmd, "--version"], capture_output=True, text=True
            )
        if result.returncode != 0:
            print(
                f"Error: {cli_cmd} CLI not found or returned error. Stderr: {result.stderr}"
            )
            sys.exit(1)
    except FileNotFoundError:
        print(
            f"Error: {cli_cmd} CLI not found. Please ensure '{cli_cmd}' is installed and in PATH"
        )
        sys.exit(1)

    agent = CodeSWEAgent(args.prompt_template, args.model, backend)

    # Run on specific instance or dataset
    if args.instance_id:
        print(f"Running on instance: {args.instance_id}")
        prediction = agent.run_on_instance(args.instance_id, args.dataset_name)
        print(f"Prediction saved: {prediction}")
    else:
        print(f"Running on dataset: {args.dataset_name}")
        predictions = agent.run_on_dataset(args.dataset_name, limit=args.limit)
        print(f"Processed {len(predictions)} instances")


if __name__ == "__main__":
    main()
