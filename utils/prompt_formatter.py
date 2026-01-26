import os
import tempfile
from pathlib import Path
from typing import Dict, Optional


class PromptFormatter:
    """Format SWE-bench issues into prompts for Claude Code."""

    def __init__(
        self,
        prompt_template_path: Optional[str] = None,
        plan_template_path: Optional[str] = None,
    ):
        self.prompt_template_path = prompt_template_path
        self.plan_template_path = plan_template_path
        self.base_template = self._load_base_template()
        self.plan_template = self._load_plan_template()

    def _load_base_template(self) -> str:
        """Load the base prompt template."""
        if self.prompt_template_path:
            try:
                with open(self.prompt_template_path, "r") as f:
                    return f.read()
            except FileNotFoundError:
                pass

        # Default template if no file provided
        return """You have access to a repository with a software issue that needs to be fixed.

Repository: {repo_name}
Cloned path: {base_path}
Issue: {issue_title}

Issue Description:
{issue_description}

Your task:
{task_block}

Important notes:
- Focus on making minimal, targeted changes
- Consider edge cases and potential side effects
- The tests should pass after applying your fix
- Output clear file edits showing exactly what needs to be changed

"""

    def _load_plan_template(self) -> str:
        """Load the plan-only prompt template."""
        if self.plan_template_path:
            try:
                with open(self.plan_template_path, "r") as f:
                    return f.read()
            except FileNotFoundError:
                pass

        return """You have access to a repository with a software issue.

Repository: {repo_name}
Cloned path: {base_path}
Issue: {issue_title}

Your task:
- You are the planner. Produce a concise execution plan (max 8 bullets) to fix this issue.
- Do NOT edit files; only return the plan bullets.
- Save the plan as /opt/claude_plans/{instance_id}.md.

Important rules:
- Do NOT implement a fix
- Do NOT propose code changes
- Do NOT suggest files to edit
- Do NOT mention exact file names, line numbers, or code snippets
- Do NOT state the precise bug location or the exact change to make
- Do NOT write patches or edits

Issue Description:
{issue_description}

"""

    def format_issue(
        self,
        instance: Dict,
        task_block_text: Optional[str] = None,
    ) -> str:
        """Format a SWE-bench instance into a prompt for Claude Code."""
        return self._format_instance(
            instance,
            self.base_template,
            task_block_text=task_block_text,
        )

    def format_plan(self, instance: Dict) -> str:
        """Format a SWE-bench instance into a planning-only prompt."""
        return self._format_instance(instance, self.plan_template)

    def _format_instance(
        self,
        instance: Dict,
        template: str,
        task_block_text: Optional[str] = None,
    ) -> str:
        """Format a SWE-bench instance into a prompt using the given template."""
        # Extract key information from the instance
        repo_name = instance.get("repo", "")
        problem_statement = instance.get("problem_statement", "") or ""
        issue_title = (
            instance.get("issue_title")
            or problem_statement.split("\n")[0]
            if problem_statement
            else ""
        )
        issue_description = instance.get("issue_body") or problem_statement
        base_commit = instance.get("base_commit", "")

        # Get instance_id for tracking
        instance_id = instance.get("instance_id", "")

        # Format the prompt
        base_path = instance.get(
            "repo_path",
            os.environ.get("SWEB_TESTBED_DIR")
            or os.environ.get("REPO_PATH")
            or f"/tmp/testbed/{instance_id}",
        )

        default_task_block = (
            "1. Understand the issue by carefully reading the description\n"
            "2. Search the codebase in cloned path to find relevant files using grep, find, or other search tools\n"
            "3. Analyze the code to understand the root cause\n"
            "4. Generate a fix that resolves the issue\n"
            "5. Ensure your fix doesn't break existing functionality\n"
        )

        prompt = template.format(
            repo_name=repo_name,
            issue_title=issue_title,
            issue_description=issue_description,
            base_path=str(base_path),
            instance_id=instance_id,
            base_commit=base_commit,
            task_block=task_block_text if task_block_text else default_task_block,
        )

        # Add any hints if available
        if "hints_text" in instance and instance["hints_text"]:
            prompt += f"\n\nHints:\n{instance['hints_text']}"

        return prompt

    def extract_instance_info(self, instance: Dict) -> Dict:
        """Extract key information from a SWE-bench instance."""
        return {
            "instance_id": instance.get("instance_id", ""),
            "repo": instance.get("repo", ""),
            "version": instance.get("version", ""),
            "base_commit": instance.get("base_commit", ""),
            "problem_statement": instance.get("problem_statement", ""),
            "hints_text": instance.get("hints_text", ""),
            "created_at": instance.get("created_at", ""),
            "test_patch": instance.get("test_patch", ""),
            "environment_setup_commit": instance.get("environment_setup_commit", ""),
        }
