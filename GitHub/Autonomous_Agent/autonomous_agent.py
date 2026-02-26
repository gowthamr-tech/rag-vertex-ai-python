#!/usr/bin/env python3
"""Autonomous build/review/PR agent.

Capabilities:
1) Detect project stack and run install/build/test/lint.
2) Review current git diff for common risk patterns.
3) Optional autonomous objective mode with iterative plan execution.
4) Optional branch/commit/push/PR creation through GitHub CLI.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


@dataclass
class CmdResult:
    command: str
    returncode: int
    stdout: str
    stderr: str


@dataclass
class PlanStep:
    name: str
    command: str


class Runner:
    def __init__(self, cwd: Path, verbose: bool = True) -> None:
        self.cwd = cwd
        self.verbose = verbose

    def run(self, command: str, check: bool = False) -> CmdResult:
        if self.verbose:
            print(f"$ {command}")
        proc = subprocess.run(
            command,
            cwd=self.cwd,
            shell=True,
            capture_output=True,
            text=True,
        )
        result = CmdResult(
            command=command,
            returncode=proc.returncode,
            stdout=proc.stdout.strip(),
            stderr=proc.stderr.strip(),
        )
        if self.verbose:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr, file=sys.stderr)
            print(f"[exit={result.returncode}]\n")
        if check and result.returncode != 0:
            raise RuntimeError(f"Command failed: {command}")
        return result


def exists(cwd: Path, *names: str) -> bool:
    return any((cwd / n).exists() for n in names)


def detect_stack(cwd: Path) -> str:
    if exists(cwd, "package.json"):
        return "node"
    if exists(cwd, "pyproject.toml", "requirements.txt", "setup.py"):
        return "python"
    if exists(cwd, "go.mod"):
        return "go"
    if exists(cwd, "Cargo.toml"):
        return "rust"
    return "unknown"


def workflow_commands(stack: str) -> List[Tuple[str, str]]:
    if stack == "node":
        return [
            ("install", "npm install"),
            ("build", "npm run build --if-present"),
            ("test", "npm test --if-present"),
            ("lint", "npm run lint --if-present"),
        ]
    if stack == "python":
        return [
            ("install", "python3 -m pip install -r requirements.txt || true"),
            ("build", "python3 -m compileall ."),
            ("test", "pytest -q || python3 -m unittest discover"),
            ("lint", "ruff check . || flake8 . || true"),
        ]
    if stack == "go":
        return [
            ("install", "go mod download"),
            ("build", "go build ./..."),
            ("test", "go test ./..."),
            ("lint", "go vet ./..."),
        ]
    if stack == "rust":
        return [
            ("install", "cargo fetch"),
            ("build", "cargo build"),
            ("test", "cargo test"),
            ("lint", "cargo clippy -- -D warnings"),
        ]
    return []


def git_diff(runner: Runner) -> str:
    res = runner.run("git diff -- . ':(exclude).agent/*'", check=False)
    return res.stdout


def review_findings(diff_text: str) -> List[str]:
    findings: List[str] = []

    rules = [
        (r"\bconsole\.log\(", "Found `console.log` in diff; confirm debug logs are intended."),
        (r"\bprint\(", "Found `print(...)` in diff; confirm debug output is intended for production."),
        (r"\bTODO\b|\bFIXME\b", "Found TODO/FIXME markers; track them in issues before merge."),
        (r"except\s*:\s*$", "Found bare `except:`; prefer catching specific exceptions."),
        (r"\beval\(", "Found `eval(...)`; verify input safety and necessity."),
        (r"\bpassword\b|\bsecret\b|\btoken\b", "Potential credential-related string detected; confirm no secrets are committed."),
    ]

    for pattern, message in rules:
        if re.search(pattern, diff_text, flags=re.IGNORECASE | re.MULTILINE):
            findings.append(message)

    if "+" not in diff_text and "-" not in diff_text:
        findings.append("No code diff found. Nothing to review yet.")

    return findings


def write_review_report(
    cwd: Path,
    stack: str,
    command_results: List[Tuple[str, CmdResult]],
    findings: List[str],
    objective: Optional[str] = None,
    plan_steps: Optional[List[PlanStep]] = None,
) -> Path:
    report_dir = cwd / ".agent"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "review.md"

    now = dt.datetime.now().isoformat(timespec="seconds")
    lines = [
        "# Autonomous Agent Review Report",
        "",
        f"Generated: `{now}`",
        f"Detected stack: `{stack}`",
    ]

    if objective:
        lines.append(f"Objective: `{objective}`")

    lines.extend(["", "## Build/Test/Lint Results", ""])

    if command_results:
        for name, res in command_results:
            status = "PASS" if res.returncode == 0 else "FAIL"
            lines.append(f"- **{name}**: `{status}` (`{res.command}`)")
    else:
        lines.append("- No executable build/test/lint workflow detected for this repository.")

    if plan_steps:
        lines.extend(["", "## Autonomous Plan", ""])
        for s in plan_steps:
            lines.append(f"- {s.name}: `{s.command}`")

    lines.extend(["", "## Diff Findings", ""])

    if findings:
        for f in findings:
            lines.append(f"- {f}")
    else:
        lines.append("- No obvious risk patterns found.")

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def write_state(cwd: Path, payload: Dict[str, object]) -> Path:
    state_dir = cwd / ".agent"
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    state_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return state_path


def in_git_repo(runner: Runner) -> bool:
    res = runner.run("git rev-parse --is-inside-work-tree", check=False)
    return res.returncode == 0 and res.stdout.strip() == "true"


def ensure_branch(runner: Runner, prefix: str = "codex/agent") -> str:
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    branch = f"{prefix}-{ts}"
    runner.run(f"git checkout -b {shlex.quote(branch)}", check=True)
    return branch


def has_changes(runner: Runner) -> bool:
    res = runner.run("git status --porcelain", check=False)
    return bool(res.stdout.strip())


def commit_changes(runner: Runner, message: str) -> None:
    # Stage only files under the current working directory to avoid
    # accidentally adding unrelated sibling projects in a parent repo.
    runner.run("git add -- .", check=True)
    runner.run(f"git commit -m {shlex.quote(message)}", check=True)


def create_pr(runner: Runner, title: str, body: str, base: str = "main") -> None:
    runner.run("git push -u origin HEAD", check=True)
    runner.run(
        "gh pr create "
        f"--base {shlex.quote(base)} "
        f"--title {shlex.quote(title)} "
        f"--body {shlex.quote(body)}",
        check=True,
    )


def safe_command(command: str) -> bool:
    allowed_prefixes = (
        "npm ",
        "python3 ",
        "pytest",
        "ruff ",
        "flake8",
        "go ",
        "cargo ",
        "make ",
        "git ",
    )
    stripped = command.strip()
    return stripped.startswith(allowed_prefixes)


def llm_plan(objective: str, stack: str, model: str) -> Optional[List[PlanStep]]:
    import os

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None

    prompt = (
        "Return strict JSON with key `steps` (array). "
        "Each step must include `name` and `command`. "
        "Commands must be safe local shell commands for code quality/build/test only. "
        f"Stack={stack}. Objective={objective}."
    )
    payload = {
        "model": model,
        "input": prompt,
    }

    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None

    text = raw.get("output_text", "")
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    steps = parsed.get("steps")
    if not isinstance(steps, list):
        return None

    plan: List[PlanStep] = []
    for idx, s in enumerate(steps, start=1):
        if not isinstance(s, dict):
            continue
        name = str(s.get("name", f"step-{idx}"))
        command = str(s.get("command", "")).strip()
        if command and safe_command(command):
            plan.append(PlanStep(name=name, command=command))

    return plan or None


def fallback_plan(stack: str, objective: str, skip_install: bool) -> List[PlanStep]:
    plan: List[PlanStep] = []

    for stage, command in workflow_commands(stack):
        if skip_install and stage == "install":
            continue
        plan.append(PlanStep(name=stage, command=command))

    objective_lower = objective.lower()
    if "security" in objective_lower:
        if stack == "python":
            plan.append(PlanStep(name="security", command="python3 -m pip list"))
        if stack == "node":
            plan.append(PlanStep(name="security", command="npm audit --audit-level=high || true"))
    if "format" in objective_lower and stack == "python":
        plan.append(PlanStep(name="format-check", command="python3 -m compileall ."))

    return plan


def recovery_commands(stack: str, step: PlanStep, result: CmdResult) -> List[str]:
    text = f"{result.stdout}\n{result.stderr}".lower()
    commands: List[str] = []

    if stack == "python":
        if "no module named pytest" in text:
            commands.append("python3 -m unittest discover")
        if "ruff: command not found" in text or "flake8: command not found" in text:
            commands.append("python3 -m pip install ruff flake8 || true")
    if stack == "node":
        if "missing script" in text and "test" in step.command:
            commands.append("npm run build --if-present")
    if "could not find a version" in text and "pip install" in step.command:
        commands.append("python3 -m pip install -r requirements.txt || true")

    return [c for c in commands if safe_command(c)]


def run_plan(
    runner: Runner,
    stack: str,
    plan: List[PlanStep],
    max_iterations: int,
) -> Tuple[List[Tuple[str, CmdResult]], bool]:
    results: List[Tuple[str, CmdResult]] = []
    all_passed = True

    for step in plan:
        if not safe_command(step.command):
            blocked = CmdResult(step.command, 1, "", "Blocked unsafe command")
            results.append((step.name, blocked))
            all_passed = False
            continue

        final_result: Optional[CmdResult] = None
        for _ in range(max_iterations):
            attempt = runner.run(step.command, check=False)
            final_result = attempt
            if attempt.returncode == 0:
                break

            repair_attempted = False
            for repair_cmd in recovery_commands(stack, step, attempt):
                repair_attempted = True
                runner.run(repair_cmd, check=False)

            if not repair_attempted:
                break

        if final_result is None:
            final_result = CmdResult(step.command, 1, "", "No execution")

        if final_result.returncode != 0:
            all_passed = False
        results.append((step.name, final_result))

    return results, all_passed


def run_build_review(cwd: Path, skip_install: bool) -> Tuple[str, List[Tuple[str, CmdResult]], Path]:
    runner = Runner(cwd)
    stack = detect_stack(cwd)
    print(f"Detected stack: {stack}")

    results: List[Tuple[str, CmdResult]] = []
    for stage, cmd in workflow_commands(stack):
        if skip_install and stage == "install":
            continue
        res = runner.run(cmd, check=False)
        results.append((stage, res))

    diff = git_diff(runner)
    findings = review_findings(diff)
    report_path = write_review_report(cwd, stack, results, findings)
    print(f"Review report written to: {report_path}")
    return stack, results, report_path


def run_autonomous(
    cwd: Path,
    objective: str,
    skip_install: bool,
    max_iterations: int,
    use_llm: bool,
    model: str,
) -> Tuple[bool, Path, Path]:
    runner = Runner(cwd)
    stack = detect_stack(cwd)
    print(f"Detected stack: {stack}")
    print(f"Objective: {objective}")

    plan = llm_plan(objective, stack, model) if use_llm else None
    if not plan:
        plan = fallback_plan(stack, objective, skip_install)

    results, passed = run_plan(runner, stack, plan, max_iterations=max_iterations)
    diff = git_diff(runner)
    findings = review_findings(diff)

    report_path = write_review_report(
        cwd,
        stack,
        results,
        findings,
        objective=objective,
        plan_steps=plan,
    )

    state_payload: Dict[str, object] = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "stack": stack,
        "objective": objective,
        "passed": passed,
        "plan": [{"name": s.name, "command": s.command} for s in plan],
        "results": [
            {
                "name": name,
                "command": res.command,
                "returncode": res.returncode,
                "stdout": res.stdout,
                "stderr": res.stderr,
            }
            for name, res in results
        ],
    }
    state_path = write_state(cwd, state_payload)

    print(f"Review report written to: {report_path}")
    print(f"Agent state written to: {state_path}")
    return passed, report_path, state_path


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous build/review/PR agent")
    sub = parser.add_subparsers(dest="cmd", required=True)

    run_cmd = sub.add_parser("run", help="Run build + review")
    run_cmd.add_argument("--skip-install", action="store_true", help="Skip dependency install step")

    pr_cmd = sub.add_parser("run-and-pr", help="Run build+review and open PR")
    pr_cmd.add_argument("--skip-install", action="store_true")
    pr_cmd.add_argument("--base", default="main")
    pr_cmd.add_argument("--title", default="chore: autonomous agent updates")
    pr_cmd.add_argument(
        "--body",
        default="Automated changes validated by build/test/lint and reviewed by autonomous agent.",
    )
    pr_cmd.add_argument("--commit-message", default="chore: apply autonomous agent updates")

    review_cmd = sub.add_parser("review", help="Only run diff review")

    auto_cmd = sub.add_parser("autonomous", help="Objective-driven autonomous execution")
    auto_cmd.add_argument("--objective", required=True, help="Goal for the agent to validate")
    auto_cmd.add_argument("--skip-install", action="store_true")
    auto_cmd.add_argument("--max-iterations", type=int, default=2)
    auto_cmd.add_argument("--use-llm", action="store_true", help="Enable LLM planner (requires OPENAI_API_KEY)")
    auto_cmd.add_argument("--model", default="gpt-4.1-mini")
    auto_cmd.add_argument("--open-pr", action="store_true")
    auto_cmd.add_argument("--base", default="main")
    auto_cmd.add_argument("--title", default="chore: autonomous objective execution")
    auto_cmd.add_argument("--body", default="Automated PR generated by autonomous objective mode.")
    auto_cmd.add_argument("--commit-message", default="chore: autonomous objective updates")

    args = parser.parse_args()
    cwd = Path.cwd()
    runner = Runner(cwd)

    if not in_git_repo(runner):
        print("Error: this command must run inside a git repository.", file=sys.stderr)
        return 2

    if args.cmd == "review":
        diff = git_diff(runner)
        findings = review_findings(diff)
        report = write_review_report(cwd, detect_stack(cwd), [], findings)
        print(f"Review report written to: {report}")
        return 0

    if args.cmd == "run":
        run_build_review(cwd, args.skip_install)
        return 0

    if args.cmd == "run-and-pr":
        run_build_review(cwd, args.skip_install)
        if not has_changes(runner):
            print("No changes detected; skipping branch/commit/PR.")
            return 0
        branch = ensure_branch(runner)
        print(f"Created branch: {branch}")
        commit_changes(runner, args.commit_message)
        create_pr(runner, args.title, args.body, args.base)
        print("PR created successfully.")
        return 0

    if args.cmd == "autonomous":
        passed, _, _ = run_autonomous(
            cwd=cwd,
            objective=args.objective,
            skip_install=args.skip_install,
            max_iterations=max(1, args.max_iterations),
            use_llm=args.use_llm,
            model=args.model,
        )

        if args.open_pr:
            if not has_changes(runner):
                print("No changes detected; skipping branch/commit/PR.")
            else:
                branch = ensure_branch(runner)
                print(f"Created branch: {branch}")
                commit_changes(runner, args.commit_message)
                create_pr(runner, args.title, args.body, args.base)
                print("PR created successfully.")

        if not passed:
            print("Autonomous run completed with failures. Check .agent/state.json and .agent/review.md")
            return 1
        print("Autonomous run completed successfully.")
        return 0

    return 1


if __name__ == "__main__":
    sys.exit(main())
