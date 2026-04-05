#!/usr/bin/env python3
"""Test runner script for the Aerospace MCP test suite.

Provides a convenience wrapper around pytest with preset configurations for
different test scopes: unit, integration, fast, slow, coverage, and all.
Supports verbose output, parallel execution (via pytest-xdist), and
targeting individual test files.

Usage:
    python run_tests.py                # Run all tests (default)
    python run_tests.py unit           # Unit tests only
    python run_tests.py coverage       # Full coverage report
    python run_tests.py fast -v        # Fast tests, verbose
    python run_tests.py -f test_plan.py  # Single file
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(cmd: list[str], description: str) -> int:
    """Run a subprocess command, print its output, and return the exit code.

    Args:
        cmd: The command and arguments to execute (e.g., ["pytest", "-v"]).
        description: Human-readable label for the test run (e.g., "Running unit tests").

    Returns:
        The subprocess exit code (0 = success, non-zero = failure).
    """
    print(f"\n🚀 {description}")
    print(f"Command: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, cwd=Path(__file__).parent)

    if result.returncode == 0:
        print(f"✅ {description} - PASSED")
    else:
        print(f"❌ {description} - FAILED")

    return result.returncode


def main():
    """Parse CLI arguments and dispatch the appropriate test run."""
    parser = argparse.ArgumentParser(description="Run Aerospace MCP tests")
    parser.add_argument(
        "test_type",
        choices=["unit", "integration", "all", "coverage", "fast", "slow"],
        nargs="?",
        default="all",
        help="Type of tests to run",
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-n", type=int, help="Run tests in parallel")
    parser.add_argument("--file", "-f", help="Run specific test file")

    args = parser.parse_args()

    # Base pytest command
    pytest_cmd = ["python3", "-m", "pytest"]

    if args.verbose:
        pytest_cmd.append("-v")
    else:
        pytest_cmd.extend(["-q", "--tb=short"])

    if args.parallel:
        pytest_cmd.extend(["-n", str(args.parallel)])

    if args.file:
        pytest_cmd.append(f"tests/{args.file}")
        return run_command(pytest_cmd, f"Running tests in {args.file}")

    exit_code = 0

    if args.test_type == "unit":
        cmd = pytest_cmd + ["-m", "unit", "tests/"]
        exit_code = run_command(cmd, "Running unit tests")

    elif args.test_type == "integration":
        cmd = pytest_cmd + ["-m", "integration", "tests/"]
        exit_code = run_command(cmd, "Running integration tests")

    elif args.test_type == "fast":
        cmd = pytest_cmd + ["-m", "not slow", "tests/"]
        exit_code = run_command(cmd, "Running fast tests")

    elif args.test_type == "slow":
        cmd = pytest_cmd + ["-m", "slow", "tests/"]
        exit_code = run_command(cmd, "Running slow tests")

    elif args.test_type == "coverage":
        cmd = pytest_cmd + [
            "--cov=aerospace_mcp",
            "--cov=app",
            "--cov-report=html",
            "--cov-report=term-missing",
            "--cov-fail-under=80",
            "tests/",
        ]
        exit_code = run_command(cmd, "Running tests with coverage")

        if exit_code == 0:
            print("\n📊 Coverage report generated:")
            print("  - Terminal: see above")
            print("  - HTML: open htmlcov/index.html")

    elif args.test_type == "all":
        # Run all tests
        cmd = pytest_cmd + ["tests/"]
        exit_code = run_command(cmd, "Running all tests")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
