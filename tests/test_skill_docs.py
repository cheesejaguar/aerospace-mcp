"""The agent skills in ``skills/<name>/SKILL.md`` must match the real tools.

Skills are installed on their own (``npx skills add`` or the Claude Code
plugin) and an agent runs their examples verbatim, so a stale parameter name
fails silently from the user's point of view: the CLI only warns about an
unknown ``--flag`` and the tool returns an error string. These tests run every
CLI example and Python snippet against the real tools and check the
parameter reference sections against the tool signatures.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import re
import shlex
import sys
from pathlib import Path

import pytest

from aerospace_mcp import cli

SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
SKILL_FILES = sorted(SKILLS_DIR.glob("*/SKILL.md"))

# Markers the CLI and the tools emit on failure (tools return error strings
# rather than raising).
ERROR_MARKERS = re.compile(
    r" error: |Error running|Error parsing|Unknown parameter|Missing required"
    r"|Unknown tool|Traceback|unexpected keyword|\"error\":|estimation failed"
)


def _blocks(path: Path, lang: str) -> list[str]:
    return re.findall(rf"```{lang}\n(.*?)```", path.read_text(), re.S)


def _cli_examples() -> list[tuple[str, list[str]]]:
    cases = []
    for path in SKILL_FILES:
        for block in _blocks(path, "bash"):
            joined = re.sub(r"\\\n", " ", block)
            for line in joined.splitlines():
                argv = shlex.split(line, comments=True)
                # uvx lines fetch from PyPI; placeholders are not runnable
                if not argv or argv[0] != "aerospace-mcp-cli" or "<" in line:
                    continue
                cases.append((path.parent.name, argv[1:]))
    return cases


CLI_EXAMPLES = _cli_examples()


def test_skills_use_discoverable_layout():
    """``npx skills`` and Claude Code plugins only load ``<name>/SKILL.md``."""
    assert SKILL_FILES, "no skills found"
    assert not list(SKILLS_DIR.glob("*.md")), "flat skills/*.md are not discovered"
    for path in SKILL_FILES:
        front = re.match(r"---\n(.*?)\n---\n", path.read_text(), re.S)
        assert front, f"{path} has no YAML frontmatter"
        fields = dict(
            line.split(": ", 1) for line in front.group(1).splitlines() if ": " in line
        )
        # Agent Skills spec: name matches the directory, lowercase-hyphenated
        assert fields.get("name") == path.parent.name
        assert re.fullmatch(r"[a-z0-9]+(-[a-z0-9]+)*", fields["name"])
        assert fields.get("description")


@pytest.mark.parametrize(
    "skill,argv", CLI_EXAMPLES, ids=[f"{s}:{' '.join(a)[:60]}" for s, a in CLI_EXAMPLES]
)
def test_cli_example_runs(skill, argv, monkeypatch, capsys):
    monkeypatch.setattr("aerospace_mcp.tools.agents.LLM_TOOLS_ENABLED", False)
    monkeypatch.setattr(sys, "argv", ["aerospace-mcp-cli", *argv])
    try:
        cli.main()
    except SystemExit as exc:
        assert exc.code in (0, None), capsys.readouterr()
    out = capsys.readouterr()
    text = out.out + out.err
    assert text.strip(), "example printed nothing"
    assert not ERROR_MARKERS.search(text), text[:2000]


@pytest.mark.parametrize("path", SKILL_FILES, ids=[p.parent.name for p in SKILL_FILES])
def test_python_snippets_run(path, monkeypatch):
    monkeypatch.setattr("aerospace_mcp.tools.agents.LLM_TOOLS_ENABLED", False)
    for i, block in enumerate(_blocks(path, "python")):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(block, f"{path}:python#{i}", "exec"), {})
        assert not ERROR_MARKERS.search(buf.getvalue()), buf.getvalue()[:2000]


@pytest.mark.parametrize("path", SKILL_FILES, ids=[p.parent.name for p in SKILL_FILES])
def test_parameter_reference_matches_signatures(path):
    sections = re.split(r"^### ", path.read_text(), flags=re.M)[1:]
    for section in sections:
        tool = section.split("\n", 1)[0].strip()
        if tool not in cli.TOOL_MAP:
            continue
        params = inspect.signature(cli.TOOL_MAP[tool]).parameters
        for documented in re.findall(r"^- `--(\w+)`", section, re.M):
            assert documented in params, f"{path.parent.name}: {tool} --{documented}"


def test_plugin_manifest_matches_package():
    """The Claude Code plugin launches the published package via uvx."""
    import json
    import tomllib

    root = SKILLS_DIR.parent
    project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
    plugin = json.loads((root / ".claude-plugin" / "plugin.json").read_text())
    marketplace = json.loads((root / ".claude-plugin" / "marketplace.json").read_text())

    # Plugin updates are keyed on this version: bump both together.
    assert plugin["version"] == project["version"]
    server = plugin["mcpServers"]["aerospace"]
    assert server["command"] == "uvx"
    assert server["args"][0] == project["name"]
    assert project["name"] in project["scripts"]  # uvx runs this script
    assert [p["source"] for p in marketplace["plugins"]] == ["."]
