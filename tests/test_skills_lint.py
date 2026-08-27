"""Executable contract between the skills' prose and the real CLI surface."""

import re
from pathlib import Path

from typer.testing import CliRunner

from evalbuilder.cli import app

SKILLS = sorted(Path("skills").glob("agent-eval-*/SKILL.md"))


def test_four_skills_exist_with_frontmatter():
    assert len(SKILLS) == 5, f"expected 5 skills, found {[str(s) for s in SKILLS]}"
    for s in SKILLS:
        text = s.read_text()
        assert text.startswith("---\n"), f"{s} missing frontmatter"
        assert "name:" in text and "description:" in text
        description = re.search(r"^description: (.+)$", text, re.M).group(1)
        assert description.startswith("Use when"), f"{s} description must start 'Use when'"


def _commands_in(text: str):
    for m in re.finditer(r"^evalbuilder (\S+)(?: (\S+))?", text, re.M):
        first, second = m.group(1), m.group(2)
        args = [first]
        # subcommand groups: dataset/agent-map/mock have a second command word
        if second and re.fullmatch(r"[a-z-]+", second):
            args.append(second)
        yield m.group(0), args


def test_every_cli_command_in_skills_is_real():
    runner = CliRunner()
    for s in SKILLS:
        for line, args in _commands_in(s.read_text()):
            result = runner.invoke(app, args + ["--help"])
            assert result.exit_code == 0, (
                f"{s}: '{line}' does not resolve to a real CLI command"
            )


def test_symlinks_resolve():
    links = sorted(Path(".claude/skills").glob("agent-eval-*"))
    assert len(links) == 5
    for link in links:
        assert (link / "SKILL.md").exists(), f"{link} does not resolve"


def test_references_exist():
    for s in SKILLS:
        text = s.read_text()
        for ref in re.finditer(r"references/[\w-]+\.md", text):
            assert (s.parent / ref.group(0)).exists(), (
                f"{s} references missing file {ref.group(0)}"
            )
