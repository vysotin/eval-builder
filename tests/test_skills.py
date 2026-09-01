"""Agent Skills: SKILL.md folders loaded into `Skill` objects, prompts, the loader tool
and the agent-map description."""

from pathlib import Path

import pytest

from evalbuilder import skills as sk

TRIAGE = """---
name: incident-triage
description: Triage a production incident before proposing remediation.
allowed-tools: [get_service_status, search_runbooks]
metadata:
  version: 2
---
# Incident triage

1. Call `get_service_status` for the affected service first.
2. Map the severity with references/severity-matrix.md.
"""

COMMS = """---
description: Word a status-page update.
---
Always name the ticket id.
"""


def _skills_dir(tmp_path: Path) -> Path:
    root = tmp_path / "skills"
    (root / "incident-triage" / "references").mkdir(parents=True)
    (root / "incident-triage" / "SKILL.md").write_text(TRIAGE)
    (root / "incident-triage" / "references" / "severity-matrix.md").write_text("# Severity matrix\n\nsev1 = outage, sev2 = degraded, sev3 = minor.\n")
    (root / "incident-triage" / "scripts").mkdir()
    (root / "incident-triage" / "scripts" / "check.sh").write_text("echo ok\n")
    (root / "comms").mkdir()
    (root / "comms" / "SKILL.md").write_text(COMMS)
    (root / "not-a-skill").mkdir()
    (root / "not-a-skill" / "README.md").write_text("nothing here")
    return root


def test_load_skills_parses_frontmatter_body_references_and_scripts(tmp_path):
    loaded = sk.load_skills(_skills_dir(tmp_path))
    assert [s.name for s in loaded] == ["comms", "incident-triage"]  # sorted by folder; name defaults to the folder
    triage = loaded[1]
    assert triage.description.startswith("Triage a production incident")
    assert triage.body.startswith("# Incident triage") and "severity-matrix.md" in triage.body
    assert triage.metadata["allowed-tools"] == ["get_service_status", "search_runbooks"]
    assert triage.metadata["metadata"] == {"version": 2}
    assert triage.path.endswith("incident-triage/SKILL.md") and triage.dir.endswith("incident-triage")
    assert triage.references == [{
        "path": "references/severity-matrix.md", "title": "Severity matrix", "chars": len("# Severity matrix\n\nsev1 = outage, sev2 = degraded, sev3 = minor.\n"),
        "excerpt": "sev1 = outage, sev2 = degraded, sev3 = minor.",
    }]
    assert triage.scripts == ["scripts/check.sh"]
    comms = loaded[0]
    assert comms.name == "comms" and comms.body == "Always name the ticket id." and comms.references == []


def test_load_skills_accepts_a_single_skill_file_and_lists_and_missing_paths(tmp_path):
    root = _skills_dir(tmp_path)
    one = sk.load_skills(root / "comms" / "SKILL.md")
    assert [s.name for s in one] == ["comms"]
    both = sk.load_skills([root / "comms", root / "incident-triage"])
    assert [s.name for s in both] == ["comms", "incident-triage"]
    assert sk.load_skills(tmp_path / "nowhere") == []
    assert sk.load_skills(None) == []


def test_prompts_list_or_inline_the_skills(tmp_path):
    loaded = sk.load_skills(_skills_dir(tmp_path))
    listing = sk.skills_prompt(loaded)
    assert "load_skill" in listing and "- incident-triage: Triage a production incident" in listing and "- comms: Word a status-page update." in listing
    assert "Call `get_service_status`" not in listing  # bodies are not disclosed in the listing
    inline = sk.skills_inline_prompt(loaded, "incident-triage")
    assert "## Skill: incident-triage" in inline and "Call `get_service_status`" in inline and "ticket id" not in inline
    assert "comms" in sk.skills_inline_prompt(loaded) and "incident-triage" in sk.skills_inline_prompt(loaded)
    assert sk.skills_prompt([]) == "" and sk.skills_inline_prompt([]) == ""
    with pytest.raises(KeyError):
        sk.skills_inline_prompt(loaded, "unknown")


def test_loader_tool_returns_the_body_and_is_recognised_as_a_skill_loader(tmp_path):
    loaded = sk.load_skills(_skills_dir(tmp_path))
    tool = sk.skill_loader_tool(loaded)
    assert tool.name == "load_skill" and "incident-triage" in tool.description and "comms" in tool.description
    assert sk.is_skill_loader(tool) and tool.metadata["kind"] == "skill_loader"
    text = tool.invoke({"name": "incident-triage"})
    assert text.startswith("# Incident triage") and "references/severity-matrix.md" in text and "Severity matrix" in text
    missing = tool.invoke({"name": "nope"})
    assert "unknown skill" in missing and "incident-triage" in missing
    named = sk.skill_loader_tool(loaded, name="read_skill")
    assert named.name == "read_skill" and sk.is_skill_loader(named)

    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def get_weather(city: str) -> dict:
        """weather"""
        return {}

    @lc_tool
    def use_skill(name: str) -> str:
        """loads a skill"""
        return ""

    assert not sk.is_skill_loader(get_weather) and sk.is_skill_loader(use_skill)


def test_describe_skill_is_the_agent_map_entry(tmp_path):
    loaded = sk.load_skills(_skills_dir(tmp_path))
    entry = sk.describe_skill(loaded[1], tool_names=["get_service_status", "search_runbooks", "create_ticket"])
    assert entry["name"] == "incident-triage" and entry["description"].startswith("Triage")
    assert entry["prompt"].startswith("# Incident triage")
    assert entry["allowed_tools"] == ["get_service_status", "search_runbooks"]
    assert entry["tools_mentioned"] == ["get_service_status"]
    assert entry["references"][0]["path"] == "references/severity-matrix.md" and entry["scripts"] == ["scripts/check.sh"]
    assert entry["used_by"] == [] and entry["evidence"] == [f"skill:{entry['name']}"]
    assert entry["path"].endswith("SKILL.md") and entry["metadata"] == {"version": 2}
