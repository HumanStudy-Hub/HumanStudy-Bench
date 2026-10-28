# Reasoning in Reference Games: Individual- vs. Population-Level Probabilistic Modeling: agent skill

Michael Franke, Judith Degen (2016). Reasoning in Reference Games: Individual- vs. Population-Level Probabilistic Modeling. PLOS ONE. https://doi.org/10.1371/journal.pone.0154854.

Built with HumanStudy Hub. `frankedegen2016/` is a skill folder: `SKILL.md`, one readable reference per study with page numbers, `program.json` and the replay package. Any agent that reads skills can use it; it needs only Python 3.8+ (no packages, no keys).

Ready to run the parts the replay package covers (see `package/frankedegen2016/task/task.json`). Results from AI participants are a simulation, not observations of people.

On Windows use `python` or `py` where these lines say `python3`.

## Install

```bash
# Claude Code
mkdir -p ~/.claude/skills && cp -R frankedegen2016 ~/.claude/skills/
# Codex
mkdir -p ~/.agents/skills && cp -R frankedegen2016 ~/.agents/skills/
# OpenCode: copy it into your project's .opencode/skills/
```

## MCP server (optional)

For clients that cannot run commands or read files. It serves the same folder:

```bash
claude mcp add frankedegen2016 -- python3 "$PWD/frankedegen2016/scripts/mcp_server.py"
codex mcp add frankedegen2016 -- python3 "$PWD/frankedegen2016/scripts/mcp_server.py"
```

OpenCode `opencode.json`:

```json
{
  "mcp": {
    "frankedegen2016": {
      "type": "local",
      "command": [
        "python3",
        "/absolute/path/frankedegen2016/scripts/mcp_server.py"
      ],
      "enabled": true
    }
  }
}
```

## Without an agent

```bash
python3 frankedegen2016/scripts/study_core.py overview
```

`MANIFEST.json` lists every file with its SHA-256. AI participants' answers are a simulation, not data from people.

## In the HumanStudy community library

This copy shares `program.json` and `package/` with the research files two folders up (`../../program.json`, `../../package/`), so they are stored once. The scripts find them there. To install the skill on its own, copy those two into the skill folder, or export the study from the desktop app (0.6.9 or later), which builds a self-contained folder.
