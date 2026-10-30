# Displacing misinformation about events: An experimental test of causal corrections: agent skill

Brendan Nyhan; Jason Reifler. Displacing misinformation about events: An experimental test of causal corrections. Journal of Experimental Political Science (forthcoming at the time of the supplement).

Built with HumanStudy Hub. `nyhan2015-causal-corrections/` is a skill folder: `SKILL.md`, one readable reference per study with page numbers, `program.json` and the replay package. Any agent that reads skills can use it; it needs only Python 3.8+ (no packages, no keys).

Ready to run the parts the replay package covers (see `package/nyhan2015/task/task.json`). Results from AI participants are a simulation, not observations of people.

On Windows use `python` or `py` where these lines say `python3`.

## Install

```bash
# Claude Code
mkdir -p ~/.claude/skills && cp -R nyhan2015-causal-corrections ~/.claude/skills/
# Codex
mkdir -p ~/.agents/skills && cp -R nyhan2015-causal-corrections ~/.agents/skills/
# OpenCode: copy it into your project's .opencode/skills/
```

## MCP server (optional)

For clients that cannot run commands or read files. It serves the same folder:

```bash
claude mcp add nyhan2015-causal-corrections -- python3 "$PWD/nyhan2015-causal-corrections/scripts/mcp_server.py"
codex mcp add nyhan2015-causal-corrections -- python3 "$PWD/nyhan2015-causal-corrections/scripts/mcp_server.py"
```

OpenCode `opencode.json`:

```json
{
  "mcp": {
    "nyhan2015-causal-corrections": {
      "type": "local",
      "command": [
        "python3",
        "/absolute/path/nyhan2015-causal-corrections/scripts/mcp_server.py"
      ],
      "enabled": true
    }
  }
}
```

## Without an agent

```bash
python3 nyhan2015-causal-corrections/scripts/study_core.py overview
```

`MANIFEST.json` lists every file with its SHA-256. AI participants' answers are a simulation, not data from people.

## In the HumanStudy community library

This copy shares `program.json`, `package/` and `capabilities.json` with the research files two folders up (`../../`), so they are stored once. The scripts find them there. To install the skill on its own, copy those three into the skill folder, or export the study from the desktop app, which builds a self-contained folder. Human data are never stored here: `study_core.py human-data` fetches them from their source.
