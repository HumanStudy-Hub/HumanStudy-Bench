# Not unreasonable: Carving vague dimensions with contraries and contradictions: agent skill

Michael Henry Tessler, Michael Franke (2018). Not unreasonable: Carving vague dimensions with contraries and contradictions. Proceedings of the Annual Meeting of the Cognitive Science Society, 40.

Built with HumanStudy Hub. `tesslerfranke2018/` is a skill folder: `SKILL.md`, one readable reference per study with page numbers, `program.json` and the replay package. Any agent that reads skills can use it; it needs only Python 3.8+ (no packages, no keys).

Ready to run the parts the replay package covers (see `package/tesslerfranke2018/task/task.json`). Results from AI participants are a simulation, not observations of people.

On Windows use `python` or `py` where these lines say `python3`.

## Install

```bash
# Claude Code
mkdir -p ~/.claude/skills && cp -R tesslerfranke2018 ~/.claude/skills/
# Codex
mkdir -p ~/.agents/skills && cp -R tesslerfranke2018 ~/.agents/skills/
# OpenCode: copy it into your project's .opencode/skills/
```

## MCP server (optional)

For clients that cannot run commands or read files. It serves the same folder:

```bash
claude mcp add tesslerfranke2018 -- python3 "$PWD/tesslerfranke2018/scripts/mcp_server.py"
codex mcp add tesslerfranke2018 -- python3 "$PWD/tesslerfranke2018/scripts/mcp_server.py"
```

OpenCode `opencode.json`:

```json
{
  "mcp": {
    "tesslerfranke2018": {
      "type": "local",
      "command": [
        "python3",
        "/absolute/path/tesslerfranke2018/scripts/mcp_server.py"
      ],
      "enabled": true
    }
  }
}
```

## Without an agent

```bash
python3 tesslerfranke2018/scripts/study_core.py overview
```

`MANIFEST.json` lists every file with its SHA-256. AI participants' answers are a simulation, not data from people.

## In the HumanStudy community library

This copy shares `program.json`, `package/`, `capabilities.json` and `effects.json` with the research files two folders up (`../../`), so they are stored once. The scripts find them there. To install the skill on its own, copy those four into the skill folder, or export the study from the desktop app, which builds a self-contained folder. Human data are never stored here: `study_core.py human-data` fetches them from their source.
