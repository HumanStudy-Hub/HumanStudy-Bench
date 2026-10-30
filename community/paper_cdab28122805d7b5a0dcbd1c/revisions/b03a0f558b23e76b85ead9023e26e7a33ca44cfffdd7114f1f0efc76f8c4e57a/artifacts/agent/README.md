# Investigating Variation in Replicability: A 'Many Labs' Replication Project: agent skill

Klein, R. A., Ratliff, K. A., Vianello, M., Adams, R. B. Jr., Bahnik, S., Bernstein, M. J., et al., Nosek, B. A. (2014). Investigating Variation in Replicability: A 'Many Labs' Replication Project. Social Psychology 2014; 45(3):142-152. https://doi.org/10.1027/1864-9335/a000178.

Built with HumanStudy Hub. `manylabs1/` is a skill folder: `SKILL.md`, one readable reference per study with page numbers, `program.json` and the replay package. Any agent that reads skills can use it; it needs only Python 3.8+ (no packages, no keys).

Ready to run the parts the replay package covers (see `package/manylabs1/task/task.json`). Results from AI participants are a simulation, not observations of people.

On Windows use `python` or `py` where these lines say `python3`.

## Install

```bash
# Claude Code
mkdir -p ~/.claude/skills && cp -R manylabs1 ~/.claude/skills/
# Codex
mkdir -p ~/.agents/skills && cp -R manylabs1 ~/.agents/skills/
# OpenCode: copy it into your project's .opencode/skills/
```

## MCP server (optional)

For clients that cannot run commands or read files. It serves the same folder:

```bash
claude mcp add manylabs1 -- python3 "$PWD/manylabs1/scripts/mcp_server.py"
codex mcp add manylabs1 -- python3 "$PWD/manylabs1/scripts/mcp_server.py"
```

OpenCode `opencode.json`:

```json
{
  "mcp": {
    "manylabs1": {
      "type": "local",
      "command": [
        "python3",
        "/absolute/path/manylabs1/scripts/mcp_server.py"
      ],
      "enabled": true
    }
  }
}
```

## Without an agent

```bash
python3 manylabs1/scripts/study_core.py overview
```

`MANIFEST.json` lists every file with its SHA-256. AI participants' answers are a simulation, not data from people.

## In the HumanStudy community library

This copy shares `program.json`, `package/` and `capabilities.json` with the research files two folders up (`../../`), so they are stored once. The scripts find them there. To install the skill on its own, copy those three into the skill folder, or export the study from the desktop app, which builds a self-contained folder. Human data are never stored here: `study_core.py human-data` fetches them from their source.
