# Many Labs 2: Investigating Variation in Replicability Across Sample and Setting: agent skill

Richard A. Klein, Michelangelo Vianello, Fred Hasselman, et al., Brian A. Nosek. Many Labs 2: Investigating Variation in Replicability Across Sample and Setting.

Built with HumanStudy Hub. `manylabs2/` is a skill folder: `SKILL.md`, one readable reference per study with page numbers, `program.json` and the replay package. Any agent that reads skills can use it; it needs only Python 3.8+ (no packages, no keys).

Not ready to run: the study still needs decisions (iss-score-zhon). Reading and querying work.

On Windows use `python` or `py` where these lines say `python3`.

## Install

```bash
# Claude Code
mkdir -p ~/.claude/skills && cp -R manylabs2 ~/.claude/skills/
# Codex
mkdir -p ~/.agents/skills && cp -R manylabs2 ~/.agents/skills/
# OpenCode: copy it into your project's .opencode/skills/
```

## MCP server (optional)

For clients that cannot run commands or read files. It serves the same folder:

```bash
claude mcp add manylabs2 -- python3 "$PWD/manylabs2/scripts/mcp_server.py"
codex mcp add manylabs2 -- python3 "$PWD/manylabs2/scripts/mcp_server.py"
```

OpenCode `opencode.json`:

```json
{
  "mcp": {
    "manylabs2": {
      "type": "local",
      "command": [
        "python3",
        "/absolute/path/manylabs2/scripts/mcp_server.py"
      ],
      "enabled": true
    }
  }
}
```

## Without an agent

```bash
python3 manylabs2/scripts/study_core.py overview
```

`MANIFEST.json` lists every file with its SHA-256. AI participants' answers are a simulation, not data from people.

## In the HumanStudy community library

This copy shares `program.json`, `package/` and `capabilities.json` with the research files two folders up (`../../`), so they are stored once. The scripts find them there. To install the skill on its own, copy those three into the skill folder, or export the study from the desktop app, which builds a self-contained folder. Human data are never stored here: `study_core.py human-data` fetches them from their source.
