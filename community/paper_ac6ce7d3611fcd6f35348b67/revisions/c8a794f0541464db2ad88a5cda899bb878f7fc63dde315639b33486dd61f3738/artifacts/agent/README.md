# Hindsight Bias Through Knowledge Updating: A Conceptual Replication of Groß et al. (2023): agent skill

Barbara K. Kreis, Antje Hermann, Thorsten Pachur, and Julia Groß. Hindsight Bias Through Knowledge Updating: A Conceptual Replication of Groß et al. (2023).

Built with HumanStudy Hub. `kreis2025hindsight/` is a skill folder: `SKILL.md`, one readable reference per study with page numbers, `program.json` and the replay package. Any agent that reads skills can use it; it needs only Python 3.8+ (no packages, no keys).

Not ready to run: the study still needs decisions (iss_filler). Reading and querying work.

On Windows use `python` or `py` where these lines say `python3`.

## Install

```bash
# Claude Code
mkdir -p ~/.claude/skills && cp -R kreis2025hindsight ~/.claude/skills/
# Codex
mkdir -p ~/.agents/skills && cp -R kreis2025hindsight ~/.agents/skills/
# OpenCode: copy it into your project's .opencode/skills/
```

## MCP server (optional)

For clients that cannot run commands or read files. It serves the same folder:

```bash
claude mcp add kreis2025hindsight -- python3 "$PWD/kreis2025hindsight/scripts/mcp_server.py"
codex mcp add kreis2025hindsight -- python3 "$PWD/kreis2025hindsight/scripts/mcp_server.py"
```

OpenCode `opencode.json`:

```json
{
  "mcp": {
    "kreis2025hindsight": {
      "type": "local",
      "command": [
        "python3",
        "/absolute/path/kreis2025hindsight/scripts/mcp_server.py"
      ],
      "enabled": true
    }
  }
}
```

## Without an agent

```bash
python3 kreis2025hindsight/scripts/study_core.py overview
```

`MANIFEST.json` lists every file with its SHA-256. AI participants' answers are a simulation, not data from people.

## In the HumanStudy community library

This copy shares `program.json` and `package/` with the research files two folders up (`../../program.json`, `../../package/`), so they are stored once. The scripts find them there. To install the skill on its own, copy those two into the skill folder, or export the study from the desktop app (0.6.9 or later), which builds a self-contained folder.
