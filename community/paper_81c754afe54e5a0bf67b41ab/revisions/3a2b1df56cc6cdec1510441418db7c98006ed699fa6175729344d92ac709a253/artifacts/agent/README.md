# Fighting COVID-19 Misinformation on Social Media: Experimental Evidence for a Scalable Accuracy-Nudge Intervention: agent skill

Gordon Pennycook, Jonathon McPhetres, Yunhao Zhang, Jackson G. Lu, and David G. Rand (2020). Fighting COVID-19 Misinformation on Social Media: Experimental Evidence for a Scalable Accuracy-Nudge Intervention. Psychological Science, 2020, 31(7), 770-780. https://doi.org/10.1177/0956797620939054.

Built with HumanStudy Hub. `pennycook2020-covid-accuracy-nudge/` is a skill folder: `SKILL.md`, one readable reference per study with page numbers, `program.json` and the replay package. Any agent that reads skills can use it; it needs only Python 3.8+ (no packages, no keys).

Ready to run the parts the replay package covers (see `package/pennycook2020/task/task.json`). Results from AI participants are a simulation, not observations of people.

On Windows use `python` or `py` where these lines say `python3`.

## Install

```bash
# Claude Code
mkdir -p ~/.claude/skills && cp -R pennycook2020-covid-accuracy-nudge ~/.claude/skills/
# Codex
mkdir -p ~/.agents/skills && cp -R pennycook2020-covid-accuracy-nudge ~/.agents/skills/
# OpenCode: copy it into your project's .opencode/skills/
```

## MCP server (optional)

For clients that cannot run commands or read files. It serves the same folder:

```bash
claude mcp add pennycook2020-covid-accuracy-nudge -- python3 "$PWD/pennycook2020-covid-accuracy-nudge/scripts/mcp_server.py"
codex mcp add pennycook2020-covid-accuracy-nudge -- python3 "$PWD/pennycook2020-covid-accuracy-nudge/scripts/mcp_server.py"
```

OpenCode `opencode.json`:

```json
{
  "mcp": {
    "pennycook2020-covid-accuracy-nudge": {
      "type": "local",
      "command": [
        "python3",
        "/absolute/path/pennycook2020-covid-accuracy-nudge/scripts/mcp_server.py"
      ],
      "enabled": true
    }
  }
}
```

## Without an agent

```bash
python3 pennycook2020-covid-accuracy-nudge/scripts/study_core.py overview
```

`MANIFEST.json` lists every file with its SHA-256. AI participants' answers are a simulation, not data from people.

## In the HumanStudy community library

This copy shares `program.json`, `package/` and `capabilities.json` with the research files two folders up (`../../`), so they are stored once. The scripts find them there. To install the skill on its own, copy those three into the skill folder, or export the study from the desktop app, which builds a self-contained folder.
