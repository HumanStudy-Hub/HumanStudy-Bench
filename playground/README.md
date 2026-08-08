# Playground

Replay one benchmark study with a model of your choice and see where the agent
matched the published human result and where it did not.

This is the engine behind the HumanStudy-Hub playground. It does not
reimplement any study: it drives each study's own `scripts/config.py` to build
trials and its own `scripts/evaluator.py` to score the run, so a playground run
is scored exactly like a benchmark run.

## Run it locally

```bash
python playground/run_playground.py --run <run-dir>
```

The run directory holds `run.json` and receives `progress.json`, `logs/`, and
`output/`:

```json
{
  "id": "local-1",
  "studyId": "study_001",
  "model": "openai/gpt-4o-mini",
  "preset": "v3_human_plus_demo",
  "systemPrompt": "",
  "participantsPerScenario": 8,
  "temperature": 1.0,
  "seed": 42,
  "demographics": { "age": 21, "gender": "female", "background": "undergraduate" }
}
```

`OPENROUTER_API_KEY` must be set. Every model — including Claude and Gemini
names — is routed through OpenRouter. Add `--simulate` to exercise the whole
path with simulated participants and no API calls.

Outputs:

| File | Contents |
|---|---|
| `output/responses.json` | Every participant response, in the shape evaluators read |
| `output/evaluation.json` | The study evaluator's own output |
| `output/analysis.json` | One row per statistical test: human result vs. agent result |
| `output/transcript_sample.json` | A few complete participant exchanges |
| `output/charts.json` | Plotly chart specifications and a written reading of the run |

## Prompts

`preset` selects a shipped participant prompt from
`src/agents/custom_methods/`: `v1_empty`, `v2_human`, `v3_human_plus_demo`, or
`v4_background`. Set `preset` to `custom` and supply `systemPrompt` to run a
prompt you wrote yourself. `demographics` overrides the sampled participant
profile — age, gender, education, background, population, persona — which is
what the demographic presets read.

A custom prompt may contain `{{age}}`, `{{gender}}`, `{{background}}`,
`{{persona}}`, or any other profile field. Placeholders are filled in per agent,
so a hand-written prompt still varies across participants. Without a
placeholder, every agent receives exactly the same prompt.

## Personas

`demographics` pins one identity onto every participant. A `personaGroup`
instead describes the population the participants are drawn from:

```json
{
  "personaGroup": {
    "name": "Nurses and students",
    "segments": [
      {
        "id": "nurses",
        "label": "Hospital nurses",
        "share": 0.3,
        "age": { "min": 28, "max": 55 },
        "gender": { "female": 0.8, "male": 0.2 },
        "background": "works night shifts in an emergency department",
        "persona": "You are a hospital nurse."
      },
      { "id": "students", "label": "Undergraduates", "share": 0.7, "age": { "min": 18, "max": 22 } }
    ]
  }
}
```

Shares are relative and are normalised, so `30`/`70` and `0.3`/`0.7` mean the
same thing. Participants are split by largest remainder, and no segment is ever
dropped when there are enough participants to go around — a segment that never
appears would be a silently ignored part of the design. Because a group is a
population rather than a fixed cast, one saved group fits a 20-session run and a
600-session run alike.

Sampling is deterministic: the same group and `seed` reproduce the same
participants exactly. The resolved cast is written to `output/profiles.json`.

Groups contributed through HumanStudy-Hub are stored in
[`profiles/`](profiles) as `<study>-<contributor>-<n>.json`.

## Charts

```bash
python playground/run_charts.py --run <run-dir>
```

Claude Code reads the finished run and writes `output/charts.json` following
[`CLAUDE.md`](CLAUDE.md). Its output is checked by `validate_charts.py` before
it is kept: charts must be plain JSON, restricted to an allowlist of trace
types. Anything invalid, missing, or late falls back to the deterministic charts
in `default_charts.py`, so a run always ends with readable charts.

## Run budget

A run on the shared HumanStudy-Hub key is capped at 60 participant sessions.
A researcher who supplies their own OpenRouter key gets 600. The web app seals
that key with AES-256-GCM under `PLAYGROUND_KEY_SECRET`, so the private jobs
repository only ever holds ciphertext, and the workflow drops it once the run
ends.

## In CI

[`run-playground.yml`](../.github/workflows/run-playground.yml) is dispatched by
HumanStudy-Hub. Repository settings it needs:

| Type | Name | Value |
|---|---|---|
| Actions secret | `OPENROUTER_API_KEY` | OpenRouter key for shared runs and the charting agent |
| Actions secret | `HUMANSTUDY_PIPELINE_TOKEN` | Token with read/write access to the private jobs repository |
| Actions secret | `PLAYGROUND_KEY_SECRET` | Shared secret that opens a researcher's own key; must match the web app |
| Actions variable | `PLAYGROUND_CHARTS_MODEL` | Optional; defaults to `anthropic/claude-sonnet-5` |
