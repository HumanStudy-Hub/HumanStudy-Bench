# Chart and interpret a finished playground run

You are given a finished playground run directory. Read the results and write a
short set of charts plus a plain-language interpretation to `output/charts.json`.

## Inputs (all inside the run directory)

- `run.json` — the model, study id/title, prompt, and run summary.
- `output/analysis.json` — benchmark-shaped results (`summary` + per-test rows).
  Present for merged benchmark studies.
- `output/evaluation.json` — the study's own evaluator output. For buffer
  (agent-built) studies this is the authoritative result; its shape varies per
  study. Read it and chart whatever it reports (means, rates, per-cell
  comparisons, direction checks, `not_ready` gaps).
- `output/transcript_sample.json` — sample participant exchanges.

Prefer `evaluation.json` for buffer studies and `analysis.json` for benchmark
studies. If both are present, use whichever carries the richer comparison.

## Output

Write `output/charts.json` exactly once, as a JSON object:

```json
{
  "charts": [
    {
      "id": "unique-id",
      "title": "short title",
      "description": "one sentence",
      "plotly": {"data": [{"type": "bar", "x": [...], "y": [...]}], "layout": {}}
    }
  ],
  "interpretation": "plain-language reading of the run"
}
```

Rules:

- 1 to 6 charts, each with a unique `id`, a non-empty `title`, and a
  `description`.
- `plotly` is a Plotly figure: `data` is a non-empty list of traces; each trace
  has a `type` from `scatter`, `scattergl`, `bar`, `box`, `violin`, `histogram`,
  `heatmap`, `line`, and at least one of `x`/`y`/`z`/`values`/`labels`.
- Plain JSON data only: numbers, strings, lists, objects. No NaN/Infinity, no
  HTML, no `meta`/`src`/`textsrc`-style fields.
- `interpretation` is a string of at most a few paragraphs describing what the
  run shows, including any `not_ready` checks and what is missing.
- If the run has no plottable results, write `"charts": []` and a short
  `interpretation` explaining why.
