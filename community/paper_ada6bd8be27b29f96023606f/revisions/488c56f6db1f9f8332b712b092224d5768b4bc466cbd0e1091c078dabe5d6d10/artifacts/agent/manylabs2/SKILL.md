---
name: manylabs2
description: "Klein et al. Many Labs 2: Investigating Variation in Replicability Across Sample and Setting: 3 studies as a HumanStudy program, with participants, design, conditions, verbatim materials, procedure, analysis, reported results and open issues, each cited to its page. Use when asked about this study, to brief or run AI participants in it, or to compare agent answers with the reported human results."
---

# Many Labs 2: Investigating Variation in Replicability Across Sample and Setting

Richard A. Klein, Michelangelo Vianello, Fred Hasselman, et al., Brian A. Nosek. Many Labs 2: Investigating Variation in Replicability Across Sample and Setting.

This folder is the study rebuilt as a program: what each participant saw and did, in order, and how the paper analysed it. Every fact comes from the paper and carries its page. `program.json` is the full machine-readable version.

## Studies

| Study | Title | Read | Conditions |
| --- | --- | --- | --- |
| `slate1` | Many Labs 2, Slate 1 (13 effects, 64 samples) | [study-slate1.md](references/study-slate1.md) | `c-huan-high`, `c-huan-low`, `c-kay-structure`, `c-kay-random`, `c-alt-easy`, `c-alt-hard`, `c-rott-certain`, `c-rott-lottery`, `c-bau-consumer`, `c-bau-individual`, `c-miya-pro`, `c-miya-anti`, `c-inba-gay`, `c-inba-couples`, `c-crit-p97`, `c-crit-p17`, `c-haus12-side`, `c-haus12-means`, `c-and-top`, `c-and-bottom` |
| `slate2` | Many Labs 2, Slate 2 (15 effects, 61 samples) | [study-slate2.md](references/study-slate2.md) | `c-geis-long`, `c-geis-short`, `c-tver-cheap`, `c-tver-costly`, `c-haus34-ned`, `c-haus34-oscar`, `c-rise-tempt`, `c-rise-notempt`, `c-sava-personal`, `c-sava-interpersonal`, `c-nore-belong`, `c-nore-similar`, `c-hsee-scarf`, `c-hsee-coat`, `c-gray-adultharm`, `c-gray-babyharm`, `c-zhon-unethical`, `c-zhon-ethical`, `c-schw-specific`, `c-schw-general`, `c-shaf-award`, `c-shaf-deny`, `c-zav-heat`, `c-zav-cold`, `c-knob-help`, `c-knob-harm`, `c-gati-sim1`, `c-gati-sim2`, `c-gati-diff1`, `c-gati-diff2` |
| `variation` | Variation across samples and settings (cross-slate analysis) | [study-variation.md](references/study-variation.md) | none recorded |

Also: [paper.md](references/paper.md) (background, findings, limitations), [open-issues.md](references/open-issues.md) (what the paper does not state) and [sources.md](references/sources.md) (source files and hashes) and [accuracy.md](references/accuracy.md) (222 quotes checked against their pages: 222 unchecked) and [capabilities.md](references/capabilities.md) (what can and cannot be done with this study, and why).

Search everything: `python scripts/study_core.py search <words>`.

## Run AI participants

Not ready to run: the study still needs decisions (iss-score-zhon). Reading and querying work.

The replay package `package/manylabs2/` holds the participant prompts (`task/adapter.py`) and the paper's analysis (`evaluation/evaluation.py`).

Each AI participant must answer in its own fresh context that sees only its prompt. Never answer several prompts yourself in this conversation: you know the hypothesis, so those answers would not be independent participants.

With an agent CLI you are signed in to (no key needed), one new process per participant:

```bash
python scripts/study_core.py run --n 5 --cli "claude -p --model haiku"
python scripts/study_core.py run --n 5 --cli "codex exec --skip-git-repo-check -s read-only -"
```

Each process gets only the prompt on stdin, in an empty folder, so it cannot read this study. With an API key instead: `--model <model-id>` (`OPENROUTER_API_KEY`, or `OPENAI_API_KEY` with `OPENAI_BASE_URL`). To collect answers some other way: `plan --n 5` lists every prompt, and `replay replies.json --n 5` scores the replies given in the same order.

## Human data

The original human data are not in this folder. `python scripts/study_core.py human-data` downloads them from their source and checks every file against its pinned SHA-256. `python scripts/study_core.py reproduce` recomputes the paper's key results from them with the package's analysis: trust the analysis on AI participants only as far as it recovers the paper here. If the data are a replication sample, say so when you report.

## Effects

`effects.json` links the paper's 28 hypotheses to 41 effects: outcome, contrast (treatment minus reference), predicted direction, the paper's reported numbers (each quoted, with its page) and how each is computed from sessions. `python scripts/study_core.py effects` lists them (filter with `--hypothesis`, `--phenomenon`, `--topic` or `--role`). Every run and every reproduction returns one result block per effect in the same shape (cells, test, metric), for human and AI-participant sessions alike.

## Rules

- Answer only from these files and cite the page given. If something is not here, say so.
- Materials marked verbatim are shown to participants exactly as written. Never paraphrase them.
- Items that read "Not stated in the paper" and open issues stay open: ask the researcher; never fill them with plausible text.
- Results from AI participants are a simulation. They are not observations of people.
- Do only what capabilities.md allows. A refused request (status `refused`) is reported with its reason; never replace it with an approximation of your own.
