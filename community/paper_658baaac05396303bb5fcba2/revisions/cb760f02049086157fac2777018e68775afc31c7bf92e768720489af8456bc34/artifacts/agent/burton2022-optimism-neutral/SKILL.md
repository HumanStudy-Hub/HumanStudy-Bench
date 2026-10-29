---
name: burton2022-optimism-neutral
description: "Burton et al. Optimism Where There is None: Asymmetric Belief Updating Observed with Valence-Neutral Life Events: 4 studies as a HumanStudy program, with participants, design, conditions, verbatim materials, procedure, analysis, reported results and open issues, each cited to its page. Use when asked about this study, to brief or run AI participants in it, or to compare agent answers with the reported human results."
---

# Optimism Where There is None: Asymmetric Belief Updating Observed with Valence-Neutral Life Events

Jason W. Burton, Adam J. L. Harris, Punit Shah, & Ulrike Hahn. Optimism Where There is None: Asymmetric Belief Updating Observed with Valence-Neutral Life Events.

This folder is the study rebuilt as a program: what each participant saw and did, in order, and how the paper analysed it. Every fact comes from the paper and carries its page. `program.json` is the full machine-readable version.

## Studies

| Study | Title | Read | Conditions |
| --- | --- | --- | --- |
| `study1` | Study 1: update method, E2 after each event, valence rated in Part 2 | [study-study1.md](references/study-study1.md) | `cond-s1-e1first`, `cond-s1-ebrfirst` |
| `study2` | Study 2: all E2s in Part 2, valence rated in Part 3 | [study-study2.md](references/study-study2.md) | `cond-s2-e1first`, `cond-s2-ebrfirst` |
| `study3` | Study 3: valence rating before E2, E2 in Part 2 | [study-study3.md](references/study-study3.md) | `cond-s3-e1first`, `cond-s3-ebrfirst` |
| `agg123` | Aggregate analysis of Studies 1-3 and alternative analytic techniques | [study-agg123.md](references/study-agg123.md) | none recorded |

Also: [paper.md](references/paper.md) (background, findings, limitations), [open-issues.md](references/open-issues.md) (what the paper does not state) and [sources.md](references/sources.md) (source files and hashes) and [accuracy.md](references/accuracy.md) (76 quotes checked against their pages: 76 unchecked) and [capabilities.md](references/capabilities.md) (what can and cannot be done with this study, and why).

Search everything: `python scripts/study_core.py search <words>`.

## Run AI participants

Not ready to run: the study still needs decisions (iss-stimuli). Reading and querying work.

The replay package `package/burton2022-optimism-neutral/` holds the participant prompts (`task/adapter.py`) and the paper's analysis (`evaluation/evaluation.py`).

Each AI participant must answer in its own fresh context that sees only its prompt. Never answer several prompts yourself in this conversation: you know the hypothesis, so those answers would not be independent participants.

With an agent CLI you are signed in to (no key needed), one new process per participant:

```bash
python scripts/study_core.py run --n 5 --cli "claude -p --model haiku"
python scripts/study_core.py run --n 5 --cli "codex exec --skip-git-repo-check -s read-only -"
```

Each process gets only the prompt on stdin, in an empty folder, so it cannot read this study. With an API key instead: `--model <model-id>` (`OPENROUTER_API_KEY`, or `OPENAI_API_KEY` with `OPENAI_BASE_URL`). To collect answers some other way: `plan --n 5` lists every prompt, and `replay replies.json --n 5` scores the replies given in the same order.

## Rules

- Answer only from these files and cite the page given. If something is not here, say so.
- Materials marked verbatim are shown to participants exactly as written. Never paraphrase them.
- Items that read "Not stated in the paper" and open issues stay open: ask the researcher; never fill them with plausible text.
- Results from AI participants are a simulation. They are not observations of people.
- Do only what capabilities.md allows. A refused request (status `refused`) is reported with its reason; never replace it with an approximation of your own.
