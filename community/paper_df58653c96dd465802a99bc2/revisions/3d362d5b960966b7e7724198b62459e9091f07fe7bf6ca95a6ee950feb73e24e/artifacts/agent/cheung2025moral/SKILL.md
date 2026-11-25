---
name: cheung2025moral
description: "Cheung et al. Large Language Models Show Amplified Cognitive Biases in Moral Decision-Making: 4 studies as a HumanStudy program, with participants, design, conditions, verbatim materials, procedure, analysis, reported results and open issues, each cited to its page. Use when asked about this study, to brief or run AI participants in it, or to compare agent answers with the reported human results."
---

# Large Language Models Show Amplified Cognitive Biases in Moral Decision-Making

Vanessa Cheung, Maximilian Maier, Falk Lieder. Large Language Models Show Amplified Cognitive Biases in Moral Decision-Making. PNAS (preprint of an article in press).

This folder is the study rebuilt as a program: what each participant saw and did, in order, and how the paper analysed it. Every fact comes from the paper and carries its page. `program.json` is the full machine-readable version.

## Studies

| Study | Title | Read | Conditions |
| --- | --- | --- | --- |
| `s1` | Study 1: LLMs versus a representative U.S. sample on 13 moral dilemmas and nine collective action problems | [study-s1.md](references/study-s1.md) | `c1_action`, `c1_omission`, `c1_p_part`, `c1_p_adv` |
| `s2` | Study 2: Yes-No and Action-Omission reframing of six moral dilemmas (preregistered) | [study-s2.md](references/study-s2.md) | `c2_orig`, `c2_yn`, `c2_om`, `c2_p_part`, `c2_p_adv`, `c2_p_exp_part`, `c2_p_exp_adv`, `c2_p_sil_part`, `c2_p_sil_adv` |
| `s3` | Study 3: Everyday (AITA) dilemmas with Yes-No and Action-Omission reframing (preregistered) | [study-s3.md](references/study-s3.md) | `c3_orig`, `c3_yn`, `c3_om`, `c3_p_part`, `c3_p_adv` |
| `s4` | Study 4: Effect of post-training on the biases (Llama 3.1 pre-trained, Llama 3.1-Instruct, Centaur) | [study-s4.md](references/study-s4.md) | `c4_pre`, `c4_inst`, `c4_cent` |

Also: [paper.md](references/paper.md) (background, findings, limitations), [open-issues.md](references/open-issues.md) (what the paper does not state) and [sources.md](references/sources.md) (source files and hashes) and [accuracy.md](references/accuracy.md) (191 quotes checked against their pages: 191 unchecked) and [capabilities.md](references/capabilities.md) (what can and cannot be done with this study, and why).

Search everything: `python scripts/study_core.py search <words>`.

## Run AI participants

Ready to run the parts the replay package covers (see `package/cheung2025moral/task/task.json`). Results from AI participants are a simulation, not observations of people.

The replay package `package/cheung2025moral/` holds the participant prompts (`task/adapter.py`) and the paper's analysis (`evaluation/evaluation.py`).

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

`effects.json` links the paper's 5 hypotheses to 75 effects: outcome, contrast (treatment minus reference), predicted direction, the paper's reported numbers (each quoted, with its page) and how each is computed from sessions. `python scripts/study_core.py effects` lists them (filter with `--hypothesis`, `--phenomenon`, `--topic` or `--role`). Every run and every reproduction returns one result block per effect in the same shape (cells, test, metric), for human and AI-participant sessions alike. `python scripts/study_core.py graph` (MCP: `study_graph`) gives the paper as one typed graph: hypotheses, effects, constructs, conditions, variables, materials, steps, analyses and results, joined by core edge types; `--node <id>` shows what one node connects to.

## Rules

- Answer only from these files and cite the page given. If something is not here, say so.
- Materials marked verbatim are shown to participants exactly as written. Never paraphrase them.
- Items that read "Not stated in the paper" and open issues stay open: ask the researcher; never fill them with plausible text.
- Results from AI participants are a simulation. They are not observations of people.
- Do only what capabilities.md allows. A refused request (status `refused`) is reported with its reason; never replace it with an approximation of your own.
