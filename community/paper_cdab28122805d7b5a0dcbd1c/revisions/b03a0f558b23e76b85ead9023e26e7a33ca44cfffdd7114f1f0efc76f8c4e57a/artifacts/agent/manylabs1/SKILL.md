---
name: manylabs1
description: "Klein et al. (2014) Investigating Variation in Replicability: A 'Many Labs' Replication Project: 13 studies as a HumanStudy program, with participants, design, conditions, verbatim materials, procedure, analysis, reported results and open issues, each cited to its page. Use when asked about this study, to brief or run AI participants in it, or to compare agent answers with the reported human results."
---

# Investigating Variation in Replicability: A 'Many Labs' Replication Project

Klein, R. A., Ratliff, K. A., Vianello, M., Adams, R. B. Jr., Bahnik, S., Bernstein, M. J., et al., Nosek, B. A. (2014). Investigating Variation in Replicability: A 'Many Labs' Replication Project. Social Psychology 2014; 45(3):142-152. https://doi.org/10.1027/1864-9335/a000178.

This folder is the study rebuilt as a program: what each participant saw and did, in order, and how the paper analysed it. Every fact comes from the paper and carries its page. `program.json` is the full machine-readable version.

## Studies

| Study | Title | Read | Conditions |
| --- | --- | --- | --- |
| `s01` | Study 1: Sunk costs | [study-s01.md](references/study-s01.md) | `s01_c_paid`, `s01_c_free` |
| `s02` | Study 2: Gain versus loss framing | [study-s02.md](references/study-s02.md) | `s02_c_gain`, `s02_c_loss` |
| `s03` | Study 3: Anchoring | [study-s03.md](references/study-s03.md) | `s03_c_low`, `s03_c_high` |
| `s04` | Study 4: Retrospective gambler's fallacy | [study-s04.md](references/study-s04.md) | `s04_c_three6`, `s04_c_two6one3` |
| `s05` | Study 5: Low-versus-high category scales | [study-s05.md](references/study-s05.md) | `s05_c_low`, `s05_c_high` |
| `s06` | Study 6: Norm of reciprocity | [study-s06.md](references/study-s06.md) | `s06_c_us_first`, `s06_c_nk_first` |
| `s07` | Study 7: Allowed/Forbidden | [study-s07.md](references/study-s07.md) | `s07_c_allow`, `s07_c_forbid` |
| `s08` | Study 8: Quote attribution | [study-s08.md](references/study-s08.md) | `s08_c_liked`, `s08_c_disliked` |
| `s09` | Study 9: Flag priming | [study-s09.md](references/study-s09.md) | `s09_c_flag`, `s09_c_control` |
| `s10` | Study 10: Currency priming | [study-s10.md](references/study-s10.md) | `s10_c_money`, `s10_c_control` |
| `s11` | Study 11: Imagined contact | [study-s11.md](references/study-s11.md) | `s11_c_contact`, `s11_c_control` |
| `s12` | Study 12: Implicit math attitudes (sex differences and implicit-explicit relation) | [study-s12.md](references/study-s12.md) | none recorded |
| `s13` | Cross-site analysis of replicability and moderation | [study-s13.md](references/study-s13.md) | none recorded |

Also: [paper.md](references/paper.md) (background, findings, limitations), [open-issues.md](references/open-issues.md) (what the paper does not state) and [sources.md](references/sources.md) (source files and hashes) and [accuracy.md](references/accuracy.md) (154 quotes checked against their pages: 154 unchecked) and [capabilities.md](references/capabilities.md) (what can and cannot be done with this study, and why).

Search everything: `python scripts/study_core.py search <words>`.

## Run AI participants

Ready to run the parts the replay package covers (see `package/manylabs1/task/task.json`). Results from AI participants are a simulation, not observations of people.

The replay package `package/manylabs1/` holds the participant prompts (`task/adapter.py`) and the paper's analysis (`evaluation/evaluation.py`).

Each AI participant must answer in its own fresh context that sees only its prompt. Never answer several prompts yourself in this conversation: you know the hypothesis, so those answers would not be independent participants.

With an agent CLI you are signed in to (no key needed), one new process per participant:

```bash
python scripts/study_core.py run --n 5 --cli "claude -p --model haiku"
python scripts/study_core.py run --n 5 --cli "codex exec --skip-git-repo-check -s read-only -"
```

Each process gets only the prompt on stdin, in an empty folder, so it cannot read this study. With an API key instead: `--model <model-id>` (`OPENROUTER_API_KEY`, or `OPENAI_API_KEY` with `OPENAI_BASE_URL`). To collect answers some other way: `plan --n 5` lists every prompt, and `replay replies.json --n 5` scores the replies given in the same order.

## Human data

The original human data are not in this folder. `python scripts/study_core.py human-data` downloads them from their source and checks every file against its pinned SHA-256. `python scripts/study_core.py reproduce` recomputes the paper's key results from them with the package's analysis: trust the analysis on AI participants only as far as it recovers the paper here. If the data are a replication sample, say so when you report.

## Rules

- Answer only from these files and cite the page given. If something is not here, say so.
- Materials marked verbatim are shown to participants exactly as written. Never paraphrase them.
- Items that read "Not stated in the paper" and open issues stay open: ask the researcher; never fill them with plausible text.
- Results from AI participants are a simulation. They are not observations of people.
- Do only what capabilities.md allows. A refused request (status `refused`) is reported with its reason; never replace it with an approximation of your own.
