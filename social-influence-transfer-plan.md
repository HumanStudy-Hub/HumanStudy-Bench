# Social Influence Transfer: Simulation-as-Calibration

## Framing

We treat agent simulation of social-influence effects as a **calibration problem**,
not a correctness problem. The target is matching human response *distributions*,
not Bayesian-optimal answers. Data contamination is not the central risk — even
when a base model can recite a paper's findings, it may still fail to reproduce
the corresponding human behavior in-context. That gap is the thing worth measuring.

Central question the paper aims to answer: **should agent simulation of social
influence be general-purpose (one adapter transfers across paradigms) or
domain-specific (each paradigm/scenario needs its own tuning)?**

Scope is unchanged: three papers, same family (group influence on individual
judgment). What changed from the original plan: C is no longer only a held-out
eval — it also gets its own directly-trained adapter, to serve as a ceiling.

## The three studies

**A — Information cascades**
Anderson, L. R., & Holt, C. A. (1997). Information Cascades in the Laboratory.
*American Economic Review*, 87(5), 847–862.
Sequential urn-guessing task. 6 positions, 2 private signal values, 2^(n-1)
possible public histories at position n → 126 possible (position, signal,
history) prompts, fully enumerable, no sampling needed.

**B — Source calibration (agreement vs. accuracy)**
Jaquiery, M., & Yeung, N. (2024). Preferences for Advisor Agreement and
Accuracy. *PLOS ONE*, 19(9), e0311211.
Use Experiment 3C only (feedback condition). Two advisors: accurate-but-
disagreeing, agreeing-but-inaccurate. ~100 simulated familiarization episodes
(15+15 exposures each, randomized), ending in an advisor choice. Exp 3B
(no-feedback) is a control, not used for training — human choice there is
~50:50 with wide individual variation, so there's no stable target distribution.
Participant-level data available via Zenodo (R package).

**C — Authority effect**
Schöbel, M., Rieskamp, J., & Huber, R. (2016). Social Influences in Sequential
Decision Making. *PLOS ONE*, 11(1), e0146536.
40 fixed medical scenarios (appendicitis vs. sigmoid diverticulitis). Assistant
physician and medical director are explicitly stated to have equal independent
diagnostic accuracy (p = 0.67 both), so rank carries no normative information —
any extra weight on the director's opinion is pure authority bias. Scenarios
group into 4 Bayesian-posterior buckets (0.50 / 0.67 / 0.80 / 0.89) crossed with
director condition (absent / supports private signal / contradicts private
signal). Aggregate human choice proportions available in paper Tables 3–6 — not
individual-level, so C training/eval works at the population-calibration level
only, not per-subject.

## Open item before writing generation code

Confirm the granularity of human data available for **A**. If Anderson & Holt
only report an aggregate cascade-formation rate rather than proportions broken
out by (public-history-strength × private-signal-conflict), we fall back to one
pooled proportion for all "conflict" prompts in A. Note as a limitation if so.

## Training approach: proportional DPO

DPO labels are binary, but human preference is a distribution. Resolve this by
generating multiple training pairs per prompt, with chosen/rejected assigned
in proportion to the human split — not a single "correct answer" label.

Example: scenario where humans choose appendicitis 70% of the time, N=20:

```python
n_major = round(N * 0.70)  # 14
for i in range(N):
    if i < n_major:
        pairs.append((prompt, chosen="A", rejected="B"))
    else:
        pairs.append((prompt, chosen="B", rejected="A"))
```

This is population-level calibration (sample many times, aggregate matches human
distribution), not individual-level (persisting one simulated subject's
consistent bias across items). v1 is population-level only; individual-level is
a limitation / future work — Schöbel's public data doesn't support it anyway
without requesting raw data, and A's granularity is uncertain (see above).

**Known math**: at the DPO loss optimum for a given prompt, model log-odds ≈
base-model log-odds + logit(human proportion) / β. This means:
- β controls how sharply the human proportion gets imprinted — the standard
  RLHF default β=0.1 will over-sharpen a 70:30 human split toward ~99.9:0.1.
  **Must sweep β** (e.g. {0.1, 0.3, 1.0}) rather than using the RLHF default.
- This is the population loss optimum; with limited LoRA capacity, small data,
  and early stopping, actual training won't hit this exactly — treat as
  directional guidance for tuning, not a guaranteed outcome.

Confidence ratings (where collected, e.g. in C) are continuous and awkward for
DPO — use them for evaluation only, not as a training signal.

## Evaluation

No sampling needed. For a single-token forced-choice prompt ending in "Answer
with a single letter, A or B.\nAnswer:", read the logits for the two answer
tokens directly, normalize, and compare to the human proportion. Fast (seconds
for all 40 C scenarios), noise-free (no sampling variance). Metric: MAE between
model proportion and human proportion, plus a scatter plot (human % vs. model %,
ideal = diagonal) — this plot doubles as the calibration-framing figure for the
paper/intro.

## Three gates (stop if a gate fails — don't proceed past a failed gate)

**Gate 0 — Base model profile (do this first, ~2 days)**
Run untrained base model over all A/B/C prompt sets, compute MAE vs. human
proportions per study, produce the scatter plot. Also run a cheap knowledge
probe: ask the model directly what each paper found (e.g. "What did Anderson &
Holt 1997 find?"). Purpose: establish that behavioral failure isn't explained by
lack of knowledge of the source material — directly supports the
"contamination isn't the real risk" framing.
*Stop condition*: if base MAE is already low (e.g. <0.10) across studies, the
model already behaves like humans and there's no calibration gap to close —
report that and stop.

**Gate 1 — Direct in-domain training on C (~3 days)**
Split the 40 C scenarios into train/test (20/20), stratified by Bayesian-
posterior bucket and director condition. Train LoRA (r=8, attention
projections) directly on C's own scenarios using the proportional-DPO scheme
above, sweeping β ∈ {0.1, 0.3, 1.0}. Evaluate MAE on held-out C scenarios.
This is the ceiling — the best a domain-specific adapter can do.
*Stop condition*: if in-domain training doesn't meaningfully reduce MAE vs.
Gate 0, the calibration approach doesn't work at all — report that and stop;
no point testing transfer.

**Gate 2 — A+B trained, zero-shot on C (~3 days)**
Train a single LoRA adapter (not two adapters merged — avoids merge-coefficient
tuning and interference as an extra failure mode) on the combined A+B
proportional-DPO data, using the β chosen in Gate 1. Evaluate zero-shot on all
40 C scenarios (C never appears in training).

Report:

| | C MAE |
|---|---|
| Base (Gate 0) | floor |
| A+B transfer (Gate 2) | ? |
| C direct training (Gate 1) | ceiling |

recovery fraction = (base − transfer) / (base − ceiling)

This number is the direct answer to "general-purpose vs. domain-specific."

## Reporting convention (do not collapse into one number)

Because "better on C" has two opposite readings — normative (smaller authority
effect = more Bayesian) vs. behavioral (larger authority effect = more human-
like) — and we've now committed to the behavioral/calibration framing, report
distance-to-human-distribution (MAE) as the primary metric throughout, and
separately note the direction of any residual authority bias (over- or under-
shooting the human curve), rather than a single "performance improved" claim.

## Fill-in-the-blank abstract (draft after Gate 0/1/2 numbers are in)

> We treat agent simulation of social influence as a calibration problem. On
> three classic paradigms, base model behavior diverges from human data by
> ___ (Gate 0), despite the model accurately recalling the papers' own
> findings — indicating the failure is not explained by missing knowledge.
> Using human response proportions as DPO preference labels closes this gap to
> ___ in-domain (Gate 1). An adapter trained only on the other two paradigms,
> transferred zero-shot to an unseen authority-effect scenario set, recovers
> ___% of the in-domain gain (Gate 2), indicating that calibrating social
> behavior ___ (can / cannot) be done in a general-purpose way.

## Compute estimate

7B base model, LoRA, ~2500 (A) + ~2000 (B) + ~400 (C) training pairs, few
epochs — single GPU, ~1-2 hours per training run. Evaluation is single-token
logit reads, seconds per study. Compute is not the bottleneck; the bottleneck
is sourcing per-scenario human proportions from the papers/data and building
prompt templates. Full pipeline (Gates 0-2) is a ~2 week timeline.
