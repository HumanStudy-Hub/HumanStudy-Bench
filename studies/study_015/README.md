# Testing the Decoy Effect to Improve Online Survey Participation: Evidence from a Field Experiment

**Authors:** Sandro T. Stoffel, Yining Sun, Yasemin Hirst, Christian von Wagner, Ivo Vlaev
**Year:** 2023
**Published:** Journal of Behavioral Decision Making

---

## Description

This field experiment tests whether adding an inferior decoy survey option increases completion of a target online questionnaire about fear of coronavirus. The study examines the classical "decoy effect" (or attraction effect) in the context of survey participation, with implications for improving response rates in social science research. A particularly novel aspect is the examination of presentation order effects: does the effect of the decoy depend on whether the target or decoy option appears first in the comparison table?

---

## Experiments Included

This benchmark contains two nested experiments:

1. **Preliminary Questionnaire Study**: Online recruitment and manipulation validation (N=210). Participants indicated preferences for question types (closed-ended vs open-ended) and payment timing (1 week vs 4 weeks).

2. **Main Field Experiment**: Between-subjects field experiment (N=203 with valid emails, randomized to 3 conditions):
   - Control condition: Only target survey option offered
   - Decoy condition (target first): Both options shown, target survey shown first
   - Decoy condition (decoy first): Both options shown, decoy survey shown first

---

## Participants

**Preliminary Study:**
- **N:** 210 who completed the preliminary questionnaire (241 started, 216 completed, 210 provided email addresses)
- **Recruitment:** Students at UK-based university via Facebook, WhatsApp, WeChat (August 2022)
- **Demographics:** Age 20-24: 26.7%, 25-29: 50.9%, 30-35: 22.4%; Male: 52.4%, Female: 47.6%; White: 57.1%, Asian: 23.8%, Black: 17.1%, Other: 1.9%; Christian: 71.9%, Other religion: 19.5%, No religion: 8.6%; Bachelor's: 51.9%, Graduate: 48.1%

**Main Experiment:**
- **N:** 203 with valid emails (210 randomized, 7 bounce; control n=101, decoy n=102 split as target-first n=52, decoy-first n=50)
- **Sample characteristics:** Age 20-24: 27.1%, 25-29: 50.2%, 30-35: 22.7%; Male: 52.2%, Female: 47.8%; White: 57.6%, Asian: 23.7%, Black: 16.7%, Other: 2.0%; Christian: 71.9%, Other religion: 8.9%, No religion: 19.2%; Bachelor's: 51.7%, Graduate: 48.3%

---

## Primary Outcomes

**Target Survey Completion Rate:**
- Control: 33/101 = 32.7%
- Decoy (overall): 57/102 = 55.9%
  - Target first: 43/52 = 82.7%
  - Decoy first: 14/50 = 28.0%

---

## Key Findings (Human Data)

### F1: Randomization Balance
- Age difference: χ², p = 0.165 (no difference)
- Gender difference: χ², p = 0.441 (no difference)
- **Ethnicity difference: χ², p = 0.004** (imbalance: decoy condition had more White participants)
- Religion difference: χ², p = 0.063 (marginal, no difference at α=0.05)
- Education difference: χ², p = 0.233 (no difference)

### F2: The Decoy Effect on Target Survey Completion
- **Chi-square test:** χ²(1, N=203) = 11.08, p < 0.001
- **Unadjusted OR:** 2.610 (95% CI 1.475-4.618), p < 0.01
- **Adjusted OR** (for age, gender, ethnicity, religion, education): 2.584 (95% CI 1.415-4.718), p < 0.01
- **Conclusion:** The decoy significantly increased target survey completion.

### F3: Strong Order Effect Within Decoy Condition
- **Target-first vs Control (unadjusted):** OR = 9.845 (95% CI 4.293-22.580), p < 0.01
- **Target-first vs Control (adjusted):** aOR = 11.177 (95% CI 4.571-27.330), p < 0.01
- **Decoy-first vs Control (unadjusted):** OR = 0.801 (95% CI 0.381-1.687), p not significant
- **Decoy-first vs Control (adjusted):** aOR = 0.746 (95% CI 0.341-1.631), p not significant
- **Conclusion:** The decoy effect on target completion only occurred when the target was shown first (82.7% vs 32.7%). When the decoy was shown first, completion (28.0%) was no higher than control.

### F4: Perceived Influence of Decoy
- Among 57 target completers in the decoy condition, 33/57 (57.9%) reported that the decoy option at least somewhat influenced their decision to participate.

### F5: Non-Response Bias and Response Behavior
- **FCQ score**: Wilcoxon-Mann-Whitney z = 0.488, p = 0.629. No difference in Fear of Coronavirus questionnaire responses between conditions.
- **Demographic composition of completers:**
  - Ethnicity among completers differed (p = 0.006): decoy condition had more White completers.
  - Age, gender, religion, education among completers: no significant differences.
- **Completers vs non-completers:**: No significant demographic differences on age, gender, religion, or education (ps > 0.05). Ethnicity comparison borderline: p = 0.661.

---

## Design Features

**Study Type:** Between-subjects field experiment with nested order manipulation

**Independent Variables:**
1. Invitation condition (Control vs Decoy)
2. Presentation order within decoy (Target first vs Decoy first)

**Dependent Variables:**
1. Primary: Completion of target survey (binary)
2. Secondary: Fear of Coronavirus Questionnaire responses (8 Likert items, summed 8-40)
3. Secondary: Demographic characteristics of completers
4. Exploratory: Self-reported influence of decoy on decision

**Control Condition:** Email invitation with only the target survey option (8 closed-ended FCQ items, demographics, debrief; £2 after 1 week)

**Decoy Condition:** Email invitation with a comparison table offering two survey options:
- Target option (closed-ended, £2 after 1 week)
- Decoy option (two open-ended questions requiring 100+ words each, £2 after 4 weeks)

---

## Materials

**source/specification.json**
- Study design, participants, procedure, variables

**source/ground_truth.json**
- All statistical tests, raw data, effect sizes

**source/materials/preliminary_questionnaire.json**
- Preliminary questionnaire items and recruitment logic

**source/materials/main_experiment_control.json**
- Control condition invitation email and questionnaire (demographics, FCQ, debrief)

**source/materials/main_experiment_decoy_target_first.json**
- Decoy condition (target-first order): comparison table, target questionnaire, decoy questionnaire

**source/materials/main_experiment_decoy_decoy_first.json**
- Decoy condition (decoy-first order): comparison table, decoy questionnaire, target questionnaire

---

## Replication Notes

**For LLM Agents:**

1. **Preliminary study:** Present questionnaire items, collect preference and willingness responses; use recorded preferences to validate suitability of decoy attributes.

2. **Main experiment - Control:** Present invitation email, agent decides whether to participate. If yes, complete target questionnaire (FCQ + demographics + debrief).

3. **Main experiment - Decoy:** Present invitation email with comparison table. Agent chooses target or decoy, then completes chosen questionnaire. Evaluate primary outcome (target completion), secondary outcomes (FCQ scores, perceived decoy influence).

4. **Critical details:**
   - The strong order effect is central: placing the target first dramatically increases the decoy effect (OR ~10), while placing the decoy first eliminates it (OR ~0.8).
   - Question type (closed vs open-ended) and payment timing (1 week vs 4 weeks) are the decoy attributes.
   - Sample completers vs non-completers show minimal non-response bias by demographic (except ethnicity imbalance).

---

## Files

**source/**
- `specification.json` — Study design, participants, procedure
- `ground_truth.json` — All findings, statistical tests, raw data
- `metadata.json` — Additional metadata
- `materials/preliminary_questionnaire.json` — Recruitment survey items
- `materials/main_experiment_control.json` — Control condition materials
- `materials/main_experiment_decoy_target_first.json` — Decoy condition (target first)
- `materials/main_experiment_decoy_decoy_first.json` — Decoy condition (decoy first)
- `decoy_effect_1.pdf` — Full paper
- `decoy_effect_1_apndx.pdf` — Paper appendix
- `exp.csv` — Participant-level data (203 participants, 36 columns)
- `analysis_18082022.do` — Stata analysis script

**scripts/**
- `config.py` — Study configuration and prompt builder
- `evaluator.py` — Evaluation logic and statistical comparisons
- `stats_lib.py` — Statistical utilities (if needed)
- `study_utils.py` — Helper functions (if needed)
