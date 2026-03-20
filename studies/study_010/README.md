# Thinking through Uncertainty: Nonconsequential Reasoning and Choice

**Authors:** Eldar Shafir, Amos Tversky  
**Year:** 1992

---

## Description

When thinking under uncertainty, people often do not consider appropriately each of the relevant branches of a decision tree, as required by consequentialism. As a result they sometimes violate Savage’s sure-thing principle. In the Prisoner’s Dilemma game, for example, many subjects compete when they know that the opponent has competed and when they know that the opponent has cooperated, but cooperate when they do not know the opponent’s response. Newcomb’s Problem and Wason’s selection task are also interpreted as manifestations of nonconsequential decision making and reasoning. The causes and implications of such behavior, and the notion of quasi-magical thinking, are discussed....

## Participants

- **N:** 80
- **Population:** Princeton undergraduates
- **By sub-study:**
  - `Experiment 1`: n = 80
  - `Experiment 2`: n = 40
  - `Experiment 3`: n = 0

## Replicated tests (human data)

- **Chi-square or proportion test (implied)**  
  Reported: p < .001  
  Significance level: 0.05
- **Descriptive frequency analysis**  
  Reported: 25% (113/444)  
  Significance level: —
- **Descriptive percentage**  
  Reported: 65%  
  Significance level: —
- **Descriptive percentage**  
  Reported: 81%  
  Significance level: —

## Effect sizes & auxiliary statistics

- F1: Cooperation rates across 444 triads of Prisoner's Dilemma games played by 80 subjects.
  Raw/summary data present in `ground_truth.json`.
- F2: Choices in a computer-simulated Newcomb's Problem.
  Raw/summary data present in `ground_truth.json`.
- F3: Information seeking behavior and subsequent choice in PD games.
  Raw/summary data present in `ground_truth.json`.

## Files

**source/**
- Prisoner.pdf

- ground_truth.json

- materials/newcombs_computer_task.json

- materials/pd_info_seeking_variation.json

- materials/pd_triad_tasks.json

- metadata.json

- specification.json

**scripts/**
- config.py

- evaluator.py

- stats_lib.py

- study_utils.py
