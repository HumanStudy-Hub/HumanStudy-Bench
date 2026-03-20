# Do Auctioneers Pick Optimal Reserve Prices?

**Authors:** Andrew M. Davis, Elena Katok, Anthony M. Kwasnica
**Year:** 2011

---

## Description

This study investigates how auctioneers set reserve prices in second-price sealed-bid auctions (described to sellers as English auctions). A well-established theoretical result, assuming risk neutrality of the seller, is that the optimal reserve price should not depend on the number of participating bidders (Roger B. Myerson 1981). In a set of controlled laboratory experiments, seller behavior often deviates from this theoretical benchmark.

Specifically, this study tests the Cuberoot distribution treatment with bidder counts in {1, 4, 7, 10} under the NoInfo condition. Computerized bidders follow the weakly dominant strategy of bidding their private valuations, drawn from the Cuberoot distribution F(v) = (v/100)^(1/3) with mean 25 and support [0, 100]. The seller's personal valuation is zero, and the risk-neutral optimal reserve price is 42.

The key finding: sellers systematically increase their reserve prices as the number of bidders grows (Pearson r = 0.42, p < .001), contrary to standard auction theory.

LLM simulation baselines for this experiment are done and reported in [Feng et al., "Noise, Adaptation, and Strategy: Assessing LLM Fidelity in Decision-Making," EMNLP 2025](https://aclanthology.org/2025.emnlp-main.391.pdf). Baselines from GPT-4o and Claude Sonnet models are available in `benchmark/`.

## Participants

- **N:** 40
- **Population:** University students

## Replicated tests (human data)

Due to the unavailability of the original human experiment data (Davis et al. 2011), we use the replicated experiment data (Davis et al. 2023) instead.

- **Pearson correlation (reserve price vs. number of bidders)**
  Reported: r = 0.42, p < .001
  Significance level: 0.05

## Effect sizes & auxiliary statistics

- F1: Mean reserve prices by number of bidders (Cuberoot distribution, NoInfo):
  1 bidder: 14.85 (SD = 19.15, n = 716),
  4 bidders: 24.30 (SD = 17.90, n = 566),
  7 bidders: 32.93 (SD = 21.25, n = 458),
  10 bidders: 39.50 (SD = 25.12, n = 660).
  Overall: mean = 27.31, SD = 23.23, sell-through rate = 0.74, mean profit = 32.83.
  Risk-neutral optimal reserve price = 42.
  Raw data present in `ground_truth.json`.

## Files

**source/**
- ground_truth.json
- materials/auction_instructions.json
- metadata.json
- specification.json

**scripts/**
- config.py
- evaluator.py
- stats_lib.py
- study_utils.py

**benchmark/**
- human_data/auction_human_data.csv
- llm_data
