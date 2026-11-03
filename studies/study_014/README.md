# Do Auctioneers Pick Optimal Reserve Prices?

**Authors:** Andrew M. Davis, Elena Katok, Anthony M. Kwasnica
**Year:** 2011

---

## Description

This study investigates how auctioneers set reserve prices in second-price sealed-bid auctions (described to sellers as English auctions). A well-established theoretical result, assuming risk neutrality of the seller, is that the optimal reserve price should not depend on the number of participating bidders (Roger B. Myerson 1981). In a set of controlled laboratory experiments, seller behavior often deviates from this theoretical benchmark.

Specifically, this study tests the Cuberoot distribution treatment with bidder counts in {1, 4, 7, 10} under the NoInfo condition. Computerized bidders follow the weakly dominant strategy of bidding their private valuations, drawn from the Cuberoot distribution F(v) = (v/100)^(1/3) with mean 25 and support [0, 100]. The seller's personal valuation is zero, and the risk-neutral optimal reserve price is 42.

The key finding: sellers systematically increase their reserve prices as the number of bidders grows, contrary to standard auction theory. Davis et al. (2011) test this with a regression of the observed reserve price on the number of bidders, with random effects for individual subjects: in Table 2, Model (1) (p. 182) the coefficient on n − mean(n) is 2.98 [SE 0.08], p < 0.01, that is, about 3 points of reserve price per additional bidder.

## Participants

- **Original study (Davis et al. 2011, Table 1):** 120 seller subjects, 20 per treatment cell; the scored Model (1) uses the 80 NoInfo subjects.
- **Agent task:** 40 simulated sellers (`specification.json`), 60 rounds each, Cuberoot, n in {1, 4, 7, 10}, NoInfo. The bidder counts and computerized bids per round come from the legacy human data file (`benchmark/human_data/auction_human_data.csv`, attributed to the Davis et al. 2023 replication); that file is not used for scoring.

## Scored test (Davis et al. 2011 only)

The study follows one paper and is scored the way that paper tests its finding (see `source/PROVENANCE.md`):

- **Human result (printed):** Davis et al. (2011), Table 2, Model (1): coefficient on n − mean(n) = 2.98 [SE 0.08], p < 0.01. Only the bound is printed, so no exact human p-value is reported.
- **Agent result (computed):** the same regression on the agent decisions: reserve price on n − mean(n) and period − mean(period), random intercept per agent. The Cube indicator is dropped because the agent task uses only the Cuberoot distribution.
- **Replication:** the agent coefficient on n − mean(n) is significant at 0.05 and positive.
- **Effect comparison:** the agent coefficient against 2.98, both in reserve-price points per additional bidder (not Cohen's d).
- **Data:** one response per round with its number of bidders. The 60-round baselines in `benchmark/llm_data` are scored offline; the playground cannot yet run this multi-round game, so a playground run is not scored.

## Descriptive statistics (not scored)

- Davis et al. (2011), Figure 2(a), NoInfo Cuberoot mean reserve prices: n = 1: 19.2, n = 2: 23.4, n = 3: 27.0, n = 4: 29.5, n = 7: 38.7, n = 10: 48.2. Risk-neutral optimal reserve price = 42.
- Legacy aggregates retained in `ground_truth.json` (`original_data_points`), each with its provenance label: by bidder count 14.85 (SD 19.15, n = 716), 24.30 (SD 17.90, n = 566), 32.93 (SD 21.25, n = 458), 39.50 (SD 25.12, n = 660); overall mean 27.31, SD 23.23, sell-through 0.74, mean profit 32.83. The legacy SD 23.23 and sell-through 0.74 differ from the 23.22 and 0.76 that Feng et al. (2025) Table 2 prints.

## See also

Feng, Choudhary & Shrestha, ["Noise, Adaptation, and Strategy: Assessing LLM Fidelity in Decision-Making,"](https://aclanthology.org/2025.emnlp-main.391.pdf) EMNLP 2025, is a later benchmark on this task. It reports a human Pearson correlation of reserve price with bidder count of r = 0.42 (Table 2, no p-value printed), and the LLM baselines from GPT-4o and Claude Sonnet models in `benchmark/llm_data/` come from it. It is a reference only and is not used in scoring.

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
