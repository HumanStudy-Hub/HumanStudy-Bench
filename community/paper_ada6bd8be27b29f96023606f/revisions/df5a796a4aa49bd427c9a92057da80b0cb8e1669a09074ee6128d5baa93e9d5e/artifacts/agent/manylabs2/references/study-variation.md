# Variation across samples and settings (cross-slate analysis)

`variation`

Analysis scope over the replication data of both slates: intra-class correlations, heterogeneity, lab versus online, WEIRD versus less WEIRD, task order and replication-success criteria.

## Analysis

### Intra-class correlations of effects and samples

`a-icc`

- **Aim**: Compare how much effect sizes depend on the effect versus on the sample (p. 64)
- **Method**: Linear mixed model with samples and studies as random effects; intra-class correlations of samples across effects and of effects across samples (p. 64)
- **Inputs [derived]**: Per-sample effect sizes of the 28 replications from both slates (p. 64)
  - how it was derived: Both slates are inputs; this scope is not an experiment.

### Heterogeneity across samples

`a-het`

- **Aim**: Test whether variability across samples exceeds that expected from sampling error (p. 64)
- **Method**: Random-effects meta-analysis per effect (REML estimator for between-study variance) with Q, Tau and I2 (p. 64)
- **Inputs [derived]**: Per-sample effect sizes of the 28 replications from both slates (p. 64)
  - how it was derived: Both slates are inputs; this scope is not an experiment.

### Lab versus online administration

`a-lab`

- **Aim**: Test moderation of effects by administration mode (p. 65)
- **Method**: Moderation tests between lab and online samples (Table 3) (p. 65)
- **Inputs [derived]**: Per-sample effect sizes of the 28 replications from both slates (p. 65)
  - how it was derived: Both slates are inputs; this scope is not an experiment.

### WEIRD versus less WEIRD samples (exploratory)

`a-weird`

- **Aim**: Explore cultural moderation of effects (p. 66)
- **Method**: Samples classified by a country-level WEIRDness score above or below the mean; heterogeneity statistics compared between the two groups (p. 66)
- **Inputs [derived]**: Per-sample effect sizes of the 28 replications from both slates (p. 66)
  - how it was derived: Both slates are inputs; this scope is not an experiment.

### Task order

`a-order`

- **Aim**: Test whether position in the randomized sequence moderates effect sizes (p. 68)
- **Method**: Effect size at each task position compared with the effect's mean effect size; trend tests (p. 68)
- **Inputs [derived]**: Per-sample effect sizes of the 28 replications from both slates (p. 68)
  - how it was derived: Both slates are inputs; this scope is not an experiment.

### Replication success criteria

`a-success`

- **Aim**: Summarise replication success under several criteria (p. 70)
- **Method**: Significance in the original direction with the replication N, the original N, 2.5x the original N and 50 per group (Table 5) (p. 70)
- **Inputs [derived]**: Per-sample effect sizes of the 28 replications from both slates (p. 70)
  - how it was derived: Both slates are inputs; this scope is not an experiment.

## Reported results

### ICC result

`r-icc`

- **Result kind**: reported (p. 64)
- **Reported result** (verbatim, p. 64):
  > In other words, to predict effect sizes across the 28 findings and dozens of samples studied here, it is very useful to know the effect being studied and barely useful to know the sample in which it is being studied.

### Heterogeneity result

`r-het`

- **Result kind**: reported (p. 65)
- **Reported result** (verbatim, p. 65):
  > Finally, with Tau, only one effect (Huang et al., 2014) showed a substantial standard deviation among effect sizes (0.24), and 8 others showed modest heterogeneity near 0.10. Most of the effects, 19 of 28 (68%) showed near zero heterogeneity as estimated by Tau.

### Lab versus online result

`r-lab`

- **Result kind**: reported (p. 65)
- **Reported result** (verbatim, p. 65):
  > Table 3 summarizes moderation tests between lab and online samples. Just one result showed a significant difference between lab and online samples (Zhong & Liljenquist, 2006).

### WEIRDness result

`r-weird`

- **Result kind**: reported (p. 68)
- **Reported result** (verbatim, p. 68):
  > Ultimately, just three effects (Huang et al., 2014, Knobe, 2003, and Norenzayan et al., 2002) showed significant evidence for moderation by WEIRDness after correcting for multiple comparisons. However, for Norenzayan et al., the cultural difference was the inverse of the original result though their original study did not have a theoretical commitment regarding the cultural differences in the condition effect that we tested.

### Task order result

`r-order`

- **Result kind**: reported (p. 68)
- **Reported result** (verbatim, p. 68):
  > Across the 28 findings, we observed little systematic evidence that effects are stronger (or weaker) when administered first compared to last. Also, there was no evidence of linear, quadratic or cubic trends by task order (see supplements for analytic details: https://osf.io/z8dqs/).

### Replication success result

`r-success`

- **Result kind**: reported (p. 62)
- **Reported result** (verbatim, p. 62):
  > Overall, 14 of the 28 effects (50%) showed significant evidence in the same direction as the original finding, 1 was weakly consistent (4%),8 and 13 (46%) showed a null effect or evidence in the opposite direction of the original finding.

## Simulation scope

### Simulation scope, cross-slate variation analysis

`sim-variation`

- **Steps [implementation]**: No steps: analysis scope over per-sample effect sizes. (p. 5)
- **Roles [implementation]**: No participants of its own. (p. 5)
- **Conditions [implementation]**: Lab versus online, WEIRD versus less WEIRD and task position cannot be varied by an AI replay of one text survey. (p. 15)
- **Outcomes [implementation]**: Heterogeneity statistics need many independent samples; a handful of simulated sessions cannot estimate them. (p. 5)
- **Limits [implementation]**: The analysis depends on the human samples of 36 countries; it is not reproduced by simulation. (p. 15)
