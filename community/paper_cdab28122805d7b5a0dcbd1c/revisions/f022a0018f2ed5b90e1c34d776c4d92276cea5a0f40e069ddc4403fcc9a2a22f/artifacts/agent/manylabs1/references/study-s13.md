# Cross-site analysis of replicability and moderation

`s13`

Aggregate, heterogeneity, intra-class correlation and moderation analyses across the 36 samples and 13 effects

## Analysis

### Aggregate effect sizes across samples

`s13_an_agg`

- **Aim**: Compare each replicated effect with its original effect size across the 36 samples (p. 6)
- **Method**: Original ES, median replication ES, weighted and unweighted ES with 99% CIs, proportion of samples rejecting the null in the expected and unexpected direction; weighted on the whole aggregated dataset, unweighted on the 36 samples (p. 6, p. 7)
- **Pre-registered plan**: A confirmatory analysis plan was specified before data collection; anchoring analyses were revised because of erratic distributions (p. 6)

### Intra-class correlations

`s13_an_icc`

- **Aim**: Attribute variability in effect sizes to samples versus to effects (p. 8)
- **Method**: ICC of samples across effects compared with ICC of effects across samples (p. 8)

### Heterogeneity tests

`s13_an_het`

- **Aim**: Test whether heterogeneity of effect sizes exceeds what is expected by measurement error (p. 8)
- **Method**: Cochran's Q and I2 per effect (R package metafor, REML) (p. 8, p. 9)

### Moderation by country and setting

`s13_an_mod`

- **Aim**: Test whether US versus international and laboratory versus online moderate each effect (p. 9)
- **Method**: Condition x Country (US or other) x Location (lab or online) ANOVA per effect; hierarchical regression for the implicit-explicit relation; study order tested as a random factor (p. 9)
- **Coding**: Moderation tests were described as exploratory in the supplement (p. 9)

### Flag priming moderator regressions

`s13_an_flag`

- **Method**: Five hierarchical regression models on US participants testing the moderators suggested by the original authors (p. 9)

## Reported results

### Aggregate replication outcome

`s13_res_agg`

- **Result kind**: reported (p. 6)
- **Descriptives**: 10 of 13 effects replicated with varying distance from the original; imagined contact weakly; flag and currency priming near zero (p. 6, p. 8)

### Variation attributable to samples versus effects

`s13_res_icc`

- **Result kind**: reported (p. 8)
- **ICC**: Samples across effects ICC = .005; effects across samples ICC = .75 (see evidence) (p. 8)

### Heterogeneity results

`s13_res_het`

- **Result kind**: reported (p. 9)
- **Heterogeneity**: Table 3 and text (p. 9)

### Moderation results

`s13_res_mod`

- **Result kind**: reported (p. 9)
- **Moderation**: 10 of 32 moderation tests significant; none exceeded partial eta-squared .022 (see evidence) (p. 9)
- **Order of presentation**: No effect moderated by position in the study procedure (p. 8, p. 9)
- **Flag priming moderators**: None moderated the effect (see evidence) (p. 9)

## Simulation scope

### Simulation scope: cross-site analysis

`sim_an`

- **Steps [implementation]**: Analysis of collected sessions only; no new participant is run for this scope.
- **Roles [implementation]**: None.
- **Conditions [implementation]**: Site moderators (lab/online, US/international) cannot be varied in a replay without site profiles.
- **Outcomes [implementation]**: Per-effect statistics from replay sessions can be set beside Table 2; heterogeneity needs multiple profiles.
- **Limits [implementation]**: Not computed in the package.

## Open issues

- **The paper's text disagrees on how many effects replicated** [check, `i_s13_ntext`, s13_res_agg]: The abstract and results say 10 of 13 effects replicated (imagined contact weak, two not replicated); the Discussion says eleven of 13 were successfully replicated. Suggested: Treat 10 of 13 (results and abstract) as the main count unless the researcher decides otherwise. (p. 6, p. 9)
