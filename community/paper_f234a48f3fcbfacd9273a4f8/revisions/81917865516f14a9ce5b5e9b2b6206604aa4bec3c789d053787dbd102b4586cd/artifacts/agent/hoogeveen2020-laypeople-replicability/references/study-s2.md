# Exploratory analyses without the lay sample: Bayes factors vs replication effect sizes, and expert accuracy

`s2`

Two analyses that re-use published data: the association between the original studies' Bayes factors and replication effect sizes, and the posterior accuracy of experts in SSRP and ML2 (Camerer et al., 2018; Forsell et al., 2018).

## Analysis

### Bayes factors vs replication effect sizes

`a-bfcorr`

- **Aim**: Check that original Bayes factors carry information about replicability (p. 5)
- **Method**: Spearman correlation between log-transformed original Bayes factors and replication effect sizes (Pearson r) (p. 5)

### Accuracy of experts in SSRP and ML2

`a-experts`

- **Aim**: Estimate expert accuracy for comparison with laypeople (p. 12)
- **Method**: Bayesian hierarchical model on expert predictions from Camerer et al. (2018) and Forsell et al. (2018), full study sets of each project (p. 12)

## Reported results

### Bayes factor correlation result

`res-bfcorr`

- **Kind**: reported (p. 5)
- **Effect**: median rho = .62, 95% CI [.33, .78]; overwhelming evidence for a positive correlation (BF+0 = 162) (p. 5)

### Expert accuracy

`res-experts`

- **Kind**: reported (p. 12)
- **Accuracy**: {"SSRP experts": ".72, 95% CI [.69, .74]", "ML2 experts": ".65, 95% CI [.62, .68]"} (p. 12)

## Open issues

- **Analyses in s2 need external data** [check, `i-s2data`, a-experts]: The expert and Bayes-factor correlation analyses use data from other projects not supplied. Suggested: Supply OSF data if a re-analysis is wanted. (p. 12)
