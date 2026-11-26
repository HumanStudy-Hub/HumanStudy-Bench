# Individual-level hierarchical modelling and Bayesian model comparison

`mc`

Latent reasoning-type model fitted to the individual choice counts of Experiments 1 and 2, with homogeneous (Gricean) versus heterogeneous population comparison. Analysis scope, not an experiment.

## Procedure

Order of steps:

- Fit hierarchical type model to retained participants' choice counts (participant sees: Trial-level record; Trial-level record) (p. 14, p. 15)
- Approximate Bayes factors, homogeneous vs heterogeneous (participant sees: Posterior reasoning-type distribution (Table 3)) (p. 18)

## Analysis

### Hierarchical latent-type model

`mc_an_model`

- **Aim [to check]**: Infer each participant's likely reasoning type (S0/S1/S2 for production, R0/R1/R2 for comprehension) and the population distribution over types, and decide whether the homogeneous Gricean model suffices. (p. 14, p. 13)
- **Inputs [to check]**: Choice counts per participant, game (simple, complex, unambiguous, ambiguous) and category (target, competitor, distractor 1, possibly distractor 2) for retained participants of Experiments 1 and 2. (p. 14)
- **Method [to check]**: Binomial likelihood with probabilities determined by the participant's type, population error rate epsilon and rationality lambda; types drawn from a categorical population distribution with a Dirichlet(1,1,1) prior; Gamma priors on epsilon and lambda; JAGS, 10000 samples after 10000 burn-in, two chains, thinning 2. (p. 15)
- **Assumptions [to check]**: Types are constant over the experiment; lambda and epsilon are shared across types. (p. 14, p. 19)
- **Salience priors for R2**: Not stated in the paper.

### Bayesian model comparison

`mc_an_bf`

- **Aim [to check]**: Compare the nested homogeneous model against the heterogeneous model by Bayes factor. (p. 18, p. 6)
- **Method [to check]**: Bayes factor of nested models approximated by simulation (Savage-Dickey based); charitable near-homogeneous null priors with tolerance epsilon (0.05 for the headline numbers, 0.01 as a check). (p. 18)
- **Criterion [to check]**: Bayes factors greater than 3 count as sufficiently strong evidence for a model. (p. 18)
- **Computation details**: Not stated in the paper.

## Reported results

### Population-level RSA fit

`mc_r_rsa`

- **Result kind**: reported (p. 13)
- **Fit [to check]**: Best-fitting homogeneous RSA parameters reproduce the observed choice frequencies with correlation r = 0.997, p < 0.0001 (Fig 4). (p. 13)

### Posterior reasoning-type distribution (Table 3)

`mc_r_post`

- **Result kind**: reported (p. 16)
- **Means (95% HDI) [to check]**: Speakers: tau0 .117 [.036, .209], tau1 .843 [.741, .940], tau2 .040 [8.43e-05, .095]. Listeners: tau0 .301 [.173, .442], tau1 .560 [.410, .703], tau2 .139 [.054, .236]. (p. 16)
- **Individual classification [to check]**: Production: one participant most likely S2 and four most likely S0; comprehension: only three participants have ambiguous classifications. (p. 16, p. 15)

### Bayes factors

`mc_r_bf`

- **Result kind**: reported (p. 18)
- **Test [to check]**: With tolerance 0.05: production BF(hom/het) about 3.955 (favours the near-homogeneous model); comprehension about 2.8e-11 (strongly favours heterogeneous). With tolerance 0.01 for production the BF is 0.004 (about 250 for heterogeneous). (p. 18)

## Simulation scope

### Simulation scope

`mc_sim`

- **Steps [implementation, to check]**: The paper's own descriptive checks (per-participant target-choice rates) can be computed from simulated sessions; the JAGS hierarchical model and Bayes factors are not run by the package. (p. 15)
- **Roles [implementation, to check]**: No participants of its own: analysis of the sessions from Experiments 1 and 2. (p. 14)
- **Conditions [implementation, to check]**: Not applicable. (p. 14)
- **Outcomes [implementation, to check]**: evaluate() returns not_ready for the model comparison and states what is missing. (p. 18)
- **Limits [implementation, to check]**: Needs JAGS, the salience priors of Experiment 3 and S5 Text. (p. 18)

## Open issues

- **Salience priors (Experiment 3) are missing** [check, `is_mc_sal`, mc_an_model]: Experiment 3 and S4 Text are not available in paper.pdf. Suggested: Supply S4 Text or Experiment 3 data.
- **Bayes factor computation is in S5 Text** [check, `is_mc_s5`, mc_an_bf]: The approximation procedure is deferred to S5 Text. Suggested: Supply S5 Text.
