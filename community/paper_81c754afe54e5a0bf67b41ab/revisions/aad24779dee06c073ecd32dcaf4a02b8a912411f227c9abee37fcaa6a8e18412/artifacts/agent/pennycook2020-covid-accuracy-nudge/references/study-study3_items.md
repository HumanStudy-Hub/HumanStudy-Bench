# Item-level analysis across Studies 1 and 2

`study3_items`

Post hoc correlation between each headline's Study 1 perceived accuracy and its Study 2 treatment effect (r(28)).

## Hypotheses

### Item-level expectation: the nudge matters most for headlines seen as clearly true or clearly false

`h_s3`

- **Statement [derived]**: Given the proposed mechanism, the treatment should change sharing of a headline in proportion to its perceived accuracy. (p. 7)
  - how it was derived: Paraphrase of the stated item-level expectation.
- **Scope**: Post hoc analysis; Study 1 accuracy ratings are used as a proxy for how accurate Study 2 participants would deem each headline. (p. 7)

## Procedure

Order of steps:

- Correlate per-headline Study 1 accuracy with per-headline Study 2 treatment effect (p. 7)

## Variables

### Headline perceived accuracy (Study 1 accuracy condition)

`v_item_acc`

- **Role [derived]**: iv (p. 7)
  - how it was derived: Predictor of the item-level treatment effect.
- **Type [derived]**: continuous (mean proportion judged accurate per headline) (p. 7)
  - how it was derived: Average accuracy rating of the yes/no responses.
- **Observation unit**: headline (30) (p. 8)

### Item-level treatment effect on sharing (Study 2)

`v_item_effect`

- **Role [derived]**: dv (p. 7)
  - how it was derived: Outcome of the item-level analysis.
- **Definition**: Average sharing intention in the treatment condition minus average sharing intention in the control condition, per headline. (p. 7)
- **Observation unit**: headline (30) (p. 8)

## Analysis

### Item-level correlation (post hoc)

`a_s3`

- **Aim**: Show that the treatment effect on sharing of a headline varies with its perceived accuracy. (p. 7)
- **Inputs**: Per headline: mean accuracy rating in the Study 1 accuracy condition and treatment minus control mean sharing in Study 2. (p. 7)
- **Method [derived]**: Correlation across the 30 headlines, using Study 1 ratings as a proxy for Study 2 perceived accuracy. (p. 7, p. 8)
  - how it was derived: Pearson r with 28 degrees of freedom is implied by r(28).

## Reported results

### Item-level correlation result

`res_s3`

- **Result kind [implementation]**: reported (p. 8)
- **Correlation**: r(28) = .76, p < .0001. (p. 8)
- **Reading**: Headlines more likely to be identified as true were more strongly positively impacted (sharing increases) by the accuracy nudge. (p. 8)

## Simulation scope

### Item-level analysis simulation scope

`sim_s3`

- **Steps [implementation]**: Computed from replayed Study 1 accuracy sessions and replayed Study 2 control and treatment sessions; there is no separate participant task. (p. 7)
- **Roles [implementation]**: No participants of its own. (p. 7)
- **Conditions [implementation]**: Needs Study 1 accuracy and Study 2 control and treatment arms. (p. 7)
- **Outcomes [implementation]**: Correlation across 30 headlines; not ready until the Study 2 treatment arm exists. (p. 8)
- **Limits [implementation]**: Depends on the blocked treatment arm; with few simulated participants item means are noisy. (p. 7)
