# Capabilities

What this study can do, each with its reason. A tool refuses anything else (`status: refused`, with the reason and what to do instead).

| Capability | Status | Why |
| --- | --- | --- |
| Read and search | yes |  |
| Run AI participants | yes |  |
| Human data | partial | source pinned in package/frankedegen2016/human/key.json (2 of 2 data files downloadable from it) |
| Authors' analysis code | no | no author-code key recorded |
| Participant profiles | no | the runner does not sample participant profiles yet |
| Paper reproduced from human data | partial | 1 of 2 reported results recovered by the package analysis from the connected human data (see humanData.source: it may be a replication sample, not the original participants) |

## Not in the human data

- **exp1**: No separate Experiment 1 (comprehension) file: exp_s007.csv and exp_s008.csv are identical and match Experiment 2's production proportions; the data has no objects/messages/choice index either.

## Where the template does not fit

| Pattern | Where | What | Fallback |
| --- | --- | --- | --- |
| modelling | exp1, exp2 | The paper's JAGS hierarchical model and Bayes factors are not computed; the logistic regression uses fixed effects only, so its p-values are not the paper's. | Report choice proportions and a fixed-effects regression, labelled as an approximation. |
| human-data-incomplete | exp1 | No separate Experiment 1 (comprehension) data file; the two published files are identical and match Experiment 2. | Compare Experiment 1 with the paper's reported numbers. |
