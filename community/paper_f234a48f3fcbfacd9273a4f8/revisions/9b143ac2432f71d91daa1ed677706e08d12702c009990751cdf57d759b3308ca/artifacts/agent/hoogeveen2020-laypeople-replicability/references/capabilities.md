# Capabilities

What this study can do, each with its reason. A tool refuses anything else (`status: refused`, with the reason and what to do instead).

| Capability | Status | Why |
| --- | --- | --- |
| Read and search | yes |  |
| Run AI participants | yes |  |
| Human data | yes | loader in package/hoogeveen2020/human/; source key not pinned yet (#45 milestone 2) |
| Authors' analysis code | no | no author-code key recorded |
| Participant profiles | no | the runner does not sample participant profiles yet |
| Paper reproduced from human data | pass | 4 of 4 reported results recovered by the package analysis from the connected human data (see humanData.source: it may be a replication sample, not the original participants) |

## Where the template does not fit

| Pattern | Where | What | Fallback |
| --- | --- | --- | --- |
| modelling | all analyses | The paper used Bayesian models (issue i-priors); the offline evaluation reports the same quantities with frequentist summaries and says so. | Frequentist summaries, labelled as an approximation. |
