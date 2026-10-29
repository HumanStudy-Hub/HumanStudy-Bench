# Capabilities

What this study can do, each with its reason. A tool refuses anything else (`status: refused`, with the reason and what to do instead).

| Capability | Status | Why |
| --- | --- | --- |
| Read and search | yes |  |
| Run AI participants | yes |  |
| Human data | yes | loader in package/tesslerfranke2018/human/; source key not pinned yet (#45 milestone 2) |
| Authors' analysis code | no | no author-code key recorded |
| Participant profiles | no | the runner does not sample participant profiles yet |
| Paper reproduced from human data | not checked | no reported result could be compared with the human data |

## Where the template does not fit

| Pattern | Where | What | Fallback |
| --- | --- | --- | --- |
| modelling | exp1, exp2 | The paper fits lme4 mixed models; the offline evaluation approximates each test with participant-level contrasts, so statistics are not numerically comparable to the paper's b and t. | Report direction and significance of the approximated contrasts, labelled as such. |
