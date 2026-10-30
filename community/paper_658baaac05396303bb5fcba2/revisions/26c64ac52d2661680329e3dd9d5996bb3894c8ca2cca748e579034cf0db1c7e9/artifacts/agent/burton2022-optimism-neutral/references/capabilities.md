# Capabilities

What this study can do, each with its reason. A tool refuses anything else (`status: refused`, with the reason and what to do instead).

| Capability | Status | Why |
| --- | --- | --- |
| Read and search | yes |  |
| Run AI participants | no | blocked by open issues: iss-stimuli |
| Human data | partial | source pinned in package/burton2022-optimism-neutral/human/key.json (3 of 3 data files downloadable from it) |
| Authors' analysis code | no | no author-code key recorded |
| Participant profiles | no | the runner does not sample participant profiles yet |
| Paper reproduced from human data | not checked | not checked |

## Not in the human data

- **study1:e1first / study1:ebrfirst / study2:... / study3:...**: The data files do not record whether E1 or eBR was asked first, so the two order arms cannot be separated. Each participant is loaded once per study with arm = condition = the study id ('study1', 'study2', 'study3').

## Where the template does not fit

| Pattern | Where | What | Fallback |
| --- | --- | --- | --- |
| materials-not-public | all studies | The full event list is in SI Table S1, which is not in the paper or its sources; only four events and two base rates are quoted (issue iss-stimuli). | Blocked: readable and searchable, not runnable. Never invent the missing events. |
| human-data-incomplete | study1-3 order arms | The data files do not record whether E1 or eBR was asked first, so the two order arms cannot be separated. | Load each participant once per study with the order arms pooled. |
