# Capabilities

What this study can do, each with its reason. A tool refuses anything else (`status: refused`, with the reason and what to do instead).

| Capability | Status | Why |
| --- | --- | --- |
| Read and search | yes |  |
| Run AI participants | no | blocked by open issues: iss_filler |
| Human data | yes | loader in package/kreis2025hindsight/human/; source key not pinned yet (#45 milestone 2) |
| Authors' analysis code | no | no author-code key recorded |
| Participant profiles | no | the runner does not sample participant profiles yet |
| Paper reproduced from human data | not checked | not checked |

## Where the template does not fit

| Pattern | Where | What | Fallback |
| --- | --- | --- | --- |
| materials-not-public | filler task | The filler text is described as unrelated to the study but not printed (issue iss_filler). | Blocked: readable and searchable, not runnable. Never invent the filler text. |
