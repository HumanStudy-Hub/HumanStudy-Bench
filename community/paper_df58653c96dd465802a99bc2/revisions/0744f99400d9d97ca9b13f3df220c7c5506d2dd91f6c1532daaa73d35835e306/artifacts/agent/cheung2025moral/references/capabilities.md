# Capabilities

What this study can do, each with its reason. A tool refuses anything else (`status: refused`, with the reason and what to do instead).

| Capability | Status | Why |
| --- | --- | --- |
| Read and search | yes |  |
| Run AI participants | partial | 1 part(s) of the paper cannot be run; see excluded |
| Human data | partial | loader in package/cheung2025moral/human/; source key not pinned yet (#45 milestone 2) |
| Authors' analysis code | no | no author-code key recorded |
| Participant profiles | no | the runner does not sample participant profiles yet |
| Paper reproduced from human data | pass | 1 of 1 reported results recovered by the package analysis from the connected human data (see humanData.source: it may be a replication sample, not the original participants) |

## Not runnable

- **studies 1-2**: The paper's respondents include LLMs as well as people; the human-versus-LLM ANOVAs are not computed.

## Not in the human data

- **s1_moral_sequential**: Study 1 human data (N = 285, 13 sequential dilemmas) are not in human-data; data_p.csv is Study 2 only.

## Where the template does not fit

| Pattern | Where | What | Fallback |
| --- | --- | --- | --- |
| non-human-participants | studies 1-2 | The paper's respondents include LLMs as well as people; the human-versus-LLM ANOVAs are not computed. | Run the human-participant design with AI participants; label the LLM-respondent comparisons not run. |
| human-data-incomplete | s1_moral_sequential | Study 1 human data (N = 285, 13 sequential dilemmas) are not in the public data; data_p.csv is Study 2 only. | Compare Study 1 with the paper's reported numbers. |
