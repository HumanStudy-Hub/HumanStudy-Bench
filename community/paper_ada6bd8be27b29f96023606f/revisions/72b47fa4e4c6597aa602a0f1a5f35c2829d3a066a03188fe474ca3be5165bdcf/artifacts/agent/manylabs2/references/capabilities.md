# Capabilities

What this study can do, each with its reason. A tool refuses anything else (`status: refused`, with the reason and what to do instead).

| Capability | Status | Why |
| --- | --- | --- |
| Read and search | yes |  |
| Run AI participants | no | blocked by open issues: iss-score-zhon |
| Human data | partial | source pinned in package/manylabs2/human/key.json (3 of 3 data files downloadable from it) |
| Authors' analysis code | no | no author-code key recorded |
| Participant profiles | no | the runner does not sample participant profiles yet |
| Paper reproduced from human data | pass | 14 of 14 reported results recovered by the package analysis from the connected human data (see humanData.source: it may be a replication sample, not the original participants) |

## Not runnable

- **effects huan, geis, haus34, nore, gray**: Picture stimuli (huan also needs a click position on a map); the evaluation marks them not_ready.

## Not in the human data

- **huan**: click position on a map picture; evaluation does not compute it (not_ready) and the vertical click coordinate is outside the page model
- **alt**: evaluation marks it not_ready (syllogism key and font manipulation absent); not mapped
- **grah**: evaluation marks it not_ready (needs ideology-by-foundation correlation); not mapped
- **inba**: evaluation marks it not_ready (needs disgust-scale scoring); not mapped
- **miya**: evaluation marks it not_ready (covariate model); not mapped
- **van**: evaluation marks it not_ready (SVO slider scoring); not mapped
- **and**: evaluation marks it not_ready (composition of the well-being index); not mapped
- **geis**: picture stimulus; evaluation marks it not_ready; not mapped
- **haus34**: picture stimulus; evaluation marks it not_ready; not mapped
- **sava**: evaluation marks it not_ready (hierarchical logistic model); not mapped
- **nore**: picture stimulus; scoring key not applied by evaluation; not mapped
- **gray**: picture stimulus; evaluation marks it not_ready; not mapped
- **zhon**: evaluation marks it not_ready (product classification); not mapped
- **gati**: evaluation marks it not_ready (country prominence list); not mapped

## Where the template does not fit

| Pattern | Where | What | Fallback |
| --- | --- | --- | --- |
| missing-information | effect zhon | Which five of the ten rated products are cleaning products is not stated (issue iss-score-zhon). This one blocking issue blocks the whole package, although the other effects could run. | Whole paper blocked today. |
| non-text-stimuli | effects huan, geis, haus34, nore, gray | Picture stimuli (huan also needs a click position on a map); the evaluation marks them not_ready. | Run the text effects; label the picture effects not run. |
| modelling | effects sava, miya, grah | Hierarchical logistic, covariate and correlation models are not computed by the evaluation (not_ready). | Label these effects not analysed. |
| missing-information | effects alt, inba, van, and, gati | Scoring keys or composition rules are absent (syllogism key, disgust scale, SVO slider, well-being index, country prominence list); the evaluation marks them not_ready. | Label these effects not analysed. |
