# Capabilities

What this study can do, each with its reason. A tool refuses anything else (`status: refused`, with the reason and what to do instead).

| Capability | Status | Why |
| --- | --- | --- |
| Read and search | yes |  |
| Run AI participants | yes |  |
| Human data | partial | source pinned in package/thomas2014/human/key.json (2 of 2 data files downloadable from it) |
| Authors' analysis code | no | no author-code key recorded |
| Participant profiles | no | the runner does not sample participant profiles yet |
| Paper reproduced from human data | fail | 0 of 4 reported results recovered by the package analysis from the connected human data (see humanData.source: it may be a replication sample, not the original participants) |

## Not in the human data

- **exp1 arms with payoff $1, $5, $10**: the data contain one payoff level only (GBP 1.60 alone)
- **exp2 (four-player game)**: not in the data
- **exp3 (similarity manipulation, Big Five, reported level)**: not in the data (no similarity or Big Five columns)

## Where the template does not fit

| Pattern | Where | What | Fallback |
| --- | --- | --- | --- |
| human-data-incomplete | exp1 ($1, $5, $10 arms), exp2, exp3 | The public replication data hold one payoff level of Experiment 1 only; Experiments 2 and 3 are not in the data. | Compare Experiment 1 at $2 with people; compare the rest with the paper's reported numbers. |
| missing-information | exp3 | The Big Five items are not printed, so personality analyses return not_ready. | Run the rest of Experiment 3; label the personality analyses not run. |
