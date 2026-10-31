# Capabilities

What this study can do, each with its reason. A tool refuses anything else (`status: refused`, with the reason and what to do instead).

| Capability | Status | Why |
| --- | --- | --- |
| Read and search | yes |  |
| Run AI participants | partial | 1 part(s) of the paper cannot be run; see excluded |
| Human data | partial | source pinned in package/manylabs1/human/key.json (3 of 3 data files downloadable from it) |
| Authors' analysis code | no | no author-code key recorded |
| Participant profiles | no | the runner does not sample participant profiles yet |
| Paper reproduced from human data | partial | 11 of 12 reported results recovered by the package analysis from the connected human data (see humanData.source: it may be a replication sample, not the original participants) |

## Not runnable

- **s12 (IAT)**: IAT reaction-time latencies are not a text response; the adapter has no parser for them.

## Not in the human data

- **s12**: IAT reaction-time latencies are not a text response and the adapter has no parser for them.
- **s03 (participants with mixed anchors)**: The anchor was randomised per question (anch1group..anch4group differ within a participant) but a session holds one condition; those participants are left out (see EXCLUSIONS).

## Where the template does not fit

| Pattern | Where | What | Fallback |
| --- | --- | --- | --- |
| unsimulable-measure | s12 (IAT) | IAT reaction-time latencies are not a text response; the adapter has no parser for them. | Label the IAT not run; never substitute a proxy. |
| other | s03 (anchoring) | The anchor was randomised per question, but a session holds one condition, so participants with mixed anchors are left out of the human comparison. | Compare participants with a single anchor condition only. |
