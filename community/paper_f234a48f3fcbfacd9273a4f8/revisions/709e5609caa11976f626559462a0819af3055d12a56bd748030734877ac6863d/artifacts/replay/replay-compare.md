# hoogeveen2020: replay vs human data

Human sessions: 233. AI participants: n=10 as the package counts it (per arm or in total) (claude -p --model haiku --strict-mcp-config --setting-sources "" --tools "" --system-prompt "You are taking part in a study. Answer as asked.").

| Effect | Reported | Human | Agent | Human≈paper | Agent agrees |
| --- | --- | --- | --- | --- | --- |
| Accuracy above chance, description only (mean accuracy - .50) | 0.09 | 0.0847 (p –) | 0.2 (p –) | ✓ | ✓ |
| Accuracy above chance, description + evidence (mean accuracy - .50) | 0.17 | 0.171 (p –) | 0.289 (p –) | ✓ | ✓ |
| AUC (SDT, from d'), description only | 0.62 | 0.617 (p –) | 0.799 (p –) | ✓ | ✓ |
| AUC (SDT, from d'), description + evidence | 0.74 | 0.735 (p –) | 0.87 (p –) | ✓ | ✓ |
| Evidence benefit: mean log Brier per participant (description only - evidence) | – | 0.154 (p 3.31e-09) | 0.105 (p 1.9e-05) | – | ✓ |

Human data reproduces the paper on 4 of 4 effects; agents agree with humans on 5 of 5.
