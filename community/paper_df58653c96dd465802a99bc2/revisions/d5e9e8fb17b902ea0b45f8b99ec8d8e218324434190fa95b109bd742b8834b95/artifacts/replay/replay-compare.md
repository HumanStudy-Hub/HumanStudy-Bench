# cheung2025moral: replay vs human data

Human sessions: 474. AI participants: n=10 as the package counts it (per arm or in total) (claude -p --model haiku --strict-mcp-config --setting-sources "" --tools "" --system-prompt "You are taking part in a study. Answer as asked.").

| Effect | Reported | Human | Agent | Human≈paper | Agent agrees |
| --- | --- | --- | --- | --- | --- |
| Study 2 yes-no bias, participants (CBR when yes=CBR minus when no=CBR; paper: 60% vs 56%, reported as no significant bias; sign flipped so positive = bias toward 'no') | -0.04 | -0.039 (p 0.48) | 0.7 (p 0.000624) | ✓ | ✗ |
| Study 2 omission bias, participants (CBR on omission minus on action) | – | 0.116 (p 0.039) | 0.5 (p 0.015) | – | ✓ |

Human data reproduces the paper on 1 of 1 effects; agents agree with humans on 1 of 2.
