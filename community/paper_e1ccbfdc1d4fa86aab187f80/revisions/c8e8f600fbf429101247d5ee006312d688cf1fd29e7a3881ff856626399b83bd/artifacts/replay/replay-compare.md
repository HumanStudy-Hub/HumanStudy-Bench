# nyhan2015: replay vs human data

Human sessions: 1000. AI participants: 10 per arm (claude -p --model haiku --strict-mcp-config --setting-sources "" --tools "" --system-prompt "You are taking part in a study. Answer as asked.").

| Effect | Reported | Human | Agent | Human≈paper | Agent agrees |
| --- | --- | --- | --- | --- | --- |
| Innuendo vs control: lower favorability of Swensen | 0.58 | 0.582 (p 0) | 0.4 (p 0.0298) | ✓ | ✓ |
| Innuendo vs control: higher belief he took bribes | 0.78 | 0.787 (p 0) | 0.5 (p 0.0071) | ✓ | ✓ |
| Denial vs innuendo: repairs favorability | – | 0.196 (p 0.053) | 0.2 (p 0.277) | – | ✓ |
| Causal correction vs innuendo: repairs favorability | – | 0.643 (p 0) | 0.7 (p 0.0001) | – | ✓ |
| Denial vs innuendo: lowers belief in bribes | – | 0.236 (p 0.0142) | -0 (p 1) | – | ✗ |
| Causal correction vs innuendo: lowers belief in bribes | – | 0.701 (p 0) | 0.7 (p 0.0002) | – | ✓ |
| Denial vs innuendo: lowers belief he resigned due to investigation | 0.25 | 0.246 (p 0.0003) | 0.2 (p 0.142) | ✓ | ✗ |
| Causal correction vs innuendo: lowers belief he resigned due to investigation | 0.58 | 0.578 (p 0) | 0.9 (p 0) | ✓ | ✓ |

Human data reproduces the paper on 4 of 4 effects; agents agree with humans on 6 of 8.
