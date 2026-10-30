# ml5-risen-gilovich-2008: replay vs human data

Human sessions: 754. AI participants: n=10 as the package counts it (per arm or in total) (claude -p --model haiku --strict-mcp-config --setting-sources "" --tools "" --system-prompt "You are taking part in a study. Answer as asked.").

| Effect | Reported | Human | Agent | Human≈paper | Agent agrees |
| --- | --- | --- | --- | --- | --- |
| Tempting fate raises likelihood (no-load condition, 0-10 points) | 0.11 | 0.108 (p 0.595) | 0.5 (p 0.398) | ✓ | ✓ |
| Tempting fate x cognitive load interaction (0-10 points) | -0.02 | -0.025 (p 0.933) | -0.9 (p 0.284) | ✓ | ✓ |

Human data reproduces the paper on 2 of 2 effects; agents agree with humans on 2 of 2.
