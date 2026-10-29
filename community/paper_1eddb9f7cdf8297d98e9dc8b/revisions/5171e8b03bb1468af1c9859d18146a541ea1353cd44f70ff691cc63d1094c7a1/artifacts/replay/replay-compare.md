# frankedegen2016: replay vs human data

Human sessions: 52. AI participants: 10 per arm (claude -p --model haiku --strict-mcp-config --setting-sources "" --tools "" --system-prompt "You are taking part in a study. Answer as asked.").

| Effect | Reported | Human | Agent | Human≈paper | Agent agrees |
| --- | --- | --- | --- | --- | --- |
| Exp 1 simple vs harder (complex+ambiguous): target-choice log-odds, higher for simple | – | nan (p –) | 21.1 (p 0.982) | – | ✗ |
| Exp 1 complex vs ambiguous: target-choice log-odds, higher for complex | – | nan (p –) | 2.55 (p 4.17e-10) | – | ✗ |
| Exp 2 simple vs harder (complex+ambiguous): target-choice log-odds, higher for simple | 1.63 | 1.66 (p 4.15e-34) | 3.61 (p 2.78e-07) | ✓ | ✓ |
| Exp 2 complex vs ambiguous: target-choice log-odds, higher for complex | 0.2 | 0.081 (p 0.492) | 0.098 (p 0.74) | ✗ | ✓ |

Human data reproduces the paper on 1 of 2 effects; agents agree with humans on 2 of 4.
