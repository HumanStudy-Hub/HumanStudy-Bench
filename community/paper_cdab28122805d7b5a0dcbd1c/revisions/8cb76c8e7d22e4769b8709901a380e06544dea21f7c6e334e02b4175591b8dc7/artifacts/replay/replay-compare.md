# manylabs1: replay vs human data

Human sessions: 61407. AI participants: 60 per arm (claude -p --model haiku --strict-mcp-config --setting-sources "" --tools "" --system-prompt "You are taking part in a study. Answer as asked.").

| Effect | Reported | Human | Agent | Human≈paper | Agent agrees |
| --- | --- | --- | --- | --- | --- |
| Sunk costs: paid vs free ticket | 0.27 | 0.272 (p 4.35e-27) | 0.595 (p 0.0248) | ✓ | ✓ |
| Gain vs loss framing: sure option chosen more in the gain frame | 0.6 | 0.599 (p 2.55e-114) | 0.632 (p 0.0195) | ✓ | ✓ |
| Anchoring: distance to NYC (high - low) | 1.17 | 1.17 (p 8.56e-34) | 0.366 (p 0.164) | ✓ | ✗ |
| Anchoring: Chicago population (high - low) | 1.79 | 1.73 (p 1.14e-62) | – (p –) | ✓ | – |
| Anchoring: Mt. Everest (high - low) | 2.23 | 2.38 (p 7.97e-98) | 1.15 (p 4.29e-05) | ✓ | ✓ |
| Anchoring: babies born per day (high - low) | 2.42 | 2.53 (p 7.28e-106) | -0.287 (p 0.273) | ✓ | ✗ |
| Retrospective gambler's fallacy: three 6s vs two 6s and a 3 | 0.61 | 0.676 (p 1.06e-136) | 0.318 (p 0.224) | ✓ | ✗ |
| Category scales: over 2.5 h more often on the low-frequency scale | 0.49 | 0.496 (p 1.92e-76) | 0.42 (p 0.112) | ✓ | ✗ |
| Norm of reciprocity: more support when the American question comes first | 0.3 | 0.297 (p 2.8e-31) | – (p –) | ✓ | – |
| Allowed/forbidden: more restrictive answers with 'allow' wording | – | 0.47 (p 1.73e-73) | – (p –) | – | – |
| Quote attribution: more agreement when attributed to the liked speaker | 0.32 | 0.322 (p 5.53e-37) | 5.04 (p 3.66e-27) | ✓ | ✓ |
| Flag priming: flag minus control | 0.03 | -0.0382 (p 0.182) | – (p –) | ✗ | – |
| Currency priming: money minus control | -0.02 | -0.038 (p 0.131) | – (p –) | ✓ | – |
| Imagined contact: contact minus control | 0.13 | – (p –) | – (p –) | – | – |

Human data reproduces the paper on 11 of 12 effects; agents agree with humans on 4 of 8.
