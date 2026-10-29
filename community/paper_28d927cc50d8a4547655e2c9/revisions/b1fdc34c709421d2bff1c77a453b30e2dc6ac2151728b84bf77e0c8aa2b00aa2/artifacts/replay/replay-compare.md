# thomas2014: replay vs human data

Human sessions: 579. AI participants: 10 per arm (claude -p --model haiku --strict-mcp-config --setting-sources "" --tools "" --system-prompt "You are taking part in a study. Answer as asked.").

| Effect | Reported | Human | Agent | Human≈paper | Agent agrees |
| --- | --- | --- | --- | --- | --- |
| Knowledge level omnibus, $2 (signed chi2, df 3) | 62.8 | 14.9 (p 0.0019) | 24.6 (p 0) | ✗ | ✓ |
| Secondary > primary knowledge, $2 (chi2) | 23.3 | 0.271 (p 0.603) | 12.4 (p 0.0004) | ✗ | ✗ |
| Tertiary vs secondary (predicted no difference), $2 (chi2) | 0.63 | 1.6 (p 0.206) | 0 (p 1) | ✗ | ✗ |
| Common > tertiary knowledge, $2 (chi2) | 7.12 | 3.34 (p 0.0675) | 2.22 (p 0.136) | ✗ | ✓ |

Human data reproduces the paper on 0 of 4 effects; agents agree with humans on 2 of 4.
