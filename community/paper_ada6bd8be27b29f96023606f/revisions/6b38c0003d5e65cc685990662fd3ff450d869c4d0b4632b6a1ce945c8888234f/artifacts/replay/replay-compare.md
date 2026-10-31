# manylabs2: replay vs human data

Human sessions: 99736. AI participants: n=30 as the package counts it (per arm or in total) (claude -p --model haiku --strict-mcp-config --setting-sources "" --tools "" --system-prompt "You are taking part in a study. Answer as asked.").

| Effect | Reported | Human | Agent | Human≈paper | Agent agrees |
| --- | --- | --- | --- | --- | --- |
| Kay: structure vs random event, willingness to pursue goal (d) | -0.02 | -0.0282 (p 0.255) | 1.05 (p 0.0112) | ✓ | ✗ |
| Rottenstreich & Hsee: affect-rich choice, 1% chance vs certain (d from odds ratio) | -0.08 | -0.0798 (p 0.00212) | – (p 0.277) | ✓ | – |
| Bauer: trust in others, individual minus consumer wording (d) | 0.12 | 0.121 (p 8.63e-07) | – (p –) | ✓ | – |
| Critcher & Gilovich: sales estimate, P97 minus P17 (d) | 0.04 | 0.0619 (p 0.0104) | -0.455 (p 0.223) | ✓ | ✗ |
| Hauser: side-effect vs greater-good permissibility (d from odds ratio) | 1.35 | 1.32 (p 0) | – (p 4.32e-08) | ✓ | – |
| Ross: false consensus, supermarket (d) | 1.18 | 1.2 (p 0) | 0.58 (p 0.203) | ✓ | ✗ |
| Ross: false consensus, traffic ticket (d) | 0.95 | 0.97 (p 0) | -0.0985 (p 0.733) | ✓ | ✗ |
| Tversky & Kahneman: framing, cheap vs costly item on sale (d from odds ratio) | 0.4 | 0.399 (p 1.81e-50) | 1.41 (p 0.0106) | ✓ | ✓ |
| Risen & Gilovich: tempting fate raises likelihood of being called on (d) | 0.18 | 0.181 (p 7.67e-16) | -0.442 (p 0.167) | ✓ | ✗ |
| Hsee: less is better, scarf minus coat generosity (d) | 0.78 | 0.774 (p 7.02e-235) | 8.57 (p 1.35e-17) | ✓ | ✓ |
| Schwarz: assimilation/contrast, specific-first minus general-first correlation (Cohen's q) | -0.07 | -0.0702 (p 0.00244) | – (p –) | ✓ | – |
| Shafir: P(award extreme) + P(deny extreme) - 1, converted to d from the z test | -0.13 | -0.138 (p 9.95e-10) | 1.2 (p 0.00475) | ✓ | ✗ |
| Zaval: heat minus cold priming, concern about global warming (d) | -0.03 | -0.02 (p 0.517) | -0.0567 (p 0.892) | ✓ | ✓ |
| Knobe: harm minus help condition, intentionality (d) | 1.75 | 1.75 (p 0) | 4.04 (p 9.76e-12) | ✓ | ✓ |

Human data reproduces the paper on 14 of 14 effects; agents agree with humans on 4 of 10.
