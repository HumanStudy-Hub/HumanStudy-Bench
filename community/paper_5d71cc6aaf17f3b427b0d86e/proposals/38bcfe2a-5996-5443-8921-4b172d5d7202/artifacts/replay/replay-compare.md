# tesslerfranke2018: replay vs human data

Human sessions: 826. AI participants: n=10 as the package counts it (per arm or in total) (claude -p --model haiku --strict-mcp-config --setting-sources "" --tools "" --system-prompt "You are taking part in a study. Answer as asked.").

| Effect | Reported | Human | Agent | Human≈paper | Agent agrees |
| --- | --- | --- | --- | --- | --- |
| Exp1: antonym-vs-negated-positive contrast, morphological minus lexical (lexical antonyms lower) | – | 0.0582 (p 3.55e-05) | 0.0335 (p 0.395) | – | ✗ |
| Exp2 single: morphological minus lexical contrast | – | 0.0219 (p 0.00624) | 0.018 (p 0.434) | – | ✗ |
| Exp2 single, morphological: negated positive minus antonym (paper: ~0) | – | -0.000288 (p 0.956) | 0.103 (p 0.000152) | – | ✗ |
| Exp2 single, lexical: negated positive minus antonym (lexical more negative) | – | 0.0217 (p 0.000388) | 0.121 (p 2.22e-05) | – | ✓ |
| Exp2 morphological: multiple-context makes antonym lower than single (single minus multiple contrast) | – | 0.0629 (p 8.78e-10) | -0.00533 (p 0.784) | – | ✗ |
| Exp2 three-way: context effect larger for morphological than lexical | – | 0.024 (p 0.135) | 0.004 (p 0.886) | – | ✓ |

Human data reproduces the paper on 0 of 0 effects; agents agree with humans on 2 of 6.
