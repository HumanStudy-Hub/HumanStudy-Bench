# pennycook2020: replay vs human data

Human sessions: 1709. AI participants: n=10 as the package counts it (per arm or in total) (claude -p --model haiku --strict-mcp-config --setting-sources "" --tools "" --system-prompt "You are taking part in a study. Answer as asked.").

| Effect | Reported | Human | Agent | Human≈paper | Agent agrees |
| --- | --- | --- | --- | --- | --- |
| Study 1: sharing reduces sensitivity to veracity (-beta of veracity x condition, z-scored yes/no) | – | 0.52 (p 0) | 0.607 (p 0.0322) | – | ✓ |
| Study 2: accuracy prompt increases sharing discernment (beta of veracity x treatment, 0-1 scale) | 0.039 | 0.0359 (p 0.000199) | 0.016 (p 0.763) | ✓ | ✗ |
| Study 2: treatment/control sharing discernment ratio | 2.8 | 2.91 (p –) | 1.28 (p –) | ✓ | ✓ |
| Item level: Study 1 accuracy rating vs Study 2 treatment effect (r) | 0.76 | 0.757 (p 1e-06) | 0.269 (p 0.15) | ✓ | ✗ |

Human data reproduces the paper on 3 of 3 effects; agents agree with humans on 2 of 4.
