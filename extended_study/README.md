# Extended Studies

This directory contains complete study environments that are still being
evaluated before possible inclusion in the default HumanStudy-Bench registry.
They are intentionally excluded from `studies/` and the website study index.

Current extensions:

- `study_016`: Anderson and Holt (1997), *Information Cascades in the Laboratory*
- `study_017`: Jaquiery and Yeung (2024), *Preferences for Advisor Agreement and Accuracy*
- `study_019`: Schobel, Rieskamp, and Huber (2016), *Social Influences in Sequential Decision Making*

Run an extension by passing its directory directly:

```bash
.venv/bin/python generation_pipeline/run.py \
  --settings config/settings.example.yaml \
  --stage 5 \
  --experiment extended_study/study_019 \
  --sim-models mock \
  --n-agents 2 \
  --mock-agent
```

Simulation outputs remain under `runs/<study_id>/<model>/`.
