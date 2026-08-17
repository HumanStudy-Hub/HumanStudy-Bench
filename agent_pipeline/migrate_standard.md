# Add the standard run interface to an existing package

An existing buffer study package was built before the standard harness interface
existed. It already has a runnable task (`task/adapter.py`) and an evaluator
(`evaluation/evaluation.py`), but they expose a study-specific interface. Add
the standard interface on top without changing the existing logic.

## Standard interface to add

- `agent_fn(input: dict) -> dict` is the injected participant. `input` holds the
  state shown to the participant for one decision; the returned dict holds the
  participant's action.
- `run_sessions(agent_fn, seed) -> list[dict]` runs the study with that agent
  across every condition/arm the adapter defines and returns one session-log
  dict per session. Each session log must carry its condition/arm labels so
  `evaluate` can compare them.
- `evaluate(sessions: list[dict]) -> dict` scores the sessions returned by
  `run_sessions`.

## What to do

1. Read `task/adapter.py`, `task/task.json`, and `evaluation/evaluation.py`.
2. Add `run_sessions(agent_fn, seed)` to `task/adapter.py` (or import it from a
   new `task/run_sessions.py`). It must:
   - adapt the standard `agent_fn(input: dict) -> dict` to whatever the existing
     adapter expects (a numeric return, a different signature, or a
     build-prompt/validate-response flow);
   - iterate every condition/arm the adapter defines and skip "blocked"/missing
     cells rather than inventing material;
   - return a list of session-log dicts (or response records) that the existing
     evaluator can consume.
3. Add `evaluate(sessions)` to `evaluation/evaluation.py` (or wrap the existing
   evaluate entry) so it scores the list returned by `run_sessions`. Do not
   change the existing checks; only add the standard entry point.
4. Verify locally that `run_sessions` and `evaluate` import and run (use a
   trivial agent_fn that returns a plausible dict).
