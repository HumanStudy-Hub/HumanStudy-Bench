#!/usr/bin/env python3
"""effects.json rules and computation: the pinned hsenv engine (V080-PLATFORM B1).

This file used to be a copy of humanstudy-desktop `pack/skills/export/agent/effects_core.py`. Bench now
reads the one engine every platform reads, `hsenv.effects_core`, at the version pinned in requirements.txt.
Import paths are kept (`scripts.research.effects_core`, and loading this file by path), so callers do not
change. Do not add logic here: a rule change is an hsenv release.

    python scripts/research/effects_core.py check <project>      same CLI as hsenv.effects_core
"""
import sys

from hsenv import effects_core as _engine

globals().update({k: v for k, v in vars(_engine).items()
                  if not (k.startswith("__") and k.endswith("__"))})
ENGINE = _engine

if __name__ == "__main__":
    sys.exit(_engine.main())
