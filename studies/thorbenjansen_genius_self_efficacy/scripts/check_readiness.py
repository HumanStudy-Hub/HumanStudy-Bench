import json
from pathlib import Path

if __name__=='__main__':
    study=Path(__file__).resolve().parents[1]
    state=json.loads((study/'source/readiness.json').read_text())
    print(json.dumps(state,ensure_ascii=False,indent=2))
    raise SystemExit(0 if state['ready_for_submission'] else 2)
