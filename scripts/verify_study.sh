#!/usr/bin/env bash
# Verify a study has required structure and run pytest on Python scripts.
# Usage: bash scripts/verify_study.sh study_XXX

set -e

STUDY_ID="${1:?Usage: bash scripts/verify_study.sh study_XXX}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STUDY_DIR="$REPO_ROOT/studies/$STUDY_ID"
PASSED=0
FAILED=0

msg_pass() { echo "  [PASS] $1"; ((PASSED++)) || true; }
msg_fail() { echo "  [FAIL] $1"; ((FAILED++)) || true; }

echo "Verifying study: $STUDY_ID"
echo "Study dir: $STUDY_DIR"
echo ""

# === Structure checks ===
echo "=== Structure ==="

if [[ -d "$STUDY_DIR/source" ]]; then
  msg_pass "source/ directory exists"
else
  msg_fail "source/ directory missing"
fi

if [[ -d "$STUDY_DIR/scripts" ]]; then
  msg_pass "scripts/ directory exists"
else
  msg_fail "scripts/ directory missing"
fi

if [[ -f "$STUDY_DIR/index.json" ]]; then
  msg_pass "index.json exists"
else
  msg_fail "index.json missing"
fi

if [[ -f "$STUDY_DIR/README.md" ]]; then
  msg_pass "README.md exists"
else
  msg_fail "README.md missing"
fi

# Validate index.json required fields
if [[ -f "$STUDY_DIR/index.json" ]]; then
  if python3 -c "
import json
with open('$STUDY_DIR/index.json') as f:
    d = json.load(f)
required = ['title', 'authors', 'year', 'description']
missing = [k for k in required if k not in d]
if missing:
    raise SystemExit('Missing fields: ' + ', '.join(missing))
if not d.get('title') or not isinstance(d.get('authors'), list) or not d.get('description'):
    raise SystemExit('title, authors (list), description must be non-empty')
" 2>/dev/null; then
    msg_pass "index.json has title, authors, year, description"
  else
    msg_fail "index.json missing required fields (title, authors, year, description)"
  fi
fi

# === Pytest on all Python scripts ===
echo ""
echo "=== Python tests ==="

PY_FILES=$(find "$STUDY_DIR" -name "*.py" 2>/dev/null)
if [[ -n "$PY_FILES" ]]; then
  if python3 -m pytest "$STUDY_DIR" --tb=short -q 2>&1; then
    msg_pass "pytest passed"
  else
    msg_fail "pytest failed"
  fi
else
  echo "  No Python files found, skipping pytest."
fi

echo ""
echo "Summary: $PASSED passed, $FAILED failed"
if [[ $FAILED -gt 0 ]]; then
  exit 1
fi
exit 0
