# v0.8 protocol releases in HumanStudy-Bench

This page covers the Bench part of the v0.8 platform design. In v0.8 a study is one folder holding `study.json` and `protocol.json`, and one engine (`hsenv`)
reads it on every platform. Bench publishes a released protocol as a community revision
(`community/paper_<id>/revisions/<sha256>/`). Bench never interprets a protocol itself and never executes
its files: every check calls the pinned `hsenv`.

## What this change does

- **B1.** `requirements.txt` pins hsenv by commit SHA; it is the only place the pin is written, and the
  sync-studies-index, research-catalog, maintain-community and validate-studies workflows read it from there.
  `scripts/research/effects_core.py` re-exports `hsenv.effects_core`, with import paths and the CLI kept.
  `tests/test_hsenv_pin.py` checks that every community revision with `effects.json` gives the same problems,
  warnings and kg digest as the Bench copy did (`tests/fixtures/effects_core_golden.json`, recorded with the
  copy at `b34d214`), and that the protocol readers never execute protocol text.
- **B2.** `scripts/check_contributors.py` (one contributor rule), `scripts/research/protocols.py` (release
  checks), a protocol branch in `validate_assets`, and the `validate-protocols` job in `validate-studies.yml`.
  The tests use an invented CC0 fixture release.
- **B3.** `build_catalog` adds a `protocol` block to the paper's `research_index.json` entry and computes
  `replayable`/`humanBenchmark` from the protocol. `studies_index.json` and `research_index.json` stay
  byte-identical for today's studies.

Not done here: **B4** (`run_protocol.py`, the playground on `kind: protocol`, quiet logs) waits for owner gate G3.
**B5** (the first releases) needs signed reviews (G2) and studies with no `release_hold`.

## What CI checks on a protocol release

`python -m scripts.research.protocols <revision-dir>... [--sessions 300]` exits 1 on any error. The steps
follow the order in design §2.1:

1. **Schema.** `hsenv.schema_check` validates `study.json` (`kind=study`) and `protocol.json` (`kind=protocol`,
   including its cross-reference pass). `schema_version` is required: legacy files are not released.
2. **Release rules.**
   - Every path must be on the allowlist: `study.json`, `protocol.json`, `effects.json`, `program.json`,
     `capabilities.json`, `review/fidelity-review-*.md`, `review/checks-latest.json`, `sources/SOURCE.json`,
     `sources/open/**`, `sourcetext/<key>*` (only when `meta.sources[key]` is under `sources/open/`),
     `variants/*.json`, `artifacts/**` and `package/**`.
   - These are never published: `sources/{researcher,private,quarantine}/**`, `runs/**`, `**/runs/**`,
     `**/records.jsonl`, `paper/**` and `bundle.json`.
   - `visibility` must be `"public"`. `quality` must be `high_fidelity`, with `fidelity.verdict` HIGH FIDELITY,
     a non-empty `fidelity.signed_by` and the review file included (G2).
   - No `release_hold` may appear, neither in `study.json` nor on any arm.
   - `study.json.protocol.sha256` must equal the canonical hash of `protocol.json`, and `study.json.arms` must
     agree with the protocol's arms.
   - An image `src` must be a URL, or an `asset:<sha16>` that points to a file in `sources/open/`.
   - `study.json.hsenv` must admit the installed engine.
3. **Checks report.** `review/checks-latest.json` must report `PASS: true` and carry `protocol_sha256`
   equal to this protocol's hash. A report without the hash, or with another hash, counts as stale.
   `--sessions N` also replays N random sessions through `hsenv.checks` (references and fuzz).
4. **Effects and capabilities.** `effects.json` goes through `hsenv.effects_core.check`, and
   `capabilities.json` through the capability rules.
5. **Digest.** The revision directory's name must equal its manifest digest (`revisions.revision_id`).

Each revision is checked through one of two paths:

- **Maintainer release PR** (direct files under `community/**`): the `validate-protocols` job in
  `validate-studies.yml` runs the checks on each changed revision that holds `protocol.json`.
- **Community proposal** (`maintain-community.yml`): `revisions.validate_assets` sees `protocol.json` and runs
  the same checks on the proposal's files. `program.json` becomes optional. The engine comes from main's
  pinned requirements and never from the proposal. A proposal that holds `protocol.json` is **never
  accepted on this path**: `integrate` always returns `needs_review` with `requiresMaintainer: true`, whatever
  the review says (an agent's or anyone's `accept`) and whatever the release declares about itself
  (`fidelity.signed_by`, a PASS `checks-latest.json`). No agent review is run for it. A maintainer publishes
  it through a release PR.

**Contributor rule.** `scripts/check_contributors.py --author <login> <dir>...` reads `index.json` for
legacy studies and `study.json` for protocol releases. The rule is unchanged: a `contributors[].github` must
equal the PR author, given as `login` or as a GitHub URL. Admin and maintain users still bypass it in the
workflow. A community proposal gets its PR author appended to `study.json` contributors when missing.

## research_index.json

Shared contract fixture: `tests/fixtures/protocol_release/research-index-protocol.json` is the exact
`protocol` block and capabilities Bench writes for the fixture release (`schemaVersion` is the protocol
schema version, `"1"`). The web repo tests its reader against the same JSON; change both together.

For a current revision with `protocol.json`, the paper entry gains these fields:

```
"protocol": {"slug", "schemaVersion", "hsenv", "fidelity": {"verdict", "signed_by"},
             "arms": [{"id", "runnable", "nPaper"}], "runExcluded": [...], "effects": n, "revision"},
"capabilities": {..., "replayable": any runnable arm without release_hold, "humanBenchmark": any effect with reported values},
"coverage": "protocol-release"        (only when the revision has no program.json)
```

The block holds counts and ids only, never protocol text. `studies_index.json` keeps its legacy shape.

The Web library checks the catalog-level `publications` map for the **Validated** label.
Maintainers record `{"kind": "maintainer-reviewed-protocol", "revision": "<sha256>"}` by paper ID in
`catalog/publications.json` after reviewing the exact release. The catalog includes the record only while
it points to the current signed runnable protocol. A new recommendation does not inherit the label.
Contributor-editable paper metadata cannot populate this map. The catalog snapshot digest covers active
publication records. Validation is for the supported protocol scope; it does not certify replication of
the original findings.

## Test fixture

`tests/fixtures/protocol_release/example2026lamps/` is an invented study under CC0-1.0. Its material is two
sentences about a lamp and one rating question, and it is no real paper, protocol or participant. Its
`review/checks-latest.json` is a real `hsenv.checks` PASS report (200 sessions, verbatim 3/3 found, keyed sign
correct), with `protocol_sha256` added. To regenerate the fixture, write it again with the same content, run
`python -m hsenv.checks <dir> --sessions 200 --source-text survey=<dir>/sourcetext/survey.txt`, and store the
output with `protocol_sha256`.

## Running locally

```bash
pip install -r requirements.txt          # installs hsenv from GitHub at the pinned commit
pytest tests/test_hsenv_pin.py tests/test_protocol_release.py tests/test_check_contributors.py -o addopts=''
python -m scripts.research.protocols community/paper_<id>/revisions/<sha> --sessions 300
```

The hsenv tests skip when hsenv is not installed.

## Open items (hsenv H1 and maintainer decisions)

- **hsenv is installed, not vendored.** `requirements.txt` pins hsenv v0.8.1 by its commit SHA
  (`db85a9c`). The desktop pins the same release (B1 is chained to it).
- **Engine schema vs design.** `study.v1.json` has `additionalProperties: false` and lacks the design's §1.1
  fields (`slug`, `paper`, `contributors`, `visibility`, `hsenv`, `fidelity`, `arms`, `paperId`, `release`).
  Bench validates those itself and removes them before calling `schema_check` (`protocols.RELEASE_FIELDS`).
  H1 should add them to the schema.
- **H1 must provide:**
  - `protocol_sha256` in the `hsenv.checks` report: today it carries none, so CI could not detect a stale report
  - `hsenv check --public`: `protocols.replay` is the interim refs and fuzz subset
  - `hsenv.release.PUBLIC_PATHS`: a test fails if it differs from Bench's list
  - `hsenv.capabilities`: `capabilities.py` stays a Bench copy until then
- **Design statement corrected.** Design §2.1 says `expr.py` has "no eval/exec (verified by grep)". In fact
  it calls `compile` and `eval`. It does so on an AST whose node types it has checked against a whitelist,
  and it runs with `__builtins__` emptied. The CI test pins that whitelist and a set of escape attempts.
- **Maintainer decisions:** who signs a release (G2), and how B4 keeps public Actions logs quiet (G3).

## Hosted community drafts

A proposal explicitly marked `kind: protocol` whose study card has `quality: draft`
uses draft intake, not formal release validation. It must include a schema-valid Program,
a matching raw protocol binding, a matching canonical study binding, hosted-source
bytes with declared redistributable licenses and hashes, and consistent paper identity.
Private/raw paths, held material, malformed schemas, stale reports and mismatched versions
remain failures. Draft intake does not require a curated fidelity signature, public
release visibility, or an execution report; missing those is not scientific approval.

Passing intake always yields `needs_review` with `requiresMaintainer: true`. No agent
review, contributed code, random session or participant model is executed. It creates no
current revision and awards no runnable/validated capability. Formal release validation
retains all its existing gates. The test fixture in `tests/fixtures/paired-draft` is an
invented CC0 lamp study exported by Desktop's paired exporter, including an honestly
blocked arm; it is not an empirical research asset.
