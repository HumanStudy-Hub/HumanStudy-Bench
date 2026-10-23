# One recommended research asset per paper

This is the first implementation of the owner-approved 2026-10-21 research plan. It is a
review branch, not a production rollout or a claim that the imported experiments are runnable.

## What is delivered

- Stable paper identities and source aliases; explicit parent paper identity when forking.
- One `current.json` per community paper, immutable content-addressed revisions, full attribution.
- Three-way integration against the actual contribution baseline; ordered arrays stay ordered.
- Schema/reference checks, package bindings, independent read-only source review, bounded workflow.
- A rebuildable research index including substudy text, evidence IDs, issues and separate capabilities.
- An initial framing concept/alias seed and a versioned taxonomy file for curator-led maintenance.
- Metadata adapters for the 180-claim reanalysis corpus and Psych-201 source directories.

The source imports retain full metadata and original URLs, exact source versions, corpus aliases,
and explicitly unverified data access, licensing and substudy coverage. Psych test sets are not
downloaded. Transcripts, raw human data and source code are never executed during an import.
Original reported results are not converted into fabricated meta-analysis estimates.

`capabilities` are **verified** capabilities: false means not yet established by this pipeline,
not proof that the research can never be run. Structural checks alone never establish replayability.

## Layout

```text
catalog/taxonomy.json                       definitions, aliases, evidence and curation status
catalog/imports/<source>.json               repeatable public metadata snapshots
co_website/data/research_index.json         derived search/catalog projection
community/<paper-key>/proposals/<uuid>/     immutable submissions, not public catalog versions
community/<paper-key>/reviews/<uuid>.json   concrete validation/review outcomes
community/<paper-key>/revisions/<hash>/     complete accepted assets + manifest + projection
community/<paper-key>/current.json          the single recommended version
```

A contribution's `paperId` comes from its original catalog entry, and `baseRevision` comes
from the snapshot actually copied locally. Neither is replaced with today's recommendation.
The worker resolves these against trusted target-branch data. Title-only matches without
explicit lineage become identity-review candidates rather than automatic merges.

Scientific `codesign` changes cannot silently replace a reconstruction. Conflicting scientific
interpretations, unavailable source evidence, invalid or unknown baselines and failed checks
retain the previous recommendation. Compatible changes can be prepared without rewriting
unrelated accepted changes. All accepted contributions remain attributable in the archive.

## Rebuild and test

```bash
python -m scripts.research.imports psych201 --output catalog/imports/psych201.json
python -m scripts.research.imports reanalysis --output catalog/imports/reanalysis.json
# Add --source-ref <recorded commit> for exact-source repeatability.
python scripts/build_studies_index.py
python -m pytest -o addopts='' tests/test_research_catalog.py tests/test_community_worker.py tests/test_build_studies_index.py -q
```

No model call is made by these commands. A fresh import is written atomically after the whole
source completes; failed or truncated source-tree reads do not publish a partial collection.
The reanalysis importer records focal claims, not a claim to cover all substudies in each article.

## Maintenance worker

`maintain-community.yml` runs trusted target-branch code. It reads PR blobs without checking out
the PR, does not execute adapters or contributor scripts, and checks contributor identity.
Only an independently reviewed exact candidate can advance `current.json`. Source quotations
are checked against available resources before model review; missing source text requires review.

The worker uses the existing OpenRouter setup with `OPENROUTER_API_KEY` and `OPENROUTER_MODEL`.
Its review has no execution tools and cannot mark behavioral execution or statistical reanalysis
as verified. Every provider failure ends with a concrete review status; there is no blind continue.
The GitHub job has a 12-minute limit and serializes recommendation updates. Final publication
is a non-forced ref update with the previous main as parent; another writer causes a retry,
not an overwrite. The contribution PR is archived/closed after integration rather than merging
untrusted code changes into the implementation.

**The workflow is off unless `COMMUNITY_MAINTAINER_ENABLED=true`. Do not enable it before the
owner reviews all coordinated PRs and approves rollout.** `COMMUNITY_BOT_LOGIN` must match
the actual bot token account in both the Hub and Bench settings.

## Coordinated rollout after approval

1. Review/merge the web, Bench and desktop PRs as a coordinated change; no desktop release is
   created by these branches.
2. Apply the reviewed web migration to a preview database, run its SQL/permission tests there,
   and sync the approved catalog. The branch's embedded PostgreSQL tests already cover the SQL.
3. Configure the web's `RESEARCH_INDEX_URL` to the approved Bench source during preview;
   production defaults to Bench `main`. A bundled copy supports rollout/offline fallback.
4. Configure `RESEARCH_INDEX_SYNC_SECRET` on web and Bench; set Bench's
   `RESEARCH_CATALOG_SYNC_URL=https://humanstudy-hub.com/api/research/sync`.
   The endpoint fetches an exact commit in the fixed Bench repository, not caller-chosen URLs.
5. Perform the first database sync with the web's `scripts/sync-research-index.mjs` in the
   server environment. It uses the current catalog snapshot as its compare-and-swap baseline.
6. Set web `COMMUNITY_CATALOG_ENABLED=true` to retire the old numbered-study pipeline publisher.
   The old `/api/contribute/*` paths already use the shared authenticated proposal handlers.
7. Verify a real source-backed contribution, a conflict, a stale baseline, a fork, and the status
   update in preview. Then enable `COMMUNITY_MAINTAINER_ENABLED` in production only with approval.
8. Ship the reviewed desktop update separately; existing projects acquire missing catalog tools,
   while customized instructions are preserved.

## Deliberately not claimed complete

- The 302 initial identities are catalog entries, not 302 completed Human Programs.
- No original trial dataset was bulk imported or validated; participant overlap is not yet resolved.
- The proposed twelve diverse conversion pilots have not been built and empirically validated.
- PSYCHE's full implementation/data availability still needs author confirmation. AutoCog and
  YuLan-OneSim are research references for later adapters, not imported human benchmarks here.
- A human taxonomy lead remains to be appointed. Future seed suggestions need accountable review.
- Runtime smoke tests and scientific numerical reference tests must run in an isolated executor
  before replayable/reanalyzable flags can become true. This read-only maintainer never asserts them.
- No production migration, live catalog sync, paid model review, release or main-branch merge was
  performed to prepare these PRs.
