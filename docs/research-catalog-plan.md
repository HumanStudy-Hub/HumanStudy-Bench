# Community research catalog implementation

Approved 2026-10-21. Branch: `codex/community-research-index` in Bench, web and desktop.
Do not merge or modify production data before the owner's explicit approval.

## Contract

- One stable paper identity, one current community revision, immutable underlying assets.
- Three-way integration against the contributor's actual baseline; conflicts are review items.
- Validate complete Human Programs and package bindings; missing scientific information is valid.
- Search is a revision-bound projection, not a second manually edited research document.
- Taxonomy has stable concepts, aliases, evidence, review status and named curation ownership.
- Public-source imports preserve source versions, coverage and access/license restrictions.
- Human observations, original reported estimates, reanalysis and simulation remain distinct.
- Preserve complete content; no model-output truncation to satisfy a program-size cap.

## Work

1. Bench: identity/projection, taxonomy, safe three-way revisions and validation, public-source adapters,
   reproducible manifests, contribution worker, index generation and tests.
2. Web: PostgreSQL migration and real database tests; search API with explicit snapshot fallback;
   canonical contribution baseline and safe default artifact publishing; existing catalog integration.
3. Desktop: richer catalog search, community identities, revision-pinned forks, local baseline metadata;
   remove program-content size caps and update build instructions.
4. Test all three, document deployment order and known source gaps, create linked PRs for review.

## Acceptance

- Searching `framing effect` or `框架效应` finds the same reviewed concept without calling an LLM.
- Each paper appears once; an imported source directory is not claimed to be a fully rebuilt paper.
- Same import is idempotent; Psych-101 overlap is preserved; gated tests are never downloaded.
- Concurrent or stale contributions cannot revert unrelated accepted changes.
- Merge conflicts, source disagreements and validation failure retain the previous recommendation.
- Anonymous/public readers cannot write the catalog or read unpublished proposals.
- Forks retrieve immutable files and record the baseline; they never claim a SHA while reading `main`.
- API results identify database/snapshot coverage and pagination; no hidden top-k completeness claim.
- No production migration, release, community auto-merge activation or code merge in this task.
