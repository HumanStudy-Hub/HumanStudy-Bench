# Community contributions

Anyone can publish a reconstruction of a published paper here, either from the HumanStudy app
(**Publish to community**) or with an ordinary pull request. One contribution is one
reconstruction of one paper by one person or team. Several people may reconstruct the same paper;
each gets their own folder, so nobody edits anyone else's files and nothing needs merging by hand.

## Layout

```text
community/
  doi_10.1111_1468-0017.00215/            one folder per paper key
    paper.json                            DOI, title, authors, year, venue (written once, by the first contributor)
    knobe-2003--hs-7f3a/                  <slug>--<contributor-id>   (a contribution)
      CONTRIBUTION.json                   attribution, licence, checks passed
      program.json                        the Study document (Human Program v2)
      package/knobe-2003/…                optional: the eight-file runnable package
      artifacts/simulations/<stamp>/      optional: report.md, evaluation.json, results.png
    knobe-2003--gh-xuanl17/
    knobe-2003--gh-xuanl17-v2/            an update by the same person; the old folder stays
```

### Paper key and folder name

The key is `doi:<doi>` (lower-case, no URL prefix) when the paper has a DOI, otherwise
`title:<normalised-title>-<year>-<first-author-surname>`. The title is lower-cased, accents are
dropped, other characters become single hyphens, and it is cut at 60 characters on a word
boundary. The folder name is the key with every character outside `A-Za-z0-9._-` replaced by `_`,
so `doi:10.1111/1468-0017.00215` becomes `doi_10.1111_1468-0017.00215`. Both functions live in
[`scripts/community.py`](../scripts/community.py) (`make_paper_key`, `paper_key_to_dir`).

### Contribution folder name

`<slug>--<contributor-id>` with an optional `-v2`, `-v3`, ... suffix. Slug and contributor id use
lower-case letters, digits and hyphens. Typical contributor ids are `gh-<github-login>` and
`hs-<short-hub-id>`. Updating your own contribution means adding a new `-vN` folder whose
`CONTRIBUTION.json` has `supersedes` set to the previous folder name.

### What is not published

`paper.pdf` and anything under `sources/`: the paper is usually under copyright. A PDF is accepted
only when `CONTRIBUTION.json` has `"includesPdf": true` (open access, or you hold the rights).
Short evidence quotes inside `program.json` stay; they are the citations.

### Licences

`program.json`, materials and reports: CC BY 4.0. Code (`task/`, `evaluation/`): MIT, like the
rest of HumanStudy-Bench. `CONTRIBUTION.json` records both under `license`.

## Files

| File | Contract |
|---|---|
| `paper.json` | [`contracts/paper.schema.json`](../contracts/paper.schema.json): `schemaVersion` 1, `paperKey`, `title`, `authors[]`, `year`, optional `doi`, `venue` |
| `program.json` | [`contracts/human-program.schema.json`](../contracts/human-program.schema.json), checked with `agent_pipeline.human_program.validate_human_program` |
| `CONTRIBUTION.json` | [`contracts/contribution.schema.json`](../contracts/contribution.schema.json) |
| `package/<slug>/` | the eight-file contract in [`agent_pipeline/CLAUDE.md`](../agent_pipeline/CLAUDE.md); `study.json.program` must equal `program.json` |

`CONTRIBUTION.json`:

```json
{
  "schemaVersion": 1,
  "paperKey": "doi:10.1111/1468-0017.00215",
  "slug": "knobe-2003",
  "contributor": {
    "name": "Ada Lovelace",
    "affiliation": "NYU",
    "orcid": "0000-0002-1825-0097",
    "github": "ada",
    "hub": { "id": "hs-7f3a", "signature": "…" }
  },
  "license": { "content": "CC-BY-4.0", "code": "MIT" },
  "includesPdf": false,
  "appVersion": "0.9.0",
  "createdAt": "2026-10-12T09:30:00Z",
  "checks": { "schema": true, "package": true, "simulation": null },
  "supersedes": "knobe-2003--gh-ada"
}
```

`affiliation`, `orcid`, `github`, `hub` and `supersedes` are optional, but a contributor needs a
`github` login or a `hub` entry. `checks.package` and `checks.simulation` are `null` when that
part was not built.

## Checking a contribution locally

```bash
python scripts/validate_contribution.py community/<paper-key>/<slug>--<contributor-id>
```

The validator checks `CONTRIBUTION.json`, `program.json`, the sibling `paper.json` and, when
`package/` exists, runs the same checks as `agent_pipeline/validate_package.py`. It also rejects
files other than `program.json`, `CONTRIBUTION.json`, `package/` and `artifacts/`, links, files
over 25 MB, and PDFs that are not declared.

## Who may change what (CI)

The workflow `.github/workflows/validate-community.yml` runs on every pull request that touches
`community/`.

- **A new contribution folder** is accepted when the PR author is `contributor.github`
  (case-insensitive), or the author is `humanstudy-bot` and `contributor.hub.signature` is
  present. CI only checks that the signature exists; the HumanStudy server verifies it when it
  opens the PR.
- **Changing an existing folder** is allowed for its owner (as recorded at the base commit),
  `humanstudy-bot`, and the maintainers listed in `MAINTAINERS` in `scripts/community.py`. For
  anyone else the check fails: corrections to somebody else's work are reviewed by the owner.
- **`paper.json`** may be added, not edited, except by the bot or a maintainer.
- A PR that only adds contribution folders and passes everything gets the `auto-merge` label.
  The label is all CI does; merging is a separate step.

`co_website/data/studies_index.json` is rebuilt after merges by `scripts/build_studies_index.py`.
Its `studies` array is unchanged; the `community` array has one entry per paper with a
`contributions[]` summary (contributor, date, number of condition nodes and steps, open issues by
severity, whether a package and a simulation are present).
