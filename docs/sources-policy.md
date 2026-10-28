# Sources policy: what a published study may contain

Applies to every study in this repository: benchmark studies (`studies/`) and community papers (`community/`).
Issue: [#34](https://github.com/HumanStudy-Hub/HumanStudy-Bench/issues/34). HumanStudy Hub is a non-commercial research
project; this policy relies on that.

This is a working policy, not legal advice. Ask the maintainers when a case is unclear; the default is to publish the
key, not the content.

## The aim

Give users as much as we can, and never redistribute what we have no right to. Everything we wrote is published.
Third-party content is published only when its license allows. Everything else is published as a **key**: enough to
find, obtain and verify the exact original.

## What may be published

| Content | Published | Condition |
|---|---|---|
| Title, authors, year, venue, DOI, links | Always | Facts |
| Our work: program, summaries, code, analyses, replay results, agent skill | Always | We wrote it (repository license) |
| Evidence quotes from the paper, with page | Yes | Short: up to about 50 words each, only what supports a field |
| Participant-facing materials printed in the paper (instructions, items, vignettes, response scales, stimuli text) | Yes, verbatim, with page and citation | Needed to run the study. Exceptions below |
| Materials posted by the authors (OSF, GitHub, supplements) | Yes | Under an open license (CC0, CC-BY, CC-BY-SA, CC-BY-NC*, CC-BY-NC-SA*, CC-BY-NC-ND* unmodified). Otherwise link only |
| Paper PDF | Yes | The exact version we built from carries an open license as above. Otherwise key only |
| Author analysis code | Yes | Open-source or CC license. Otherwise key only |
| Raw human data | Never in git | Open license: mirror in a separate data store that supports takedown. Otherwise key only |

\* NC licenses are allowed because the Hub is non-commercial. If that changes, NC content must be reviewed.

**Exceptions: always location only, never the text:**
- **Commercial instruments.** Scales and tests sold by a publisher, e.g. Pearson, Mind Garden, PAR, MHS, Hogrefe test
  editions, or marked "©" with a permission requirement. Record the name, publisher, citation, item count and scoring
  rule.
- **Third-party works reproduced as stimuli.** News articles, stories, essays or passages longer than about 300 words
  that the paper reprints from another publisher, unless openly licensed.

"Public" or "on OSF" is not a license. OSF projects carry no license unless the creator chose one.

## Material states

Every material item in a program has one of three states:

- `included`: the text is in the program, under a row above.
- `linked`: the text is available elsewhere but not ours to redistribute; we store a pinned link.
- `location`: restricted; we store where it is and what it is, never the text.

## The key: what to publish instead

Each study has a source record (`source/SOURCE.json` for benchmark studies, `sources/SOURCE.json` for community
papers):

```json
{
  "schemaVersion": 1,
  "paper": {"doi": "10.1371/journal.pone.0154854", "title": "…", "year": 2016},
  "pdf": {
    "status": "included",
    "license": "cc-by",
    "filename": "paper.pdf",
    "sha256": "…",
    "bytes": 1234567,
    "builtFrom": "the version the program's page numbers refer to",
    "licenseEvidence": "license statement on page 1 of this file"
  },
  "openAccess": [{"url": "…", "license": "cc-by", "host": "PLoS ONE"}],
  "obtain": "How a user gets the PDF when it is not included",
  "policy": "docs/sources-policy.md"
}
```

- `pdf.status` is one of:
  - `included`: open license confirmed for this exact file;
  - `not-included`: closed, or the license is not confirmed for the version we used;
  - `removed`: it was published before this policy.
- `sha256` is always recorded, so a user's own copy can be checked against the version the page numbers refer to.
- `openAccess` lists legal free copies. They may be a different version from the one we built from; say so when known.

When a study is opened in the desktop app without its PDF, the app asks the user for their own copy, checks the hash,
and keeps it local. Source highlighting and select-to-explain work as soon as it is there.

## Human data and author code

These are published as keys too: source, pinned version (OSF file version, Zenodo DOI, commit), sha256 and license. The
app downloads the pinned version from the original location and checks the hash.

- **The authors publish a new version:** the user chooses old or new, and results record which one was used.
- **The data are withdrawn:** with an open license, a mirror in the data store is used and marked "withdrawn by the
  authors". Without one, the study stays readable and runnable with AI participants, but the human comparison cannot
  run. Earlier results keep their version label.

## Takedown

Anyone may ask us to remove content: open an issue or email the maintainers. We remove it from `main` (and from the
data store) within 7 days while we look into it, and replace it with the key. Git history is not rewritten unless
legally required.

## Checks

The build and the audit check this policy (audit items S3, X9, C3 in humanstudy-desktop `docs/audit-book.md`):
- no PDF without `pdf.status: included` and a license;
- no human data in git;
- no commercial instrument text;
- every study has a source record.
