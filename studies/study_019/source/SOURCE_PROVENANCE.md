# Source Provenance

## Primary Sources

- Article DOI: <https://doi.org/10.1371/journal.pone.0146536>
- PLOS JATS manuscript:
  <https://journals.plos.org/plosone/article/file?id=10.1371/journal.pone.0146536&type=manuscript>
- Supporting Bayesian text:
  <https://doi.org/10.1371/journal.pone.0146536.s001>
- Figshare dataset:
  <https://doi.org/10.6084/m9.figshare.1597662.v1>

The PLOS article and Figshare dataset are published under CC BY 4.0.

## Figshare Files

- `Raw Data_Study1.xlsx`, Figshare file 3559196, published MD5
  `e6f623799cce4b8331b806b55563b31c`
- `RawData_Study2.xlsx`, Figshare file 3559199, published MD5
  `c29df321e159c9cd0afbfa3defa73159`
- `Variables_Coding_Raw_DATA_Study 1.pdf`, Figshare file 3657090,
  published MD5 `e81db199e990bfe86a6bd421041aad87`
- `Variables_Coding_RawData_Study2.pdf`, Figshare file 3657087,
  published MD5 `f04a7d802b1bce27355ba8b808e7916a`

`materials/build_scenarios.py` verifies the two workbook MD5 values before
compiling `materials/scenarios.json`.

## Scenario Identity

The Study 1 workbook columns follow the variable-coding PDF, not the article's
table order. Variables 1-12 are the base scenarios; variables 13-24 are their
mirrored presentations. The raw workbook recodes the mirrored choices to the
base orientation. The compiler records both raw coding and the actual visible
Urn A/Urn B orientation.

The Study 2 workbook columns similarly follow the variable-coding PDF. The
compiler stores both each raw variable ID and the corresponding article-table
scenario ID. Medical roles, diagnoses, private symptoms, and posterior groups
are transcribed from the public coding document and checked against the PLOS
JATS tables.

## Missing Values

The public workbooks encode missing values as `9`. Choice and confidence
missingness are independent in five source cells. The compiler therefore keeps
separate `n_choice`, `n_confidence`, and `n_paired` counts. Choice rates use all
valid choices, confidence means use all valid confidence judgments, and
choice-conditioned probability judgments require a valid pair. This recovers
the reported `838/1119` medical-director alignment count without imputing data.

The article reports nominal cascade denominators (`120/160` for Study 1 and
`230/280` for Study 2), while the workbooks contain 159 and 279 valid cascade
choices respectively. The valid raw-data rates are therefore `120/159` and
`230/279`. The package preserves the article values as reported ground truth
and the independently counted workbook denominators in compiled material
statistics.

## Wording Limitation

The public archive does not include the original questionnaire or German
participant text. Runtime instructions are a semantic reconstruction from the
published Procedure sections. No missing scenario content is inferred, but
the package does not claim verbatim linguistic fidelity.
