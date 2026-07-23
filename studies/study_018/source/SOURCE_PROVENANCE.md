# Source Provenance

## Primary Sources

- Article and supplement: `molleman_et_al_2020_strategies_disparate_social_information.pdf`
- Open article: <https://pmc.ncbi.nlm.nih.gov/articles/PMC7739494/>
- OSF project: <https://osf.io/rmcuy/>
- Original LIONESS software archive: <https://osf.io/download/jgk2q/>
- Original behavioral data: <https://osf.io/download/dasn2/>
- Article DOI: <https://doi.org/10.1098/rspb.2020.2413>
- Supplement collection: <https://doi.org/10.6084/m9.figshare.c.5202231>

The article was published in *Proceedings of the Royal Society B: Biological
Sciences*, volume 287, issue 1939, article 20202413, under CC BY 4.0.
The LIONESS archive does not state a separate software license. This package
therefore records its provenance and compiles only the schedule and lookup
constants required by the runtime rather than redistributing the archive.

The downloaded LIONESS ZIP had SHA-256:

`ea0b6e8908abb8c42cf3d06c0d7f72eeaac4e6ca22f0ce82bea50d4b9f7d9d8e`

The downloaded behavioral CSV had SHA-256:

`ef7757e63f133e53327a64d0ab074b6dcf3df4964a5c7848e83c104b8415f5e3`

The committed article-plus-supplement PDF had SHA-256:

`dfcaa01ffe678dfd614a9c1ee068d194d68b3b6314376e569cb672c07fb5f6bb`

## Compiled Lookup

`materials/peer_lookup.json` is compiled without executing PHP or JavaScript from:

- `stage32741.php`: fixed species, true counts, and condition schedule
- `stage32743.php`: main-task peer estimates indexed by round and first estimate
- `stage32725.php`: four-peer control anchor pools
- `stage32751.php`: four-peer control peer estimates
- `setup_tables.sql`: image size 50 px, overlap ratio 0.6, and border 50 px

The original runtime indexes these tables as
`peer[round - 1][firstEstimate - 1]`. Valid slider responses are 1 through
150, so only the first 150 values of each row are reachable. The compiler
records original row lengths and removes unreachable trailing values.

`materials/build_peer_lookup.py` performs this compilation without executing
the downloaded PHP or JavaScript. It also checks dimensions, estimate ranges,
and the identity of control rows with their corresponding main rounds.

Two control anchor pools contain a zero sentinel. Zero is outside the slider
domain and cannot safely index the published lookup. The runtime rejects those
two values rather than reproducing the known invalid historical row.

## Visual Stimuli

The LIONESS source references external animal sprites that are not included in
the OSF archive and whose historical URLs no longer resolve. The 30 committed
PNGs are deterministic regenerated stimuli, not copies of the original art.
They preserve:

- the exact published species sequence;
- the exact animal count in every round;
- the 50 px image size, 0.6 overlap ratio, 50 px border, and source background;
- the original JavaScript's deterministic placement formula, default
  lexicographic sort, and source period seed (`period=1`);
- no count in the image file name or agent-visible prompt.

Because the original canvas used half of each participant's screen dimensions,
the committed PNGs use a documented 1920 by 1080 reference screen. Runtime
semantics do not depend on that reference viewport.

Run `source/stimuli/generate_stimuli.py` to regenerate them from the compiled
lookup.
