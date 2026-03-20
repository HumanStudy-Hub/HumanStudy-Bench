## co_website (HumanStudy-Bench web app)

Next.js app for:
- Homepage sections (`Overview`, `Leaderboard`, `Dataset`)
- Study catalog and study detail pages
- Contribute flow (upload + PR creation)
- Docs tab (`/docs`)

## Local preview

```bash
cd co_website
npm install
npm run dev
```

Open `http://localhost:3000`.

## Build checks

```bash
cd co_website
npm run lint
npm run build
```

## Study ZIP download

ZIP downloads are served by:
- `GET /api/studies/:studyId/zip`

This endpoint packages `studies/:studyId` directly from the repository.

## Refreshing effect-size plot data

Homepage effect-size scatter reads:
- `public/data/effects/gemini_flash_v4_effect_data.json`

To refresh:
1. From repo root: `python scripts/plot_single_fig3_effects.py --export-data /absolute/path/to/co_website/public/data/effects/gemini_flash_v4_effect_data.json`
2. Rebuild/redeploy the site.
