# Note for cognithor-site deployment

After v0.93.0 is released, the site repo needs to add a new page
`/integrations` that:

1. Fetches `docs/integrations/catalog.json` at build time from this repo
   (Octokit fetch at build time, analogous to the existing pack fetch).
2. Renders a grid of integration cards, grouped by category.
3. Highlights the `dach_specific: true` entries in a dedicated DACH section.
4. Links each card to `docs/quickstart/02-first-tool.md` as "build your own".

No additional API keys needed — the catalog.json is a public file in main.
