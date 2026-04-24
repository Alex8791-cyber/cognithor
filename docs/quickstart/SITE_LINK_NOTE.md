# Note for cognithor.ai deployment

The `cognithor.ai` site repo should link the Quickstart at `cognithor.ai/quickstart` — rendering the same Markdown from this repo's `docs/quickstart/` via Octokit fetch at build time.

Both DE and EN versions should be available with a language switcher. Suggested URL structure:

- `cognithor.ai/quickstart` → redirects to user's locale
- `cognithor.ai/de/quickstart` → renders `docs/quickstart/README.md`
- `cognithor.ai/en/quickstart` → renders `docs/quickstart/README.en.md`
- `cognithor.ai/de/quickstart/{00..07}-*` → renders individual DE pages
- `cognithor.ai/en/quickstart/{00..07}-*` → renders individual EN pages

The Markdown in this repo uses relative paths (e.g. `](01-first-crew.md)`). The site renderer must rewrite these to absolute site URLs (`/de/quickstart/01-first-crew`) at build time.

Spec §3.5: site deployment is in scope for the Feature 7 site-PR, not here. This note captures the target URL structure for the site maintainer.

Spec §2.4 (acceptance): `cognithor.ai` Startseite verlinkt prominent auf `docs/quickstart/00-installation.md` — this remains the canonical entry point.
