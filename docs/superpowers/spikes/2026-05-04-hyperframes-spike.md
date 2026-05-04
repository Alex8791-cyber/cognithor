# Spike — HyperFrames as Cognithor's Default Video Renderer (Option C)

**Date:** 2026-05-04
**Sprint:** 27 — IDE-Integration (parallel HF track)
**Owner:** Alexander Söllner
**Status:** Research output for HF-1 (#127). Decisions land in HF-2..HF-5.

---

## TL;DR

HyperFrames (HeyGen, Apache-2.0, Apr 2026) ships an HTML/CSS/JS-native
video composition + render pipeline designed for AI-agent authoring.
Apache-2.0 makes it license-compatible with Cognithor. Adapter
licenses are mostly safe (MIT) **except GSAP**, which has a free
license that flips to a paid commercial tier when the using product
charges multiple customers — a constraint that affects Cognithor's
**paid Agent Packs**, but NOT the free Apache-2.0 core. Recommendation:
ship HyperFrames as the default renderer behind a thin
`RendererABC` abstraction (Option C from the research note);
default-disable the GSAP adapter for paid-pack render paths and
default-enable Anime.js / CSS / Lottie / WAAPI which are MIT.

## Repo facts (from github.com/heygen-com/hyperframes)

| Attribute | Value |
|---|---|
| License | **Apache-2.0** |
| Node version | **>= 22** |
| Install | `npx hyperframes init my-video` |
| CLI surface | `init`, `preview`, `render`, `lint`, `transcribe`, `tts`, `doctor`, `add` |
| Wire format | HTML with `data-composition-id`, `data-start`, `data-duration`, `data-track-index`, `data-volume`, `data-width`, `data-height` |
| Output | MP4 (Puppeteer + FFmpeg via `@hyperframes/engine`) |

### Modular npm packages

* `@hyperframes/core` — types, parsers, generators, linter, runtime, adapters
* `@hyperframes/engine` — page-to-video capture (Puppeteer + FFmpeg)
* `@hyperframes/producer` — rendering pipeline (capture + encode + audio)
* `@hyperframes/studio` — browser editor UI
* `@hyperframes/player` — embeddable web component
* `@hyperframes/shader-transitions` — WebGL transitions

### Bundled Frame Adapters

| Adapter | Use-case | License | Cognithor verdict |
|---|---|---|---|
| **CSS Animations** | keyframe discovery + seeking | n/a (browser native) | ✅ ship by default |
| **WAAPI** | Web Animations API via `document.getAnimations()` | n/a (browser native) | ✅ ship by default |
| **Anime.js** | timeline animations | **MIT** | ✅ ship by default |
| **Lottie** | `lottie-web` + dotLottie | **MIT** | ✅ ship by default |
| **Three.js** | 3D scene rendering | **MIT** | ✅ ship by default |
| **GSAP** | timeline animations | **GreenSock Standard License** — free for most uses, but a "Business Green" tier ($199-$1999/yr) is required when *the using product collects fees from multiple customers* | ⚠️ disable in paid-pack render paths; OK in free packs / core |

### GSAP license analysis (the only real catch)

Webflow took over GreenSock and made GSAP "100 % free for commercial
use" for the broad case (since 2024). The only remaining trigger for
the paid tier is "if you charge multiple customers a usage / access /
license fee for the app/product/game/site that uses GreenSock tools."

**Cognithor mapping:**

* Cognithor core (Apache-2.0, no charge) — GSAP usage is fine.
* Free packs (Hacker News / Discord / RSS Lead Hunter — Apache-2.0,
  no charge) — fine.
* **Paid packs** (Reddit Lead Hunter Pro €75 indie / €179 commercial,
  Deep Research Analyst, future video-rendering packs) — these
  *do* match the GSAP "Business Green" trigger when the pack's
  rendered output is sold to the pack's customers. Two ways out:
  1. **Default off.** Disable the GSAP adapter in paid-pack renderer
     configs. Anime.js + CSS + WAAPI + Lottie + Three.js cover ~95 %
     of timeline animation needs MIT-clean.
  2. **Cognithor-side license.** Buy a Business Green license at the
     org level if a paid pack genuinely needs GSAP-only features.
     Owner-decision when that need arises.

Default for Sprint-27 HF-2..HF-5: **adapter allowlist excludes
GSAP**. Owner-flag `--enable-gsap` to opt in for free-pack contexts.
Documented in the renderer's `DEFAULT_ALLOWED_ADAPTERS` set.

## How Cognithor will integrate (Option C — abstraction + default impl)

```
src/cognithor/video/
├── __init__.py
├── renderer_base.py         # RendererABC, RenderRequest, RenderResult, RenderError
├── renderers/
│   ├── __init__.py          # registry dict {"hyperframes": HyperFramesRenderer}
│   └── hyperframes.py       # subprocess-spawning npx hyperframes
└── compose/
    ├── __init__.py
    └── templates.py          # HF-5: explainer / social-cut / caption-overlay templates
```

`RendererABC` is the swap-out seam — when (if) Remotion or a
homegrown renderer lands, only this layer changes. The MCP tools
`video_compose` and `video_render` (HF-3) are renderer-agnostic.

### Subprocess contract (HyperFramesRenderer)

The Python side spawns:

```
npx --yes hyperframes render <composition.html> --out <output.mp4>
```

with cwd = a sandboxed temp dir under `~/.cognithor/render/<run_id>/`,
`PATH` inherited (npx must be on PATH — checked by `node --version`
+ `npx --version` at startup), and a 5-minute default timeout. STDIN
closed. STDOUT line-parsed for the progress JSON HyperFrames emits
(documented in the engine package).

### Threat model + Gatekeeper coverage

* **Arbitrary HTML** the Planner produces is treated as RED until
  validated against a whitelist of allowed adapters and DOM verbs.
  TRUST-2 DecisionExplanation populated (`rule_id="render.html.untrusted"`,
  `rule_source="cognithor.video.renderer_base"`).
* **Rendered MP4** lands under `~/.cognithor/render/<run_id>/` only —
  scope filesystem axis is the existing user-doc allowlist + this
  one render dir; never the whole filesystem.
* **Network during render** — Puppeteer must NOT load arbitrary
  external URLs; the Planner-supplied composition must self-contain
  all assets (reference local images / videos in the
  `~/.cognithor/vault/` or `~/.cognithor/render/<run_id>/assets/`).
  Enforced by a strict CSP injected before page load.

## Risks + open questions

1. **Node 22 dependency.** Cognithor target is Python; we currently
   require Node only for the optional Flutter UI build. Adding
   HyperFrames means Node 22+ is a runtime dep for video features.
   Mitigation: render is opt-in (the `video_*` MCP tools fail with a
   clear error if `npx --version` < 22).
2. **Puppeteer / Chrome footprint.** `@hyperframes/engine` pulls
   Puppeteer + a headless Chrome (~250 MB). Plus FFmpeg (~50 MB).
   Mitigation: separate `cognithor[video]` extra so the install is
   opt-in, mirrored in the Windows installer wizard.
3. **License-chain audit.** `@hyperframes/*` themselves are
   Apache-2.0, but their transitive deps include GPL-LGPL libs
   (FFmpeg LGPL, ffprobe LGPL). Cognithor already ships LGPL FFmpeg
   in the Windows installer (per v0.92.7 video-input pipeline) so
   precedent + tooling exist.
4. **Render determinism on different OS / Chrome versions.** HF
   claims same-input-same-output but only if the headless Chrome
   version is pinned. Mitigation: pin via `npm ci` + commit
   lockfile (same as the extension scaffold).

## Decision

**Go Option C.** Implement HF-2 ..HF-5 against the abstraction.
Default adapter allowlist: CSS, WAAPI, Anime.js, Lottie, Three.js
(MIT-clean). GSAP gated behind opt-in `--enable-gsap`. Threat model
above is the floor for Gatekeeper rules in HF-3.

## Next step

HF-2 (#128) — implement `cognithor.video` package with
`RendererABC` + `HyperFramesRenderer`. Land before HF-3 (MCP tools).

## Sources

- [HyperFrames repo — github.com/heygen-com/hyperframes](https://github.com/heygen-com/hyperframes)
- [GSAP licensing — gsap.com/licensing/](https://gsap.com/licensing/)
- [GreenSock "Why" license blog](https://gsap.com/blog/why-license/)
- HyperFrames research note (Tomi, 2026-05-04, `~/Downloads/hyperframes-research-note.md`)
