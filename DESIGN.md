# NGIS · Nalanda — Design System

One heritage design language, two acts, separated by a session gate:

1. **The Threshold** — a Madhubani/Mithila splash screen (`splash.py`), shown once per browser session.
2. **The Manuscript Console** — the parchment dashboard system (master CSS block in `app.py`, header comment ≈L292). *Chosen 2026-07-17; supersedes the earlier "Midnight Command Console" dark system.*

---

## 1. The Threshold — Madhubani splash

### Concept

A parchment canvas blooms out of darkness and draws itself in Mithila line-art — a double-line border frame, corner spirals, a central sun-mandala, a mirrored fish pair, a lotus, a peacock — then resolves into the bilingual title (Devanagari first: "नालंदा") and a single threshold action: **प्रवेश करें · Enter Console**. Clicking dips to ink-brown and hands off to the console's own entrance choreography.

### Gating

```
session_state["entered"]      absent → render_splash(); st.stop()   (app.py, right after set_page_config)
CTA on_click                  sets entered=True, just_entered=True
next rerun                    dashboard renders; just_entered popped → one-shot .enter-veil dip to ink
hard refresh                  new session → splash again (intended)
```

The gate precedes the master CSS block, so **splash CSS and dashboard CSS never coexist** — every splash override (chrome hiding, parchment background, button styling) is automatically scoped.

### Heritage tokens (scoped on `.stApp` in splash CSS; mirrored as `--sp-*` in the dashboard `:root` so motif SVGs render in both worlds)

| Token | Value | Role |
|---|---|---|
| `--sp-parchment` | #F4E7C8 | canvas |
| `--sp-parchment-2` | #EAD7AE | vignette edge |
| `--sp-ink` | #33210F | outlines, title (≈11:1 on parchment) |
| `--sp-terracotta` | #C3532D | primary accent, fish, inner frame |
| `--sp-madder` | #93312B | deep red, subtitle, CTA hover |
| `--sp-indigo` | #2C4B8F | peacock, hatch lines, strips |
| `--sp-saffron` | #E9A21B | sun fill, rule, CTA glow |
| `--sp-leaf` | #4F7C3A | lotus pad |
| `--sp-e` | cubic-bezier(.16,1,.3,1) | house easing (= console `--e-out`) |

### Storyboard (~4.8 s to resolve; ambient loops continue)

| t (s) | Element | Animation |
|---|---|---|
| 0.0–0.9 | stage | parchment blooms from the dark first frame |
| 0.2–1.8 | double frame | `sp-draw` hand-around-the-page (top → right → bottom → left) |
| 1.0–1.9 | tick strips | scale in from centre |
| 1.3–2.6 | corner spirals | draw, staggered TL→TR→BR→BL |
| 0.95–2.6 | mandala | centre-out draw; fills bloom 2.5–3.2; rotation (150 s) from 3 s |
| 1.8–3.6 | fish pair, lotus, peacock | draw + delayed fills; gentle float loops after |
| 2.8–3.6 | "नालंदा" | fade-up + heritage-ink text sheen loop |
| 3.3–4.5 | subtitle, saffron rule, kicker | staggered fade-up |
| 4.1–4.8 | Enter CTA | fade-up; then sheen sweep + saffron glow pulse |

### Motion contract

- **All motion is pure CSS** — `st.markdown` is DOMPurify-sanitised, `<script>` never executes, `components.html` is iframed/wiped (same contract as the console).
- Every SVG shape carries `pathLength="1"`; one universal rule (`stroke-dasharray:1` → `stroke-dashoffset:1→0`, class `d`) animates every draw. Fills are pre-set with `fill-opacity:0` and revealed by class `f` (never animate `fill` itself). Delays arrive as inline `--d` / `--fd` custom properties.
- `prefers-reduced-motion: reduce` ⇒ the **finished painting, static, CTA visible**. Content is never lost to motion.
- Performance: no **animated** CSS filters (grain is a pre-encoded feTurbulence data-URI tile). One *static* `backdrop-filter:blur(9px)` is sanctioned: the readability blob behind the title stack (`.sp-center::before`), edge-feathered by a mask so both the fill and the blur fall off gradually. Loops are transform/opacity only; zero raster images on the splash run.

### Caveats (do not "fix" these)

- **`div.stButton>button` is styled directly.** Streamlit 1.35 has no `st-key-*` classes. This is safe *only* because splash CSS is injected exclusively while `entered` is False. The wrapper is centred with `left:0;right:0;display:flex;justify-content:center` — NOT `left:50% + translateX`, because the wrapper's shrink-to-fit width is not the button's width in Streamlit's DOM.
- **`.sp-title` carries `padding:.25em .15em; margin:-.25em -.15em`.** With `background-clip:text`, glyph parts outside the element's background box paint nothing — without this headroom the anusvara of नालंदा silently disappears. Latin text never exposes the bug; keep the headroom on any clipped Devanagari.
- **Sanitisation allowlist** (verified against the 1.35 bundle): `<svg> <path> <circle> <g> <line> <rect> <polygon> <defs>` and the `pathlength` attribute survive; **`<use>` does not** — every motif inlines its geometry.
- **Frame edges avoid `vector-effect`**: each edge is its own non-uniformly stretched SVG line; stretching along the line's own axis never distorts its stroke.
- The dark first frame before the splash CSS parses is deliberate ("bloom from dark"), matching `.streamlit/config.toml`'s dark base… which is now light parchment — the first frame is simply parchment now; the bloom keyframe still runs.

## 2. The Manuscript Console

The splash's manuscript IS the app. Parchment canvas, manuscript-ink text, terracotta primary, and the splash's own frame furniture carried into the chrome.

### Decision record

2026-07-17: full parchment-light conversion chosen over (a) keeping the dark Midnight Command Console and (b) a dark-heritage hybrid. The UX4G-violet lineage argument was retired with the dark theme; the primary is now Madhubani terracotta `#C3532D` (matches `.streamlit/config.toml` `primaryColor`).

### Tokens (`:root` in the master CSS block — NAMES are historical, values are heritage)

| Token | Value | Meaning now |
|---|---|---|
| `--ink` / `--ink-2` | #F4E7C8 / #EAD7AE | canvas / deep parchment |
| `--surf` / `--surf-2` | #F9F0DC / #EFE2C2 | raised paper / sidebar surface |
| `--violet` / `--violet-dim` | #C3532D / #93312B | primary terracotta / madder |
| `--cyan` | #2C4B8F | indigo (info) |
| `--saffron` | #E9A21B | saffron — **decorative only** |
| `--green` / `--red` | #4F7C3A / #93312B | leaf success / madder danger |
| `--tx` / `--tx-dim` / `--tx-mute` | #33210F / rgba(51,33,15,.78) / rgba(51,33,15,.58) | manuscript ink |
| `--hair` / `--hair-2` | rgba(51,33,15,.16) / .09 | ink hairlines |

Python chart constants (`NAVY SAFF GREEN RED TEAL MUTED …`, ≈L250) mirror the same values for Plotly; names there are historical too.

### Contrast rules

- Ink on parchment ≈ 11:1 — body text.
- Terracotta: large text and UI surfaces (≈4.4:1). Madder for small accent text.
- **Saffron is never text on parchment** — rules, fills, dots only.
- Focus outlines are terracotta (saffron fails the 3:1 non-text minimum).
- The hero is a **dark photographic plate** (sepia + ink/terracotta duotone wash) — its overlay text is pinned to light parchment values (`#F7EDD6`, `rgba(244,231,200,.85)`), NOT the `--tx` tokens.

### Splash furniture carried into the chrome

- Double-line viewport frame: `.stApp::before/::after` (ink outer, terracotta inner, fixed, pointer-transparent).
- Corner spirals: `corner_svg_static()` from `splash.py`, four fixed `.ms-corner` divs. Splash motif SVGs reference `--sp-*` tokens, which the dashboard `:root` also defines; `.ms-motif-static` neutralises the draw/fill choreography classes (`.d`, `.f`) so motifs arrive pre-drawn.
- Sidebar fish watermark: `fish_svg_static()` under the filters (`.sb-fish`).
- `.tricolor-strip` is now the heritage strip (terracotta/saffron/indigo) — the flag's white band vanishes on parchment.
- Grain: same feTurbulence tile as the splash, `multiply` at .16.

### Nav veil (page-transition loader)

Streamlit reruns are server-side, so a pre-navigation spinner is impossible without JS. The veil rides in ON the incoming page: `st.session_state["_last_page"]` is compared to the option_menu selection; on a page *change* a one-shot `.nav-veil` (parchment field + terracotta sun-ring spinner) holds ~.45 s and fades. It never fires on first arrival (the `.enter-veil` owns that) or on same-page reruns (filters, uploads). This tracker is the single sanctioned session_state addition of the re-theme. Reduced motion ⇒ `display:none`.

### Structural notes

- `.streamlit/config.toml` theme is load-bearing: `secondaryBackgroundColor` (#EFE2C2) must equal the sidebar surface so the `option_menu` iframe blends (injected CSS cannot cross iframe boundaries). Config changes need a server restart.
- The map (`Analytics Suite`) uses an inline Mapbox style (`_parchment_canvas`) — parchment background, zero tile/glyph/sprite requests. Its `sources` block contains a dummy non-empty geojson source (`ngis-blank`): **Streamlit's frontend strips empty objects from the figure spec**, and a missing `sources` fails mapbox-gl style validation as an opaque "Mapbox error.". Do not "simplify" it away.
- Content entrances are time-based; scroll-driven timelines decorate only.
- Reduced-motion guard near the end of the master CSS block; mirror any new animation there.
