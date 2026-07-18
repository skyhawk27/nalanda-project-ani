# नालंदा · NGIS — Nalanda Grievance Intelligence System

**A streamlined Grievance Management System for Nalanda district, Bihar.**

🔗 Live demo: [nalandagis.streamlit.app](https://nalandagis.streamlit.app)

NGIS is a Streamlit dashboard that ingests citizen grievances (in Hindi and English), auto-classifies them by department and government scheme, maps them to Nalanda's 20 administrative blocks, and visualizes patterns across the district — all wrapped in a heritage Madhubani/Mithila-art design system.

---

## What it does

- **Bilingual complaint classification** (`classifier.py`) — free-text grievances (Hindi, English, or mixed) are run through a hand-tuned keyword-scoring engine (no external NLP/ML dependency): each candidate category has a `{keyword: weight}` dict (weights 1–3), multi-word keyword matches get a 1.5× bonus, and the category with the highest cumulative score wins. Confidence is normalized as `score / (sum of top-3 keyword weights × 1.5)`, capped at 1.0. Ties/no-matches fall back to `Other / Anya`.
- **Scheme & jurisdiction mapping** — each of the 9 substantive categories (Water/Jal, Roads/Sadak, Ration/PDS, Land/Bhumi, Health/Swasthya, Electricity/Bijli, Education/Shiksha, MGNREGA/Rozgar, Pension/Samajik Suraksha) carries its own `department`, an `->`-chained `jurisdiction` escalation path (e.g. Ward Committee → Panchayat Secretary → Junior Engineer), plus lists of applicable `central_schemes` and `bihar_schemes`.
- **Priority detection** — a separate `priority_keywords` list per category (e.g. "गंभीर", "emergency", "दुर्घटना", "months pending") flags a grievance `High` priority via simple substring matching against the lower-cased complaint text.
- **Entity extraction** — regex-based helpers pull structured fields out of raw grievance text:
  - `_extract_block()` — matches against `BLOCK_MAPPING`, a bilingual (Devanagari/Latin) alias table for all 20 Nalanda blocks.
  - `_extract_village()` — Hindi/English patterns (`ग्राम`, `गाँव`, `village:` …).
  - `_extract_name()` — Hindi patterns (`मेरा नाम`, `श्री/श्रीमती`) and English (`I am` / `My name is`).
  - `_extract_date()` — `dd/mm/yyyy`-style and Hindi month-name dates, defaulting to today if none found.
- **Reference/demographic data layer** (`real_data.py`) — static, hand-compiled dictionaries per block (all 20 blocks) covering:
  - `BLOCK_CENSUS` — population, literacy %, sex ratio, child sex ratio, SC %, rural %, households.
  - `JJM_COVERAGE` — Jal Jeevan Mission tap-water coverage/functional % and villages covered vs. total.
  - `MGNREGA_DATA` — average wage-payment delay (days), job cards issued, active workers, pending wages (₹ lakh), scheme completion %.
  - `HILSA_STATS` / `BENCHMARKS` — a detailed profile for the Hilsa block plus literacy/sex-ratio/JJM/MGNREGA benchmark comparisons against Nalanda, Bihar, and national averages.
  - `get_blocks_df()` merges all of the above into a single `pandas.DataFrame`, one row per block, for use in the dashboard's charts and tables.
- **Block coordinates** (`fetch_coords.py`) — a one-off/offline utility script using `geopy`'s `Nominatim` geocoder to resolve each block name (with `", Nalanda, Bihar, India"` appended, `", Nawada, Bihar, India"` for the Warisaliganj special case) to lat/lon, written out to `block_coords.json`. Not run at app startup — `block_coords.json` is committed and read directly by `app.py`.
- **GIS visualization** — an interactive Plotly/Mapbox map plots block-level metrics against `hilsa_boundaries.geojson`, using the pre-fetched `block_coords.json` for point placement.
- **Image handling** (`image_loader.py`) — `get_base64_image()` (Streamlit-cached) inlines local JPEGs from `img/` as base64 data URIs for CSS `background-image`; `hero_css_bg()` prefers a local file (`nalanda_ruins`, `nalanda_monument`) and falls back to a curated set of remote Unsplash/CDN URLs (e.g. `rajgir`, `pawapuri`, `mithila_art`) keyed by name, with a generic default image if the key is unrecognized.
- **Heritage UI** — a two-act experience: a one-time animated Madhubani/Mithila-style splash screen (`splash.py`, pure-CSS/SVG animation, no JS) gated by `session_state["entered"]`, which hands off into a "Manuscript Console" parchment-themed dashboard (master CSS block in `app.py`). Full token values, motion contract, and accessibility/contrast rules are documented in [`DESIGN.md`](DESIGN.md).

## Tech stack

| Layer | Tool | Notes |
|---|---|---|
| App framework | [Streamlit](https://streamlit.io) `1.35.0` | pinned exact version — some CSS hooks in `app.py`/`splash.py` rely on 1.35's DOM (no `st-key-*` classes) |
| Navigation | `streamlit-option-menu` | sidebar/menu widget, rendered in an iframe |
| Data | `pandas 2.2.2`, `numpy 1.26.4` | powers `real_data.py`'s `get_blocks_df()` |
| Charts / GIS | `plotly 5.22.0` (Mapbox), `statsmodels` | analytics dashboard + inline Mapbox GL map |
| Geocoding (offline tooling) | `geopy` | used only by `fetch_coords.py`; **not** in `requirements.txt`, install separately if you need to regenerate `block_coords.json` |
| Classification | Custom keyword-scoring engine (`classifier.py`) | pure Python/`re`, no external NLP/ML dependency |
| Language | Python (100%) | |

## Repository structure

```
nalanda-project/
├── app.py                     # Main Streamlit app — page routing, Mapbox GIS view, analytics, master CSS ("Manuscript Console")
├── splash.py                  # One-time animated Madhubani/Mithila splash screen; owns the "entered" session gate + corner_svg_static()/fish_svg_static() motifs
├── classifier.py               # Bilingual grievance classifier — BLOCK_MAPPING, SCHEMA (department/jurisdiction/schemes/keywords), classify_complaint(), entity extractors
├── real_data.py                 # Static per-block reference data (census, JJM coverage, MGNREGA stats, benchmarks) + get_blocks_df()
├── fetch_coords.py              # Offline script: geopy/Nominatim geocoding of the 20 blocks -> block_coords.json (not run by the app itself)
├── image_loader.py              # base64 image inlining + hero background-image resolver (local file first, remote URL fallback)
├── block_coords.json            # Pre-generated lat/lon for Nalanda's 20 administrative blocks (output of fetch_coords.py)
├── hilsa_boundaries.geojson      # GeoJSON boundaries used for the GIS map
├── img/                          # Local image assets (nalanda_ruins.jpg, nalanda_monument.jpg, etc.)
├── .streamlit/                   # Streamlit theme/config (config.toml — primaryColor, secondaryBackgroundColor must match the CSS tokens, see DESIGN.md)
├── requirements.txt              # Python dependencies (streamlit, streamlit-option-menu, pandas, plotly, numpy, statsmodels)
├── DESIGN.md                     # Full design system documentation (splash + dashboard: tokens, motion contract, contrast rules, caveats)
└── AGENTS.md                     # Notes/instructions for AI coding agents working on this repo
```

## Getting started

### Prerequisites
- Python 3.12

### Installation

```bash
git clone https://github.com/AA3111s/nalanda-project.git
cd nalanda-project
pip install -r requirements.txt
```

### Run locally

```bash
streamlit run app.py
```

The app will open in your browser (default: `http://localhost:8501`). On first load you'll see the Madhubani splash screen — click **प्रवेश करें · Enter Console** to proceed to the dashboard.

### Regenerating block coordinates (optional)

`block_coords.json` is committed to the repo, so this step isn't required to run the app. If you need to rebuild it:

```bash
pip install geopy   # not in requirements.txt — only needed for this script
python fetch_coords.py
```

This geocodes all 20 blocks via OpenStreetMap's Nominatim API (rate-limited to 1 request/sec) and overwrites `block_coords.json`.

### Note on `.streamlit/config.toml`

Per [`DESIGN.md`](DESIGN.md), the theme config is load-bearing, not cosmetic: `secondaryBackgroundColor` must match the sidebar surface color used in `app.py`'s injected CSS, since the `option_menu` widget renders in an iframe that injected CSS can't reach. Changing `config.toml` requires a Streamlit server restart to take effect.

## Grievance categories

The classifier currently routes complaints into the following categories, each mapped to its department, jurisdiction chain, and relevant central/Bihar state schemes:

-  Water / Jal — PHED, Jal Jeevan Mission
-  Roads / Sadak — RWD, PMGSY
-  Ration / PDS — Food & Consumer Protection Dept., PMGKAY / NFSA
-  Land / Bhumi — Revenue & Land Reforms Dept., SVAMITVA
-  Health / Swasthya — Health Dept., Ayushman Bharat / NHM
-  Electricity / Bijli — BSPHCL, RDSS
-  Education / Shiksha — Education Dept., Samagra Shiksha Abhiyan
-  MGNREGA / Rozgar — Rural Development Dept.
-  Pension / Samajik Suraksha — Social Welfare Dept., NSAP
-  Other / Anya — General Administration / SDO Office

## Design system

The visual language — a Madhubani/Mithila-inspired splash screen transitioning into a parchment-toned "Manuscript Console" dashboard — is fully documented in [`DESIGN.md`](DESIGN.md), including color tokens, motion contracts, accessibility/contrast rules, and implementation caveats.

## Contributing

Issues and pull requests are welcome. If you're an AI coding agent, check [`AGENTS.md`](AGENTS.md) for repo-specific conventions before making changes.

## License

No license file is currently published in this repository. Contact the repository owner ([AA3111s](https://github.com/AA3111s)) for reuse permissions.
