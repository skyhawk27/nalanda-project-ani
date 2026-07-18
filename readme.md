# नालंदा · NGIS — Nalanda Grievance Intelligence System

**A streamlined Grievance Management System for Nalanda district, Bihar.**

🔗 Live demo: [nalandagis.streamlit.app](https://nalandagis.streamlit.app)

NGIS is a Streamlit dashboard that ingests citizen grievances (in Hindi and English), auto-classifies them by department and government scheme, maps them to Nalanda's 20 administrative blocks, and visualizes patterns across the district — all wrapped in a heritage Madhubani/Mithila-art design system.

---

## What it does

NGIS helps officials in Nalanda district handle citizen complaints more easily. People can write in a problem — a broken handpump, a pothole, a stopped pension, a ration card issue — in Hindi, English, or a mix of both. The system reads it, figures out what kind of problem it is, and tells you which department and official should handle it, plus the relevant government scheme. Complaints that sound urgent (accidents, contamination, emergencies, months-long delays) get flagged automatically. It also pulls out details like block, village, name, and date straight from the text, and shows district-wide patterns on an interactive map using real demographic and scheme-coverage data for each block.

The whole app is wrapped in a Madhubani/Mithila folk-art-inspired design — an animated splash screen leads into a parchment-styled dashboard, documented in [`DESIGN.md`](DESIGN.md).

**In short:**
- Reads Hindi/English complaints and sorts them into the right category automatically
- Points to the responsible department, official, and government scheme
- Flags urgent complaints as high priority
- Pulls out block, village, name, and date from the complaint text
- Maps block-level stats — literacy, water coverage, MGNREGA delays — for the whole district
- Wrapped in an animated, heritage-art-inspired design

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
