import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime, timedelta
from PIL import Image
import json
import io
import time
import warnings
import html as _html  # ← for safe escaping of dynamic values into HTML

# ══════════════════════════════════════════════════════════════════════
# HEIC SUPPORT
# ══════════════════════════════════════════════════════════════════════
_HEIC_OK  = False
_HEIC_ERR = ""
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
    _HEIC_OK = True
except ImportError:
    _HEIC_ERR = "pillow-heif not installed.\nFix: pip install pillow-heif → restart Streamlit."
except Exception as _e:
    _HEIC_ERR = f"pillow-heif registration failed: {_e}"

# ── ctypes libheif fallback ────────────────────────────────────────
_LIBHEIF_OK = False
try:
    import ctypes, ctypes.util
    _libheif = ctypes.CDLL("libheif.so.1")
    _libheif.heif_context_alloc.restype            = ctypes.c_void_p
    _libheif.heif_context_get_primary_image_handle.restype = ctypes.c_int
    _libheif.heif_decode_image.restype             = ctypes.c_int
    _libheif.heif_image_get_plane_readonly.restype = ctypes.c_void_p
    _libheif.heif_image_handle_get_width.restype   = ctypes.c_int
    _libheif.heif_image_handle_get_height.restype  = ctypes.c_int
    _LIBHEIF_OK = True
except Exception:
    pass

def _open_heic_ctypes(raw_bytes: bytes):
    import ctypes, tempfile, os
    tmp = tempfile.NamedTemporaryFile(suffix=".heic", delete=False)
    tmp.write(raw_bytes); tmp.close()
    try:
        ctx    = ctypes.c_void_p(_libheif.heif_context_alloc())
        _libheif.heif_context_read_from_file(ctx, tmp.name.encode(), None)
        handle = ctypes.c_void_p()
        _libheif.heif_context_get_primary_image_handle(ctx, ctypes.byref(handle))
        w      = _libheif.heif_image_handle_get_width(handle)
        h      = _libheif.heif_image_handle_get_height(handle)
        img_ptr= ctypes.c_void_p()
        _libheif.heif_decode_image(handle, ctypes.byref(img_ptr), 1, 10, None)
        stride = ctypes.c_int(0)
        plane  = _libheif.heif_image_get_plane_readonly(img_ptr, 10, ctypes.byref(stride))
        if not plane or stride.value == 0:
            raise RuntimeError("libheif plane decode returned null")
        raw = (ctypes.c_uint8 * (stride.value * h)).from_address(plane)
        pil  = Image.frombuffer("RGB", (w, h), bytes(raw), "raw", "RGB", stride.value, 1)
        _libheif.heif_context_free(ctx)
        return pil.copy()
    finally:
        os.unlink(tmp.name)


def open_image_safe(raw_bytes: bytes, filename: str = "") -> tuple:
    fname   = filename.lower()
    is_heic = fname.endswith((".heic", ".heif"))
    if is_heic and _HEIC_OK:
        try:
            return Image.open(io.BytesIO(raw_bytes)).convert("RGB"), None
        except Exception:
            pass
    if is_heic and _LIBHEIF_OK:
        try:
            return _open_heic_ctypes(raw_bytes), None
        except Exception as e:
            return None, f"libheif decode failed: {e}"
    if is_heic:
        return None, _HEIC_ERR or "No HEIC decoder available."
    try:
        img = Image.open(io.BytesIO(raw_bytes))
        return img.convert("RGB"), None
    except Exception as exc:
        return None, str(exc)


# ══════════════════════════════════════════════════════════════════════
# LOCAL IMPORTS
# ══════════════════════════════════════════════════════════════════════
from real_data import get_blocks_df, HILSA_STATS, BLOCK_CENSUS, JJM_COVERAGE, MGNREGA_DATA
from classifier import classify_complaint, SCHEMA
from image_loader import hero_css_bg

@st.cache_data
def load_hilsa_boundaries():
    try:
        with open("hilsa_boundaries.geojson", "r") as f:
            return json.load(f)
    except:
        return {}

# ══════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="NGIS · Nalanda",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════
# GEMINI
# ══════════════════════════════════════════════════════════════════════
_GEMINI_CHAIN = ["gemini-2.5-flash","gemini-2.0-flash","gemini-2.0-flash-lite","gemini-1.5-pro"]
_RETRY_ERRORS = ("404","not found","deprecated","unavailable","429","quota","exceeded","resource_exhausted")
_SKIP_ERRORS  = ("permission_denied","invalid_api_key","api_key_invalid","auth","invalid argument")

def _is_retryable(err_str):
    s = err_str.lower()
    if any(x in s for x in _SKIP_ERRORS): return False
    return any(x in s for x in _RETRY_ERRORS)

def _is_quota(err_str):
    s = err_str.lower()
    return any(x in s for x in ("429","quota","exceeded","resource_exhausted"))

def init_gemini(api_key):
    if not api_key or api_key.strip() == "": return None, None
    key = api_key.strip()
    try:
        from google import genai as _gai
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            client = _gai.Client(api_key=key)
        active = _GEMINI_CHAIN[1]
        try:
            available = [m.name.split("/")[-1] for m in client.models.list()
                         if "generateContent" in (m.supported_actions or [])]
            for pref in _GEMINI_CHAIN:
                if pref in available: active = pref; break
        except Exception: pass
        st.session_state["_gemini_active"] = active
        return client, "new"
    except ImportError: pass
    try:
        import google.generativeai as _old
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _old.configure(api_key=key)
        st.session_state["_gemini_active"] = _GEMINI_CHAIN[1]
        return _old, "legacy"
    except Exception: return None, None

def run_gemini_ocr(client_tuple, image_bytes):
    client, sdk_type = (client_tuple if isinstance(client_tuple, tuple) else (client_tuple, "legacy"))
    active_model = st.session_state.get("_gemini_active", _GEMINI_CHAIN[1])
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=90)
        jpeg_bytes = buf.getvalue()
    except Exception as e: return {"error": f"Image preprocessing failed: {e}"}

    prompt = """You are an OCR assistant for Bihar government district administration.
Read this handwritten Hindi grievance letter EXACTLY as written.
STRICT RULES:
1. Transcribe ONLY what is actually written. Do NOT infer or add information.
2. If a word is unclear write [अस्पष्ट] rather than guessing.
3. Preserve original Hindi text exactly.
Return ONLY a valid JSON object — no markdown, no backticks:
{
  "transcription": "Complete Hindi text exactly as written",
  "complainant_name": "Name of person who signed (or Unknown)",
  "village": "Village/Gram name mentioned (or Unknown)",
  "block": "Block/Thana/Panchayat mentioned (or Unknown)",
  "date_filed": "Date written in letter (or Unknown)",
  "issue_summary": "One sentence summary of core problem in English"
}"""

    models_to_try = [active_model] + [m for m in _GEMINI_CHAIN if m != active_model]
    _last_err = None; quota_hit = False

    if sdk_type == "new":
        try:
            from google.genai import types as _types
        except ImportError:
            from google import genai as _gai_mod; _types = _gai_mod.types
        image_part = _types.Part.from_bytes(data=jpeg_bytes, mime_type="image/jpeg")
        for model_name in models_to_try:
            try:
                response = client.models.generate_content(model=model_name, contents=[prompt, image_part])
                raw = response.text.strip().replace("```json","").replace("```","").strip()
                try: return json.loads(raw)
                except json.JSONDecodeError:
                    return {"transcription":response.text,"complainant_name":"Unknown","village":"Unknown",
                            "block":"Unknown","date_filed":datetime.now().strftime("%Y-%m-%d"),
                            "issue_summary":"Could not parse JSON"}
            except Exception as _e:
                _s = str(_e); _last_err = _s
                if _is_quota(_s): quota_hit = True; time.sleep(5); continue
                if _is_retryable(_s): continue
                return {"error": f"Fatal Gemini error: {_s}"}
        if quota_hit:
            return {"error": "🚫 Gemini free-tier quota exhausted.\n\nFix: Generate a new API key at https://aistudio.google.com/apikey\n\nLast error: " + str(_last_err)}
        return {"error": f"All models exhausted. Last error: {_last_err}"}

    import google.generativeai as _old_genai
    for model_name in models_to_try:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = _old_genai.GenerativeModel(model_name)
                response = m.generate_content([prompt, img])
            raw = response.text.strip().replace("```json","").replace("```","").strip()
            try: return json.loads(raw)
            except json.JSONDecodeError:
                return {"transcription":response.text,"complainant_name":"Unknown","village":"Unknown",
                        "block":"Unknown","date_filed":datetime.now().strftime("%Y-%m-%d"),
                        "issue_summary":"Could not parse JSON"}
        except Exception as _e:
            _s = str(_e); _last_err = _s
            if _is_quota(_s): quota_hit = True; time.sleep(5); continue
            if _is_retryable(_s): continue
            return {"error": f"Fatal Gemini error: {_s}"}
    if quota_hit:
        return {"error": "🚫 Quota exhausted.\n\nGet new key: https://aistudio.google.com/apikey\n\nLast error: " + str(_last_err)}
    return {"error": f"All models exhausted. Last: {_last_err}"}


# ══════════════════════════════════════════════════════════════════════
# THEME
# ══════════════════════════════════════════════════════════════════════
# Midnight Command Console — dark console palette.
# Hues are carried over from UX4G rather than discarded: the primary is the
# UX4G violet #613AF5 lifted to #7C5CFF so it clears 4.5:1 on a #0A0B14
# surface. Names are unchanged so every existing call site re-themes for free.
NAVY   = "#7C5CFF"  # Primary violet (UX4G #613AF5, lifted for dark surfaces)
SAFF   = "#F5A524"  # Heritage saffron
GREEN  = "#34D399"  # Success
RED    = "#FB5E5E"  # Danger
AMBER  = "#F5A524"  # Warning
TEAL   = "#22D3EE"  # Info cyan
MUTED  = "#8A90A8"  # Muted slate
LGRID  = "#232741"  # Hairline grid
LBG    = "#11142290" # Translucent surface
WHITE  = "#E7E9F2"  # Primary text on dark
TXT    = "#E7E9F2"  # Body text on dark

# Dark canvas tokens
INK    = "#07080F"  # Deepest background
SURF   = "#0F1220"  # Raised surface
GLASS  = "rgba(255,255,255,.045)"
HAIR   = "rgba(255,255,255,.09)"

def ct(fig, title="", h=None):
    kw = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, family="'Plus Jakarta Sans', sans-serif", size=11),
        title=dict(text=title, font=dict(family="'Instrument Serif', serif", size=17, color=WHITE),
                   x=0, xanchor="left") if title else {},
        xaxis=dict(gridcolor=LGRID, linecolor=LGRID, tickcolor=MUTED, tickfont=dict(size=10),
                   zerolinecolor=LGRID),
        yaxis=dict(gridcolor=LGRID, linecolor=LGRID, tickcolor=MUTED, tickfont=dict(size=10),
                   zerolinecolor=LGRID),
        margin=dict(t=46 if title else 14, b=14, l=8, r=8),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
                    font=dict(color=MUTED, size=11), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor="#161A2E", bordercolor=NAVY,
                        font=dict(color=WHITE, family="'Fira Code', monospace")),
        colorway=[NAVY, TEAL, SAFF, GREEN, RED, "#C4B5FD"],
    )
    if h: kw["height"] = h
    fig.update_layout(**kw)
    return fig


# ══════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — "Midnight Command Console"
# Presentation only. No markup below carries data or logic.
#
# Direction: Modern Dark (Cinema) + Aurora UI + Glassmorphism, carrying the
# Nalanda heritage masthead and Devanagari-first display type.
# The UX4G violet is retained as a hue (#613AF5 → #7C5CFF, lifted to clear
# 4.5:1 on #07080F) rather than discarded, so the government lineage reads.
#
# Streamlit constraints this works within:
#   · st.markdown is sanitised by DOMPurify — <script> never executes.
#   · components.html is iframed and wiped on every rerun.
# So all motion is CSS. Content entrances are TIME-based (guaranteed to
# finish visible); scroll-driven timelines are used ONLY for decoration,
# where non-support degrades to "no motion" and never to "no content".
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital,wght@0,400;1,400&family=Noto+Serif+Devanagari:wght@400;500;600;700&family=Noto+Sans+Devanagari:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@200;300;400;500;600;700;800&family=Fira+Code:wght@400;500;600&display=swap');

/* ── Registered properties: enable true numeric interpolation ──
   Fallback contract: every element also declares --kpi-n inline, so if
   @property is unsupported the counter still resolves to the real value. */
@property --kpi-n   { syntax:'<integer>'; initial-value:0; inherits:false }
@property --ring-a  { syntax:'<angle>';   initial-value:0deg; inherits:false }
@property --glow    { syntax:'<number>';  initial-value:0; inherits:false }

:root{
  --ink:#07080F; --ink-2:#0A0B14; --surf:#0F1220; --surf-2:#141834;
  --violet:#7C5CFF; --violet-dim:#5B3FD1; --cyan:#22D3EE;
  --saffron:#F5A524; --green:#34D399; --red:#FB5E5E;
  --tx:#E7E9F2; --tx-dim:#A7AECB; --tx-mute:#8A90A8;
  --hair:rgba(255,255,255,.09); --hair-2:rgba(255,255,255,.055);
  --glass:rgba(255,255,255,.045); --glass-2:rgba(255,255,255,.028);

  --e-out:cubic-bezier(.16,1,.3,1);
  --e-back:cubic-bezier(.34,1.56,.64,1);
  --e-soft:cubic-bezier(.4,0,.2,1);
  --d-fast:150ms; --d:280ms; --d-slow:520ms;

  --r:14px; --r-sm:8px; --r-lg:20px;
}

*,*::before,*::after{box-sizing:border-box}
footer{visibility:hidden}header{visibility:hidden}.stDeployButton{display:none}
#MainMenu{visibility:hidden}

/* ══ CANVAS ══════════════════════════════════════════════════════════ */
.stApp{background:var(--ink)!important;color:var(--tx)}

/* Aurora field — drifts continuously behind every surface */
[data-testid="stAppViewContainer"]::before{
  content:"";position:fixed;inset:-25%;z-index:0;pointer-events:none;
  background:
    radial-gradient(38% 38% at 18% 22%, rgba(124,92,255,.34), transparent 68%),
    radial-gradient(32% 32% at 82% 12%, rgba(34,211,238,.22), transparent 68%),
    radial-gradient(42% 42% at 68% 82%, rgba(245,165,36,.15), transparent 68%),
    radial-gradient(30% 30% at 12% 88%, rgba(52,211,153,.13), transparent 68%);
  filter:blur(70px) saturate(1.25);
  animation:aurora 26s var(--e-soft) infinite alternate;
}
/* Film grain — kills gradient banding, adds a photographic surface */
[data-testid="stAppViewContainer"]::after{
  content:"";position:fixed;inset:0;z-index:0;pointer-events:none;opacity:.22;
  mix-blend-mode:overlay;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='.85' numOctaves='3' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}
.main .block-container{
  padding:0 46px 90px!important;max-width:1560px!important;
  position:relative;z-index:1;
}

/* ══ TYPOGRAPHY ══════════════════════════════════════════════════════ */
html,body,[class*="css"]{font-family:'Plus Jakarta Sans',sans-serif}
h1,h2,h3,h4{color:var(--tx)!important;font-weight:600!important}
.stMarkdown p{color:var(--tx-dim)}
.dv{font-family:'Noto Serif Devanagari','Noto Sans Devanagari',serif}

/* ══ MASTHEAD ════════════════════════════════════════════════════════ */
.ngis-mast{
  display:flex;align-items:flex-end;justify-content:space-between;gap:24px;
  padding:26px 2px 18px;border-bottom:1px solid var(--hair);
  margin-bottom:26px;position:relative;
  animation:fade-down 700ms var(--e-out) both;
}
.ngis-mast::after{
  content:"";position:absolute;left:0;bottom:-1px;height:1px;width:100%;
  background:linear-gradient(90deg,var(--violet),var(--cyan),transparent 72%);
  transform:scaleX(0);transform-origin:left;
  animation:draw-x 1100ms var(--e-out) 260ms both;
}
.mast-l{display:flex;align-items:center;gap:16px}
.mast-seal{
  width:46px;height:46px;border-radius:50%;flex-shrink:0;
  display:grid;place-items:center;font-size:21px;
  background:conic-gradient(from var(--ring-a),var(--violet),var(--cyan),var(--saffron),var(--violet));
  animation:ring-spin 9s linear infinite;
  position:relative;
}
.mast-seal::before{
  content:"";position:absolute;inset:2px;border-radius:50%;background:var(--ink-2);
}
.mast-seal span{position:relative;z-index:1}
.mast-hi{
  font-family:'Noto Serif Devanagari',serif;font-size:30px;font-weight:600;
  letter-spacing:-.5px;
  background:linear-gradient(92deg,#FFFFFF 8%,var(--violet) 52%,var(--cyan) 96%);
  -webkit-background-clip:text;background-clip:text;color:transparent;
  background-size:220% 100%;
  animation:sheen-text 7s var(--e-soft) infinite;
  /* Devanagari matras (anusvara, ि, े …) sit ABOVE the em box. With
     background-clip:text + color:transparent, anything outside the element's
     background box paints nothing — line-height:1 silently ate the bindu in
     नालंदा. Latin never exposes this; the line box must clear the matras. */
  line-height:1.5;
  padding-top:.06em;
}
.mast-en{
  font-size:9.5px;letter-spacing:5.5px;text-transform:uppercase;
  color:var(--tx-mute);margin-top:6px;font-weight:600;
}
.mast-r{text-align:right;font-family:'Fira Code',monospace;font-size:10px;
  color:var(--tx-mute);line-height:1.9;letter-spacing:.4px}
.mast-r b{color:var(--saffron);font-weight:500}
.mast-live{display:inline-flex;align-items:center;gap:6px;color:var(--green)}
.mast-live i{
  width:6px;height:6px;border-radius:50%;background:var(--green);
  box-shadow:0 0 0 0 rgba(52,211,153,.7);animation:beat 2s var(--e-soft) infinite;
}

/* ══ TRICOLOUR ═══════════════════════════════════════════════════════ */
.tricolor-strip{
  height:2px;width:100%;position:relative;overflow:hidden;border-radius:2px;
  background:linear-gradient(90deg,#FF9933 0 33%,#F3F3F3 33% 66%,#138808 66% 100%);
  opacity:.85;transform-origin:left;
  animation:draw-x 800ms var(--e-out) both;
}
.tricolor-strip::after{
  content:"";position:absolute;inset:0;width:34%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.9),transparent);
  animation:sheen 3.6s var(--e-soft) 900ms infinite;
}

/* ══ HERO ════════════════════════════════════════════════════════════ */
.ngis-hero{
  position:relative;width:100%;height:390px;overflow:hidden;
  border-radius:var(--r-lg);margin:18px 0 30px;
  border:1px solid var(--hair);
  box-shadow:0 30px 80px -20px rgba(0,0,0,.85), 0 0 0 1px rgba(255,255,255,.03) inset;
  animation:hero-in 900ms var(--e-out) both;
  isolation:isolate;
}
.ngis-hero-bg{
  width:100%;height:100%;background-size:cover;background-position:center;
  filter:saturate(.55) contrast(1.08) brightness(.62);
  transform-origin:center;
  animation:ken 30s var(--e-soft) infinite alternate;
  will-change:transform;
}
/* Duotone wash — unifies four unrelated photographs into one palette */
.ngis-hero::before{
  content:"";position:absolute;inset:0;z-index:1;pointer-events:none;
  background:
    linear-gradient(105deg, rgba(7,8,15,.97) 0%, rgba(10,11,20,.82) 42%, rgba(124,92,255,.30) 78%, rgba(34,211,238,.26) 100%);
  mix-blend-mode:multiply;
}
.ngis-hero::after{
  content:"";position:absolute;inset:0;z-index:2;pointer-events:none;
  background:
    radial-gradient(90% 130% at 0% 100%, rgba(124,92,255,.24), transparent 62%),
    linear-gradient(0deg, rgba(7,8,15,.9) 0%, transparent 55%);
}
.ngis-hero-over{
  position:absolute;inset:0;z-index:3;display:flex;align-items:flex-end;
  padding:44px 48px;
}
.hero-kicker{
  display:inline-flex;align-items:center;gap:9px;margin-bottom:16px;
  padding:6px 13px;border-radius:100px;
  border:1px solid rgba(255,255,255,.16);
  background:rgba(255,255,255,.06);
  backdrop-filter:blur(14px);-webkit-backdrop-filter:blur(14px);
  font-family:'Fira Code',monospace;font-size:9.5px;letter-spacing:2.4px;
  text-transform:uppercase;color:var(--tx-dim);
  animation:fade-up 700ms var(--e-out) 180ms both;
}
.hero-kicker i{width:5px;height:5px;border-radius:50%;background:var(--cyan);
  box-shadow:0 0 10px var(--cyan);animation:beat 2.4s var(--e-soft) infinite}
.ngis-hero h1{
  font-family:'Instrument Serif',serif!important;
  font-size:clamp(2.6rem,5vw,4.4rem)!important;font-weight:400!important;
  line-height:.98!important;letter-spacing:-1.6px!important;margin:0 0 6px!important;
  color:#fff!important;
  animation:fade-up 800ms var(--e-out) 300ms both;
}
.ngis-hero .hero-hi{
  display:block;font-family:'Noto Serif Devanagari',serif;
  font-size:clamp(1.1rem,1.7vw,1.55rem);font-weight:500;
  letter-spacing:0;margin-bottom:8px;
  background:linear-gradient(92deg,var(--saffron),#FFD59B 60%,var(--saffron));
  -webkit-background-clip:text;background-clip:text;color:transparent;
  animation:fade-up 800ms var(--e-out) 220ms both;
  line-height:1.55;   /* clears Devanagari matras — see .mast-hi */
}
.ngis-hero p{
  color:var(--tx-dim)!important;font-size:13px!important;margin:10px 0 0!important;
  letter-spacing:.5px;max-width:62ch;line-height:1.7;
  animation:fade-up 800ms var(--e-out) 420ms both;
}
.hero-rule{
  height:1px;width:0;margin:16px 0 0;
  background:linear-gradient(90deg,var(--violet),var(--cyan),transparent);
  animation:rule-grow 1200ms var(--e-out) 520ms both;
}
/* Decorative parallax only — if scroll timelines are unsupported this is
   simply absent; no content depends on it. */
@supports (animation-timeline: view()){
  .ngis-hero-bg{
    animation:ken 30s var(--e-soft) infinite alternate, hero-drift linear both;
    animation-timeline:auto, view();
    animation-range:normal, entry 0% exit 100%;
  }
}

/* ══ GLASS PRIMITIVE ═════════════════════════════════════════════════ */
.brief-card,.kpi-card,.reg-wrap,.scheme-entry,.reg-ticket,.processing-note{
  background:linear-gradient(180deg,var(--glass),var(--glass-2))!important;
  backdrop-filter:blur(18px) saturate(1.3);-webkit-backdrop-filter:blur(18px) saturate(1.3);
  border:1px solid var(--hair)!important;
}

.brief-card{
  border-radius:var(--r)!important;border-left:2px solid var(--violet)!important;
  padding:16px 20px!important;color:var(--tx-dim)!important;font-size:13px!important;
  line-height:1.8!important;margin-bottom:24px!important;
  animation:fade-up var(--d-slow) var(--e-out) 120ms both;
  transition:transform var(--d) var(--e-soft),box-shadow var(--d) var(--e-soft);
}
.brief-card:hover{transform:translateY(-2px);box-shadow:0 18px 44px -18px rgba(124,92,255,.6)}
.brief-card strong{color:var(--tx)!important}

/* ══ KPI ═════════════════════════════════════════════════════════════ */
.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:30px}
.kpi-card{
  position:relative;overflow:hidden;border-radius:var(--r)!important;
  padding:20px 20px 18px!important;
  border-left:1px solid var(--hair)!important;
  animation:pop-in 620ms var(--e-back) both;
  transition:transform var(--d) var(--e-soft),box-shadow var(--d) var(--e-soft),
             border-color var(--d) var(--e-soft);
}
.kpi-row .kpi-card:nth-child(1){animation-delay:180ms}
.kpi-row .kpi-card:nth-child(2){animation-delay:250ms}
.kpi-row .kpi-card:nth-child(3){animation-delay:320ms}
.kpi-row .kpi-card:nth-child(4){animation-delay:390ms}
/* Accent seam */
.kpi-card::before{
  content:"";position:absolute;left:0;top:0;bottom:0;width:2px;
  background:linear-gradient(180deg,var(--violet),transparent);
  transform:scaleY(0);transform-origin:top;
  animation:draw-y 800ms var(--e-out) 520ms both;
}
.kpi-card.k-red::before{background:linear-gradient(180deg,var(--red),transparent)}
.kpi-card.k-green::before{background:linear-gradient(180deg,var(--green),transparent)}
.kpi-card.k-amb::before{background:linear-gradient(180deg,var(--saffron),transparent)}
/* Cursor-tracked bloom is impossible without JS; a static top bloom reads
   the same at rest and costs nothing. */
.kpi-card::after{
  content:"";position:absolute;inset:0;pointer-events:none;opacity:0;
  background:radial-gradient(60% 90% at 50% 0%, rgba(124,92,255,.30), transparent 70%);
  transition:opacity var(--d-slow) var(--e-soft);
}
.kpi-card:hover::after{opacity:1}
.kpi-card.k-red:hover::after{background:radial-gradient(60% 90% at 50% 0%, rgba(251,94,94,.28), transparent 70%)}
.kpi-card.k-green:hover::after{background:radial-gradient(60% 90% at 50% 0%, rgba(52,211,153,.26), transparent 70%)}
.kpi-card.k-amb:hover::after{background:radial-gradient(60% 90% at 50% 0%, rgba(245,165,36,.26), transparent 70%)}
.kpi-card:hover{transform:translateY(-6px);border-color:rgba(255,255,255,.2)!important;
  box-shadow:0 28px 60px -22px rgba(0,0,0,.9)}
.kpi-label{
  font-size:9px!important;text-transform:uppercase!important;letter-spacing:2.2px!important;
  color:var(--tx-mute)!important;font-weight:700!important;margin-bottom:14px!important;
  position:relative;z-index:1;
}
.kpi-label .dv,.kpi-label span{font-family:'Noto Sans Devanagari',sans-serif}
/* Count-up. The element also carries --kpi-n inline, so the counter resolves
   to the true value even where @property is unsupported — it can never
   render a wrong number, only a non-animated one. */
.kpi-value{
  position:relative;z-index:1;
  font-family:'Instrument Serif',serif!important;
  font-size:46px!important;font-weight:400!important;line-height:1!important;
  letter-spacing:-1.5px;margin-bottom:8px!important;
  color:transparent!important;
  counter-reset:kpi var(--kpi-n);
  animation:fade-up 600ms var(--e-out) 420ms both;
}
.kpi-value::after{
  content:counter(kpi);position:absolute;left:0;top:0;
  background:linear-gradient(180deg,#FFFFFF,var(--violet));
  -webkit-background-clip:text;background-clip:text;color:transparent;
}
.kpi-value.pct::after{content:counter(kpi) "%"}
.k-red .kpi-value::after{background:linear-gradient(180deg,#FFFFFF,var(--red));
  -webkit-background-clip:text;background-clip:text}
.k-green .kpi-value::after{background:linear-gradient(180deg,#FFFFFF,var(--green));
  -webkit-background-clip:text;background-clip:text}
.k-amb .kpi-value::after{background:linear-gradient(180deg,#FFFFFF,var(--saffron));
  -webkit-background-clip:text;background-clip:text}
.kpi-delta{font-size:10.5px!important;font-family:'Fira Code',monospace!important;
  color:var(--tx-mute)!important;position:relative;z-index:1}
.kpi-delta.up{color:var(--green)!important}
.kpi-delta.down{color:var(--red)!important}

/* ══ SECTION LABEL ═══════════════════════════════════════════════════ */
.sec-label{
  position:relative;border-bottom:none!important;
  font-size:10px!important;letter-spacing:2.6px!important;text-transform:uppercase;
  color:var(--tx-dim)!important;font-weight:700!important;
  padding-bottom:12px!important;margin:34px 0 18px!important;
  display:flex;align-items:center;gap:8px;
  font-family:'Plus Jakarta Sans',sans-serif!important;
  animation:fade-in var(--d) var(--e-soft) both;
}
.sec-label::before{
  content:"";width:5px;height:5px;border-radius:50%;background:var(--violet);
  box-shadow:0 0 10px var(--violet);flex-shrink:0;
  animation:beat 2.6s var(--e-soft) infinite;
}
.sec-label::after{
  content:"";position:absolute;left:0;bottom:0;width:100%;height:1px;
  background:linear-gradient(90deg,var(--violet) 0%,rgba(124,92,255,.28) 34%,transparent 78%);
  transform:scaleX(0);transform-origin:left;
  animation:draw-x 900ms var(--e-out) 140ms both;
}
.ngis-hr{height:1px;background:var(--hair-2);margin:30px 0;border:none}
.ngis-body{padding:22px 0 0}

/* ══ REGISTER TABLE ══════════════════════════════════════════════════ */
.reg-wrap{
  overflow-x:auto;border-radius:var(--r)!important;
  animation:fade-up var(--d-slow) var(--e-out) 160ms both;
}
.reg-table{width:100%;border-collapse:collapse;font-size:12.5px;
  font-family:'Plus Jakarta Sans',sans-serif}
.reg-table thead tr{background:rgba(255,255,255,.035)!important;border:none}
.reg-table thead th{
  padding:14px 16px!important;text-align:left;border:none!important;
  border-bottom:1px solid var(--hair)!important;
  color:var(--tx-mute)!important;font-size:9px!important;font-weight:700!important;
  letter-spacing:1.4px;text-transform:uppercase;white-space:nowrap;
}
.reg-table thead th .hi{display:block;font-family:'Noto Sans Devanagari',sans-serif;
  font-size:10.5px;font-weight:600;color:var(--tx-dim);letter-spacing:.3px;text-transform:none}
.reg-table thead th .en{display:block;font-size:8px;opacity:.6;margin-top:2px}
.reg-table tbody tr{
  border-bottom:1px solid var(--hair-2);
  animation:row-in 340ms var(--e-out) both;
}
.reg-table tbody tr:nth-child(1){animation-delay:220ms}
.reg-table tbody tr:nth-child(2){animation-delay:250ms}
.reg-table tbody tr:nth-child(3){animation-delay:280ms}
.reg-table tbody tr:nth-child(4){animation-delay:310ms}
.reg-table tbody tr:nth-child(5){animation-delay:340ms}
.reg-table tbody tr:nth-child(6){animation-delay:370ms}
.reg-table tbody tr:nth-child(7){animation-delay:400ms}
.reg-table tbody tr:nth-child(8){animation-delay:430ms}
.reg-table tbody tr:nth-child(9){animation-delay:460ms}
.reg-table tbody tr:nth-child(10){animation-delay:490ms}
.reg-table tbody tr:nth-child(n+11){animation-delay:520ms}
.reg-table tbody tr:nth-child(odd) td,
.reg-table tbody tr:nth-child(even) td{background:transparent!important}
.reg-table tbody td{
  padding:13px 16px!important;border:none!important;vertical-align:middle;
  color:var(--tx-dim)!important;
  transition:background var(--d-fast) var(--e-soft),color var(--d-fast) var(--e-soft);
}
.reg-table tbody tr:hover td{background:rgba(124,92,255,.11)!important;color:var(--tx)!important}
.reg-table tbody tr:hover td:first-child{box-shadow:inset 2px 0 0 var(--violet)}
.p-high{color:var(--red)!important;font-weight:600;font-family:'Fira Code',monospace;font-size:9.5px}
.p-med{color:var(--saffron)!important;font-weight:600;font-family:'Fira Code',monospace;font-size:9.5px}
.p-low{color:var(--green)!important;font-weight:600;font-family:'Fira Code',monospace;font-size:9.5px}
.age-red{color:var(--red)!important;font-family:'Fira Code',monospace}
.age-amb{color:var(--saffron)!important;font-family:'Fira Code',monospace}
.age-ok{color:var(--tx-mute)!important;font-family:'Fira Code',monospace}
.src-badge{
  background:rgba(124,92,255,.14)!important;color:var(--violet)!important;
  border:1px solid rgba(124,92,255,.3)!important;border-radius:100px!important;
  padding:3px 9px!important;font-size:8.5px!important;
  font-family:'Fira Code',monospace;white-space:nowrap;letter-spacing:.4px;
  transition:all var(--d-fast) var(--e-soft);
}
.reg-table tbody tr:hover .src-badge{background:var(--violet)!important;color:#fff!important;
  box-shadow:0 0 18px rgba(124,92,255,.6)}

/* ══ REGISTER TICKET ═════════════════════════════════════════════════ */
.reg-ticket{
  border-radius:var(--r)!important;overflow:hidden;
  border:1px solid rgba(124,92,255,.3)!important;
  animation:pop-in 620ms var(--e-out) both;
  box-shadow:0 30px 70px -26px rgba(0,0,0,.9);
}
.reg-ticket-header{
  background:linear-gradient(120deg,var(--violet-dim),var(--surf-2) 70%)!important;
  padding:16px!important;text-align:center;position:relative;overflow:hidden;
}
.reg-ticket-header::after{
  content:"";position:absolute;inset:0;width:32%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.16),transparent);
  animation:sheen 4.2s var(--e-soft) 600ms infinite;
}
.reg-ticket-h1{font-family:'Noto Serif Devanagari',serif!important;font-size:15px!important;
  font-weight:600!important;color:#fff!important}
.reg-ticket-h2{color:rgba(255,255,255,.6)!important;font-size:10px!important;
  letter-spacing:.6px;margin-top:3px}
.reg-ticket-h3{color:var(--saffron)!important;font-size:11.5px!important;font-weight:600;
  margin-top:7px;font-family:'Noto Serif Devanagari',serif!important}
.reg-ticket table{width:100%;border-collapse:collapse}
.reg-ticket .lbl{
  background:rgba(255,255,255,.03)!important;
  border:1px solid var(--hair-2)!important;padding:9px 12px!important;width:135px;
  font-size:8.5px!important;color:var(--tx-mute)!important;font-weight:700!important;
  text-transform:uppercase;letter-spacing:1px;vertical-align:top;
}
.reg-ticket .lbl .hi{display:block;font-family:'Noto Sans Devanagari',sans-serif;
  font-size:10.5px;font-weight:600;text-transform:none;letter-spacing:0;color:var(--tx-dim)}
.reg-ticket .lbl .en{display:block;font-size:8px;opacity:.6;margin-top:2px}
.reg-ticket .val{
  border:1px solid var(--hair-2)!important;padding:9px 12px!important;
  font-size:12px!important;color:var(--tx)!important;vertical-align:top;line-height:1.7;
}
.reg-ticket .val.mono{font-family:'Fira Code',monospace}
.reg-ticket .section-hdr{
  background:rgba(124,92,255,.16)!important;padding:7px 12px!important;
  font-size:8.5px!important;font-weight:700!important;color:var(--violet)!important;
  text-transform:uppercase;letter-spacing:1.6px;
  border-top:1px solid var(--hair-2);border-bottom:1px solid var(--hair-2);
}
.prashan-row{border:1px solid var(--hair-2)!important;padding:11px 12px!important;
  display:flex;flex-wrap:wrap;gap:7px;align-items:center}
.prashan-box{
  border:1px solid rgba(124,92,255,.35)!important;border-radius:100px!important;
  padding:5px 12px!important;font-size:9.5px!important;font-weight:600;
  color:var(--violet)!important;background:rgba(124,92,255,.1)!important;
  letter-spacing:.3px;
  transition:transform var(--d-fast) var(--e-back),background var(--d-fast) var(--e-soft),
             color var(--d-fast) var(--e-soft),box-shadow var(--d-fast) var(--e-soft);
}
.prashan-box.highlight{background:var(--violet)!important;color:#fff!important;
  box-shadow:0 0 22px rgba(124,92,255,.55)}
.prashan-box:hover{transform:translateY(-2px);background:var(--violet)!important;color:#fff!important;
  box-shadow:0 0 22px rgba(124,92,255,.55)}
.jur-chain-light{display:flex;align-items:center;gap:5px;flex-wrap:wrap}
.jur-step-light{
  background:rgba(255,255,255,.05)!important;color:var(--tx-dim)!important;
  padding:4px 10px!important;border-radius:100px!important;font-size:9.5px!important;
  font-weight:500;font-family:'Fira Code',monospace;
  border:1px solid var(--hair)!important;
  transition:transform var(--d-fast) var(--e-back),background var(--d-fast) var(--e-soft),
             color var(--d-fast) var(--e-soft);
}
.jur-step-light:hover{transform:translateY(-2px);background:var(--cyan)!important;color:#04121A!important}
.pri-stamp{
  display:inline-block;border:1px solid;padding:4px 14px;border-radius:100px;
  font-size:10px;font-weight:700;letter-spacing:1.4px;text-transform:uppercase;
}
.pri-high{border-color:var(--red)!important;color:var(--red)!important;
  background:rgba(251,94,94,.12)!important;animation:stamp 2.6s var(--e-soft) infinite}
.pri-normal{border-color:var(--green)!important;color:var(--green)!important;
  background:rgba(52,211,153,.12)!important}

/* ══ ALERTS ══════════════════════════════════════════════════════════ */
.alert-high,.alert-normal,.alert-warn{
  border-radius:var(--r-sm)!important;padding:13px 16px!important;margin-top:10px!important;
  font-size:11.5px!important;font-family:'Fira Code',monospace!important;
  white-space:pre-wrap;line-height:1.7;backdrop-filter:blur(12px);
  animation:alert-in var(--d) var(--e-out) both;
}
.alert-high{background:rgba(251,94,94,.1)!important;border:1px solid rgba(251,94,94,.28)!important;
  border-left:2px solid var(--red)!important;color:#FFC9C9!important}
.alert-normal{background:rgba(52,211,153,.1)!important;border:1px solid rgba(52,211,153,.28)!important;
  border-left:2px solid var(--green)!important;color:#A7F3D8!important}
.alert-warn{background:rgba(245,165,36,.1)!important;border:1px solid rgba(245,165,36,.28)!important;
  border-left:2px solid var(--saffron)!important;color:#FFE0AC!important}
.processing-note{
  border-radius:var(--r-sm)!important;padding:14px!important;font-size:9.5px!important;
  font-family:'Fira Code',monospace!important;color:var(--tx-mute)!important;
  margin-top:12px!important;line-height:2;
  animation:fade-up var(--d-slow) var(--e-out) 180ms both;
}

/* ══ SCHEME CARDS ════════════════════════════════════════════════════ */
.scheme-entry{
  border-radius:var(--r)!important;overflow:hidden;margin-bottom:16px!important;
  animation:pop-in 600ms var(--e-back) both;
  transition:transform var(--d) var(--e-soft),box-shadow var(--d) var(--e-soft),
             border-color var(--d) var(--e-soft);
}
/* Each .scheme-entry is an only child of its own st.markdown wrapper, so
   nth-child on the card never stages. Columns are true siblings. */
div[data-testid="column"]:nth-child(1) .scheme-entry{animation-delay:120ms}
div[data-testid="column"]:nth-child(2) .scheme-entry{animation-delay:200ms}
div[data-testid="column"]:nth-child(3) .scheme-entry{animation-delay:280ms}
.scheme-entry:hover{transform:translateY(-6px);border-color:rgba(124,92,255,.4)!important;
  box-shadow:0 30px 60px -24px rgba(0,0,0,.9)}
.scheme-entry-header{
  background:linear-gradient(115deg,rgba(124,92,255,.3),rgba(34,211,238,.1))!important;
  padding:12px 14px!important;display:flex;align-items:center;gap:10px;
  border-bottom:1px solid var(--hair);position:relative;overflow:hidden;
}
.scheme-entry-header::after{
  content:"";position:absolute;inset:0;width:38%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.2),transparent);
  transform:translateX(-140%) skewX(-18deg);
  transition:transform 780ms var(--e-soft);
}
.scheme-entry:hover .scheme-entry-header::after{transform:translateX(320%) skewX(-18deg)}
.scheme-central-tag{
  background:rgba(124,92,255,.1)!important;border:1px solid rgba(124,92,255,.24)!important;
  border-radius:var(--r-sm)!important;padding:6px 10px!important;font-size:10px!important;
  color:var(--tx-dim)!important;margin-bottom:5px!important;
  transition:transform var(--d-fast) var(--e-soft);
}
.scheme-state-tag{
  background:rgba(245,165,36,.09)!important;border:1px solid rgba(245,165,36,.24)!important;
  border-radius:var(--r-sm)!important;padding:6px 10px!important;font-size:10px!important;
  color:#FFE0AC!important;margin-bottom:5px!important;
  transition:transform var(--d-fast) var(--e-soft);
}
.scheme-entry:hover .scheme-central-tag,.scheme-entry:hover .scheme-state-tag{transform:translateX(4px)}

/* ══ SIDEBAR ═════════════════════════════════════════════════════════ */
/* Solid, and matched to theme secondaryBackgroundColor (.streamlit/config.toml)
   so the option_menu component iframe blends in seamlessly. A gradient here
   would band against that flat iframe rectangle. */
section[data-testid="stSidebar"]{
  background:#0B0D18!important;
  border-right:1px solid var(--hair)!important;
}
section[data-testid="stSidebar"]>div{background:transparent!important}
section[data-testid="stSidebar"] iframe{background:transparent!important;color-scheme:dark}
section[data-testid="stSidebar"] h1,section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,section[data-testid="stSidebar"] .stMarkdown *{
  color:var(--tx)!important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] *,
section[data-testid="stSidebar"] input,section[data-testid="stSidebar"] textarea{
  color:var(--tx)!important;
}
.sb-brand{padding:20px 14px 14px;border-bottom:1px solid var(--hair);
  animation:fade-down 600ms var(--e-out) both}
.sb-seal{
  width:40px;height:40px;border-radius:50%;flex-shrink:0;display:grid;place-items:center;
  font-size:19px;position:relative;
  background:conic-gradient(from var(--ring-a),var(--violet),var(--cyan),var(--saffron),var(--violet));
  animation:ring-spin 11s linear infinite;
}
.sb-seal::before{content:"";position:absolute;inset:2px;border-radius:50%;background:#0B0E1B}
.sb-seal span{position:relative;z-index:1}
.sb-meta{font-family:'Fira Code',monospace;font-size:9.5px;color:var(--tx-mute);
  line-height:2.1;padding-left:4px;letter-spacing:.3px}
.sb-meta b{color:var(--tx-dim);font-weight:500}
.sb-cap{font-size:8.5px;color:var(--tx-mute);text-transform:uppercase;letter-spacing:2px;
  padding-left:4px;margin-bottom:7px;font-weight:700}
.sb-div{height:1px;background:var(--hair-2);margin:14px -8px}

/* ══ CONTROLS ════════════════════════════════════════════════════════ */
.stButton>button{
  position:relative;overflow:hidden;
  background:linear-gradient(120deg,var(--violet),var(--violet-dim))!important;
  color:#fff!important;font-weight:600!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;
  border:1px solid rgba(255,255,255,.14)!important;border-radius:var(--r-sm)!important;
  letter-spacing:.3px!important;padding:11px 18px!important;
  box-shadow:0 10px 30px -10px rgba(124,92,255,.7);
  transition:transform var(--d-fast) var(--e-soft),box-shadow var(--d-fast) var(--e-soft),
             filter var(--d-fast) var(--e-soft)!important;
}
.stButton>button:hover{transform:translateY(-2px);filter:brightness(1.14);
  box-shadow:0 18px 44px -12px rgba(124,92,255,.9)!important}
.stButton>button:active{transform:translateY(0) scale(.985)}
.stButton>button::after{
  content:"";position:absolute;top:0;bottom:0;left:0;width:34%;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.35),transparent);
  transform:translateX(-140%) skewX(-18deg);
  transition:transform 700ms var(--e-soft);
}
.stButton>button:hover::after{transform:translateX(320%) skewX(-18deg)}

.stTabs [data-baseweb="tab-list"]{background:transparent!important;
  border-bottom:1px solid var(--hair)!important;gap:2px!important}
.stTabs [data-baseweb="tab"]{
  color:var(--tx-mute)!important;font-family:'Plus Jakarta Sans',sans-serif!important;
  font-size:12.5px!important;font-weight:600!important;padding:11px 20px!important;
  border-bottom:2px solid transparent!important;border-radius:var(--r-sm) var(--r-sm) 0 0!important;
  transition:color var(--d-fast) var(--e-soft),background var(--d-fast) var(--e-soft)!important;
}
.stTabs [data-baseweb="tab"]:hover{color:var(--tx)!important;background:rgba(255,255,255,.04)!important}
.stTabs [aria-selected="true"]{color:var(--violet)!important;border-bottom-color:var(--violet)!important;
  background:rgba(124,92,255,.07)!important}

.stSelectbox>div>div,.stTextArea textarea,.stTextInput input{
  background:rgba(255,255,255,.045)!important;
  border:1px solid var(--hair)!important;color:var(--tx)!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;font-size:12.5px!important;
  border-radius:var(--r-sm)!important;
  transition:border-color var(--d-fast) var(--e-soft),box-shadow var(--d-fast) var(--e-soft),
             background var(--d-fast) var(--e-soft)!important;
}
.stSelectbox>div>div:hover,.stTextArea textarea:hover,.stTextInput input:hover{
  background:rgba(255,255,255,.07)!important;border-color:rgba(255,255,255,.18)!important}
.stSelectbox>div>div:focus-within,.stTextArea textarea:focus,.stTextInput input:focus{
  border-color:var(--violet)!important;box-shadow:0 0 0 3px rgba(124,92,255,.28)!important}
div[data-baseweb="popover"] li{background:var(--surf)!important;color:var(--tx)!important}
div[data-baseweb="popover"] li:hover{background:rgba(124,92,255,.24)!important}

div[data-testid="metric-container"]{
  background:var(--glass)!important;border:1px solid var(--hair)!important;
  border-left:2px solid var(--violet)!important;border-radius:var(--r)!important;
  padding:16px!important;backdrop-filter:blur(16px);
  animation:pop-in 600ms var(--e-back) both;
  transition:transform var(--d) var(--e-soft),box-shadow var(--d) var(--e-soft)!important;
}
div[data-testid="metric-container"]:hover{transform:translateY(-4px);
  box-shadow:0 24px 50px -20px rgba(0,0,0,.9)!important}
[data-testid="metric-container"] label{color:var(--tx-mute)!important;font-size:9px!important;
  text-transform:uppercase!important;letter-spacing:1.8px!important}
[data-testid="metric-container"] [data-testid="metric-value"]{
  color:var(--tx)!important;font-family:'Instrument Serif',serif!important;font-weight:400!important}

.stDataFrame,[data-testid="stDataFrame"]{
  font-family:'Fira Code',monospace!important;font-size:11px!important;
  border:1px solid var(--hair)!important;border-radius:var(--r)!important;overflow:hidden;
  animation:fade-up var(--d-slow) var(--e-out) both;
}
.streamlit-expanderHeader,[data-testid="stExpander"] summary{
  background:var(--glass)!important;border:1px solid var(--hair)!important;
  font-family:'Plus Jakarta Sans',sans-serif!important;color:var(--tx-dim)!important;
  border-radius:var(--r-sm)!important;font-size:12px!important;
  transition:background var(--d-fast) var(--e-soft)!important;
}
.streamlit-expanderHeader:hover,[data-testid="stExpander"] summary:hover{
  background:rgba(124,92,255,.14)!important;color:var(--tx)!important}
[data-testid="stExpander"]{border:none!important;background:transparent!important}
.streamlit-expanderContent{background:var(--glass-2)!important;
  border:1px solid var(--hair)!important;border-top:none!important}
.stSpinner>div{border-color:var(--violet) transparent transparent transparent!important}
.stRadio label{color:var(--tx-dim)!important;font-family:'Plus Jakarta Sans',sans-serif!important}
.stRadio [role="radiogroup"]{gap:8px}
[data-testid="stFileUploader"],[data-testid="stFileUploadDropzone"]{
  background:var(--glass)!important;border:1px dashed var(--hair)!important;
  border-radius:var(--r)!important;color:var(--tx-dim)!important;
  transition:border-color var(--d) var(--e-soft),background var(--d) var(--e-soft)!important;
}
[data-testid="stFileUploadDropzone"]:hover{border-color:var(--violet)!important;
  background:rgba(124,92,255,.08)!important}
[data-testid="stPlotlyChart"],[data-testid="stImage"]{
  border-radius:var(--r);overflow:hidden;
  animation:fade-up var(--d-slow) var(--e-out) 160ms both;
}
[data-testid="stImage"] img{border-radius:var(--r);border:1px solid var(--hair)}
[data-testid="stAlert"]{background:var(--glass)!important;border:1px solid var(--hair)!important;
  border-radius:var(--r-sm)!important;color:var(--tx)!important;backdrop-filter:blur(12px)}
hr{border-color:var(--hair-2)!important}

/* Keyboard focus is never traded away for aesthetics */
.stButton>button:focus-visible,.stTextInput input:focus-visible,
.stTextArea textarea:focus-visible,.stTabs [data-baseweb="tab"]:focus-visible,
[data-testid="stFileUploadDropzone"]:focus-visible{
  outline:3px solid var(--saffron)!important;outline-offset:3px!important;
}

/* ══ KEYFRAMES ═══════════════════════════════════════════════════════ */
@keyframes aurora{
  0%  {transform:translate3d(0,0,0) rotate(0deg) scale(1)}
  50% {transform:translate3d(3%,-2%,0) rotate(4deg) scale(1.09)}
  100%{transform:translate3d(-3%,2%,0) rotate(-3deg) scale(1.04)}
}
@keyframes ken{from{transform:scale(1.02) translate3d(0,0,0)}
               to{transform:scale(1.16) translate3d(-1.6%,-1.6%,0)}}
@keyframes hero-drift{from{transform:translateY(-3%)}to{transform:translateY(3%)}}
@keyframes hero-in{from{opacity:0;transform:translateY(22px) scale(.985)}to{opacity:1;transform:none}}
@keyframes fade-up{from{opacity:0;transform:translate3d(0,18px,0)}to{opacity:1;transform:none}}
@keyframes fade-down{from{opacity:0;transform:translate3d(0,-14px,0)}to{opacity:1;transform:none}}
@keyframes fade-in{from{opacity:0}to{opacity:1}}
@keyframes pop-in{from{opacity:0;transform:translate3d(0,18px,0) scale(.94)}to{opacity:1;transform:none}}
@keyframes row-in{from{opacity:0;transform:translate3d(-10px,0,0)}to{opacity:1;transform:none}}
@keyframes alert-in{from{opacity:0;transform:translate3d(-12px,0,0)}to{opacity:1;transform:none}}
@keyframes draw-x{from{transform:scaleX(0)}to{transform:scaleX(1)}}
@keyframes draw-y{from{transform:scaleY(0)}to{transform:scaleY(1)}}
@keyframes rule-grow{from{width:0}to{width:220px}}
@keyframes sheen{0%{transform:translateX(-140%) skewX(-18deg)}
                 100%{transform:translateX(360%) skewX(-18deg)}}
@keyframes sheen-text{0%,100%{background-position:0% 50%}50%{background-position:100% 50%}}
@keyframes ring-spin{to{--ring-a:360deg}}
@keyframes beat{0%,100%{box-shadow:0 0 0 0 rgba(52,211,153,.6)}70%{box-shadow:0 0 0 7px rgba(52,211,153,0)}}
@keyframes stamp{0%,100%{box-shadow:0 0 0 0 rgba(251,94,94,.4)}70%{box-shadow:0 0 0 8px rgba(251,94,94,0)}}

/* ══ REDUCED MOTION — hold every end state, drop the movement ═══════ */
@media (prefers-reduced-motion:reduce){
  *,*::before,*::after{
    animation-duration:.01ms!important;animation-iteration-count:1!important;
    animation-delay:0ms!important;transition-duration:.01ms!important;
    scroll-behavior:auto!important;
  }
  [data-testid="stAppViewContainer"]::before,.ngis-hero-bg,.mast-seal,.sb-seal{animation:none!important}
  .tricolor-strip::after,.reg-ticket-header::after,.stButton>button::after{display:none!important}
  .hero-rule{width:220px!important}
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PRESENTATION HELPERS
# Chrome only — these render no data and hold no state. The Devanagari
# strings here are display copy; the English page keys that drive routing
# are untouched.
# ══════════════════════════════════════════════════════════════════════
def heritage_masthead():
    """Bilingual Nalanda masthead — Devanagari leads, English is the subtitle."""
    st.markdown(f"""
    <div class="ngis-mast">
      <div class="mast-l">
        <div class="mast-seal"><span>☸</span></div>
        <div>
          <div class="mast-hi">नालंदा</div>
          <div class="mast-en">Grievance Intelligence</div>
        </div>
      </div>
      <div class="mast-r">
        <span class="mast-live"><i></i> LIVE · {datetime.now().strftime('%H:%M')} IST</span><br>
        हिलसा अनुमंडल · <b>EST. 427 CE</b>
      </div>
    </div>""", unsafe_allow_html=True)


def hero(title_en: str, title_hi: str, subtitle: str, image: str, kicker: str):
    """Cinematic hero: duotone-washed photograph, Ken Burns drift, layered type."""
    st.markdown(f"""
    <div class="ngis-hero">
      <div class="ngis-hero-bg" style="background-image:{hero_css_bg(image)};position:absolute;inset:0"></div>
      <div class="ngis-hero-over">
        <div>
          <div class="hero-kicker"><i></i> {kicker}</div>
          <span class="hero-hi">{title_hi}</span>
          <h1>{title_en}</h1>
          <div class="hero-rule"></div>
          <p>{subtitle}</p>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# SESSION STATE
# ══════════════════════════════════════════════════════════════════════
if "grievances" not in st.session_state:
    blocks  = list(BLOCK_CENSUS.keys())
    cats    = list(SCHEMA.keys())
    depts   = [SCHEMA[c]["department"].split("(")[0].strip() for c in cats]
    n       = 90
    np.random.seed(42)
    dates   = [(datetime.now()-timedelta(days=int(np.random.randint(0,45)))).strftime("%Y-%m-%d") for _ in range(n)]
    ci      = np.random.choice(len(cats)-1, n)
    bi      = np.random.choice(len(blocks), n)
    pris    = np.random.choice(["High","Medium","Low"], n, p=[0.22,0.48,0.30])
    stats_  = np.random.choice(["Open","In Progress","Resolved"], n, p=[0.50,0.25,0.25])
    ages    = np.random.randint(1, 35, n)
    st.session_state.grievances = pd.DataFrame({
        "ID":         [f"NLD-{i+1:03d}" for i in range(n)],
        "Date":       dates,
        "Category":   [cats[i] for i in ci],
        "Department": [depts[i] for i in ci],
        "Block":      [blocks[i] for i in bi],
        "Priority":   pris.tolist(),
        "Status":     stats_.tolist(),
        "Days_Open":  ages.tolist(),
        "Source":     ["Manual"]*n,
    })

for _k in ("ocr_result","classify_result","_gemini_active","gemini_key"):
    if _k not in st.session_state:
        st.session_state[_k] = None if _k != "_gemini_active" else _GEMINI_CHAIN[1]
        if _k == "gemini_key": st.session_state[_k] = ""

df        = st.session_state.grievances
blocks_df = get_blocks_df()
resolved  = len(df[df["Status"]=="Resolved"])


# ══════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="tricolor-strip" style="margin:-1rem -1rem 0"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class='sb-brand'>
      <div style='display:flex;align-items:center;gap:12px'>
        <div class='sb-seal'><span>⚖</span></div>
        <div>
          <div class='dv' style='font-size:14px;font-weight:600;color:#E7E9F2;line-height:1.2'>बिहार सरकार</div>
          <div style='font-size:8px;color:#8A90A8;letter-spacing:2.4px;
               text-transform:uppercase;margin-top:3px;font-weight:700'>Government of Bihar</div>
        </div>
      </div>
      <div style='margin-top:14px;padding-top:12px;border-top:1px solid rgba(255,255,255,.055)'>
        <div style="font-family:'Instrument Serif',serif;font-size:19px;color:#F5A524;
             letter-spacing:.5px;line-height:1">NGIS</div>
        <div style='font-size:9.5px;color:#A7AECB;letter-spacing:.3px;margin-top:4px;line-height:1.5'>
             Nalanda Grievance<br>Intelligence System</div>
        <div style="font-size:9px;color:#8A90A8;margin-top:6px;
             font-family:'Fira Code',monospace">Hilsa Sub-Division</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Today's Brief","Analytics Suite","Field Capture","Scheme Intelligence"],
        icons=["grid-fill","bar-chart-fill","camera-fill","building-fill"],
        default_index=0,
        styles={
            "container":         {"padding":"0px 4px","background-color":"transparent","backgroundColor":"transparent"},
            "icon":              {"color":"#F5A524","font-size":"13px"},
            "nav-link":          {"font-size":"12.5px","color":"#A7AECB",
                                  "font-family":"Plus Jakarta Sans, sans-serif",
                                  "font-weight":"600","padding":"11px 14px",
                                  "margin-bottom":"5px","border-radius":"10px",
                                  "letter-spacing":".2px",
                                  "border":"1px solid rgba(255,255,255,.055)",
                                  "background-color":"rgba(255,255,255,.03)","backgroundColor":"rgba(255,255,255,.03)",
                                  "transition":"all 280ms cubic-bezier(.16,1,.3,1)",
                                  "--hover-color":"rgba(124,92,255,.22)"},
            "nav-link-selected": {"background":"linear-gradient(120deg,#7C5CFF,#5B3FD1)",
                                  "background-color":"#7C5CFF","backgroundColor":"#7C5CFF",
                                  "color":"#FFFFFF","font-weight":"700",
                                  "border":"1px solid rgba(255,255,255,.16)",
                                  "box-shadow":"0 12px 30px -8px rgba(124,92,255,.85)",
                                  "transition":"all 280ms cubic-bezier(.16,1,.3,1)"},
        },
    )

    st.markdown("<div class='sb-div'></div>", unsafe_allow_html=True)

    st.markdown("<div class='sb-cap'>Gemini API Key</div>", unsafe_allow_html=True)
    raw_key = st.text_input("Gemini API Key", type="password", placeholder="AIza…",
                             value=st.session_state.get("gemini_key",""), label_visibility="collapsed")
    if raw_key and raw_key.strip():
        st.session_state["gemini_key"] = raw_key.strip()
        active_display = st.session_state.get("_gemini_active", _GEMINI_CHAIN[1])
        st.markdown(f"<div class='sb-meta' style='color:#34D399'>✓ Key set · {active_display}</div>", unsafe_allow_html=True)
    else:
        st.session_state["gemini_key"] = ""
        st.markdown("<div class='sb-meta' style='color:#FB5E5E'>⚠ No key — OCR disabled</div>", unsafe_allow_html=True)
        st.markdown("<div class='sb-meta'>aistudio.google.com/apikey</div>", unsafe_allow_html=True)

    st.markdown("<div class='sb-div'></div>", unsafe_allow_html=True)

    st.markdown("<div class='sb-cap'>फ़िल्टर · Filters</div>", unsafe_allow_html=True)
    gp = st.selectbox("Priority", ["All","High","Medium","Low"], label_visibility="collapsed")
    gb = st.selectbox("Block", ["All"]+sorted(df["Block"].unique().tolist()), label_visibility="collapsed")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    now = datetime.now()
    st.markdown(f"<div class='sb-meta'>{now.strftime('%d %b %Y · %H:%M')}<br>जनसंख्या <b>1,97,309</b><br>क्षेत्र <b>140 km²</b><br>ग्राम <b>56</b> · प्रखंड <b>20</b></div>", unsafe_allow_html=True)

fdf = df.copy()
if gp != "All": fdf = fdf[fdf["Priority"]==gp]
if gb != "All": fdf = fdf[fdf["Block"]==gb]


# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — TODAY'S BRIEF
# ══════════════════════════════════════════════════════════════════════
if selected == "Today's Brief":

    st.markdown('<div class="tricolor-strip"></div>', unsafe_allow_html=True)
    heritage_masthead()
    hero("Today's Brief", "आज का विवरण",
         f'Live grievance status · Nalanda District · {datetime.now().strftime("%d %B %Y, %A")}',
         "nalanda_ruins", "Situation Report")

    st.markdown('<div class="ngis-body">', unsafe_allow_html=True)

    today_str   = datetime.now().strftime("%Y-%m-%d")
    today_n     = len(df[df["Date"]==today_str])
    high_n      = len(df[df["Priority"]=="High"])
    pending_n   = len(df[df["Status"]=="Open"])
    res_rate    = int(resolved/len(df)*100) if len(df) else 0
    top_block   = fdf["Block"].value_counts().idxmax() if len(fdf) else "—"
    top_cat     = fdf["Category"].value_counts().idxmax().split("/")[0].strip() if len(fdf) else "—"
    total_len   = len(fdf)
    high_pct    = int(high_n/len(df)*100) if len(df) else 0
    pending_pct = int(pending_n/len(df)*100) if len(df) else 0

    st.markdown(f"""
    <div class="brief-card">
      <strong>सिस्टम विवरण (System Brief) —</strong>
      कुल {total_len} शिकायतें · आज {today_n} प्राप्त ·
      सर्वाधिक: <strong style="color:#7C5CFF">{_html.escape(top_block)}</strong> ·
      प्रमुख श्रेणी: <strong style="color:#22D3EE">{_html.escape(top_cat)}</strong> ·
      {high_n} अत्यावश्यक · निराकरण दर <strong style="color:#34D399">{res_rate}%</strong>
    </div>""", unsafe_allow_html=True)

    # ── Count-up choreography ──────────────────────────────────────────
    # CSS can interpolate a registered <integer>, so the KPI figures tick up
    # from zero instead of appearing. Targets are per-card, so the keyframes
    # are generated from the live values rather than hardcoded.
    #
    # Correctness contract: each .kpi-value ALSO declares --kpi-n inline and
    # keeps the true figure as (transparent) DOM text. Where @property is
    # unsupported, the counter still resolves against the inline value, so the
    # card renders the real number un-animated — it can never show a wrong one,
    # and screen readers always read the real text.
    _kpi_targets = [len(df), high_n, pending_n, res_rate]
    _kpi_frames  = "".join(
        f"@keyframes kpi-c{i+1}{{from{{--kpi-n:0}}to{{--kpi-n:{v}}}}}"
        f".kpi-row .kpi-card:nth-child({i+1}) .kpi-value{{"
        f"animation:fade-up 600ms var(--e-out) {380+i*70}ms both,"
        f"kpi-c{i+1} 1500ms var(--e-out) {380+i*70}ms both}}"
        for i, v in enumerate(_kpi_targets)
    )
    st.markdown(f"<style>{_kpi_frames}</style>", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-label"><span class="dv">कुल शिकायतें</span> · Total</div>
        <div class="kpi-value" style="--kpi-n:{len(df)}">{len(df)}</div>
        <div class="kpi-delta">{today_n} आज प्राप्त</div>
      </div>
      <div class="kpi-card k-red">
        <div class="kpi-label"><span class="dv">अत्यावश्यक</span> · High Priority</div>
        <div class="kpi-value" style="--kpi-n:{high_n}">{high_n}</div>
        <div class="kpi-delta down">{high_pct}% of total</div>
      </div>
      <div class="kpi-card k-amb">
        <div class="kpi-label"><span class="dv">लंबित</span> · Pending</div>
        <div class="kpi-value" style="--kpi-n:{pending_n}">{pending_n}</div>
        <div class="kpi-delta">{pending_pct}% of total</div>
      </div>
      <div class="kpi-card k-green">
        <div class="kpi-label"><span class="dv">निराकरण दर</span> · Resolution</div>
        <div class="kpi-value pct" style="--kpi-n:{res_rate}">{res_rate}%</div>
        <div class="kpi-delta up">of all registered</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown('<div class="ngis-hr"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">शिकायत रजिस्टर · Grievance Register (Oldest Unresolved)</div>', unsafe_allow_html=True)

    oldest = fdf[fdf["Status"]!="Resolved"].sort_values("Days_Open",ascending=False).head(15)
    rows_html = ""
    for _, row in oldest.iterrows():
        pri_cls     = "p-high" if row["Priority"]=="High" else "p-med" if row["Priority"]=="Medium" else "p-low"
        age_cls     = "age-red" if row["Days_Open"]>20 else "age-amb" if row["Days_Open"]>10 else "age-ok"
        pri_sym     = "▲" if row["Priority"]=="High" else "►" if row["Priority"]=="Medium" else "●"
        cat_short   = _html.escape(row["Category"].split("/")[0].strip())
        dept_short  = _html.escape(row["Department"].split("(")[0].strip()[:25])
        src         = _html.escape(str(row.get("Source","Manual")))
        block_esc   = _html.escape(str(row["Block"]))
        status_color= RED if row["Status"]=="Open" else SAFF if row["Status"]=="In Progress" else GREEN
        status_esc  = _html.escape(str(row["Status"]))
        rows_html  += (
            f"<tr>"
            f'<td style="font-family:Fira Code,monospace;color:{NAVY};font-weight:600;text-align:center">{_html.escape(str(row["ID"]))}</td>'
            f'<td style="font-family:Fira Code,monospace;font-size:11px">{_html.escape(str(row["Date"]))}</td>'
            f'<td style="color:{TXT};font-weight:500">{cat_short}</td>'
            f'<td style="color:#8A90A8;font-size:10.5px">{dept_short}</td>'
            f'<td style="color:{TXT}">{block_esc}</td>'
            f'<td><span class="{pri_cls}">{pri_sym} {row["Priority"]}</span></td>'
            f'<td class="{age_cls}">{row["Days_Open"]}d</td>'
            f'<td style="color:{status_color};font-size:11px;font-weight:500">{status_esc}</td>'
            f'<td><span class="src-badge">{src}</span></td>'
            f"</tr>"
        )

    st.markdown(f"""
    <div class="reg-wrap">
      <table class="reg-table">
        <thead><tr>
          <th style="text-align:center"><span class="hi">क्र.सं.</span><span class="en">Sl. No.</span></th>
          <th><span class="hi">दिनांक</span><span class="en">Date</span></th>
          <th><span class="hi">श्रेणी</span><span class="en">Category</span></th>
          <th><span class="hi">विभाग</span><span class="en">Department</span></th>
          <th><span class="hi">प्रखंड</span><span class="en">Block</span></th>
          <th><span class="hi">प्राथमिकता</span><span class="en">Priority</span></th>
          <th><span class="hi">आयु (दिन)</span><span class="en">Age (Days)</span></th>
          <th><span class="hi">स्थिति</span><span class="en">Status</span></th>
          <th><span class="hi">स्रोत</span><span class="en">Source</span></th>
        </tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYTICS SUITE
# ══════════════════════════════════════════════════════════════════════
elif selected == "Analytics Suite":

    st.markdown('<div class="tricolor-strip"></div>', unsafe_allow_html=True)
    heritage_masthead()
    hero("Analytics Suite", "विश्लेषण प्रणाली",
         "Power BI-style drill-down · scheme compliance · predictive forecasting · risk intelligence",
         "rajgir", "Spatial Intelligence")

    st.markdown('<div class="ngis-body">', unsafe_allow_html=True)

    # Heatmap & Drill-down for Hilsa Sub-division
    st.markdown('<div class="sec-label">हिलसा अनुमंडल · Grievance Heatmap</div>', unsafe_allow_html=True)

    hilsa_blocks = ['Hilsa', 'Chandi', 'Ekangarsarai', 'Islampur', 'Karai Parsurai', 'Parbalpur', 'Tharthari']
    
    # Filter for Hilsa sub-division
    hilsa_fdf = fdf[fdf['Block'].isin(hilsa_blocks)].copy()
    
    # Load boundaries
    geojson_data = load_hilsa_boundaries()

    # Aggregate
    if not hilsa_fdf.empty:
        agg_df = hilsa_fdf.groupby('Block').agg(
            Total_Reports=('ID', 'count'),
            High_Priority=('Priority', lambda x: (x == 'High').sum())
        ).reset_index()
        
        # The original used mapbox_style="white-bg" specifically to avoid
        # fetching basemap tiles. A hosted dark style would undo that, so this
        # is an inline Mapbox style spec with an empty `sources` block: still
        # zero network requests, but the canvas is dark instead of white.
        _dark_canvas = {
            "version": 8, "sources": {},
            "layers": [{"id": "bg", "type": "background",
                        "paint": {"background-color": "#0B0E1B"}}],
        }
        fig = px.choropleth_mapbox(
            agg_df,
            geojson=geojson_data,
            locations="Block",
            featureidkey="properties.Block",
            color="High_Priority",
            color_continuous_scale=[[0, "#1E2140"], [0.5, "#7C5CFF"], [1, "#FB5E5E"]],
            mapbox_style="white-bg", # Hides underlying base map tiles for a clean look
            zoom=9.5,
            center={"lat": 25.25, "lon": 85.35},
            hover_name="Block",
            hover_data={"Total_Reports": True, "High_Priority": True},
        )
        fig.update_traces(marker_line_color="rgba(255,255,255,.28)", marker_line_width=1)
        fig.update_layout(
            margin={"r":0,"t":0,"l":0,"b":0},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            mapbox_style=_dark_canvas,
            font=dict(color=MUTED, family="'Plus Jakarta Sans', sans-serif", size=11),
            hoverlabel=dict(bgcolor="#161A2E", bordercolor=NAVY,
                            font=dict(color=WHITE, family="'Fira Code', monospace")),
            coloraxis_colorbar=dict(
                title=dict(text="High<br>Priority", font=dict(color=MUTED, size=9)),
                tickfont=dict(color=MUTED, size=9), outlinewidth=0,
                thickness=9, len=.75, xpad=6,
            ),
            dragmode=False # Disables panning and zooming to make it "static"
        )
        
        st.markdown("<p style='color:#8A90A8;font-size:12px;margin-bottom:12px'>Click a political boundary to drill into its grievances.</p>", unsafe_allow_html=True)
        
        # We use on_select to capture clicks
        selection = st.plotly_chart(fig, use_container_width=True, on_select="rerun")
        
        selected_block = None
        if selection and 'selection' in selection and 'points' in selection['selection']:
            points = selection['selection']['points']
            if len(points) > 0:
                point_index = points[0]['point_index']
                if point_index < len(agg_df):
                    selected_block = agg_df.iloc[point_index]['Block']
        
        if selected_block:
            st.markdown(f"---")
            st.markdown(f"<div class='sec-label'>प्रखंड विवरण · Detailed Problems — {selected_block}</div>", unsafe_allow_html=True)
            block_data = hilsa_fdf[hilsa_fdf['Block'] == selected_block].copy()
            # Sort by priority High to Low (High: 0, Medium: 1, Low: 2)
            block_data['Priority_Rank'] = block_data['Priority'].map({'High': 0, 'Medium': 1, 'Low': 2})
            block_data = block_data.sort_values(by=['Priority_Rank', 'Days_Open'], ascending=[True, False]).drop(columns=['Priority_Rank'])
            
            st.dataframe(block_data[['ID', 'Category', 'Priority', 'Days_Open', 'Status']], use_container_width=True, hide_index=True)
            
    else:
        st.info("No reports found for Hilsa sub-division.")

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 3 — FIELD CAPTURE
# ══════════════════════════════════════════════════════════════════════
elif selected == "Field Capture":
    st.markdown('<div class="tricolor-strip"></div>', unsafe_allow_html=True)
    heritage_masthead()
    hero("Field Capture", "क्षेत्र संग्रह",
         "Janata Darbar digitization terminal · Gemini OCR · auto-classification · digital register",
         "pawapuri", "Capture Terminal")
    st.markdown('<div class="ngis-body">', unsafe_allow_html=True)

    input_mode=st.radio("Input Mode",["📷  Scan Letter (Image)","✍️  Type / Paste Text"],horizontal=True,label_visibility="collapsed")
    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    left, right = st.columns([1,1.5], gap="large")
    gemini_key_val = st.session_state.get("gemini_key","")
    active_model   = st.session_state.get("_gemini_active", _GEMINI_CHAIN[1])

    with left:
        if input_mode == "📷  Scan Letter (Image)":
            st.markdown('<div class="sec-label">दस्तावेज़ अपलोड करें · Upload Document</div>', unsafe_allow_html=True)
            upload_option=st.radio("",["📁  Upload file","📷  Camera"],horizontal=True,label_visibility="collapsed")
            img_bytes=None
            heic_ok = _HEIC_OK or _LIBHEIF_OK

            if upload_option == "📁  Upload file":
                if not heic_ok:
                    st.markdown('<div class="alert-warn">⚠ HEIC support unavailable — JPG &amp; PNG work fine\nFix: pip install pillow-heif → restart Streamlit</div>', unsafe_allow_html=True)
                    allowed_types=["jpg","jpeg","png"]
                else:
                    allowed_types=["jpg","jpeg","png","heic","heif"]
                uploaded=st.file_uploader("",type=allowed_types,label_visibility="collapsed")
                if uploaded:
                    raw=uploaded.read()
                    img,err=open_image_safe(raw,uploaded.name)
                    if err: st.error(f"Could not open image: {err}")
                    else:
                        buf=io.BytesIO(); img.save(buf,format="JPEG",quality=85); img_bytes=buf.getvalue()
                        st.image(img_bytes,caption=f"✅ {uploaded.name} — ready for OCR",use_column_width=True)
            else:
                cam=st.camera_input("",label_visibility="collapsed")
                if cam:
                    raw_cam=cam.read(); img_cam,err_cam=open_image_safe(raw_cam,"camera.jpg")
                    if err_cam: st.error(f"Camera error: {err_cam}")
                    else:
                        buf=io.BytesIO(); img_cam.save(buf,format="JPEG",quality=85); img_bytes=buf.getvalue()
                        st.image(img_bytes,use_column_width=True)

            if img_bytes:
                st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
                if not gemini_key_val:
                    st.markdown('<div class="alert-high">⚠ Add Gemini API key in sidebar to enable OCR.\nGet free key: https://aistudio.google.com/apikey</div>', unsafe_allow_html=True)
                else:
                    if st.button("🔍 पत्र पढ़ें · Process with Gemini OCR", use_container_width=True):
                        with st.spinner(f"Gemini ({active_model}) is reading the Hindi letter…"):
                            client_tuple=init_gemini(gemini_key_val)
                            client,sdk_type=client_tuple
                            if client is None:
                                st.markdown('<div class="alert-high">⚠ Failed to initialise Gemini.\nCheck API key: https://aistudio.google.com/apikey</div>', unsafe_allow_html=True)
                            else:
                                result=run_gemini_ocr(client_tuple,img_bytes)
                                if "error" not in result:
                                    st.session_state.ocr_result=result
                                    clf=classify_complaint(result.get("transcription",""))
                                    for field in ("block","village","complainant_name","date_filed"):
                                        if result.get(field,"Unknown")!="Unknown": clf[field]=result[field]
                                    st.session_state.classify_result=clf
                                    st.rerun()
                                else:
                                    err_msg=result["error"]
                                    cls="alert-warn" if "quota" in err_msg.lower() or "429" in err_msg else "alert-high"
                                    st.markdown(f'<div class="{cls}">{_html.escape(err_msg)}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="sec-label">शिकायत पाठ दर्ज करें · Input Grievance Text</div>', unsafe_allow_html=True)
            raw_text=st.text_area("",height=220,placeholder="हमारे गाँव में पानी की सप्लाई बंद है…",label_visibility="collapsed")
            if st.button("वर्गीकृत करें · Classify & Route →", use_container_width=True):
                if raw_text.strip():
                    clf=classify_complaint(raw_text)
                    st.session_state.ocr_result={"transcription":raw_text,"issue_summary":"Manual entry","complainant_name":clf.get("complainant_name","Unknown"),"village":clf.get("village","Unknown"),"block":clf.get("block","Unknown"),"date_filed":clf.get("date_filed","Unknown")}
                    st.session_state.classify_result=clf
                    st.rerun()

        heic_ok     = _HEIC_OK or _LIBHEIF_OK
        heic_status = "✓ HEIC · JPG · PNG · Camera" if heic_ok else "✗ HEIC unavail · JPG · PNG · Camera"
        heic_color  = GREEN if heic_ok else RED
        key_status  = f"✓ Key set ({active_model})" if gemini_key_val else "✗ No key — set in sidebar"
        key_color   = GREEN if gemini_key_val else RED
        st.markdown(f"""
        <div class="processing-note">
          OCR ENGINE  · Google Gemini (auto-fallback chain)<br>
          <span style="color:{heic_color}">FORMATS     · {heic_status}</span><br>
          <span style="color:{key_color}">API KEY     · {key_status}</span><br>
          LANGUAGES   · Hindi Devanagari + English<br>
          OUTPUT      · Register format (जनता दरबार पत्रावली)<br>
          CATEGORIES  · 9 categories + प्रेषण (forwarding)<br>
          SCHEMES     · Central + State + NITI Aayog
        </div>""", unsafe_allow_html=True)

    with right:
        ocr = st.session_state.ocr_result
        clf = st.session_state.classify_result

        if ocr and clf:
            new_id   = f"NLD-{len(st.session_state.grievances)+1:03d}"
            reg_date = datetime.now().strftime("%d/%m/%Y")

            # ── Safely escape ALL dynamic values before they touch HTML ──
            e_name    = _html.escape(str(clf.get("complainant_name","Unknown")))
            e_village = _html.escape(str(clf.get("village","Unknown")))
            e_block   = _html.escape(str(clf.get("block","Unknown")))
            e_dept    = _html.escape(str(clf["department"]))
            e_cat     = _html.escape(f"{clf['icon']} {clf['category']}")
            e_id      = _html.escape(new_id)
            priority  = clf["priority"]
            conf      = int(clf["confidence"]*100)
            pri_cls   = "pri-high" if priority=="High" else "pri-normal"

            # Summary — escape it; if it contains <> from Gemini output it won't corrupt HTML
            raw_summary = ocr.get("issue_summary","")
            if raw_summary and raw_summary not in ("Manual entry","Manually entered text"):
                e_summary = _html.escape(raw_summary)
            else:
                e_summary = "—"

            # Jurisdiction chain
            jur_steps  = clf.get("jurisdiction","").split(" ➔ ")
            jur_html   = "".join(
                f'<span class="jur-step-light">{_html.escape(j)}</span>'
                f'<span style="color:{MUTED};font-size:12px"> › </span>'
                for j in jur_steps[:-1]
            )
            if jur_steps:
                jur_html += f'<span class="jur-step-light">{_html.escape(jur_steps[-1])}</span>'

            # Prashan boxes — safe .get() with fallback
            prashan_list = clf.get("prashan_to", ["SDO"])
            prashan_html = "".join(
                f'<span class="prashan-box{"  highlight" if i==0 else ""}">'
                f'{_html.escape(str(p))}</span>'
                for i, p in enumerate(prashan_list)
            )
            first_prashan = _html.escape(str(prashan_list[0])) if prashan_list else "SDO"

            # Schemes
            schemes_html = "".join(
                f'<div style="background:rgba(124,92,255,.1);border:1px solid rgba(124,92,255,.24);border-radius:8px;'
                f'padding:6px 10px;font-size:10.5px;color:#C9C4FF;margin-bottom:4px">'
                f'<strong>केंद्रीय:</strong> {_html.escape(str(s))}</div>'
                for s in clf.get("central_schemes",[])
            )
            schemes_html += "".join(
                f'<div style="background:rgba(245,165,36,.09);border:1px solid rgba(245,165,36,.24);border-radius:8px;'
                f'padding:6px 10px;font-size:10.5px;color:#FFE0AC;margin-bottom:4px">'
                f'<strong>बिहार:</strong> {_html.escape(str(s))}</div>'
                for s in clf.get("bihar_schemes",[])
            )

            # OCR transcription (collapsible) — use st.text_area for safe rendering
            if ocr.get("transcription"):
                st.markdown('<div class="sec-label">मूल पाठ · OCR Transcription</div>', unsafe_allow_html=True)
                with st.expander("▼ पूर्ण हिंदी पाठ देखें · View full Hindi transcription"):
                    st.markdown(
                        f'<div class="dv" style="background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.09);'
                        f'border-radius:10px;padding:16px 18px;'
                        f'font-size:14px;color:#E7E9F2;line-height:2;white-space:pre-wrap">'
                        f'{_html.escape(str(ocr["transcription"]))}</div>',
                        unsafe_allow_html=True
                    )

            st.markdown('<div class="sec-label" style="margin-top:14px">पत्रावली प्रविष्टि · Register Entry</div>', unsafe_allow_html=True)

            # ── THE REGISTER TICKET  (NO HTML comments — that was the bug) ──
            st.markdown(f"""
<div class="reg-ticket">
  <div class="reg-ticket-header">
    <div class="reg-ticket-h1">अनुमंडल कार्यालय, हिलसा (नालंदा)</div>
    <div class="reg-ticket-h2">Sub-Divisional Office, Hilsa (Nalanda) · Bihar Government</div>
    <div class="reg-ticket-h3">जनता दरबार — शिकायत पत्रावली रजिस्टर</div>
  </div>
  <table>
    <tr>
      <td class="lbl"><span class="hi">क्र.सं.</span><span class="en">Sl. No.</span></td>
      <td class="val mono" style="font-weight:600;color:{NAVY};font-size:16px">{e_id}</td>
      <td class="lbl"><span class="hi">दिनांक</span><span class="en">Date</span></td>
      <td class="val mono">{reg_date}</td>
    </tr>
    <tr>
      <td class="lbl"><span class="hi">आवेदक</span><span class="en">Applicant</span></td>
      <td class="val">{e_name}</td>
      <td class="lbl"><span class="hi">ग्राम / प्रखंड</span><span class="en">Village / Block</span></td>
      <td class="val">{e_village} · {e_block}</td>
    </tr>
    <tr>
      <td class="lbl"><span class="hi">शिकायत श्रेणी</span><span class="en">Category</span></td>
      <td class="val">{e_cat} &nbsp;<span style="font-size:10px;color:{MUTED}">Conf: {conf}%</span></td>
      <td class="lbl"><span class="hi">प्राथमिकता</span><span class="en">Priority</span></td>
      <td class="val"><span class="pri-stamp {pri_cls}">{priority}</span></td>
    </tr>
    <tr>
      <td class="lbl"><span class="hi">विषय</span><span class="en">Subject</span></td>
      <td class="val" colspan="3" style="font-family:Noto Sans,sans-serif">{e_summary}</td>
    </tr>
    <tr>
      <td class="lbl"><span class="hi">प्रेषित विभाग</span><span class="en">Dept. Forwarded</span></td>
      <td class="val" colspan="3">{e_dept}</td>
    </tr>
  </table>
  <div class="section-hdr">प्रेषण · Forwarding Officers (प्रथम: {first_prashan})</div>
  <div class="prashan-row">{prashan_html}</div>
  <div class="section-hdr">न्यायाधिकार क्षेत्र · Jurisdiction Chain</div>
  <div style="border:1px solid rgba(255,255,255,.055);padding:12px">
    <div class="jur-chain-light">{jur_html}</div>
  </div>
  <div class="section-hdr">संबंधित योजनाएं · Applicable Schemes</div>
  <div style="border:1px solid rgba(255,255,255,.055);padding:12px">{schemes_html}</div>
</div>
""", unsafe_allow_html=True)

            # Escalation / routing instruction
            dept_route = clf["department"].split("(")[0].strip()
            if priority == "High":
                esc_msg = "⚠ अत्यावश्यक — SDO कार्यालय को 24 घंटे के भीतर अग्रेषित करें। Collector साप्ताहिक समीक्षा में शामिल करें।\nHIGH PRIORITY — Escalate to SDO within 24 hours. Flag for Collector's weekly review."
                esc_cls = "alert-high"
            else:
                esc_msg = f"→ {dept_route} को अग्रेषित करें। VB-GRAM-G अनुसार मानक SLA लागू।\nRoute to {dept_route}. Standard SLA applies under VB-GRAM-G mandate."
                esc_cls = "alert-normal"
            st.markdown(f'<div class="{esc_cls}">{_html.escape(esc_msg)}</div>', unsafe_allow_html=True)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            if st.button("✅ रजिस्टर में सहेजें · Save to NGIS Register", use_container_width=True):
                date_val = clf.get("date_filed","Unknown")
                try: datetime.strptime(date_val,"%Y-%m-%d")
                except (ValueError,TypeError): date_val = datetime.now().strftime("%Y-%m-%d")
                new_row=pd.DataFrame([{
                    "ID":new_id,"Date":date_val,
                    "Category":clf["category"],"Department":clf["department"],
                    "Block":clf.get("block","Unknown"),"Priority":clf["priority"],
                    "Status":"Open","Days_Open":0,"Source":"OCR / Gemini"
                }])
                st.session_state.grievances=pd.concat([st.session_state.grievances,new_row],ignore_index=True)
                st.session_state.ocr_result=None; st.session_state.classify_result=None
                st.success(f"✅ {new_id} — रजिस्टर में सहेजा गया। Today's Brief में देखें।")
                st.rerun()

        else:
            st.markdown("""
            <div class="brief-card" style="text-align:center;padding:72px 20px;border-left:1px solid rgba(255,255,255,.09)">
              <div style="font-size:44px;margin-bottom:14px;opacity:.5">📋</div>
              <div class="dv" style="font-size:17px;font-weight:600;color:#E7E9F2;margin-bottom:8px">
                जनता दरबार शिकायत पत्रावली
              </div>
              <div style="font-size:12px;color:#8A90A8">Upload a HEIC/JPG/PNG letter image or paste text on the left</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — SCHEME INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════
elif selected == "Scheme Intelligence":
    st.markdown('<div class="tricolor-strip"></div>', unsafe_allow_html=True)
    heritage_masthead()
    hero("Scheme Intelligence", "योजना बुद्धिमत्ता",
         "Scheme mapping · SLA targets · jurisdiction escalation chains · NITI Aayog indicators",
         "nalanda_monument", "Programme Registry")
    st.markdown('<div class="ngis-body">', unsafe_allow_html=True)
    st.markdown('<div class="sec-label">सभी 9 श्रेणियां — योजना मानचित्रण · All 9 Categories — Scheme Mapping</div>', unsafe_allow_html=True)

    cols = st.columns(3, gap="large")
    for idx,(cat,data) in enumerate(SCHEMA.items()):
        with cols[idx%3]:
            c_html="".join([
                f'<div class="scheme-central-tag"><strong>केंद्रीय / Central</strong> — {_html.escape(str(s))}</div>'
                for s in data.get("central_schemes",[])
            ])
            s_html="".join([
                f'<div class="scheme-state-tag"><strong>बिहार / Bihar</strong> — {_html.escape(str(s))}</div>'
                for s in data.get("bihar_schemes",[])
            ])
            prashan_list  = data.get("prashan_to", [])
            prashan_boxes = "".join([
                f'<span class="prashan-box" style="font-size:9px">{_html.escape(str(p))}</span>'
                for p in prashan_list
            ])
            jur_steps = data.get("jurisdiction","").split(" ➔ ")
            jur_html  = " › ".join([
                f'<span class="jur-step-light">{_html.escape(j)}</span>'
                for j in jur_steps
            ])
            cat_short  = _html.escape(cat.split("/")[0].strip())
            dept_short = _html.escape(data["department"].split("(")[0].strip())
            icon_esc   = data["icon"]  # emoji — safe, no escape needed

            st.markdown(f"""
<div class="scheme-entry">
  <div class="scheme-entry-header">
    <span style="font-size:20px">{icon_esc}</span>
    <div>
      <div style="color:#E7E9F2;font-weight:700;font-size:12.5px">{cat_short}</div>
      <div style="color:#8A90A8;font-size:9px;font-family:'Fira Code',monospace;margin-top:2px">{dept_short}</div>
    </div>
  </div>
  <div style="padding:10px 12px">
    {c_html}{s_html}
    <div style="margin-top:8px;font-size:8.5px;color:#8A90A8;text-transform:uppercase;letter-spacing:1.8px;font-weight:700;margin-bottom:6px">प्रेषण · Forwarding</div>
    <div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px">{prashan_boxes}</div>
    <div style="font-size:8.5px;color:#8A90A8;text-transform:uppercase;letter-spacing:1.8px;font-weight:700;margin-bottom:6px">न्यायाधिकार · Jurisdiction</div>
    <div style="line-height:2;font-size:10px">{jur_html}</div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)