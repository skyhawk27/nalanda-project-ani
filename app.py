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
NAVY   = "#613AF5"  # UX4G Primary Purple-Blue
SAFF   = "#B77224"  # UX4G Warning Orange-Amber
GREEN  = "#3C9718"  # UX4G Success Green
RED    = "#B7131A"  # UX4G Danger Red
AMBER  = "#B77224"  # UX4G Warning Saffron
TEAL   = "#00AAFF"  # UX4G Info Cyan
MUTED  = "#5E5E5E"  # UX4G Muted Gray
LGRID  = "#dee2e6"  # UX4G Border Gray
LBG    = "#F8F9FA"  # UX4G Light background
WHITE  = "#FFFFFF"
TXT    = "#212121"  # UX4G Body Text Dark

def ct(fig, title="", h=None):
    kw = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=MUTED, family="'Noto Sans', sans-serif", size=11),
        title=dict(text=title, font=dict(family="'Noto Serif', serif", size=14, color=TXT),
                   x=0, xanchor="left") if title else {},
        xaxis=dict(gridcolor=LGRID, linecolor=LGRID, tickcolor=MUTED, tickfont=dict(size=10)),
        yaxis=dict(gridcolor=LGRID, linecolor=LGRID, tickcolor=MUTED, tickfont=dict(size=10)),
        margin=dict(t=46 if title else 14, b=14, l=8, r=8),
        legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(0,0,0,0)",
                    font=dict(color=MUTED, size=11), orientation="h",
                    yanchor="bottom", y=1.02, xanchor="right", x=1),
        hoverlabel=dict(bgcolor=WHITE, bordercolor=LGRID, font=dict(color=TXT)),
    )
    if h: kw["height"] = h
    fig.update_layout(**kw)
    return fig


# ══════════════════════════════════════════════════════════════════════
# CSS  (unchanged from document 7)
# ══════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url("https://cdn.ux4g.gov.in/UX4G@2.0.8/css/ux4g-min.css");
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans:wght@300;400;500;600;700&family=Noto+Serif:wght@400;600;700&family=Noto+Sans+Devanagari:wght@400;500;600&family=Fira+Code:wght@400;500&family=Instrument+Serif:ital,wght@0,400;0,600;1,400&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

*,*::before,*::after{box-sizing:border-box}
footer{visibility:hidden}header{visibility:hidden}.stDeployButton{display:none}

.stApp{background:#F1F3F5!important}
.main .block-container{padding:24px 52px 60px 52px!important;max-width:100%!important}
section[data-testid="stSidebar"]{background:#1E293B!important;border-right:1px solid #334155!important}
section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown *,
section[data-testid="stSidebar"] span:not([data-testid="stMarkdownContainer"] *) {
    color: #F8FAFC !important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] *,
section[data-testid="stSidebar"] input,
section[data-testid="stSidebar"] textarea {
    color: #212121 !important;
}
h1,h2,h3{font-family:'Noto Sans',sans-serif!important;color:#1E293B!important;font-weight:600!important}

.tricolor-strip{height:5px;background:linear-gradient(90deg,#FF9933 33%,#FFFFFF 33%,#FFFFFF 66%,#138808 66%);width:100%}

.ngis-hero { position: relative; width: 100%; height: 320px; overflow: hidden; border-radius: 12px; box-shadow: 0 6px 16px rgba(0,0,0,0.08); margin-bottom: 24px; }
.ngis-hero-bg { width: 100%; height: 100%; background-size: cover; background-position: center; filter: saturate(0.85) brightness(0.95); }
.ngis-hero-over { position: absolute; inset: 0; background: linear-gradient(90deg, rgba(0,0,0,0.85) 0%, rgba(0,0,0,0.45) 50%, rgba(0,0,0,0) 100%); display: flex; align-items: flex-end; padding: 36px 52px; }
.ngis-hero h1 { font-family: 'Instrument Serif', serif !important; font-size: 3.8rem !important; color: #FFFFFF !important; margin: 0 0 8px !important; line-height: 1.1 !important; font-weight: 600 !important; text-shadow: 2px 2px 6px rgba(0,0,0,0.6); }
.ngis-hero p { color: #F8F5F0 !important; font-size: 14px !important; margin: 0 !important; letter-spacing: 0.5px; font-family: 'Plus Jakarta Sans', sans-serif !important; font-weight: 500 !important; text-shadow: 1px 1px 4px rgba(0,0,0,0.6); }
.gov-header-top{background:#613AF5;padding:10px 52px;display:flex;align-items:center;
  gap:16px;border-bottom:1px solid rgba(255,255,255,.12)}
.gov-emblem{width:44px;height:44px;background:rgba(255,255,255,.15);border-radius:50%;
  display:flex;align-items:center;justify-content:center;font-size:22px;flex-shrink:0}
.gov-title-hi{font-family:'Noto Sans Devanagari','Noto Sans',sans-serif;font-size:13px;
  font-weight:600;color:#FFFFFF;letter-spacing:.3px;line-height:1.3}
.gov-title-en{font-family:'Noto Sans',sans-serif;font-size:11px;
  color:rgba(255,255,255,.65);letter-spacing:.5px;margin-top:1px}
.gov-page-bar{background:#5231D1;padding:8px 52px;display:flex;align-items:center;gap:12px}
.gov-page-title{font-family:'Noto Sans',sans-serif;font-size:16px;font-weight:600;color:#FFFFFF}
.gov-page-sub{font-family:'Noto Sans',sans-serif;font-size:11px;
  color:rgba(255,255,255,.55);margin-left:auto;letter-spacing:.3px}

.ngis-body{padding:24px 0 0}
.ngis-hr{height:1px;background:#dee2e6;margin:20px 0}

.kpi-row{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:22px}
.kpi-card{background:#FFFFFF;border:1px solid #dee2e6;border-left:4px solid #613AF5;
  border-radius:4px;padding:14px 16px 12px;transition:box-shadow .2s}
.kpi-card:hover{box-shadow:0 4px 12px rgba(97,58,245,.08)}
.kpi-card.k-red{border-left-color:#B7131A}
.kpi-card.k-green{border-left-color:#3C9718}
.kpi-card.k-amb{border-left-color:#B77224}
.kpi-label{font-size:9px;text-transform:uppercase;letter-spacing:1.8px;color:#5E5E5E;
  font-weight:600;font-family:'Noto Sans',sans-serif;margin-bottom:8px}
.kpi-value{font-size:32px;font-weight:600;color:#613AF5;line-height:1;
  font-family:'Noto Sans',sans-serif;margin-bottom:5px}
.kpi-delta{font-size:11px;font-family:'Fira Code',monospace;color:#5E5E5E}
.kpi-delta.up{color:#3C9718}.kpi-delta.down{color:#B7131A}

.sec-label{font-size:10px;text-transform:uppercase;letter-spacing:2px;color:#613AF5;
  font-weight:700;font-family:'Noto Sans',sans-serif;
  padding-bottom:8px;border-bottom:2px solid #613AF5;margin-bottom:14px;
  display:flex;align-items:center;gap:6px}

.reg-wrap{overflow-x:auto;border:1px solid #dee2e6;border-radius:4px}
.reg-table{width:100%;border-collapse:collapse;font-size:13px;font-family:'Noto Sans',sans-serif}
.reg-table thead tr{background:#F8F9FA;border-bottom:2px solid #dee2e6}
.reg-table thead th{padding:12px 16px;text-align:left;color:#212121;
  border:1px solid #dee2e6;font-size:10px;font-weight:600;letter-spacing:.5px;white-space:nowrap}
.reg-table thead th .hi{display:block;font-family:'Noto Sans Devanagari',sans-serif;
  font-size:11px;font-weight:600}
.reg-table thead th .en{display:block;font-size:9px;opacity:.7;letter-spacing:.5px;margin-top:1px}
.reg-table tbody tr{border-bottom:1px solid #dee2e6}
.reg-table tbody tr:nth-child(odd) td{background:#FFFFFF}
.reg-table tbody tr:nth-child(even) td{background:#F8F9FA}
.reg-table tbody tr:hover td{background:#f1f3f5}
.reg-table tbody td{padding:12px 16px;border-right:1px solid #dee2e6;
  vertical-align:middle;color:#212121}
.reg-table tbody td:last-child{border-right:none}
.p-high{color:#B7131A;font-weight:600;font-family:'Fira Code',monospace;font-size:10px}
.p-med{color:#B77224;font-weight:600;font-family:'Fira Code',monospace;font-size:10px}
.p-low{color:#3C9718;font-weight:600;font-family:'Fira Code',monospace;font-size:10px}
.age-red{color:#B7131A;font-family:'Fira Code',monospace}
.age-amb{color:#B77224;font-family:'Fira Code',monospace}
.age-ok{color:#5E5E5E;font-family:'Fira Code',monospace}
.src-badge{background:#e9ecef;color:#613AF5;border:1px solid #ced4da;
  border-radius:2px;padding:1px 6px;font-size:9px;font-family:'Fira Code',monospace;white-space:nowrap}

.reg-ticket{border:2px solid #613AF5;border-radius:4px;overflow:hidden;
  font-family:'Noto Sans',sans-serif}
.reg-ticket-header{background:#613AF5;padding:12px 16px;text-align:center}
.reg-ticket-h1{font-family:'Noto Sans Devanagari','Noto Sans',sans-serif;
  font-size:14px;font-weight:700;color:#FFFFFF;letter-spacing:.3px}
.reg-ticket-h2{color:rgba(255,255,255,.65);font-size:11px;margin-top:2px;letter-spacing:.5px}
.reg-ticket-h3{color:#FFC53F;font-size:12px;font-weight:600;
  margin-top:5px;font-family:'Noto Sans Devanagari',sans-serif}
.reg-ticket table{width:100%;border-collapse:collapse}
.reg-ticket .lbl{background:#F8F9FA;border:1px solid #dee2e6;padding:7px 10px;
  width:130px;font-size:10px;color:#613AF5;font-weight:600;
  text-transform:uppercase;letter-spacing:.5px;vertical-align:top}
.reg-ticket .lbl .hi{display:block;font-family:'Noto Sans Devanagari',sans-serif;
  font-size:11px;font-weight:600;text-transform:none;letter-spacing:0}
.reg-ticket .lbl .en{display:block;font-size:9px;opacity:.65;margin-top:1px}
.reg-ticket .val{border:1px solid #dee2e6;padding:7px 10px;
  font-size:12px;color:#212121;vertical-align:top;line-height:1.6}
.reg-ticket .val.mono{font-family:'Fira Code',monospace}
.reg-ticket .section-hdr{background:#613AF5;padding:5px 10px;
  font-size:10px;font-weight:600;color:#FFFFFF;
  text-transform:uppercase;letter-spacing:1px;font-family:'Noto Sans',sans-serif}
.prashan-row{border:1px solid #dee2e6;padding:8px 10px;
  display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.prashan-box{border:1px solid #613AF5;border-radius:2px;padding:4px 10px;
  font-size:10px;font-weight:600;color:#613AF5;background:#e9ecef;
  font-family:'Noto Sans',sans-serif;letter-spacing:.3px}
.prashan-box.highlight{background:#613AF5;color:#FFFFFF}
.pri-stamp{display:inline-block;border:2px solid;padding:3px 12px;border-radius:2px;
  font-size:11px;font-weight:700;font-family:'Noto Sans',sans-serif;
  letter-spacing:1px;text-transform:uppercase}
.pri-high{border-color:#B7131A;color:#B7131A;background:#FFEEEA}
.pri-normal{border-color:#3C9718;color:#3C9718;background:#EDF7E6}

.alert-high{background:#FFEEEA;border:1px solid #FFCDC0;border-left:4px solid #B7131A;
  border-radius:2px;padding:10px 14px;margin-top:8px;
  font-size:12px;font-family:'Fira Code',monospace;color:#741010;white-space:pre-wrap}
.alert-normal{background:#EDF7E6;border:1px solid #E3F2D9;border-left:4px solid #3C9718;
  border-radius:2px;padding:10px 14px;margin-top:8px;
  font-size:12px;font-family:'Fira Code',monospace;color:#044400}
.alert-warn{background:#FEF1E7;border:1px solid #F9D7B9;border-left:4px solid #B77224;
  border-radius:2px;padding:10px 14px;margin-top:8px;
  font-size:12px;font-family:'Fira Code',monospace;color:#573000;white-space:pre-wrap}
.processing-note{background:#F8F9FA;border:1px solid #dee2e6;border-radius:2px;
  padding:10px 12px;font-size:10px;font-family:'Fira Code',monospace;
  color:#5E5E5E;margin-top:10px;line-height:1.8}

.brief-card{background:#FFFFFF;border:1px solid #dee2e6;border-left:4px solid #613AF5;
  border-radius:2px;padding:12px 16px;margin-bottom:18px;
  font-family:'Noto Sans',sans-serif;font-size:13px;color:#212121;line-height:1.7}

.scheme-entry{background:#FFFFFF;border:1px solid #dee2e6;border-radius:2px;
  overflow:hidden;margin-bottom:14px}
.scheme-entry-header{background:#613AF5;padding:8px 12px;
  display:flex;align-items:center;gap:8px}
.scheme-central-tag{background:#e9ecef;border:1px solid #ced4da;border-radius:2px;
  padding:5px 9px;font-size:11px;color:#613AF5;margin-bottom:4px}
.scheme-state-tag{background:#FEF1E7;border:1px solid #F9D7B9;border-radius:2px;
  padding:5px 9px;font-size:11px;color:#573000;margin-bottom:4px}
.jur-chain-light{display:flex;align-items:center;gap:4px;flex-wrap:wrap}
.jur-step-light{background:#e9ecef;color:#613AF5;padding:3px 8px;border-radius:2px;
  font-size:10px;font-weight:600;font-family:'Fira Code',monospace;border:1px solid #ced4da}

.stTabs [data-baseweb="tab-list"]{background:#FFFFFF!important;
  border-bottom:2px solid #613AF5!important;gap:0!important}
.stTabs [data-baseweb="tab"]{color:#5E5E5E!important;font-family:'Noto Sans',sans-serif!important;
  font-size:13px!important;font-weight:500!important;padding:10px 22px!important;
  border-bottom:2px solid transparent!important}
.stTabs [aria-selected="true"]{color:#613AF5!important;border-bottom-color:#613AF5!important;
  background:transparent!important;font-weight:600!important}
.stButton>button{background:#613AF5!important;color:#FFFFFF!important;
  font-weight:600!important;font-family:'Noto Sans',sans-serif!important;
  border:none!important;border-radius:4px!important;letter-spacing:.3px!important}
.stButton>button:hover{background:#774BFF!important}
.stSelectbox>div>div,.stTextArea textarea,.stTextInput input{
  background:#FFFFFF!important;border-color:#dee2e6!important;
  color:#212121!important;font-family:'Noto Sans',sans-serif!important;
  font-size:13px!important;border-radius:4px!important}
div[data-testid="metric-container"]{background:#FFFFFF!important;
  border:1px solid #dee2e6!important;border-left:4px solid #613AF5!important;
  border-radius:4px!important;padding:12px 14px!important}
[data-testid="metric-container"] label{color:#5E5E5E!important;
  font-family:'Noto Sans',sans-serif!important;font-size:9px!important;
  text-transform:uppercase!important;letter-spacing:1.5px!important}
[data-testid="metric-container"] [data-testid="metric-value"]{
  color:#613AF5!important;font-weight:600!important}
.stDataFrame{font-family:'Fira Code',monospace!important;font-size:11px!important}
.streamlit-expanderHeader{background:#F8F9FA!important;border:1px solid #dee2e6!important;
  font-family:'Noto Sans',sans-serif!important;color:#613AF5!important;border-radius:4px!important}
.streamlit-expanderContent{background:#FFFFFF!important;border:1px solid #dee2e6!important;
  border-top:none!important}
.stSpinner>div{border-color:#613AF5 transparent transparent transparent!important}
.stRadio label{color:#212121!important;font-family:'Noto Sans',sans-serif!important}
</style>
""", unsafe_allow_html=True)


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
    st.markdown('<div style="height:5px;background:linear-gradient(90deg,#FF9933 33%,#fff 33%,#fff 66%,#138808 66%);margin:-1rem -1rem 0 -1rem"></div>', unsafe_allow_html=True)

    st.markdown("""
    <div style='padding:16px 14px 10px;border-bottom:1px solid rgba(255,255,255,.15)'>
      <div style='display:flex;align-items:center;gap:10px'>
        <div style='width:40px;height:40px;background:rgba(255,255,255,.15);border-radius:50%;
             display:flex;align-items:center;justify-content:center;font-size:20px;flex-shrink:0'>⚖</div>
        <div>
          <div style='font-family:Noto Sans Devanagari,Noto Sans,sans-serif;font-size:13px;
               font-weight:700;color:#FFFFFF;line-height:1.2'>बिहार सरकार</div>
          <div style='font-size:9px;color:rgba(255,255,255,.7);letter-spacing:1px;
               text-transform:uppercase;margin-top:1px'>Government of Bihar</div>
        </div>
      </div>
      <div style='margin-top:10px;padding-top:10px;border-top:1px solid rgba(255,255,255,.15)'>
        <div style='font-family:Noto Sans,sans-serif;font-size:12px;font-weight:600;color:#FFC53F'>NGIS</div>
        <div style='font-size:9px;color:rgba(255,255,255,.75);letter-spacing:.5px;
             font-family:Noto Sans,sans-serif'>Nalanda Grievance Intelligence System</div>
        <div style='font-size:10px;color:rgba(255,255,255,.5);margin-top:2px;
             font-family:Fira Code,monospace'>Hilsa Sub-Division · Nalanda</div>
      </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["Today's Brief","Analytics Suite","Field Capture","Scheme Intelligence"],
        icons=["grid-fill","bar-chart-fill","camera-fill","building-fill"],
        default_index=0,
        styles={
            "container":         {"padding":"0px 4px","background-color":"#1E293B","backgroundColor":"#1E293B"},
            "icon":              {"color":"#FFC53F","font-size":"14px"},
            "nav-link":          {"font-size":"13px","color":"rgba(255,255,255,.85)",
                                  "font-family":"Noto Sans","padding":"10px 14px",
                                  "margin-bottom":"4px","border-radius":"6px",
                                  "background-color":"rgba(255,255,255,.05)","backgroundColor":"rgba(255,255,255,.05)"},
            "nav-link-selected": {"background-color":"#613AF5","backgroundColor":"#613AF5","color":"#FFFFFF","font-weight":"600"},
        },
    )

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:rgba(255,255,255,.15);margin:0 -8px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:9px;color:rgba(255,255,255,.6);text-transform:uppercase;letter-spacing:1.5px;font-family:Noto Sans;padding-left:4px;margin-bottom:5px'>Gemini API Key</div>", unsafe_allow_html=True)
    raw_key = st.text_input("Gemini API Key", type="password", placeholder="AIza…",
                             value=st.session_state.get("gemini_key",""), label_visibility="collapsed")
    if raw_key and raw_key.strip():
        st.session_state["gemini_key"] = raw_key.strip()
        active_display = st.session_state.get("_gemini_active", _GEMINI_CHAIN[1])
        st.markdown(f"<div style='font-size:10px;color:#6EE89A;font-family:Fira Code,monospace;padding-left:4px'>✓ Key set · {active_display}</div>", unsafe_allow_html=True)
    else:
        st.session_state["gemini_key"] = ""
        st.markdown("<div style='font-size:10px;color:#F4A0A3;font-family:Fira Code,monospace;padding-left:4px'>⚠ No key — OCR disabled</div>", unsafe_allow_html=True)
        st.markdown("<div style='font-size:9px;color:rgba(255,255,255,.4);font-family:Fira Code,monospace;padding-left:4px;margin-top:3px'>aistudio.google.com/apikey</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:1px;background:rgba(255,255,255,.15);margin:0 -8px'></div>", unsafe_allow_html=True)
    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    st.markdown("<div style='font-size:9px;color:rgba(255,255,255,.6);text-transform:uppercase;letter-spacing:1.5px;font-family:Noto Sans;padding-left:4px;margin-bottom:6px'>फ़िल्टर · Filters</div>", unsafe_allow_html=True)
    gp = st.selectbox("Priority", ["All","High","Medium","Low"], label_visibility="collapsed")
    gb = st.selectbox("Block", ["All"]+sorted(df["Block"].unique().tolist()), label_visibility="collapsed")

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    now = datetime.now()
    st.markdown(f"<div style='font-family:Fira Code,monospace;font-size:10px;color:rgba(255,255,255,.6);line-height:2;padding-left:4px'>{now.strftime('%d %b %Y  %H:%M')}<br>जनसंख्या: 1,97,309<br>क्षेत्र: 140 km²<br>ग्राम: 56 · प्रखंड: 20</div>", unsafe_allow_html=True)

fdf = df.copy()
if gp != "All": fdf = fdf[fdf["Priority"]==gp]
if gb != "All": fdf = fdf[fdf["Block"]==gb]


# ══════════════════════════════════════════════════════════════════════
# PAGE 1 — TODAY'S BRIEF
# ══════════════════════════════════════════════════════════════════════
if selected == "Today's Brief":

    st.markdown('<div class="tricolor-strip"></div>', unsafe_allow_html=True)
    bg1 = hero_css_bg("nalanda_ruins")
    st.markdown(f"""
    <div class="ngis-hero">
      <div class="ngis-hero-bg" style="background-image:{bg1};position:absolute;inset:0"></div>
      <div class="ngis-hero-over">
        <div>
          <h1>Today's Brief</h1>
          <p>Live grievance status · Nalanda District · {datetime.now().strftime("%d %B %Y, %A")}</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

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
      सर्वाधिक: <strong style="color:#1B3764">{_html.escape(top_block)}</strong> ·
      प्रमुख श्रेणी: <strong style="color:#1B6B7B">{_html.escape(top_cat)}</strong> ·
      {high_n} अत्यावश्यक · निराकरण दर <strong style="color:#1A7A3C">{res_rate}%</strong>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="kpi-row">
      <div class="kpi-card">
        <div class="kpi-label"><span style="font-family:Noto Sans Devanagari,sans-serif">कुल शिकायतें</span> · Total</div>
        <div class="kpi-value">{len(df)}</div>
        <div class="kpi-delta">{today_n} आज प्राप्त</div>
      </div>
      <div class="kpi-card k-red">
        <div class="kpi-label"><span style="font-family:Noto Sans Devanagari,sans-serif">अत्यावश्यक</span> · High Priority</div>
        <div class="kpi-value">{high_n}</div>
        <div class="kpi-delta down">{high_pct}% of total</div>
      </div>
      <div class="kpi-card k-amb">
        <div class="kpi-label"><span style="font-family:Noto Sans Devanagari,sans-serif">लंबित</span> · Pending</div>
        <div class="kpi-value">{pending_n}</div>
        <div class="kpi-delta">{pending_pct}% of total</div>
      </div>
      <div class="kpi-card k-green">
        <div class="kpi-label"><span style="font-family:Noto Sans Devanagari,sans-serif">निराकरण दर</span> · Resolution</div>
        <div class="kpi-value">{res_rate}%</div>
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
        status_color= "#C02020" if row["Status"]=="Open" else "#C8820A" if row["Status"]=="In Progress" else "#1A7A3C"
        status_esc  = _html.escape(str(row["Status"]))
        rows_html  += (
            f"<tr>"
            f'<td style="font-family:Fira Code,monospace;color:{NAVY};font-weight:600;text-align:center">{_html.escape(str(row["ID"]))}</td>'
            f'<td style="font-family:Fira Code,monospace;font-size:11px">{_html.escape(str(row["Date"]))}</td>'
            f'<td style="color:{TXT};font-weight:500">{cat_short}</td>'
            f'<td style="color:#5C5048;font-size:11px">{dept_short}</td>'
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

    bg2 = hero_css_bg("rajgir")
    st.markdown(f"""
    <div class="ngis-hero">
      <div class="ngis-hero-bg" style="background-image:{bg2};position:absolute;inset:0"></div>
      <div class="ngis-hero-over">
        <div>
          <h1>Analytics Suite</h1>
          <p>Power BI-style drill-down · scheme compliance · predictive forecasting · risk intelligence</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="ngis-body">', unsafe_allow_html=True)

    # Heatmap & Drill-down for Hilsa Sub-division
    st.markdown("""<h3 style="font-family: 'Instrument Serif', serif; font-size: 28px; color: #1E3A8A; margin-bottom: 20px;">Hilsa Sub-division Grievance Heatmap</h3>""", unsafe_allow_html=True)

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
        
        fig = px.choropleth_mapbox(
            agg_df, 
            geojson=geojson_data,
            locations="Block", 
            featureidkey="properties.Block",
            color="High_Priority",
            color_continuous_scale="Reds",
            mapbox_style="white-bg", # Hides underlying base map tiles for a clean look
            zoom=9.5,
            center={"lat": 25.25, "lon": 85.35},
            hover_name="Block",
            hover_data={"Total_Reports": True, "High_Priority": True},
            title="Hilsa Sub-division Static Boundaries"
        )
        fig.update_layout(
            margin={"r":0,"t":40,"l":0,"b":0},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            dragmode=False # Disables panning and zooming to make it "static"
        )
        
        st.markdown("<p style='color: #6B7280; font-size: 14px; margin-bottom: 10px;'>Click on a political boundary below to view detailed grievances.</p>", unsafe_allow_html=True)
        
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
            st.markdown(f"<h4 style='color: #1E3A8A; font-family: \"Instrument Serif\", serif;'>Detailed Problems for {selected_block}</h4>", unsafe_allow_html=True)
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
    bg3 = hero_css_bg("pawapuri")
    st.markdown(f"""
    <div class="ngis-hero">
      <div class="ngis-hero-bg" style="background-image:{bg3};position:absolute;inset:0"></div>
      <div class="ngis-hero-over">
        <div>
          <h1>Field Capture</h1>
          <p>Janata Darbar digitization terminal · Gemini OCR · auto-classification · digital register</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
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
        heic_color  = "#1A7A3C" if heic_ok else "#C02020"
        key_status  = f"✓ Key set ({active_model})" if gemini_key_val else "✗ No key — set in sidebar"
        key_color   = "#1A7A3C" if gemini_key_val else "#C02020"
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
                f'<div style="background:#EBF0FA;border:1px solid #B8CCE8;border-radius:2px;'
                f'padding:4px 8px;font-size:11px;color:#1B3764;margin-bottom:3px">'
                f'<strong>केंद्रीय:</strong> {_html.escape(str(s))}</div>'
                for s in clf.get("central_schemes",[])
            )
            schemes_html += "".join(
                f'<div style="background:#FEF3EB;border:1px solid #F5C89A;border-radius:2px;'
                f'padding:4px 8px;font-size:11px;color:#7A3A0A;margin-bottom:3px">'
                f'<strong>बिहार:</strong> {_html.escape(str(s))}</div>'
                for s in clf.get("bihar_schemes",[])
            )

            # OCR transcription (collapsible) — use st.text_area for safe rendering
            if ocr.get("transcription"):
                st.markdown('<div class="sec-label">मूल पाठ · OCR Transcription</div>', unsafe_allow_html=True)
                with st.expander("▼ पूर्ण हिंदी पाठ देखें · View full Hindi transcription"):
                    st.markdown(
                        f'<div style="background:#F5F3EE;border:1px solid #D4C9B8;border-radius:2px;'
                        f'padding:12px 14px;font-family:Noto Sans Devanagari,Noto Sans,sans-serif;'
                        f'font-size:13px;color:#1A1A2E;line-height:1.9;white-space:pre-wrap">'
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
  <div style="border:1px solid #D4C9B8;padding:10px 12px">
    <div class="jur-chain-light">{jur_html}</div>
  </div>
  <div class="section-hdr">संबंधित योजनाएं · Applicable Schemes</div>
  <div style="border:1px solid #D4C9B8;padding:10px 12px">{schemes_html}</div>
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
            <div style="text-align:center;padding:60px 20px;font-family:'Noto Sans',sans-serif;color:#8A7A6A">
              <div style="font-size:48px;margin-bottom:12px">📋</div>
              <div style="font-size:15px;font-weight:600;color:#1B3764;margin-bottom:6px">
                जनता दरबार शिकायत पत्रावली
              </div>
              <div style="font-size:13px">Upload a HEIC/JPG/PNG letter image or paste text on the left</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════
# PAGE 4 — SCHEME INTELLIGENCE
# ══════════════════════════════════════════════════════════════════════
elif selected == "Scheme Intelligence":
    st.markdown('<div class="tricolor-strip"></div>', unsafe_allow_html=True)
    bg4 = hero_css_bg("nalanda_monument")
    st.markdown(f"""
    <div class="ngis-hero">
      <div class="ngis-hero-bg" style="background-image:{bg4};position:absolute;inset:0"></div>
      <div class="ngis-hero-over">
        <div>
          <h1>Scheme Intelligence</h1>
          <p>Scheme mapping · SLA targets · jurisdiction escalation chains · NITI Aayog indicators</p>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
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
      <div style="color:#FFFFFF;font-weight:600;font-size:13px;font-family:Noto Sans,sans-serif">{cat_short}</div>
      <div style="color:rgba(255,255,255,.5);font-size:10px;font-family:Fira Code,monospace">{dept_short}</div>
    </div>
  </div>
  <div style="padding:10px 12px">
    {c_html}{s_html}
    <div style="margin-top:8px;font-size:9px;color:#1B3764;text-transform:uppercase;letter-spacing:1px;font-family:Noto Sans,sans-serif;font-weight:600;margin-bottom:4px">प्रेषण · Forwarding</div>
    <div style="display:flex;flex-wrap:wrap;gap:4px;margin-bottom:8px">{prashan_boxes}</div>
    <div style="font-size:9px;color:#1B3764;text-transform:uppercase;letter-spacing:1px;font-family:Noto Sans,sans-serif;font-weight:600;margin-bottom:4px">न्यायाधिकार · Jurisdiction</div>
    <div style="line-height:2;font-size:10px">{jur_html}</div>
  </div>
</div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)