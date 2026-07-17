# ══════════════════════════════════════════════════════════════════════
# "THE THRESHOLD" — Madhubani splash screen for NGIS · Nalanda
# Presentation only. Renders once per session, before the dashboard.
#
# A parchment canvas draws itself in Mithila line-art — border frame,
# corner spirals, a central sun-mandala, fish pair, lotus, peacock —
# then resolves into the bilingual title and a single Enter threshold.
#
# Constraints honoured (same contract as app.py's master CSS):
#   · st.markdown is sanitised by DOMPurify — all motion is pure CSS.
#     Verified against the Streamlit 1.35 bundle: <svg>/<path>/<circle>/
#     <g>/<line> and the pathLength attribute survive sanitisation;
#     <use> does NOT, so every motif inlines its own geometry.
#   · Every shape carries pathLength="1" so one universal rule
#     (stroke-dasharray:1 → stroke-dashoffset 1→0) animates all draws.
#   · div.stButton>button is safe to style ONLY here: Streamlit 1.35 has
#     no st-key-* classes, and this CSS is injected exclusively while
#     st.session_state["entered"] is False — it never coexists with the
#     dashboard's buttons.
#   · Reduced motion ⇒ the finished painting, static, CTA visible.
# ══════════════════════════════════════════════════════════════════════
import math

import streamlit as st

# Heritage ink tokens (scoped on .stApp in SPLASH_CSS, never :root)
INK  = "var(--sp-ink)"
TER  = "var(--sp-terracotta)"
MAD  = "var(--sp-madder)"
IND  = "var(--sp-indigo)"
SAF  = "var(--sp-saffron)"
LEAF = "var(--sp-leaf)"


# ── geometry helpers ──────────────────────────────────────────────────
def _pt(cx, cy, r, deg):
    """Polar → 'x,y' (SVG coords, y down; deg measured anticlockwise from +x)."""
    a = math.radians(deg)
    return f"{cx + r * math.cos(a):.1f},{cy - r * math.sin(a):.1f}"


def _sty(d=None, dd=None, fd=None, fo=None):
    """Inline custom properties driving the draw/fill choreography."""
    s = []
    if d is not None:
        s.append(f"--d:{d:.2f}s")
    if dd is not None:
        s.append(f"--dd:{dd:.2f}s")
    if fd is not None:
        s.append(f"--fd:{fd:.2f}s")
    if fo is not None:
        s.append(f"--fo:{fo:.2f}")
    return ";".join(s)


def _path(d, sty, stroke=INK, sw=2.4, fill="none", cls="d"):
    return (f'<path pathLength="1" class="{cls}" d="{d}" stroke="{stroke}" '
            f'stroke-width="{sw}" fill="{fill}" stroke-linecap="round" '
            f'stroke-linejoin="round" style="{sty}"/>')


def _circle(cx, cy, r, sty, stroke=INK, sw=2.4, fill="none", cls="d"):
    return (f'<circle pathLength="1" class="{cls}" cx="{cx:.1f}" cy="{cy:.1f}" '
            f'r="{r:.1f}" stroke="{stroke}" stroke-width="{sw}" fill="{fill}" '
            f'style="{sty}"/>')


def _petal(cx, cy, deg, r, hw, rb=14):
    """Two-quadratic petal from a base ring rb to a tip at radius r."""
    b1 = _pt(cx, cy, rb, deg - hw * 2)
    b2 = _pt(cx, cy, rb, deg + hw * 2)
    c1 = _pt(cx, cy, r * 0.72, deg - hw)
    c2 = _pt(cx, cy, r * 0.72, deg + hw)
    tip = _pt(cx, cy, r, deg)
    return f"M{b1} Q{c1} {tip} Q{c2} {b2}"


# ── motifs ────────────────────────────────────────────────────────────
def _mandala_svg():
    """Central sun-mandala: hub → hatch → petal ring → kite rays → rim.
    Draw order is centre-out (t 0.95 → 2.6s); fills bloom 2.5 → 3.2s."""
    C = 260
    e = []

    # hub — filled sun disc + double ring
    e.append(_circle(C, C, 16, _sty(d=0.95, dd=0.7, fd=2.55, fo=0.85),
                     sw=2.6, fill=SAF, cls="d f"))
    e.append(_circle(C, C, 30, _sty(d=1.05, dd=0.8)))
    e.append(_circle(C, C, 38, _sty(d=1.15, dd=0.8), sw=1.6, stroke=TER))

    # quadrant cross-hatch between the hub and the petal ring
    for q in range(4):
        for k in range(5):
            ang = q * 90 + 20 + k * 12.5
            d = f"M{_pt(C, C, 46, ang)} L{_pt(C, C, 84, ang)}"
            e.append(_path(d, _sty(d=1.2 + (q * 5 + k) * 0.015, dd=0.5),
                           sw=1.3, stroke=IND))
    e.append(_circle(C, C, 44, _sty(d=1.1, dd=0.9), sw=1.6))
    e.append(_circle(C, C, 88, _sty(d=1.35, dd=0.9), sw=1.6))

    # petal ring — 20 lotus petals, alternating warm fills
    for i in range(20):
        th = i * 18
        fill = SAF if i % 2 == 0 else TER
        fo = 0.26 if i % 2 == 0 else 0.20
        e.append(_path(_petal(C, C, th, 162, 7, rb=98),
                       _sty(d=1.45 + i * 0.03, dd=0.9, fd=2.5 + i * 0.02, fo=fo),
                       sw=2.0, fill=fill, cls="d f"))

    # mid double ring
    e.append(_circle(C, C, 170, _sty(d=1.9, dd=1.0)))
    e.append(_circle(C, C, 176, _sty(d=2.0, dd=1.0), sw=1.4, stroke=TER))

    # 16 kite rays, alternating terracotta / indigo
    for i in range(16):
        th = i * 22.5
        tip = _pt(C, C, 238, th)
        left = _pt(C, C, 200, th - 5)
        base = _pt(C, C, 184, th)
        right = _pt(C, C, 200, th + 5)
        fill = TER if i % 2 == 0 else IND
        e.append(_path(f"M{tip} L{left} L{base} L{right} Z",
                       _sty(d=2.05 + i * 0.035, dd=0.7,
                            fd=2.7 + i * 0.02, fo=0.30),
                       sw=2.0, fill=fill, cls="d f"))

    # madder dots in the gaps between ray tips
    for i in range(16):
        th = i * 22.5 + 11.25
        x, y = _pt(C, C, 214, th).split(",")
        e.append(_circle(float(x), float(y), 3.0,
                         _sty(fd=2.9 + i * 0.03, fo=0.9),
                         sw=0, stroke="none", fill=MAD, cls="f"))

    # outer double rim
    e.append(_circle(C, C, 244, _sty(d=2.35, dd=1.1), sw=2.6))
    e.append(_circle(C, C, 250, _sty(d=2.5, dd=1.1), sw=1.4, stroke=TER))

    return ('<svg viewBox="0 0 520 520" xmlns="http://www.w3.org/2000/svg">'
            + "".join(e) + "</svg>")


def _fish_svg(t0):
    """Madhubani fish: double outline, ringed eye, gill, scale arcs, fan tail."""
    e = [
        _path("M16,62 C40,26 122,22 152,58", _sty(d=t0, dd=1.0), sw=2.6),
        _path("M16,62 C40,98 122,102 152,66", _sty(d=t0 + 0.06, dd=1.0), sw=2.6),
        _path("M28,62 C48,38 116,34 142,60", _sty(d=t0 + 0.12, dd=0.9),
              sw=1.5, stroke=TER),
        _path("M28,62 C48,86 116,90 142,64", _sty(d=t0 + 0.18, dd=0.9),
              sw=1.5, stroke=TER),
        _path("M152,62 L188,30 C180,52 180,72 188,94 Z",
              _sty(d=t0 + 0.24, dd=0.9, fd=t0 + 1.2, fo=0.25),
              sw=2.4, fill=TER, cls="d f"),
        _circle(46, 58, 6.5, _sty(d=t0 + 0.3, dd=0.6), sw=2.0),
        _circle(46, 58, 2.4, _sty(fd=t0 + 1.0, fo=1), sw=0, stroke="none",
                fill=INK, cls="f"),
        _path("M60,40 Q52,62 60,84", _sty(d=t0 + 0.36, dd=0.7), sw=1.8, stroke=MAD),
        _path("M78,36 Q69,62 78,88", _sty(d=t0 + 0.42, dd=0.7), sw=1.4, stroke=IND),
        _path("M94,33 Q85,62 94,91", _sty(d=t0 + 0.48, dd=0.7), sw=1.4, stroke=IND),
        _path("M110,34 Q101,62 110,90", _sty(d=t0 + 0.54, dd=0.7), sw=1.4, stroke=IND),
        _path("M126,40 Q117,62 126,84", _sty(d=t0 + 0.6, dd=0.7), sw=1.4, stroke=IND),
        _path("M66,33 Q88,14 112,27", _sty(d=t0 + 0.66, dd=0.7), sw=2.0),
        _path("M66,91 Q88,110 112,97", _sty(d=t0 + 0.72, dd=0.7), sw=2.0),
    ]
    return ('<svg viewBox="0 0 200 120" xmlns="http://www.w3.org/2000/svg">'
            + "".join(e) + "</svg>")


def _lotus_svg(t0):
    """Layered lotus over a leaf-pad boat curve."""
    e = [
        _path("M34,116 Q90,142 146,116", _sty(d=t0, dd=0.9), sw=2.4, stroke=LEAF),
        _path("M48,120 Q90,136 132,120", _sty(d=t0 + 0.08, dd=0.8),
              sw=1.5, stroke=LEAF),
    ]
    cx, cy = 90, 114
    for i, ang in enumerate((30, 60, 90, 120, 150)):          # outer layer
        e.append(_path(_petal(cx, cy, ang, 74, 8),
                       _sty(d=t0 + 0.15 + i * 0.05, dd=0.8,
                            fd=t0 + 1.0 + i * 0.04, fo=0.18),
                       sw=2.2, fill=MAD, cls="d f"))
    for i, ang in enumerate((45, 75, 105, 135)):              # mid layer
        e.append(_path(_petal(cx, cy, ang, 52, 8),
                       _sty(d=t0 + 0.4 + i * 0.05, dd=0.7,
                            fd=t0 + 1.2 + i * 0.04, fo=0.25),
                       sw=2.0, fill=SAF, cls="d f"))
    for i, ang in enumerate((60, 90, 120)):                   # inner layer
        e.append(_path(_petal(cx, cy, ang, 32, 8),
                       _sty(d=t0 + 0.6 + i * 0.05, dd=0.6,
                            fd=t0 + 1.4 + i * 0.04, fo=0.30),
                       sw=1.8, fill=TER, cls="d f"))
    return ('<svg viewBox="0 0 180 150" xmlns="http://www.w3.org/2000/svg">'
            + "".join(e) + "</svg>")


def _peacock_svg(t0):
    """Stylised Mithila peacock: teardrop body, crest, fanned tail arcs
    with saffron eye-dots. Faces left; the tail fans up-right."""
    e = []
    tx, ty = 168, 150                                        # tail-fan centre
    for i, r in enumerate((34, 54, 74, 94, 112)):            # tail arcs
        start = _pt(tx, ty, r, 155)
        end = _pt(tx, ty, r, 25)
        col = IND if i % 2 == 0 else TER
        e.append(_path(f"M{start} A{r},{r} 0 0 1 {end}",
                       _sty(d=t0 + 0.15 + i * 0.12, dd=0.9),
                       sw=2.6, stroke=col))
    for i in range(7):                                       # eye-dots on the fan
        ang = 30 + i * 20
        x, y = _pt(tx, ty, 124, ang).split(",")
        e.append(_circle(float(x), float(y), 5.5,
                         _sty(d=t0 + 0.8 + i * 0.05, dd=0.5), sw=1.6))
        e.append(_circle(float(x), float(y), 2.6,
                         _sty(fd=t0 + 1.3 + i * 0.04, fo=0.95),
                         sw=0, stroke="none", fill=SAF, cls="f"))
    # body — upright teardrop
    e.append(_path("M118,96 C100,118 98,162 114,196 C122,212 142,214 150,200 "
                   "C160,180 156,138 142,110 C136,98 126,88 118,96 Z",
                   _sty(d=t0, dd=1.1, fd=t0 + 1.5, fo=0.22),
                   sw=2.6, fill=IND, cls="d f"))
    # head, beak, crest
    e.append(_circle(116, 88, 10, _sty(d=t0 + 0.3, dd=0.6), sw=2.4))
    e.append(_path("M106,86 L94,90 L107,94 Z",
                   _sty(d=t0 + 0.45, dd=0.5, fd=t0 + 1.1, fo=0.9),
                   sw=1.8, fill=SAF, cls="d f"))
    for i, (x1, x2) in enumerate(((112, 106), (118, 118), (124, 130))):
        e.append(_path(f"M{x1},76 L{x2},60",
                       _sty(d=t0 + 0.5 + i * 0.06, dd=0.5), sw=1.6))
        e.append(_circle(x2, 58, 2.2, _sty(fd=t0 + 1.2 + i * 0.05, fo=0.9),
                         sw=0, stroke="none", fill=MAD, cls="f"))
    # wing hatch
    e.append(_path("M114,132 Q128,142 140,134", _sty(d=t0 + 0.6, dd=0.6),
                   sw=1.4, stroke=MAD))
    e.append(_path("M112,152 Q128,162 142,154", _sty(d=t0 + 0.66, dd=0.6),
                   sw=1.4, stroke=MAD))
    e.append(_path("M112,172 Q128,182 140,174", _sty(d=t0 + 0.72, dd=0.6),
                   sw=1.4, stroke=MAD))
    # legs
    e.append(_path("M124,210 L122,232", _sty(d=t0 + 0.8, dd=0.5), sw=2.0))
    e.append(_path("M136,212 L136,234", _sty(d=t0 + 0.86, dd=0.5), sw=2.0))
    return ('<svg viewBox="0 0 260 250" xmlns="http://www.w3.org/2000/svg">'
            + "".join(e) + "</svg>")


def _corner_svg():
    """Corner ornament (authored for top-left; the other three are CSS
    rotations): archimedean spiral + three petals + madder dots."""
    cx = cy = 44
    pts = []
    for k in range(0, 55):                                   # 810° spiral
        ang = k * 15
        r = 4 + ang * 0.045
        a = math.radians(ang)
        pts.append(f"{cx + r * math.cos(a):.1f},{cy + r * math.sin(a):.1f}")
    spiral = "M" + " L".join(pts)
    e = [_path(spiral, _sty(dd=1.0), sw=2.2, stroke=TER)]
    for i, ang in enumerate((20, 45, 70)):                   # petals point inward
        a = math.radians(ang)
        tip = f"{cx + 66 * math.cos(a):.1f},{cy + 66 * math.sin(a):.1f}"
        b1a, b2a = math.radians(ang - 16), math.radians(ang + 16)
        c1a, c2a = math.radians(ang - 8), math.radians(ang + 8)
        b1 = f"{cx + 14 * math.cos(b1a):.1f},{cy + 14 * math.sin(b1a):.1f}"
        b2 = f"{cx + 14 * math.cos(b2a):.1f},{cy + 14 * math.sin(b2a):.1f}"
        c1 = f"{cx + 47 * math.cos(c1a):.1f},{cy + 47 * math.sin(c1a):.1f}"
        c2 = f"{cx + 47 * math.cos(c2a):.1f},{cy + 47 * math.sin(c2a):.1f}"
        e.append(_path(f"M{b1} Q{c1} {tip} Q{c2} {b2}",
                       _sty(d=0.25 + i * 0.08, dd=0.8, fd=1.4 + i * 0.06, fo=0.2),
                       sw=2.0, fill=SAF, cls="d f"))
        dx = f"{cx + 78 * math.cos(a):.1f}"
        dy = f"{cy + 78 * math.sin(a):.1f}"
        e.append(_circle(float(dx), float(dy), 2.6,
                         _sty(fd=1.6 + i * 0.06, fo=0.9),
                         sw=0, stroke="none", fill=MAD, cls="f"))
    return ('<svg viewBox="0 0 130 130" xmlns="http://www.w3.org/2000/svg">'
            + "".join(e) + "</svg>")


def _frame_edges():
    """Double border frame drawn hand-around-the-page: each edge is its own
    stretched SVG line (x-stretch never distorts a horizontal stroke, so
    vector-effect isn't needed). Outer ink pass then inner terracotta pass."""
    # (cls, line, viewBox, delay) — line coords ordered to set draw direction
    edges = [
        ("sp-f-t", '<line pathLength="1" class="d" x1="0" y1="4" x2="100" y2="4"',
         "0 0 100 8", 0.20),
        ("sp-f-r", '<line pathLength="1" class="d" x1="4" y1="0" x2="4" y2="100"',
         "0 0 8 100", 0.55),
        ("sp-f-b", '<line pathLength="1" class="d" x1="100" y1="4" x2="0" y2="4"',
         "0 0 100 8", 0.90),
        ("sp-f-l", '<line pathLength="1" class="d" x1="4" y1="100" x2="4" y2="0"',
         "0 0 8 100", 1.25),
    ]
    out = []
    for cls, line, vb, t in edges:
        for ring, (stroke, sw, dt) in enumerate(
                ((INK, 2.5, 0.0), (TER, 1.4, 0.30))):
            out.append(
                f'<svg class="sp-frame {cls} sp-ring-{ring}" viewBox="{vb}" '
                f'preserveAspectRatio="none" xmlns="http://www.w3.org/2000/svg">'
                f'{line} stroke="{stroke}" stroke-width="{sw}" '
                f'style="{_sty(d=t + dt, dd=0.55)}"/></svg>')
    return "".join(out)


# ── assembled scene ───────────────────────────────────────────────────
_CORNER = _corner_svg()

SPLASH_HTML = (
    '<div class="splash-stage">'
    '<div class="sp-grain"></div>'
    '<div class="sp-vignette"></div>'
    + _frame_edges()
    + '<div class="sp-strip sp-strip-t"></div>'
    '<div class="sp-strip sp-strip-b"></div>'
    '<div class="sp-strip sp-strip-l"></div>'
    '<div class="sp-strip sp-strip-r"></div>'
    f'<div class="sp-corner sp-c-tl">{_CORNER}</div>'
    f'<div class="sp-corner sp-c-tr">{_CORNER}</div>'
    f'<div class="sp-corner sp-c-br">{_CORNER}</div>'
    f'<div class="sp-corner sp-c-bl">{_CORNER}</div>'
    f'<div class="sp-mandala">{_mandala_svg()}</div>'
    f'<div class="sp-motif sp-fish-l"><div class="sp-float">{_fish_svg(1.9)}</div></div>'
    f'<div class="sp-motif sp-fish-r"><div class="sp-float">{_fish_svg(2.05)}</div></div>'
    f'<div class="sp-motif sp-lotus"><div class="sp-float">{_lotus_svg(2.2)}</div></div>'
    f'<div class="sp-motif sp-peacock"><div class="sp-float">{_peacock_svg(2.35)}</div></div>'
    '<div class="sp-center">'
    '<div class="sp-title">नालंदा</div>'
    '<div class="sp-sub">Grievance Intelligence System</div>'
    '<div class="sp-rule"></div>'
    '<div class="sp-kicker">हिलसा अनुमंडल · Government of Bihar · Est. 427 CE</div>'
    '</div>'
    '</div>'
)

# Film-grain tile — same feTurbulence data-URI technique as the dashboard.
_GRAIN = ("url(\"data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' "
          "width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence "
          "type='fractalNoise' baseFrequency='.85' numOctaves='3' "
          "stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' "
          "height='100%25' filter='url(%23n)'/%3E%3C/svg%3E\")")

SPLASH_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Instrument+Serif:ital,wght@0,400;1,400&family=Noto+Serif+Devanagari:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

/* ── Heritage tokens — scoped to .stApp; the dashboard's :root tokens
      are never loaded while this stylesheet exists (gate precedes them) */
.stApp{
  --sp-parchment:#F4E7C8; --sp-parchment-2:#EAD7AE;
  --sp-ink:#33210F; --sp-terracotta:#C3532D; --sp-madder:#93312B;
  --sp-indigo:#2C4B8F; --sp-saffron:#E9A21B; --sp-leaf:#4F7C3A;
  --sp-e:cubic-bezier(.16,1,.3,1);
}

/* ── Chrome: full-bleed, no Streamlit furniture, no scroll */
*,*::before,*::after{box-sizing:border-box}
html,body{overflow:hidden}
header,footer,#MainMenu,.stDeployButton,
[data-testid="stHeader"],[data-testid="stToolbar"],[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stSidebar"],[data-testid="collapsedControl"],
[data-testid="stSidebarCollapsedControl"]{display:none!important}
.stApp{background:#0A0805!important;overflow:hidden}
.main .block-container{padding:0!important;max-width:100%!important}
[data-testid="stAppViewContainer"]{overflow:hidden!important}

/* ── Stage: parchment blooms out of the dark first frame */
.splash-stage{
  position:fixed;inset:0;z-index:100;overflow:hidden;
  background:
    radial-gradient(90% 70% at 50% 38%, #F7EDD6 0%, var(--sp-parchment) 55%, var(--sp-parchment-2) 100%);
  animation:sp-fade-in .9s ease-out both;
}
.sp-grain{position:absolute;inset:0;opacity:.16;mix-blend-mode:multiply;
  background-image:GRAIN_URL}
.sp-vignette{position:absolute;inset:0;pointer-events:none;
  background:radial-gradient(120% 95% at 50% 42%, transparent 52%,
    rgba(147,49,43,.07) 76%, rgba(51,33,15,.18) 100%)}

/* ── Universal draw / fill choreography.
      Every shape carries pathLength="1"; delays arrive as inline --d/--fd. */
.splash-stage .d{
  stroke-dasharray:1;stroke-dashoffset:1;
  animation:sp-draw var(--dd,1.1s) var(--sp-e) var(--d,0s) forwards;
}
.splash-stage .f{
  fill-opacity:0;
  animation:sp-fill .8s var(--sp-e) var(--fd,2.8s) forwards;
}
.splash-stage .d.f{
  fill-opacity:0;stroke-dasharray:1;stroke-dashoffset:1;
  animation:sp-draw var(--dd,1.1s) var(--sp-e) var(--d,0s) forwards,
            sp-fill .8s var(--sp-e) var(--fd,2.8s) forwards;
}

/* ── Double border frame (each edge = one stretched SVG line) */
.sp-frame{position:absolute;display:block}
.sp-f-t.sp-ring-0{top:16px;left:16px;right:16px;height:8px;width:auto}
.sp-f-b.sp-ring-0{bottom:16px;left:16px;right:16px;height:8px;width:auto}
.sp-f-l.sp-ring-0{left:16px;top:16px;bottom:16px;width:8px;height:auto}
.sp-f-r.sp-ring-0{right:16px;top:16px;bottom:16px;width:8px;height:auto}
.sp-f-t.sp-ring-1{top:26px;left:26px;right:26px;height:8px;width:auto}
.sp-f-b.sp-ring-1{bottom:26px;left:26px;right:26px;height:8px;width:auto}
.sp-f-l.sp-ring-1{left:26px;top:26px;bottom:26px;width:8px;height:auto}
.sp-f-r.sp-ring-1{right:26px;top:26px;bottom:26px;width:8px;height:auto}

/* ── Tick strips just inside the frame */
.sp-strip{position:absolute;opacity:0;animation:sp-strip-in .8s var(--sp-e) 1.0s forwards}
.sp-strip-t,.sp-strip-b{left:40px;right:40px;height:5px;transform:scaleX(0);
  background:repeating-linear-gradient(90deg,rgba(195,83,45,.45) 0 2px,transparent 2px 9px)}
.sp-strip-t{top:38px}.sp-strip-b{bottom:38px}
.sp-strip-l,.sp-strip-r{top:40px;bottom:40px;width:5px;transform:scaleY(0);
  animation-name:sp-strip-in-y;animation-delay:1.15s;
  background:repeating-linear-gradient(0deg,rgba(44,75,143,.40) 0 2px,transparent 2px 9px)}
.sp-strip-l{left:38px}.sp-strip-r{right:38px}

/* ── Corner ornaments (authored TL; rotated for the rest) */
.sp-corner{position:absolute;width:min(13vmin,128px);height:min(13vmin,128px)}
.sp-c-tl{top:30px;left:30px}
.sp-c-tr{top:30px;right:30px;transform:rotate(90deg)}
.sp-c-br{bottom:30px;right:30px;transform:rotate(180deg)}
.sp-c-bl{bottom:30px;left:30px;transform:rotate(270deg)}
.sp-c-tl svg path,.sp-c-tl svg circle{--co:0s}
.sp-c-tr svg path,.sp-c-tr svg circle{--co:.15s}
.sp-c-br svg path,.sp-c-br svg circle{--co:.30s}
.sp-c-bl svg path,.sp-c-bl svg circle{--co:.45s}
.sp-corner .d{animation-delay:calc(var(--d,0s) + 1.3s + var(--co))}
.sp-corner .f{animation-delay:calc(var(--fd,1.4s) + 1.3s + var(--co))}
.sp-corner .d.f{animation-delay:calc(var(--d,0s) + 1.3s + var(--co)),
                calc(var(--fd,1.4s) + 1.3s + var(--co))}

/* ── Central sun-mandala — slow ceremonial rotation once drawn */
.sp-mandala{
  position:absolute;left:50%;top:47%;
  width:min(74vmin,640px);height:min(74vmin,640px);
  margin:calc(min(74vmin,640px)/-2) 0 0 calc(min(74vmin,640px)/-2);
  z-index:3;opacity:0;animation:sp-appear .8s ease-out .85s forwards;
}
.sp-mandala svg{width:100%;height:100%;animation:sp-rotate 150s linear 3s infinite}

/* ── Side motifs */
.sp-motif{position:absolute;z-index:4;opacity:0;animation:sp-appear .7s ease-out forwards}
.sp-float{width:100%;height:100%}
.sp-fish-l{left:7vw;bottom:11vh;width:min(17vmin,190px);animation-delay:1.8s}
.sp-fish-l .sp-float{animation:sp-float 7s ease-in-out 3.4s infinite alternate}
.sp-fish-r{right:7vw;bottom:11vh;width:min(17vmin,190px);transform:scaleX(-1);animation-delay:1.95s}
.sp-fish-r .sp-float{animation:sp-float 7s ease-in-out 3.9s infinite alternate}
.sp-lotus{left:5vw;top:32vh;width:min(16vmin,175px);animation-delay:2.1s}
.sp-lotus .sp-float{animation:sp-float 8s ease-in-out 4.2s infinite alternate}
.sp-peacock{right:4vw;top:28vh;width:min(21vmin,240px);animation-delay:2.25s}
.sp-peacock .sp-float{animation:sp-float 9s ease-in-out 4.6s infinite alternate}
.sp-motif svg{width:100%;height:auto;display:block}

/* ── Title stack (halo ::before lifts it off the mandala lines) */
.sp-center{
  position:absolute;inset:0;z-index:6;display:flex;flex-direction:column;
  align-items:center;justify-content:center;text-align:center;
  pointer-events:none;padding-bottom:6vh;
}
.sp-center::before{
  content:"";position:absolute;left:50%;top:47%;width:min(66vmin,580px);
  height:min(56vmin,470px);transform:translate(-50%,-50%);z-index:-1;
  border-radius:50%;
  background:rgba(247,237,214,.85);
  /* static backdrop blur (never animated — the no-live-filters rule bans
     animated filters); the mask feathers both the fill and the blur edge
     so the mandala lines soften gradually instead of cutting off. */
  backdrop-filter:blur(9px);-webkit-backdrop-filter:blur(9px);
  -webkit-mask:radial-gradient(closest-side,#000 45%,transparent 98%);
  mask:radial-gradient(closest-side,#000 45%,transparent 98%);
}
.sp-title{
  font-family:'Noto Serif Devanagari',serif;font-weight:700;
  font-size:clamp(64px,13vmin,132px);line-height:1.15;color:transparent;
  /* background-clip:text paints glyphs only where the background box is;
     the anusvara of नालंदा ascends above the line box, so without this
     headroom the dot renders transparent. padding+negative margin keep
     layout identical while extending the paint area over the ascender. */
  padding:.25em .15em;margin:-.25em -.15em;
  background:linear-gradient(105deg,
    var(--sp-ink) 0%, var(--sp-ink) 34%, var(--sp-madder) 46%,
    var(--sp-terracotta) 52%, var(--sp-indigo) 60%, var(--sp-ink) 72%, var(--sp-ink) 100%);
  background-size:300% 100%;background-position:100% 0;
  -webkit-background-clip:text;background-clip:text;
  opacity:0;
  animation:sp-fade-up .9s var(--sp-e) 2.8s forwards,
            sp-sheen-text 7s linear 4.2s infinite;
}
.sp-sub{
  font-family:'Instrument Serif',serif;font-size:clamp(17px,2.6vmin,26px);
  letter-spacing:.34em;text-indent:.34em;text-transform:uppercase;
  color:var(--sp-madder);margin-top:1.2vmin;
  opacity:0;animation:sp-fade-up .8s var(--sp-e) 3.3s forwards;
}
.sp-rule{
  width:0;height:2px;background:var(--sp-saffron);margin:2.2vmin auto 0;
  box-shadow:0 0 12px rgba(233,162,27,.5);
  animation:sp-rule .6s var(--sp-e) 3.6s forwards;
}
.sp-kicker{
  font-family:'Plus Jakarta Sans',sans-serif;font-weight:500;
  font-size:clamp(12px,1.7vmin,15px);letter-spacing:.18em;
  color:rgba(51,33,15,.72);margin-top:2vmin;
  opacity:0;animation:sp-fade-up .8s var(--sp-e) 3.8s forwards;
}

/* ── Enter CTA (real st.button; selector rationale in header comment) */
/* full-width flex centring: immune to whatever width Streamlit gives the
   wrapper (left:50% + translateX only centres a shrink-to-fit box) */
div.stButton{position:fixed;left:0;right:0;bottom:9vh;
  display:flex;justify-content:center;z-index:130;pointer-events:none}
div.stButton>button{pointer-events:auto}
div.stButton>button{
  position:relative;overflow:hidden;cursor:pointer;
  font-family:'Instrument Serif','Noto Serif Devanagari',serif;
  font-size:1.14rem;letter-spacing:.05em;
  color:#F7EDD6;background:var(--sp-ink);
  border:1px solid var(--sp-ink);border-radius:999px;
  padding:.85rem 2.4rem;
  box-shadow:0 0 0 3px var(--sp-parchment),0 0 0 5px var(--sp-ink),
             0 10px 28px rgba(51,33,15,.28);
  transition:background .25s ease,transform .2s ease;
  opacity:0;
  animation:sp-fade-up .8s var(--sp-e) 4.1s forwards,
            sp-cta-glow 3.6s ease-in-out 5.2s infinite;
}
div.stButton>button:hover{background:var(--sp-madder);border-color:var(--sp-madder)}
div.stButton>button:active{transform:scale(.97)}
div.stButton>button:focus-visible{outline:3px solid var(--sp-saffron);outline-offset:4px}
div.stButton>button p{color:#F7EDD6;font-size:inherit;font-family:inherit;margin:0}
div.stButton>button::before{
  content:"";position:absolute;inset:0;pointer-events:none;
  background:linear-gradient(110deg,transparent 30%,rgba(255,255,255,.22) 50%,transparent 70%);
  transform:translateX(-130%);
  animation:sp-sheen 4.2s ease 5.6s infinite;
}
/* dim the painting during the click round-trip */
body:has(div.stButton>button:active) .splash-stage{opacity:.35;transition:opacity .3s}

/* ── Keyframes */
@keyframes sp-draw{to{stroke-dashoffset:0}}
@keyframes sp-fill{to{fill-opacity:var(--fo,1)}}
@keyframes sp-fade-in{from{opacity:0}to{opacity:1}}
@keyframes sp-appear{from{opacity:0}to{opacity:1}}
@keyframes sp-fade-up{from{opacity:0;transform:translateY(18px)}
                      to{opacity:1;transform:none}}
@keyframes sp-strip-in{from{opacity:0;transform:scaleX(0)}
                       to{opacity:.85;transform:scaleX(1)}}
@keyframes sp-strip-in-y{from{opacity:0;transform:scaleY(0)}
                         to{opacity:.85;transform:scaleY(1)}}
@keyframes sp-rotate{to{transform:rotate(360deg)}}
@keyframes sp-float{from{transform:translateY(-5px)}to{transform:translateY(6px)}}
@keyframes sp-rule{to{width:min(22vmin,190px)}}
@keyframes sp-sheen-text{from{background-position:100% 0}to{background-position:-200% 0}}
@keyframes sp-sheen{0%{transform:translateX(-130%)}55%,100%{transform:translateX(130%)}}
@keyframes sp-cta-glow{
  0%,100%{box-shadow:0 0 0 3px var(--sp-parchment),0 0 0 5px var(--sp-ink),
          0 10px 28px rgba(51,33,15,.28)}
  50%{box-shadow:0 0 0 3px var(--sp-parchment),0 0 0 5px var(--sp-ink),
      0 10px 28px rgba(51,33,15,.28),0 0 30px rgba(233,162,27,.5)}
}

/* ── Smaller viewports: retire the side motifs, let the mandala breathe */
@media (max-width:1100px){
  .sp-peacock,.sp-lotus{display:none}
}
@media (max-width:780px){
  .sp-fish-l,.sp-fish-r,.sp-strip{display:none}
  .sp-mandala{width:92vmin;height:92vmin;margin:-46vmin 0 0 -46vmin}
}

/* ── Reduced motion: the finished painting, instantly */
@media (prefers-reduced-motion:reduce){
  .splash-stage *,.splash-stage,div.stButton>button{
    animation-duration:.01ms!important;animation-delay:0ms!important;
    animation-iteration-count:1!important;
  }
  .sp-mandala svg,.sp-float,div.stButton>button::before{animation:none!important}
  .splash-stage .d,.splash-stage .d.f{stroke-dashoffset:0!important}
  .splash-stage .f,.splash-stage .d.f{fill-opacity:var(--fo,1)!important}
}
</style>
""".replace("GRAIN_URL", _GRAIN)


def corner_svg_static():
    """Corner ornament for reuse in the dashboard chrome. The draw/fill
    choreography classes stay in the markup; the dashboard's
    .ms-motif-static rules render them as the finished painting."""
    return _CORNER


def fish_svg_static():
    """Madhubani fish for the dashboard sidebar watermark (see above)."""
    return _fish_svg(0)


def _enter():
    st.session_state["entered"] = True
    st.session_state["just_entered"] = True


def render_splash():
    """Render the full-viewport Threshold scene and the Enter CTA."""
    st.markdown(SPLASH_CSS, unsafe_allow_html=True)
    st.markdown(SPLASH_HTML, unsafe_allow_html=True)
    st.button("प्रवेश करें · Enter Console", key="enter_console", on_click=_enter)
