"""Builds dark.svg / light.svg — the animated terminal-window GitHub profile banner."""
import numpy as np
from scipy.optimize import linear_sum_assignment
from lib import (
    load_portrait_crop, to_processed_gray, to_processed_gray_inverted, foreground_mask,
    floyd_steinberg_serpentine, dots_to_runs, runs_to_path_d,
    sample_points_from_glyph, evenness_metric, GRID_W, GRID_H,
)

# ---------------------------------------------------------------- layout ----
W, H = 1180, 610
TITLEBAR_H = 40
PAD = 24
PANEL_Y = TITLEBAR_H + PAD
PANEL_W = 430
PANEL_H = round(PANEL_W * GRID_H / GRID_W)  # 487, keeps 300:340 aspect
PANEL_X = PAD
CELL = PANEL_W / GRID_W
DOT = CELL * 0.85

INFO_X = PANEL_X + PANEL_W + PAD
INFO_W = W - PAD - INFO_X
INFO_TOP = PANEL_Y

N_INTRO_GROUPS = 60
N_TRAVELERS = 900
LOOP_DUR = 14.2
INTRO_DUR = 3.2

# keyTimes (fractions of the 14.2s loop) for the four-transition cycle
T_PORTRAIT, T_TR1, T_LOGO1, T_TR2, T_LOGO2, T_TR3, T_LOGO3, T_TR4 = (
    0.0, 3.0, 4.3, 6.3, 7.6, 9.6, 10.9, 12.9
)
KEYTIMES = [t / LOOP_DUR for t in (0.0, T_TR1, T_LOGO1, T_TR2, T_LOGO2, T_TR3, T_LOGO3, T_TR4, LOOP_DUR)]

PALETTE = {
    "dark": dict(bg="#0A101F", panel="#0D1424", border="#1E2A44", chrome="#22D3EE",
                 chrome_dim="#0891B2", portrait="#A78BFA", accent="#10B981",
                 text="#E2E8F0", text_dim="#64748B", empty=None),
    "light": dict(bg="#F8FAFC", panel="#FFFFFF", border="#CBD5E1", chrome="#0891B2",
                  chrome_dim="#22D3EE", portrait="#7C3AED", accent="#059669",
                  text="#0F172A", text_dim="#64748B", empty=None),
}

RNG = np.random.default_rng(42)
GROUP_MAP = RNG.integers(0, N_INTRO_GROUPS, size=(GRID_H, GRID_W))


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def build_portrait_dot_layer(dot_mask, color, layer_id, seed=7):
    """One <g> per intro-fade group; each fades in once via SMIL, staggered so
    dots appear scattered across the whole portrait rather than region by
    region. Groups are assigned per RUN (not per cell) so long solid runs
    (e.g. the sweater) stay as single path commands instead of exploding
    into one square per pixel -- keeps file size sane without losing scatter,
    since dithered/detailed areas already have short runs of their own."""
    runs = dots_to_runs(dot_mask)
    rng = np.random.default_rng(seed)
    run_groups = rng.integers(0, N_INTRO_GROUPS, size=len(runs))

    # verify scatter: per-group centroid spread (cell-weighted) vs random reference
    positions = [[] for _ in range(N_INTRO_GROUPS)]
    for (y, x0, length), g in zip(runs, run_groups):
        for i in range(length):
            positions[g].append((x0 + i, y))
    positions = [np.array(p, dtype=float) for p in positions if p]
    even = evenness_metric(None, GRID_W, GRID_H, positions)

    parts = [f'<g id="{layer_id}" fill="{color}" shape-rendering="crispEdges">']
    stagger = (INTRO_DUR - 2.0) / (N_INTRO_GROUPS - 1)
    for g in range(N_INTRO_GROUPS):
        group_runs = [r for r, gg in zip(runs, run_groups) if gg == g]
        if not group_runs:
            continue
        d = runs_to_path_d(group_runs, CELL, DOT, PANEL_X, PANEL_Y)
        begin = g * stagger
        parts.append(
            f'<g opacity="0">'
            f'<animate attributeName="opacity" values="0;1" dur="2s" '
            f'begin="{begin:.3f}s" fill="freeze" calcMode="spline" '
            f'keySplines="0.2 0 0.2 1"/>'
            f'<path d="{d}"/></g>'
        )
    parts.append("</g>")
    return "".join(parts), even


def build_portrait_wrapper(inner_svg, logo1_centroid):
    cx = PANEL_X + PANEL_W / 2
    cy = PANEL_Y + PANEL_H / 2
    dx = (logo1_centroid[0] - cx) * 0.42
    dy = (logo1_centroid[1] - cy) * 0.42
    op_vals = ";".join(["1", "0", "0", "0", "0", "0", "0", "0", "1"])
    tr_vals = ";".join([f"{v[0]:.1f},{v[1]:.1f}" for v in
                         [(0, 0), (dx, dy), (dx, dy), (dx, dy), (dx, dy), (dx, dy), (dx, dy), (dx, dy), (0, 0)]])
    kt = ";".join(f"{t:.5f}" for t in KEYTIMES)
    return (
        f'<g id="portrait-loop">'
        f'<animateTransform attributeName="transform" type="translate" '
        f'values="{tr_vals}" keyTimes="{kt}" dur="{LOOP_DUR}s" repeatCount="indefinite"/>'
        f'<animate attributeName="opacity" values="{op_vals}" keyTimes="{kt}" '
        f'dur="{LOOP_DUR}s" repeatCount="indefinite"/>'
        f'{inner_svg}</g>'
    )


def build_logo_points(n, box_w, box_h, ox, oy):
    pts = {}
    for i, name in enumerate(["python", "react", "github"]):
        p = sample_points_from_glyph(f"logos/{name}.png", n, box=1.0, seed=i)
        p = p * [box_w, box_h] + [ox, oy]
        pts[name] = p
    return pts


def match_sequence(p_python, p_react, p_github):
    def hungarian(a, b):
        cost = np.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=2)
        r, c = linear_sum_assignment(cost)
        return b[c]
    logo1 = p_python
    logo2 = hungarian(logo1, p_react)
    logo3 = hungarian(logo2, p_github)
    return logo1, logo2, logo3


def build_travelers_layer(logo1, logo2, logo3, color):
    kt = ";".join(f"{t:.5f}" for t in KEYTIMES)
    op_vals = "0;0;1;1;1;1;1;1;0"
    parts = [
        f'<g id="travelers" fill="{color}" opacity="0">',
        f'<animate attributeName="opacity" values="{op_vals}" keyTimes="{kt}" '
        f'dur="{LOOP_DUR}s" repeatCount="indefinite"/>',
    ]
    r = 1.15
    for i in range(len(logo1)):
        x1, y1 = logo1[i]
        x2, y2 = logo2[i]
        x3, y3 = logo3[i]
        cxs = ";".join(f"{v:.1f}" for v in [x1, x1, x1, x1, x2, x2, x3, x3, x3])
        cys = ";".join(f"{v:.1f}" for v in [y1, y1, y1, y1, y2, y2, y3, y3, y3])
        parts.append(
            f'<circle r="{r:.2f}">'
            f'<animate attributeName="cx" values="{cxs}" keyTimes="{kt}" '
            f'dur="{LOOP_DUR}s" repeatCount="indefinite"/>'
            f'<animate attributeName="cy" values="{cys}" keyTimes="{kt}" '
            f'dur="{LOOP_DUR}s" repeatCount="indefinite"/>'
            f'</circle>'
        )
    parts.append("</g>")
    return "".join(parts)


# --------------------------------------------------------- info panel ------
INFO_ROWS = [
    ("Subject", "Piyush Agarwal"),
    ("Role", "AI/ML & Full-Stack Dev"),
    ("Origin", "Jaipur, India"),
    ("Education", "B.Tech, Manipal Univ. Jaipur"),
    ("Status", "Learning + Building + Shipping"),
    ("ToolChain", "VS Code, Git, Figma, Canva"),
    ("Core.Lang", "C++, C, Python"),
    ("Core.Frontend", "React, TailwindCSS, HTML5/CSS3"),
    ("Core.Backend", "Node.js, FastAPI, Spring"),
    ("Core.Database", "MongoDB, MySQL, Firebase"),
    ("Core.Infra", "Git, GitHub, Firebase, Vercel"),
    ("Grid.Mail", "piyushagarwal5525@gmail.com"),
    ("Grid.Portfolio", "coming soon"),
    ("Grid.LinkedIn", "piyush-agarwal-97b731316"),
    ("Grid.GitHub", "Piyush5525"),
]
ROW_FONT = 14
ROW_SPACING = 23
LABEL_CHARW = ROW_FONT * 0.60
VALUE_CHARW = ROW_FONT * 0.60


def build_info_panel(pal):
    header_y = INFO_TOP + 6
    rows_top = header_y + 34
    parts = []
    parts.append(
        f'<text x="{INFO_X}" y="{header_y+10}" font-size="13" font-weight="600" '
        f'letter-spacing="2" fill="{pal["chrome"]}" font-family="ui-monospace,Menlo,Consolas,monospace">SYSTEM.INFO</text>'
    )
    # LIVE badge (pulsing)
    live_x = INFO_X + INFO_W - 74
    parts.append(
        f'<circle cx="{live_x}" cy="{header_y+6}" r="4" fill="#EF4444">'
        f'<animate attributeName="opacity" values="1;0.25;1" dur="1.6s" repeatCount="indefinite"/>'
        f'</circle>'
        f'<text x="{live_x+10}" y="{header_y+10}" font-size="12" font-weight="600" '
        f'letter-spacing="1.5" fill="#EF4444" font-family="ui-monospace,Menlo,Consolas,monospace">LIVE</text>'
    )
    # handle pill
    pill_y = header_y + 24
    pill_text = "@Piyush5525"
    pill_w = len(pill_text) * 14 * 0.62 + 24
    parts.append(
        f'<rect x="{INFO_X}" y="{pill_y}" width="{pill_w:.1f}" height="26" rx="13" '
        f'fill="{pal["accent"]}" opacity="0.15" stroke="{pal["accent"]}" stroke-width="1"/>'
        f'<text x="{INFO_X+12}" y="{pill_y+17.5}" font-size="14" fill="{pal["accent"]}" '
        f'font-family="ui-monospace,Menlo,Consolas,monospace">{pill_text}</text>'
    )
    rows_top = pill_y + 44
    right_edge = INFO_X + INFO_W
    for i, (label, value) in enumerate(INFO_ROWS):
        y = rows_top + i * ROW_SPACING
        label_w = len(label) * LABEL_CHARW
        value_w = len(value) * VALUE_CHARW
        leader_x0 = INFO_X + label_w + 6
        leader_x1 = right_edge - value_w - 6
        parts.append(
            f'<text x="{INFO_X}" y="{y}" font-size="{ROW_FONT}" fill="{pal["text_dim"]}" '
            f'font-family="ui-monospace,Menlo,Consolas,monospace" '
            f'textLength="{label_w:.1f}" lengthAdjust="spacingAndGlyphs">{esc(label)}</text>'
        )
        if leader_x1 > leader_x0:
            parts.append(
                f'<line x1="{leader_x0:.1f}" y1="{y-4.5}" x2="{leader_x1:.1f}" y2="{y-4.5}" '
                f'stroke="{pal["border"]}" stroke-width="1" stroke-dasharray="1.5,3"/>'
            )
        parts.append(
            f'<text x="{right_edge}" y="{y}" font-size="{ROW_FONT}" fill="{pal["text"]}" '
            f'text-anchor="end" font-family="ui-monospace,Menlo,Consolas,monospace" '
            f'textLength="{value_w:.1f}" lengthAdjust="spacingAndGlyphs">{esc(value)}</text>'
        )
    return "".join(parts)


def build_titlebar(pal):
    parts = [f'<rect x="0" y="0" width="{W}" height="{TITLEBAR_H}" fill="{pal["panel"]}"/>']
    for i, c in enumerate(["#EF4444", "#F59E0B", "#10B981"]):
        parts.append(f'<circle cx="{20+i*18}" cy="{TITLEBAR_H/2}" r="5" fill="{c}"/>')
    parts.append(
        f'<text x="{W/2}" y="{TITLEBAR_H/2+5}" text-anchor="middle" font-size="13" '
        f'fill="{pal["text_dim"]}" font-family="ui-monospace,Menlo,Consolas,monospace">profile.sh --live</text>'
    )
    parts.append(f'<line x1="0" y1="{TITLEBAR_H}" x2="{W}" y2="{TITLEBAR_H}" stroke="{pal["border"]}" stroke-width="1"/>')
    return "".join(parts)


def build_svg(theme, dot_mask, logo1, logo2, logo3):
    pal = PALETTE[theme]
    portrait_inner, even = build_portrait_dot_layer(dot_mask, pal["portrait"], f"portrait-dots-{theme}")
    logo1_centroid = logo1.mean(axis=0)
    portrait_group = build_portrait_wrapper(portrait_inner, logo1_centroid)
    travelers = build_travelers_layer(logo1, logo2, logo3, pal["chrome"])
    titlebar = build_titlebar(pal)
    info = build_info_panel(pal)

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}">
<rect width="{W}" height="{H}" fill="{pal["bg"]}"/>
{titlebar}
<rect x="{PANEL_X}" y="{PANEL_Y}" width="{PANEL_W}" height="{PANEL_H}" fill="{pal["panel"]}" stroke="{pal["border"]}" stroke-width="1" rx="6"/>
<text x="{PANEL_X}" y="{PANEL_Y-10}" font-size="12" letter-spacing="2" fill="{pal["chrome_dim"]}" font-family="ui-monospace,Menlo,Consolas,monospace">VISUAL.MAP</text>
<clipPath id="clip-{theme}"><rect x="{PANEL_X}" y="{PANEL_Y}" width="{PANEL_W}" height="{PANEL_H}" rx="6"/></clipPath>
<g clip-path="url(#clip-{theme})">
{portrait_group}
{travelers}
</g>
{info}
</svg>'''
    return svg, even


def main():
    crop = load_portrait_crop("source_photo.png")
    gray = to_processed_gray(crop)
    dots_full = floyd_steinberg_serpentine(gray)
    mask_fg = foreground_mask(crop)
    gray_inv = to_processed_gray_inverted(crop)
    dots_dark = floyd_steinberg_serpentine(gray_inv) & mask_fg

    logo_pts = build_logo_points(N_TRAVELERS, PANEL_W * 0.7, PANEL_H * 0.7,
                                  PANEL_X + PANEL_W * 0.15, PANEL_Y + PANEL_H * 0.15)
    logo1, logo2, logo3 = match_sequence(logo_pts["python"], logo_pts["react"], logo_pts["github"])

    for theme, dots in [("light", dots_full), ("dark", dots_dark)]:
        svg, even = build_svg(theme, dots, logo1, logo2, logo3)
        out_path = f"../{theme}.svg"
        with open(out_path, "w") as f:
            f.write(svg)
        size_kb = len(svg.encode()) / 1024
        print(f"{theme}: {size_kb:.0f} KB, ink cells {dots.sum()}, intro evenness {even:.3f}")


if __name__ == "__main__":
    main()
