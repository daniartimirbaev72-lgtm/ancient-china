

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Ellipse
from matplotlib.colors import LinearSegmentedColormap

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
BG_TOP     = "#0d0908"
BG_BOTTOM  = "#1c1210"
SCALE_DARK = "#3d0f14"
SCALE_MID  = "#7c1f2b"
SCALE_LIT  = "#c9505a"
GOLD       = "#c9a24b"
GOLD_LIGHT = "#e8c77a"
CREAM      = "#e7dcc0"
FLAME      = "#e8935a"

SCALES_PER_ROW = 30
BODY_MIN_W, BODY_MAX_W = 12, 36
FIG_W, FIG_H, DPI = 15, 8, 220
rng = np.random.default_rng(7)

# ----------------------------------------------------------------------
# Bezier helpers
# ----------------------------------------------------------------------
def cubic_bezier(p0, p1, p2, p3, n=60):
    t = np.linspace(0, 1, n)[:, None]
    return ((1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 +
             3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3)

def build_spine(segments, n_per_seg=70):
    pts = []
    for seg in segments:
        p0, p1, p2, p3 = [np.array(p, dtype=float) for p in seg]
        pts.append(cubic_bezier(p0, p1, p2, p3, n_per_seg))
    return np.vstack(pts)

def tangents_normals(curve):
    d = np.gradient(curve, axis=0)
    ang = np.arctan2(d[:, 1], d[:, 0])
    normal = np.stack([-np.sin(ang), np.cos(ang)], axis=1)
    return ang, normal

def rot(pts, angle, origin):
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])
    return (np.array(pts, dtype=float) - origin) @ R.T + origin

# ----------------------------------------------------------------------
# Spine: tail -> head, a long coiling S-curve
# ----------------------------------------------------------------------
SEGMENTS = [
    ((40, 40),   (110, 130), (70, 200),  (150, 210)),
    ((150, 210), (210, 218), (195, 130), (260, 115)),
    ((260, 115), (310, 103), (300, 190), (365, 195)),
    ((365, 195), (415, 199), (405, 120), (465, 128)),
    ((465, 128), (510, 134), (500, 200), (560, 195)),
    ((560, 195), (610, 191), (600, 115), (660, 128)),
    ((660, 128), (700, 137), (705, 175), (750, 165)),
    ((750, 165), (790, 157), (800, 110), (845, 110)),
]
spine = build_spine(SEGMENTS)
angles, normals = tangents_normals(spine)
n = len(spine)
t_param = np.linspace(0, 1, n)
width = BODY_MIN_W + (BODY_MAX_W - BODY_MIN_W) * (t_param ** 0.75)

def offset_curve(curve, normal, offset):
    return curve + normal * offset[:, None] if np.ndim(offset) else curve + normal * offset

# ----------------------------------------------------------------------
# Figure & night-sky background
# ----------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(FIG_W, FIG_H), dpi=DPI)
ax.set_xlim(0, 1000); ax.set_ylim(-60, 300)
ax.set_aspect("equal"); ax.axis("off")

grad = np.linspace(0, 1, 256).reshape(-1, 1)
bg_cmap = LinearSegmentedColormap.from_list("bg", [BG_BOTTOM, BG_TOP])
ax.imshow(grad, extent=(0, 1000, -60, 300), aspect="auto", cmap=bg_cmap,
          zorder=0, interpolation="bicubic")

moon_c = (900, 245)
for r, a, c in [(78, 0.05, "#3a2c1e"), (58, 0.10, "#5a4630"),
                (42, 0.9, "#f3e6b8"), (34, 1.0, GOLD_LIGHT)]:
    ax.add_patch(Circle(moon_c, r, color=c, alpha=a, zorder=1, linewidth=0))

for cx, cy, w, h in [(120,240,220,30),(300,200,260,26),(520,255,200,24),
                      (150,50,260,22),(600,30,300,26),(700,190,180,20)]:
    for k in range(4):
        ax.add_patch(Ellipse((cx+rng.uniform(-15,15), cy+rng.uniform(-8,8)),
                              w*(1-0.12*k), h*(1-0.15*k),
                              color="#241814", alpha=0.10, zorder=1, linewidth=0))

# ----------------------------------------------------------------------
# Body base silhouette
# ----------------------------------------------------------------------
top_edge = spine + normals * (width/2)[:, None]
bot_edge = spine - normals * (width/2)[:, None]
ax.add_patch(Polygon(np.vstack([top_edge, bot_edge[::-1]]), closed=True,
                      facecolor=SCALE_DARK, edgecolor="none", zorder=2))

# ----------------------------------------------------------------------
# Scale texture: shingled, shaded top->bottom for a rounded, tube-like look
# ----------------------------------------------------------------------
scale_cmap = LinearSegmentedColormap.from_list("scale", [SCALE_DARK, SCALE_MID, SCALE_LIT])
idxs = np.linspace(5, n-5, SCALES_PER_ROW).astype(int)
row_defs = [(-0.30, 0.75, 0.55), (0.0, 0.45, 0.80), (0.30, 0.15, 0.65)]  # (offset, shade_t, size_mul)
for roff, shade_t, size_mul in row_defs:
    for j, i in enumerate(idxs):
        w = width[i]
        center = spine[i] + normals[i] * roff * w
        color = scale_cmap(shade_t)
        e = Ellipse(center, w*0.46*size_mul, w*0.30*size_mul,
                    angle=np.degrees(angles[i]),
                    facecolor=color, edgecolor=SCALE_DARK, linewidth=0.35,
                    zorder=3, alpha=0.97)
        ax.add_patch(e)

sheen_top = spine + normals * (width/2 - 2)[:, None]
ax.plot(sheen_top[:,0], sheen_top[:,1], color=GOLD_LIGHT, linewidth=1.6,
        alpha=0.45, zorder=6, solid_capstyle="round")

# ----------------------------------------------------------------------
# Dorsal spikes
# ----------------------------------------------------------------------
for i in np.linspace(10, n-35, 20).astype(int):
    w = width[i]
    base = spine[i] + normals[i] * (w/2)
    tip  = spine[i] + normals[i] * (w/2 + 6 + w*0.55)
    side = np.array([-normals[i][1], normals[i][0]]) * (w*0.16 + 1.5)
    ax.add_patch(Polygon([base-side, tip, base+side], closed=True,
                          facecolor=GOLD, edgecolor=SCALE_DARK, linewidth=0.4, zorder=7))

# ----------------------------------------------------------------------
# Legs: anchored at troughs of the spine so they hang straight down
# ----------------------------------------------------------------------
dy = np.diff(spine[:, 1])
sgn = np.sign(dy)
minima = np.where((sgn[:-1] < 0) & (sgn[1:] > 0))[0] + 1
minima = [m for m in minima if 20 < m < n-40]

def draw_leg(i, length=70):
    w = width[i]
    p0 = spine[i] - np.array([0, w/2])
    p1 = p0 + np.array([6, -length*0.55])
    p2 = p1 + np.array([-4, -length*0.45])
    xs, ys = [p0[0],p1[0],p2[0]], [p0[1],p1[1],p2[1]]
    ax.plot(xs, ys, color=SCALE_DARK, linewidth=8, solid_capstyle="round", zorder=7, alpha=0.55)
    ax.plot(xs, ys, color=GOLD, linewidth=5, solid_capstyle="round", zorder=8)
    # small gold flame tuft at the shoulder
    tuft = p0 + np.array([-10, 6])
    ax.plot([p0[0], tuft[0]], [p0[1], tuft[1]], color=GOLD, linewidth=3,
            solid_capstyle="round", zorder=8, alpha=0.85)
    for da in (-30, 0, 30):
        rad = np.radians(da - 90)
        claw = p2 + 13*np.array([np.cos(rad), np.sin(rad)])
        ax.plot([p2[0],claw[0]], [p2[1],claw[1]], color=CREAM, linewidth=2.6,
                solid_capstyle="round", zorder=9)

for m in (minima[:2] if len(minima) >= 2 else minima):
    draw_leg(m)

# ----------------------------------------------------------------------
# Head, horns, brow flames, mane, whiskers, pearl
# ----------------------------------------------------------------------
head_c = spine[-1]
head_dir = angles[-1]
def W(pts):  # local head-space -> world
    return rot(pts, head_dir, np.array([0, 0])) + head_c

_head_pts = [
    (-14, -10), (-17, 2), (-15, 12), (-4, 22), (10, 27), (24, 25),
    (36, 19), (48, 11), (58, 3), (64, -6), (58, -14), (48, -19),
    (36, -20), (24, -19), (12, -18), (0, -16),
]
head_local = [(x*1.35, y*1.35) for x, y in _head_pts]
ax.add_patch(Polygon(W(head_local), closed=True, facecolor=SCALE_MID,
                      edgecolor=GOLD, linewidth=2.2, zorder=10))
# jaw shadow crease
crease = W([((0)*1.35, (-15)*1.35), ((16)*1.35, (-17)*1.35), ((30)*1.35, (-10)*1.35)])
ax.plot(crease[:,0], crease[:,1], color=SCALE_DARK, linewidth=1.6, zorder=11, alpha=0.7)

# horns: thick, sweeping mostly upward with a slight back lean, small fork near the tip
for sign in (1, -1):
    base = W([((8)*1.35, (sign*15)*1.35)])[0]
    c1   = W([((6)*1.35, (sign*34)*1.35)])[0]
    c2   = W([((-4)*1.35, (sign*50)*1.35)])[0]
    tip  = W([((-8)*1.35, (sign*64)*1.35)])[0]
    curve = cubic_bezier(base, c1, c2, tip, 40)
    ax.plot(curve[:,0], curve[:,1], color=GOLD, linewidth=4.2,
            solid_capstyle="round", zorder=12)
    ax.plot(curve[:,0], curve[:,1], color=GOLD_LIGHT, linewidth=1.4,
            alpha=0.6, zorder=12)
    fork_tip = W([((8)*1.35, (sign*58)*1.35)])[0]
    ax.plot([c2[0], fork_tip[0]], [c2[1], fork_tip[1]], color=GOLD,
            linewidth=2.8, solid_capstyle="round", zorder=12)

# brow "flame" ridges above the eyes (classic dragon-art motif)
for sign in (1,):
    brow = W([((18)*1.35, (12)*1.35), ((24)*1.35, (18)*1.35), ((32)*1.35, (15)*1.35), ((26)*1.35, (10)*1.35)])
    ax.add_patch(Polygon(brow, closed=True, facecolor=GOLD, edgecolor=SCALE_DARK,
                          linewidth=0.4, zorder=13, alpha=0.95))

# eye
eye = W([((24)*1.35, (9)*1.35)])[0]
ax.add_patch(Circle(eye, 3.2, facecolor="#140c0a", zorder=14))
ax.add_patch(Circle(eye + np.array([0.9,1.1]), 1.0, facecolor=CREAM, zorder=15))

# nostril
nostril = W([((50)*1.35, (5)*1.35)])[0]
ax.add_patch(Circle(nostril, 1.4, facecolor="#140c0a", zorder=14))

# open mouth with fangs and a curling tongue
mouth = W([((34)*1.35, (-15)*1.35), ((44)*1.35, (-10)*1.35), ((50)*1.35, (-2)*1.35)])
ax.plot(mouth[:,0], mouth[:,1], color="#140c0a", linewidth=2, zorder=13)
upper_fang = W([((43)*1.35, (-8)*1.35), ((41)*1.35, (-19)*1.35)])
lower_fang = W([((35)*1.35, (-13)*1.35), ((37)*1.35, (-4)*1.35)])
ax.plot(upper_fang[:,0], upper_fang[:,1], color=CREAM, linewidth=2.6, zorder=14, solid_capstyle="round")
ax.plot(lower_fang[:,0], lower_fang[:,1], color=CREAM, linewidth=2.2, zorder=14, solid_capstyle="round")
tongue = cubic_bezier(W([((34)*1.35, (-15)*1.35)])[0], W([((30)*1.35, (-24)*1.35)])[0], W([((22)*1.35, (-26)*1.35)])[0], W([((16)*1.35, (-20)*1.35)])[0], 30)
ax.plot(tongue[:,0], tongue[:,1], color="#a8323f", linewidth=3, zorder=13, solid_capstyle="round")

# whiskers — long, soft, drooping down and back from the chin
chin = W([((-2)*1.35, (-15)*1.35)])[0]
for k, dy2 in enumerate((-14, -30)):
    ctrl = chin + np.array([-20, dy2])
    end  = chin + np.array([-40, dy2-16])
    curve = cubic_bezier(chin, chin+(ctrl-chin)*0.5, ctrl, end, 30)
    ax.plot(curve[:,0], curve[:,1], color=GOLD, linewidth=2.2-0.4*k,
            alpha=0.85, zorder=11, solid_capstyle="round")

# mane — short flame-like gold tufts hugging the top of the neck, well
# behind the head so they don't tangle with the horns
neck_idxs = np.linspace(n*0.80, n*0.92, 4).astype(int)
for k, i in enumerate(neck_idxs):
    w = width[i]
    up = normals[i] if normals[i][1] > 0 else -normals[i]
    base = spine[i] + up*(w/2)
    back = -np.array([np.cos(angles[i]), np.sin(angles[i])])
    ctrl = base + back*4 + up*(8+k*2)
    tip  = base + back*(6+k*2) + up*(16+k*3)
    curve = cubic_bezier(base, base+(ctrl-base)*0.6, ctrl, tip, 20)
    ax.plot(curve[:,0], curve[:,1], color=GOLD, linewidth=3.0-0.3*k,
            alpha=0.9, zorder=9, solid_capstyle="round")

# flaming pearl of wisdom, in front of the snout
pearl_c = W([((78)*1.35, (6)*1.35)])[0]
for r, a, c in [(15,0.15,FLAME), (10,0.35,GOLD_LIGHT), (6,1.0,CREAM)]:
    ax.add_patch(Circle(pearl_c, r, facecolor=c, alpha=a, zorder=16, linewidth=0))
for ang_deg in range(0, 360, 30):
    rad = np.radians(ang_deg)
    p1 = pearl_c + 7*np.array([np.cos(rad), np.sin(rad)])
    p2 = pearl_c + (12+5*((ang_deg//30) % 2))*np.array([np.cos(rad), np.sin(rad)])
    ax.plot([p1[0],p2[0]], [p1[1],p2[1]], color=FLAME, linewidth=1.4,
            alpha=0.85, zorder=16, solid_capstyle="round")

# ----------------------------------------------------------------------
# Title
# ----------------------------------------------------------------------
ax.text(450, -35, "The Dragon of the Middle Kingdom", ha="center", va="center",
        fontsize=17, color=GOLD_LIGHT, family="serif", weight="bold", zorder=20)

plt.tight_layout(pad=0)
fig.savefig("/mnt/user-data/outputs/chinese_dragon.png", facecolor=fig.get_facecolor())
fig.savefig("/mnt/user-data/outputs/chinese_dragon.svg", facecolor=fig.get_facecolor())
print("Saved chinese_dragon.png and chinese_dragon.svg")
