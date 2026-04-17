"""
Conference-style pipeline diagram for InternVL3-8B SFT + GRPO on GeoQA.

Usage:
    python scripts/plot_pipeline.py
    python scripts/plot_pipeline.py --out out/geoqa/pipeline.png
"""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch


def rbox(ax, cx, cy, w, h, fc_light, ec, lw=2.0, radius=0.3, zorder=3):
    """Rounded box: light fill + colored border."""
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad=0.1,rounding_size={radius}",
        facecolor=fc_light, edgecolor=ec, linewidth=lw,
        zorder=zorder,
    ))


def inner_box(ax, cx, cy, w, h, fc, ec, lw=1.2, radius=0.15, zorder=5):
    ax.add_patch(FancyBboxPatch(
        (cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        zorder=zorder,
    ))


def arrow(ax, x1, y1, x2, y2, color='#555555', lw=1.8, rad=0.0):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
        arrowprops=dict(
            arrowstyle='->', color=color, lw=lw, mutation_scale=16,
            connectionstyle=f'arc3,rad={rad}',
        ), zorder=12)


def txt(ax, x, y, s, fs=9, c='#1a1a1a', bold=False, italic=False,
        ha='center', va='center', zorder=13):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=c,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal', zorder=zorder)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='out/geoqa/pipeline.png')
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(16, 6.2))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6.2)
    ax.axis('off')
    fig.patch.set_facecolor('white')

    # ── palette ──────────────────────────────────────────────────────────────
    BLUE_D  = '#1F5FA6';  BLUE_L  = '#EBF3FB'
    GREEN_D = '#2E7D32';  GREEN_L = '#F1F8E9'
    ORG_D   = '#BF5A0A';  ORG_L   = '#FFF3E0'
    PUR_D   = '#6A1B9A';  PUR_L   = '#F3E5F5'
    GRAY    = '#555555'

    BW, BH = 2.8, 4.6   # box width / height
    BY     = 3.0         # box center y
    TITLE_Y = BY + BH/2 - 0.38   # section title y

    # ── Section positions ────────────────────────────────────────────────────
    POS = {'data': 1.8, 'base': 5.2, 'sft': 9.0, 'grpo': 13.2}

    # ══════════════════════════════════════════════════════════════════════════
    # Title
    # ══════════════════════════════════════════════════════════════════════════
    txt(ax, 7.5, 5.88,
        'Fine-tuning Pipeline for Geometric Visual Reasoning (GeoQA)',
        fs=13, bold=True, c='#111111')

    # ══════════════════════════════════════════════════════════════════════════
    # 1. GeoQA Dataset
    # ══════════════════════════════════════════════════════════════════════════
    dx = POS['data']
    rbox(ax, dx, BY, BW, BH, GREEN_L, GREEN_D, lw=2.2)
    txt(ax, dx, TITLE_Y,        'GeoQA Dataset',         fs=11, bold=True,   c=GREEN_D)
    txt(ax, dx, TITLE_Y - 0.42, 'Geometry MCQ (Chinese)', fs=8.5, italic=True, c=GRAY)

    # mini geometry image
    inner_box(ax, dx, 3.35, 1.9, 1.1, '#C8E6C9', GREEN_D, lw=1.2, radius=0.12)
    tri = plt.Polygon([[dx-0.5, 2.9], [dx+0.5, 2.9], [dx, 3.75]],
                       closed=True, facecolor='#66BB6A', edgecolor=GREEN_D,
                       linewidth=1.2, zorder=6)
    ax.add_patch(tri)
    txt(ax, dx, 2.65, 'geometry image', fs=7.5, c=GREEN_D, italic=True)

    for t, y in [('• Multiple choice  (A/B/C/D)', 2.2),
                 ('• 3,500 train  /  400 test',   1.85)]:
        txt(ax, dx, y, t, fs=8.5, c='#2E7D32', ha='center')

    # ══════════════════════════════════════════════════════════════════════════
    # 2. InternVL3-8B Base
    # ══════════════════════════════════════════════════════════════════════════
    bx = POS['base']
    rbox(ax, bx, BY, BW, BH, BLUE_L, BLUE_D, lw=2.2)
    txt(ax, bx, TITLE_Y,        'InternVL3-8B',      fs=11, bold=True,   c=BLUE_D)
    txt(ax, bx, TITLE_Y - 0.42, 'Pre-trained Base',  fs=8.5, italic=True, c=GRAY)

    for label_t, cy, fc in [
        ('Vision Encoder (ViT-6B)', 3.6, '#BBDEFB'),
        ('MLP Projector',           2.85, '#BBDEFB'),
        ('LLM (InternLM2-7B)',      2.1, '#BBDEFB'),
    ]:
        inner_box(ax, bx, cy, 2.35, 0.52, fc, BLUE_D, radius=0.10)
        txt(ax, bx, cy, label_t, fs=8.5, c='#0D47A1')
    for ya, yb in [(3.34, 3.12), (2.59, 2.37)]:
        arrow(ax, bx, ya, bx, yb, color=BLUE_D, lw=1.1)

    # ══════════════════════════════════════════════════════════════════════════
    # 3. SFT
    # ══════════════════════════════════════════════════════════════════════════
    sx = POS['sft']
    rbox(ax, sx, BY, BW, BH, ORG_L, ORG_D, lw=2.2)
    txt(ax, sx, TITLE_Y,        'Stage 1 — SFT',           fs=11, bold=True,   c=ORG_D)
    txt(ax, sx, TITLE_Y - 0.42, 'Supervised Fine-Tuning',  fs=8.5, italic=True, c=GRAY)

    sft_rows = [
        ('Adapter:  LoRA  (r = 16)',    3.62),
        ('Frozen:   ViT + Projector',   3.1),
        ('Trainable: LLM adapters',     2.62),
        ('Loss:  Cross-Entropy',        2.12),
        ('Target: assistant tokens',    1.68),
    ]
    for t, y in sft_rows:
        txt(ax, sx, y, t, fs=8.5, c='#4E2500')

    # ══════════════════════════════════════════════════════════════════════════
    # 4. GRPO
    # ══════════════════════════════════════════════════════════════════════════
    gx = POS['grpo']
    rbox(ax, gx, BY, BW, BH, PUR_L, PUR_D, lw=2.2)
    txt(ax, gx, TITLE_Y,        'Stage 2 — GRPO',              fs=11, bold=True,   c=PUR_D)
    txt(ax, gx, TITLE_Y - 0.42, 'Group Relative Policy Opt.',  fs=8.5, italic=True, c=GRAY)

    grpo_rows = [
        ('Adapter:  LoRA  (r = 16)',      3.62, '#3E0059', False),
        ('K = 4 rollouts / prompt',       3.12, '#3E0059', False),
        ('Reward  (max = 3.0):',          2.62, '#3E0059', False),
        ('+1   format   <think>…</think>',2.15, '#B71C1C', True),
        ('+2   correct  <answer>X</answer>', 1.68, '#B71C1C', True),
    ]
    for t, y, c, b in grpo_rows:
        txt(ax, gx, y, t, fs=8.5, c=c, bold=b)

    # ══════════════════════════════════════════════════════════════════════════
    # Arrows
    # ══════════════════════════════════════════════════════════════════════════
    # Dataset → Base
    arrow(ax, dx + BW/2, BY, POS['base'] - BW/2, BY, color=GREEN_D, lw=1.8)

    # Base → SFT
    arrow(ax, POS['base'] + BW/2, BY, sx - BW/2, BY, color=BLUE_D, lw=1.8)
    txt(ax, 7.1, BY + 0.28, 'SFT init', fs=8, c=GRAY, italic=True)

    # SFT → GRPO
    arrow(ax, sx + BW/2, BY, gx - BW/2, BY, color=ORG_D, lw=1.8)
    txt(ax, 11.1, BY + 0.28, 'GRPO init', fs=8, c=GRAY, italic=True)

    # Dataset curved to SFT (data also feeds SFT directly)
    arrow(ax, dx + BW/2 - 0.15, BY - BH/2 + 0.25,
              sx - BW/2 + 0.15, BY - BH/2 + 0.25,
          color=GREEN_D, lw=1.1, rad=-0.25)

    # Dataset curved to GRPO
    arrow(ax, dx + BW/2 - 0.15, BY - BH/2 + 0.05,
              gx - BW/2 + 0.15, BY - BH/2 + 0.05,
          color=GREEN_D, lw=1.1, rad=-0.18)

    # ══════════════════════════════════════════════════════════════════════════
    # Legend
    # ══════════════════════════════════════════════════════════════════════════
    handles = [
        mpatches.Patch(facecolor=GREEN_L, edgecolor=GREEN_D, linewidth=1.5, label='Dataset'),
        mpatches.Patch(facecolor=BLUE_L,  edgecolor=BLUE_D,  linewidth=1.5, label='Base Model'),
        mpatches.Patch(facecolor=ORG_L,   edgecolor=ORG_D,   linewidth=1.5, label='Stage 1: SFT'),
        mpatches.Patch(facecolor=PUR_L,   edgecolor=PUR_D,   linewidth=1.5, label='Stage 2: GRPO'),
    ]
    ax.legend(handles=handles, loc='lower center', ncol=4,
              fontsize=9, framealpha=0.8, edgecolor='#cccccc',
              bbox_to_anchor=(0.5, -0.04))

    plt.tight_layout(pad=0.4)
    plt.savefig(args.out, dpi=180, bbox_inches='tight', facecolor='white')
    print(f'Pipeline diagram saved → {args.out}')


if __name__ == '__main__':
    main()
