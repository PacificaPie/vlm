"""
Transformer-paper-style architecture diagram for InternVL3-8B + LoRA + GeoQA.

Usage:
    python scripts/plot_pipeline.py --out out/geoqa/pipeline.png
"""
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe


# ─────────────────────── drawing helpers ────────────────────────────────────

def box(ax, cx, cy, w, h, fc, ec='#333333', lw=1.4, r=0.18, zorder=4, alpha=1.0):
    p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
        boxstyle=f"round,pad=0.05,rounding_size={r}",
        facecolor=fc, edgecolor=ec, linewidth=lw,
        zorder=zorder, alpha=alpha)
    ax.add_patch(p)

def txt(ax, x, y, s, fs=9, c='#111111', bold=False, italic=False,
        ha='center', va='center', zorder=15):
    ax.text(x, y, s, ha=ha, va=va, fontsize=fs, color=c,
            fontweight='bold' if bold else 'normal',
            fontstyle='italic' if italic else 'normal', zorder=zorder)

def varrow(ax, x, y0, y1, c='#333333', lw=1.4):
    """Vertical arrow."""
    ax.annotate('', xy=(x, y1), xytext=(x, y0),
        arrowprops=dict(arrowstyle='->', color=c, lw=lw, mutation_scale=13),
        zorder=14)

def harrow(ax, x0, x1, y, c='#333333', lw=1.4, rad=0.0):
    ax.annotate('', xy=(x1, y), xytext=(x0, y),
        arrowprops=dict(arrowstyle='->', color=c, lw=lw, mutation_scale=13,
                        connectionstyle=f'arc3,rad={rad}'),
        zorder=14)

def bracket(ax, x, y, label, c='#555555'):
    """Small 'Nx' badge."""
    box(ax, x, y, 0.55, 0.34, '#FFFFFF', c, lw=1.2, r=0.08, zorder=16)
    txt(ax, x, y, label, fs=8.5, c=c, bold=True, zorder=17)


# ─────────────────────── main ───────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='out/geoqa/pipeline.png')
    args = parser.parse_args()

    W, H = 14.0, 10.5
    fig, ax = plt.subplots(figsize=(W, H))
    ax.set_xlim(0, W); ax.set_ylim(0, H)
    ax.axis('off'); fig.patch.set_facecolor('white')

    # ── colour palette (close to "Attention is All You Need" vibes) ─────────
    C_INPUT  = '#DAE8FC'   # light blue  – inputs
    C_VIT    = '#D5E8D4'   # light green – vision encoder
    C_PROJ   = '#FFF2CC'   # light yellow – projector / connector
    C_LLM    = '#E1D5E7'   # light purple – LLM
    C_LORA   = '#FFE6CC'   # light orange – LoRA adapters
    C_OUT    = '#F8CECC'   # light pink  – output / reward
    C_SFT    = '#DAE8FC'
    C_GRPO   = '#D5E8D4'
    C_TITLE  = '#F5F5F5'   # near-white  – section headers

    E_BLUE   = '#1F5FA6'
    E_GREEN  = '#2E7D32'
    E_YEL    = '#B07D00'
    E_PUR    = '#6A1B9A'
    E_ORG    = '#BF5A0A'
    E_RED    = '#C62828'
    E_GRAY   = '#444444'

    # ════════════════════════════════════════════════════════════════════════
    # LEFT COLUMN  –  Model Architecture   (x ≈ 0 … 8)
    # ════════════════════════════════════════════════════════════════════════
    MX = 4.0   # architecture column centre-x

    # ── Title ──
    txt(ax, MX, 10.1, 'InternVL3-8B Architecture', fs=12, bold=True, c='#111111')

    # ── Inputs ──────────────────────────────────────────────────────────────
    IMG_X, TXT_X, INP_Y = 2.1, 5.9, 9.35

    box(ax, IMG_X, INP_Y, 2.6, 0.55, C_INPUT, E_BLUE, lw=1.6)
    txt(ax, IMG_X, INP_Y + 0.02, 'Geometry Image', fs=9.5, bold=True, c=E_BLUE)
    txt(ax, IMG_X, INP_Y - 0.22, '(H × W × 3)', fs=8, c='#555555', italic=True)

    box(ax, TXT_X, INP_Y, 2.6, 0.55, C_INPUT, E_BLUE, lw=1.6)
    txt(ax, TXT_X, INP_Y + 0.02, 'Question + Options', fs=9.5, bold=True, c=E_BLUE)
    txt(ax, TXT_X, INP_Y - 0.22, '(text tokens)', fs=8, c='#555555', italic=True)

    # ── Vision Encoder  ──────────────────────────────────────────────────────
    VIT_Y = 8.1
    box(ax, IMG_X, VIT_Y, 2.6, 0.7, C_VIT, E_GREEN, lw=1.6)
    txt(ax, IMG_X, VIT_Y + 0.1, 'Vision Encoder', fs=9.5, bold=True, c=E_GREEN)
    txt(ax, IMG_X, VIT_Y - 0.18, 'ViT-6B  (frozen)', fs=8, c='#555555', italic=True)
    varrow(ax, IMG_X, INP_Y - 0.28, VIT_Y + 0.35, c=E_GREEN)

    # ── Text Embed  ───────────────────────────────────────────────────────────
    box(ax, TXT_X, VIT_Y, 2.6, 0.7, C_VIT, E_GREEN, lw=1.6)
    txt(ax, TXT_X, VIT_Y + 0.1, 'Text Embedding', fs=9.5, bold=True, c=E_GREEN)
    txt(ax, TXT_X, VIT_Y - 0.18, 'Tokenizer  (frozen)', fs=8, c='#555555', italic=True)
    varrow(ax, TXT_X, INP_Y - 0.28, VIT_Y + 0.35, c=E_GREEN)

    # ── MLP Projector  ────────────────────────────────────────────────────────
    PROJ_Y = 7.05
    box(ax, IMG_X, PROJ_Y, 2.6, 0.55, C_PROJ, E_YEL, lw=1.6)
    txt(ax, IMG_X, PROJ_Y + 0.06, 'MLP Projector', fs=9.5, bold=True, c=E_YEL)
    txt(ax, IMG_X, PROJ_Y - 0.17, '(frozen)', fs=8, c='#555555', italic=True)
    varrow(ax, IMG_X, VIT_Y - 0.35, PROJ_Y + 0.28, c=E_YEL)

    # ── Merge visual + text tokens  ───────────────────────────────────────────
    MERGE_Y = 6.12
    box(ax, MX, MERGE_Y, 5.6, 0.52, C_PROJ, E_YEL, lw=1.5)
    txt(ax, MX, MERGE_Y, 'Concatenate  [Visual Tokens | Text Tokens]',
        fs=9.5, bold=True, c=E_YEL)
    # arrows into merge box
    varrow(ax, IMG_X, PROJ_Y - 0.28, MERGE_Y + 0.26, c=E_YEL)
    varrow(ax, TXT_X, VIT_Y - 0.35, MERGE_Y + 0.26, c=E_YEL)

    # ── LLM block  ────────────────────────────────────────────────────────────
    LLM_CY = 4.35
    LLM_W, LLM_H = 5.8, 2.6
    box(ax, MX, LLM_CY, LLM_W, LLM_H, C_LLM, E_PUR, lw=2.0, r=0.22, zorder=3)
    txt(ax, MX, LLM_CY + 1.05, 'Language Model  (InternLM2-7B)', fs=10, bold=True, c=E_PUR)
    varrow(ax, MX, MERGE_Y - 0.26, LLM_CY + LLM_H/2, c=E_PUR)

    # inner: Self-Attn + LoRA
    ATTN_Y = LLM_CY + 0.28
    box(ax, MX, ATTN_Y, 4.8, 0.55, '#F3EEF8', E_PUR, lw=1.2, r=0.12)
    txt(ax, MX - 0.3, ATTN_Y, 'Multi-Head Self-Attention', fs=9, c='#4A235A')
    box(ax, MX + 1.7, ATTN_Y, 0.95, 0.38, C_LORA, E_ORG, lw=1.4, r=0.10)
    txt(ax, MX + 1.7, ATTN_Y, '+ LoRA', fs=8.5, bold=True, c=E_ORG)

    # inner: FFN + LoRA
    FFN_Y = LLM_CY - 0.55
    box(ax, MX, FFN_Y, 4.8, 0.55, '#F3EEF8', E_PUR, lw=1.2, r=0.12)
    txt(ax, MX - 0.3, FFN_Y, 'Feed-Forward Network', fs=9, c='#4A235A')
    box(ax, MX + 1.7, FFN_Y, 0.95, 0.38, C_LORA, E_ORG, lw=1.4, r=0.10)
    txt(ax, MX + 1.7, FFN_Y, '+ LoRA', fs=8.5, bold=True, c=E_ORG)

    # "× 32" badge
    bracket(ax, MX + LLM_W/2 - 0.1, LLM_CY, '×32')

    # ── Output  ───────────────────────────────────────────────────────────────
    OUT_Y = 2.7
    box(ax, MX, OUT_Y, 4.0, 0.55, C_OUT, E_RED, lw=1.6)
    txt(ax, MX, OUT_Y, 'Output Logits  →  Answer  (A / B / C / D)',
        fs=9.5, bold=True, c=E_RED)
    varrow(ax, MX, LLM_CY - LLM_H/2, OUT_Y + 0.28, c=E_RED)

    # ════════════════════════════════════════════════════════════════════════
    # RIGHT COLUMN  –  Training Stages   (x ≈ 9 … 14)
    # ════════════════════════════════════════════════════════════════════════
    RX = 11.4
    txt(ax, RX, 10.1, 'Training Stages', fs=12, bold=True, c='#111111')

    # ── Stage 1 : SFT ─────────────────────────────────────────────────────────
    SFT_CY = 7.8
    box(ax, RX, SFT_CY, 4.4, 3.8, C_SFT, E_BLUE, lw=2.0, r=0.25, zorder=3)

    # header band
    box(ax, RX, SFT_CY + 1.7, 4.4, 0.48, '#BDD7EE', E_BLUE, lw=0, r=0.22, zorder=4)
    txt(ax, RX, SFT_CY + 1.72, 'Stage 1  —  SFT', fs=11, bold=True, c=E_BLUE)

    sft_lines = [
        ('Supervised Fine-Tuning',          1.15, 8.5, True),
        ('Adapter: LoRA  (r = 16)',          0.62, 8.5, False),
        ('Frozen: ViT + MLP Projector',      0.18, 8.5, False),
        ('Trainable: LLM LoRA adapters',    -0.25, 8.5, False),
        ('',                                -0.55, 8.5, False),   # spacer
        ('Loss:',                           -0.78, 9.0, True),
    ]
    for t, dy, fs, b in sft_lines:
        if t:
            txt(ax, RX, SFT_CY + dy, t, fs=fs, c='#1F3864', bold=b)

    # loss formula box
    box(ax, RX, SFT_CY - 1.28, 3.7, 0.48, '#FFFFFF', E_BLUE, lw=1.3, r=0.12)
    txt(ax, RX, SFT_CY - 1.28,
        r'$\mathcal{L}_\mathrm{SFT} = -\sum_{t} \log p_\theta(y_t \mid y_{<t}, x)$',
        fs=9, c='#1F3864')

    # ── Stage 2 : GRPO ────────────────────────────────────────────────────────
    GRP_CY = 3.55
    box(ax, RX, GRP_CY, 4.4, 4.8, C_GRPO, E_GREEN, lw=2.0, r=0.25, zorder=3)

    box(ax, RX, GRP_CY + 2.15, 4.4, 0.48, '#C8E6C9', E_GREEN, lw=0, r=0.22, zorder=4)
    txt(ax, RX, GRP_CY + 2.17, 'Stage 2  —  GRPO', fs=11, bold=True, c=E_GREEN)

    grpo_desc = [
        ('Group Relative Policy Opt.',   1.6,  8.5, True),
        ('Init from SFT checkpoint',     1.1,  8.5, False),
        ('K = 4 rollouts / prompt',      0.65, 8.5, False),
        ('Adapter: LoRA  (r = 16)',      0.2,  8.5, False),
        ('',                            -0.1,  8.5, False),
        ('Reward Function:',            -0.38, 9.0, True),
        ('+1   format   <think>…</think>', -0.82, 8.5, False),
        ('+2   correct  <answer>X</answer>', -1.22, 8.5, False),
    ]
    for t, dy, fs, b in grpo_desc:
        if t:
            col = '#C62828' if t.startswith('+') else '#1B5E20'
            txt(ax, RX, GRP_CY + dy, t, fs=fs, c=col, bold=b)

    # objective box
    box(ax, RX, GRP_CY - 2.0, 3.9, 0.56, '#FFFFFF', E_GREEN, lw=1.3, r=0.12)
    txt(ax, RX, GRP_CY - 1.88,
        r'$\mathcal{L}_\mathrm{GRPO} = -\mathbb{E}\left[\min(r_t \hat{A}_t,\ \mathrm{clip}(r_t)\hat{A}_t)\right]$',
        fs=8.5, c='#1B5E20')

    # ════════════════════════════════════════════════════════════════════════
    # Connector arrows  (architecture → training stages)
    # ════════════════════════════════════════════════════════════════════════
    # LLM LoRA → SFT box
    ax.annotate('', xy=(RX - 2.2, SFT_CY), xytext=(MX + LLM_W/2, LLM_CY + 0.5),
        arrowprops=dict(arrowstyle='->', color=E_BLUE, lw=1.5, mutation_scale=13,
                        connectionstyle='arc3,rad=-0.25'), zorder=14)
    txt(ax, 8.5, 7.1, 'SFT', fs=8, c=E_BLUE, italic=True, bold=True)

    # LLM LoRA → GRPO box
    ax.annotate('', xy=(RX - 2.2, GRP_CY + 0.5), xytext=(MX + LLM_W/2, LLM_CY - 0.5),
        arrowprops=dict(arrowstyle='->', color=E_GREEN, lw=1.5, mutation_scale=13,
                        connectionstyle='arc3,rad=0.2'), zorder=14)
    txt(ax, 8.5, 4.1, 'GRPO', fs=8, c=E_GREEN, italic=True, bold=True)

    # SFT → GRPO init arrow
    varrow(ax, RX, SFT_CY - 1.9 - 0.05, GRP_CY + 4.8/2 + 0.08, c=E_BLUE, lw=1.5)
    txt(ax, RX + 0.7, (SFT_CY - 1.9 + GRP_CY + 2.4) / 2, 'init', fs=8, italic=True, c=E_GRAY)

    # ── LoRA legend ───────────────────────────────────────────────────────────
    box(ax, 1.8, 1.3, 2.4, 0.52, C_LORA, E_ORG, lw=1.4, r=0.12)
    txt(ax, 1.8, 1.3, 'LoRA  =  W + A·B   (r ≪ d)', fs=8.5, c=E_ORG, bold=False)

    # ── GeoQA data note ───────────────────────────────────────────────────────
    box(ax, 5.5, 1.3, 3.2, 0.52, '#F5F5F5', E_GRAY, lw=1.2, r=0.12)
    txt(ax, 5.5, 1.3, 'GeoQA  ~3,500 train / ~400 test  |  Chinese geometry MCQ',
        fs=8.5, c=E_GRAY)

    plt.tight_layout(pad=0.5)
    plt.savefig(args.out, dpi=180, bbox_inches='tight', facecolor='white')
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
