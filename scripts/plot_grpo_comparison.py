"""
GRPO1 vs GRPO2 训练曲线对比图。

Usage:
    python scripts/plot_grpo_comparison.py \
        --log1 logs/grpo_1527972.out \
        --log2 logs/grpo2_1530767.out \
        --out  out/geoqa/grpo_comparison.png
"""

import re
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# ── 解析 ──────────────────────────────────────────────────────────────────────

def parse_log(log_path: str):
    """解析日志，自动处理多次 preempt/requeue，返回最长一段的 (global_steps, rewards, losses, kls)。"""
    pattern = re.compile(
        r'Epoch\[(\d+)/\d+\]\s+Step\[(\d+)/(\d+)\]\s+loss=(\S+)\s+reward=(\S+)\s+kl=(\S+)'
    )
    fallback = re.compile(
        r'Step\[(\d+)/(\d+)\]\s+loss=(\S+)\s+reward=(\S+)\s+kl=(\S+)'
    )

    segments, current = [], []

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                epoch_cur = int(m.group(1))
                step      = int(m.group(2))
                steps_tot = int(m.group(3))
                loss      = float(m.group(4))
                reward    = float(m.group(5))
                kl        = float(m.group(6))
                gstep     = (epoch_cur - 1) * steps_tot + step
            else:
                m2 = fallback.search(line)
                if not m2:
                    continue
                step      = int(m2.group(1))
                steps_tot = int(m2.group(2))
                loss      = float(m2.group(3))
                reward    = float(m2.group(4))
                kl        = float(m2.group(5))
                gstep     = step

            if current and gstep < current[-1][0] - steps_tot:
                segments.append(current)
                current = []
            current.append((gstep, loss, reward, kl))

    if current:
        segments.append(current)
    if not segments:
        raise ValueError(f'No Step records in {log_path}')

    data = max(segments, key=lambda s: s[-1][0])
    return (
        [d[0] for d in data],
        [d[2] for d in data],   # reward
        [d[1] for d in data],   # loss
        [d[3] for d in data],   # kl
    )


def smooth(values, window=15):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


# ── 绘图 ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log1', required=True, help='GRPO1 日志（全量数据）')
    parser.add_argument('--log2', required=True, help='GRPO2 日志（Pass@K 筛选）')
    parser.add_argument('--out',  default='grpo_comparison.png')
    args = parser.parse_args()

    steps1, rew1, loss1, kl1 = parse_log(args.log1)
    steps2, rew2, loss2, kl2 = parse_log(args.log2)

    print(f'GRPO1: {len(steps1)} records, step {steps1[0]}→{steps1[-1]}')
    print(f'GRPO2: {len(steps2)} records, step {steps2[0]}→{steps2[-1]}')

    C1, C2 = '#2E75B6', '#C55A11'   # blue / orange

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('GRPO Training Curves: Full Dataset vs Pass@K Filtered',
                 fontsize=13, fontweight='bold', y=1.01)

    # ── Reward ──
    ax = axes[0]
    ax.plot(steps1, rew1,          alpha=0.18, color=C1, lw=0.8)
    ax.plot(steps1, smooth(rew1),  color=C1, lw=2.0, label='GRPO1  (full dataset, 1 epoch)')
    ax.plot(steps2, rew2,          alpha=0.25, color=C2, lw=0.8)
    ax.plot(steps2, smooth(rew2),  color=C2, lw=2.0,
            label='GRPO2  (Pass@K filtered, 2 epochs)')

    ax.axhline(3.0, color='gray', ls='--', lw=0.9, alpha=0.6, label='Max reward = 3.0')
    ax.set_title('Mean Reward per Step', fontsize=11)
    ax.set_xlabel('Training Step'); ax.set_ylabel('Mean Reward')
    ax.set_ylim(-0.1, 3.4)
    ax.legend(fontsize=8.5, loc='lower right')

    # annotation: step-0 reward
    r0_1 = rew1[0]; r0_2 = rew2[0]
    ax.annotate(f'step 0: {r0_1:.2f}', xy=(steps1[0], r0_1),
                xytext=(steps1[-1]*0.05, r0_1 - 0.35),
                fontsize=8, color=C1,
                arrowprops=dict(arrowstyle='->', color=C1, lw=1.0))
    ax.annotate(f'step 0: {r0_2:.2f}', xy=(steps2[0], r0_2),
                xytext=(steps2[-1]*0.3 + 10, r0_2 - 0.5),
                fontsize=8, color=C2,
                arrowprops=dict(arrowstyle='->', color=C2, lw=1.0))

    # ── Loss ──
    ax = axes[1]
    ax.plot(steps1, loss1,          alpha=0.18, color=C1, lw=0.8)
    ax.plot(steps1, smooth(loss1),  color=C1, lw=2.0, label='GRPO1')
    ax.plot(steps2, loss2,          alpha=0.25, color=C2, lw=0.8)
    ax.plot(steps2, smooth(loss2),  color=C2, lw=2.0, label='GRPO2')
    ax.set_title('Policy Loss', fontsize=11)
    ax.set_xlabel('Training Step'); ax.set_ylabel('Loss')
    ax.legend(fontsize=8.5)

    # ── KL ──
    ax = axes[2]
    ax.plot(steps1, kl1,          alpha=0.18, color=C1, lw=0.8)
    ax.plot(steps1, smooth(kl1),  color=C1, lw=2.0, label='GRPO1')
    ax.plot(steps2, kl2,          alpha=0.25, color=C2, lw=0.8)
    ax.plot(steps2, smooth(kl2),  color=C2, lw=2.0, label='GRPO2')
    ax.set_title('KL vs Reference', fontsize=11)
    ax.set_xlabel('Training Step'); ax.set_ylabel('KL Divergence')
    ax.legend(fontsize=8.5)

    plt.tight_layout()
    plt.savefig(args.out, dpi=180, bbox_inches='tight', facecolor='white')
    print(f'Saved → {args.out}')


if __name__ == '__main__':
    main()
