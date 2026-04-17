"""
解析 GRPO 训练日志，绘制 reward / loss / kl 曲线。

用法:
    python scripts/plot_grpo_curves.py --log logs/grpo_1527972.out
    python scripts/plot_grpo_curves.py --log logs/grpo_1527972.out --out out/geoqa/grpo_curves.png
"""

import re
import argparse
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def parse_log(log_path: str):
    """
    解析日志，返回 (steps, losses, rewards, kls)。
    若日志中存在多次 run（preempt 后 requeue），只保留最后一段。
    """
    pattern = re.compile(
        r'Step\[(\d+)/\d+\]\s+loss=(\S+)\s+reward=(\S+)\s+kl=(\S+)'
    )

    segments = []   # 每次 run 的数据
    current  = []

    with open(log_path) as f:
        for line in f:
            m = pattern.search(line)
            if not m:
                continue
            step   = int(m.group(1))
            loss   = float(m.group(2))
            reward = float(m.group(3))
            kl     = float(m.group(4))

            # step 归零说明新一轮 run 开始
            if current and step < current[-1][0]:
                segments.append(current)
                current = []
            current.append((step, loss, reward, kl))

    if current:
        segments.append(current)

    if not segments:
        raise ValueError(f'No Step records found in {log_path}')

    if len(segments) > 1:
        print(f'检测到 {len(segments)} 段 run（preempt/requeue），使用最后一段')

    data   = segments[-1]
    steps   = [d[0] for d in data]
    losses  = [d[1] for d in data]
    rewards = [d[2] for d in data]
    kls     = [d[3] for d in data]
    return steps, losses, rewards, kls


def smooth(values, window=20):
    """简单滑动平均平滑"""
    out = []
    for i in range(len(values)):
        lo = max(0, i - window // 2)
        hi = min(len(values), i + window // 2 + 1)
        out.append(sum(values[lo:hi]) / (hi - lo))
    return out


def plot(steps, losses, rewards, kls, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('GRPO Training Curves (InternVL3-8B on GeoQA)', fontsize=13)

    # --- Reward ---
    ax = axes[0]
    ax.plot(steps, rewards, alpha=0.25, color='steelblue', linewidth=0.8)
    ax.plot(steps, smooth(rewards), color='steelblue', linewidth=1.8, label='smoothed')
    ax.set_xlabel('Step')
    ax.set_ylabel('Mean Reward')
    ax.set_title('Reward (max=3.0)')
    ax.set_ylim(-0.1, 3.3)
    ax.axhline(3.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.6)
    ax.legend(fontsize=9)

    # --- Loss ---
    ax = axes[1]
    ax.plot(steps, losses, alpha=0.25, color='tomato', linewidth=0.8)
    ax.plot(steps, smooth(losses), color='tomato', linewidth=1.8, label='smoothed')
    ax.set_xlabel('Step')
    ax.set_ylabel('GRPO Loss')
    ax.set_title('Policy Loss')
    ax.legend(fontsize=9)

    # --- KL ---
    ax = axes[2]
    ax.plot(steps, kls, alpha=0.25, color='seagreen', linewidth=0.8)
    ax.plot(steps, smooth(kls), color='seagreen', linewidth=1.8, label='smoothed')
    ax.set_xlabel('Step')
    ax.set_ylabel('KL Divergence')
    ax.set_title('KL vs Reference')
    ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f'图像已保存 → {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--log', type=str, required=True, help='训练日志路径')
    parser.add_argument('--out', type=str, default='grpo_curves.png', help='输出图像路径')
    args = parser.parse_args()

    steps, losses, rewards, kls = parse_log(args.log)
    print(f'共解析 {len(steps)} 条记录，step {steps[0]} → {steps[-1]}')
    plot(steps, losses, rewards, kls, args.out)


if __name__ == '__main__':
    main()
