#!/usr/bin/env python3
"""Run bio-inspired simulation with internal drives.

This script runs the Bio version of Cogito agent with:
    - Hunger drive (internal motivation to seek food)
    - Fear drive (internal motivation to avoid danger)
    - Scent fields for food detection
    - Intrinsic reward from internal state changes

Usage:
    python run_bio.py --steps 10000
    python run_bio.py --steps 50000 --checkpoint-interval 5000
    python run_bio.py --visual  # Show visualization
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from cogito.config import Config
from cogito.core.bio_simulation import BioSimulation


def print_banner() -> None:
    """Print startup banner."""
    print()
    print("=" * 70)
    print("🐛 Cogito Bio Simulation - 内在驱动力版本")
    print("=" * 70)
    print()


def print_legend() -> None:
    """Print legend for progress output."""
    print("📊 日志参数说明:")
    print("-" * 70)
    print("  Step  = 当前步数")
    print("  Life  = 平均寿命 (死亡前存活步数)")
    print("  E     = 平均能量 (0-100)")
    print("  H     = 平均饥饿感 (0=饱足, 1=饥饿)")
    print("  F     = 平均恐惧感 (0=平静, 1=恐惧)")
    print("  R     = 平均内在奖励 (负=不适, 正=满足)")
    print("  Food  = 累计吃掉的食物")
    print("  D     = 累计死亡次数")
    print("-" * 70)
    print()


def print_final_stats(stats: dict, elapsed: float, total_steps: int) -> None:
    """Print final statistics summary."""
    print()
    print("=" * 70)
    print("📊 最终统计")
    print("=" * 70)

    # 基础统计
    print()
    print("⏱️  运行信息")
    print(f"   总步数:     {stats['total_steps']}")
    print(f"   运行时间:   {elapsed:.1f} 秒")
    print(f"   运行速度:   {total_steps / elapsed:.1f} 步/秒")

    # 行为统计
    print()
    print("🍎 行为统计")
    print(f"   死亡次数:   {stats['total_deaths']}")
    print(f"   平均寿命:   {stats['avg_lifespan']:.1f} 步")
    print(f"   吃掉食物:   {stats['total_food_eaten']}")
    print(f"   食物效率:   {stats['food_rate']:.2f} 食物/千步")

    # 动作分布
    action_names = ["上", "下", "左", "右", "吃", "等"]
    action_dist = stats['action_distribution']
    print()
    print("🎮 动作分布")
    dist_str = "   "
    for i, (name, pct) in enumerate(zip(action_names, action_dist)):
        dist_str += f"{name}:{pct*100:4.1f}%  "
        if i == 2:  # 换行
            print(dist_str)
            dist_str = "   "
    print(dist_str)

    # 学习统计
    print()
    print("📈 学习统计")
    print(f"   生存损失:   {stats['avg_survival_loss']:.4f}")
    print(f"   预测损失:   {stats['avg_prediction_loss']:.4f}")
    print(f"   总损失:     {stats['avg_total_loss']:.4f}")
    print(f"   策略熵:     {stats['avg_entropy']:.3f} (0=确定, 1.79=随机)")

    # 内部状态统计
    print()
    print("🧠 内部状态")
    print(f"   隐状态范数: {stats['avg_hidden_norm']:.3f}")
    print(f"   隐状态方差: {stats['avg_hidden_var']:.6f}")
    print(f"   平均能量:   {stats['avg_energy']:.1f}")

    # Bio特有统计
    print()
    print("🐛 生物驱动统计")
    print(f"   平均饥饿感: {stats['avg_hunger']:.3f}")
    print(f"   平均恐惧感: {stats['avg_fear']:.3f}")
    print(f"   平均奖励:   {stats['avg_intrinsic_reward']:.3f}")
    print(f"   满足事件:   {stats['satisfaction_events']} (饥饿减少)")
    print(f"   宽慰事件:   {stats['relief_events']} (恐惧减少)")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Run bio-inspired Cogito simulation"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Number of simulation steps (default: 10000)",
    )
    parser.add_argument(
        "--visual",
        action="store_true",
        help="Show visualization (default: headless)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=5000,
        help="Steps between checkpoints (default: 5000)",
    )
    parser.add_argument(
        "--snapshot-interval",
        type=int,
        default=2000,
        help="Steps between state snapshots (default: 2000)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/bio",
        help="Output directory for checkpoints and logs",
    )

    args = parser.parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    snapshot_dir = output_dir / "snapshots"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # Set random seed
    rng = np.random.default_rng(args.seed)

    # Print banner
    print_banner()
    print(f"📍 配置信息")
    print(f"   步数:           {args.steps}")
    print(f"   模式:           {'可视化' if args.visual else '无头'}")
    print(f"   随机种子:       {args.seed if args.seed else '随机'}")
    print(f"   检查点间隔:     {args.checkpoint_interval}")
    print(f"   输出目录:       {output_dir}")
    print()

    # Create simulation
    sim = BioSimulation(
        config=Config,
        rng=rng,
        headless=not args.visual,
        render_interval=10,
    )

    # Tracking
    start_time = time.time()
    last_checkpoint_step = 0

    print_legend()
    print("🚀 开始运行...")
    print()

    # Run in chunks for checkpointing
    remaining_steps = args.steps
    chunk_size = min(1000, args.steps)

    while remaining_steps > 0:
        current_chunk = min(chunk_size, remaining_steps)

        # Run chunk
        stats = sim.run(current_chunk, verbose=False)

        # Print progress
        elapsed = time.time() - start_time
        steps_done = args.steps - remaining_steps + current_chunk
        steps_per_sec = steps_done / elapsed if elapsed > 0 else 0

        print(
            f"Step {steps_done:6d}/{args.steps} | "
            f"{steps_per_sec:5.1f} st/s | "
            f"Life: {stats['avg_lifespan']:5.0f} | "
            f"E: {stats['avg_energy']:5.1f} | "
            f"H: {stats['avg_hunger']:.2f} | "
            f"F: {stats['avg_fear']:.2f} | "
            f"R: {stats['avg_intrinsic_reward']:+.2f} | "
            f"Food: {stats['total_food_eaten']:3d} | "
            f"D: {stats['total_deaths']:3d}"
        )

        # Save checkpoint
        if sim.step_count - last_checkpoint_step >= args.checkpoint_interval:
            checkpoint_path = checkpoint_dir / f"bio_step_{sim.step_count:06d}.pt"
            sim.save_checkpoint(str(checkpoint_path))
            last_checkpoint_step = sim.step_count

        remaining_steps -= current_chunk

    # Final summary
    elapsed = time.time() - start_time
    print_final_stats(stats, elapsed, args.steps)

    # Save final checkpoint
    final_path = checkpoint_dir / "final_bio.pt"
    sim.save_checkpoint(str(final_path))
    print(f"✅ 检查点已保存: {final_path}")

    sim.close()
    print()
    print("🎉 Bio 仿真完成!")
    print()


if __name__ == "__main__":
    main()