"""Generate baseline analysis report from maturation run."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure


class BaselineReport:
    """Generate comprehensive baseline analysis report."""
    
    def __init__(self, data_dir: str = "data"):
        """Initialize report generator.
        
        Args:
            data_dir: Path to data directory
        """
        self.data_dir = Path(data_dir)
        self.analysis_dir = self.data_dir / "analysis"
        self.analysis_dir.mkdir(parents=True, exist_ok=True)
        
        self.behavior_data = None
        self.learning_data = None
        self.internal_states = None
        
    def load_data(self) -> None:
        """Load all collected data."""
        # Load behavior data from SQLite
        db_path = self.data_dir / "behavior.db"
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cursor = conn.cursor()
            
            # Load behavior log
            try:
                cursor.execute("SELECT * FROM behavior_log ORDER BY step")
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                self.behavior_data = {
                    col: [row[i] for row in rows]
                    for i, col in enumerate(columns)
                }
            except sqlite3.OperationalError:
                self.behavior_data = {}
            
            # Load learning log
            try:
                cursor.execute("SELECT * FROM learning_log ORDER BY step")
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                self.learning_data = {
                    col: [row[i] for row in rows]
                    for i, col in enumerate(columns)
                }
            except sqlite3.OperationalError:
                self.learning_data = {}
            
            conn.close()
        
        # Load internal states
        states_path = self.data_dir / "internal_states.npy"
        if states_path.exists():
            self.internal_states = np.load(str(states_path))
    
    def generate_learning_curves(self) -> Figure:
        """Generate learning curves figure."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Prediction loss over time
        ax = axes[0, 0]
        if self.learning_data and 'step' in self.learning_data:
            steps = self.learning_data['step']
            pred_loss = self.learning_data.get('prediction_loss', [])
            if pred_loss:
                ax.plot(steps, pred_loss, 'b-', alpha=0.7)
                ax.set_xlabel('Step')
                ax.set_ylabel('Prediction Loss')
                ax.set_title('Prediction Loss Over Time')
                ax.grid(True, alpha=0.3)
        
        # 2. Survival loss over time
        ax = axes[0, 1]
        if self.learning_data and 'step' in self.learning_data:
            steps = self.learning_data['step']
            surv_loss = self.learning_data.get('survival_loss', [])
            if surv_loss:
                ax.plot(steps, surv_loss, 'r-', alpha=0.7)
                ax.set_xlabel('Step')
                ax.set_ylabel('Survival Loss')
                ax.set_title('Survival Loss Over Time')
                ax.grid(True, alpha=0.3)
        
        # 3. Energy over time
        ax = axes[1, 0]
        if self.behavior_data and 'step' in self.behavior_data:
            steps = self.behavior_data['step']
            energy = self.behavior_data.get('energy', [])
            if energy:
                # Smooth with rolling average
                window = min(100, len(energy) // 10)
                if window > 1:
                    energy_smooth = np.convolve(
                        energy, np.ones(window)/window, mode='valid'
                    )
                    steps_smooth = steps[:len(energy_smooth)]
                    ax.plot(steps_smooth, energy_smooth, 'g-', alpha=0.7)
                else:
                    ax.plot(steps, energy, 'g-', alpha=0.7)
                ax.set_xlabel('Step')
                ax.set_ylabel('Energy')
                ax.set_title('Energy Over Time (Smoothed)')
                ax.grid(True, alpha=0.3)
        
        # 4. Action entropy over time
        ax = axes[1, 1]
        if self.behavior_data and 'step' in self.behavior_data:
            steps = self.behavior_data['step']
            entropy = self.behavior_data.get('action_entropy', [])
            if entropy:
                window = min(100, len(entropy) // 10)
                if window > 1:
                    entropy_smooth = np.convolve(
                        entropy, np.ones(window)/window, mode='valid'
                    )
                    steps_smooth = steps[:len(entropy_smooth)]
                    ax.plot(steps_smooth, entropy_smooth, 'm-', alpha=0.7)
                else:
                    ax.plot(steps, entropy, 'm-', alpha=0.7)
                ax.set_xlabel('Step')
                ax.set_ylabel('Action Entropy')
                ax.set_title('Action Entropy Over Time')
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def generate_behavior_analysis(self) -> Figure:
        """Generate behavior analysis figure."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Action frequency
        ax = axes[0, 0]
        if self.behavior_data and 'action' in self.behavior_data:
            actions = self.behavior_data['action']
            action_names = ['Up', 'Down', 'Left', 'Right', 'Eat', 'Wait']
            counts = [actions.count(i) for i in range(6)]
            ax.bar(action_names, counts, color='steelblue')
            ax.set_xlabel('Action')
            ax.set_ylabel('Count')
            ax.set_title('Action Frequency Distribution')
        
        # 2. Lifespan distribution
        ax = axes[0, 1]
        if self.behavior_data and 'current_lifespan' in self.behavior_data:
            lifespans = self.behavior_data['current_lifespan']
            # Take max lifespan per "life"
            ax.hist(lifespans, bins=50, color='coral', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Lifespan (steps)')
            ax.set_ylabel('Frequency')
            ax.set_title('Lifespan Distribution')
        
        # 3. Energy distribution
        ax = axes[1, 0]
        if self.behavior_data and 'energy' in self.behavior_data:
            energy = self.behavior_data['energy']
            ax.hist(energy, bins=50, color='forestgreen', edgecolor='black', alpha=0.7)
            ax.set_xlabel('Energy')
            ax.set_ylabel('Frequency')
            ax.set_title('Energy Distribution')
        
        # 4. Position heatmap
        ax = axes[1, 1]
        if self.behavior_data and 'pos_x' in self.behavior_data:
            pos_x = self.behavior_data['pos_x']
            pos_y = self.behavior_data['pos_y']
            # Create 2D histogram
            heatmap, xedges, yedges = np.histogram2d(
                pos_x, pos_y, bins=20
            )
            ax.imshow(heatmap.T, origin='lower', cmap='hot', aspect='auto')
            ax.set_xlabel('X Position')
            ax.set_ylabel('Y Position')
            ax.set_title('Position Heatmap')
        
        plt.tight_layout()
        return fig
    
    def generate_tsne_evolution(self) -> Figure | None:
        """Generate t-SNE evolution figure."""
        if self.internal_states is None or len(self.internal_states) < 100:
            return None
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        from sklearn.manifold import TSNE
        
        n_samples = len(self.internal_states)
        third = n_samples // 3
        
        for i, (ax, title) in enumerate(zip(axes, ['Early', 'Middle', 'Late'])):
            start = i * third
            end = start + third
            states_chunk = self.internal_states[start:end]
            
            # Sample if too many
            if len(states_chunk) > 500:
                indices = np.random.choice(len(states_chunk), 500, replace=False)
                states_chunk = states_chunk[indices]
            
            # t-SNE
            tsne = TSNE(n_components=2, perplexity=30, random_state=42)
            coords = tsne.fit_transform(states_chunk)
            
            ax.scatter(coords[:, 0], coords[:, 1], alpha=0.5, s=5)
            ax.set_xlabel('t-SNE 1')
            ax.set_ylabel('t-SNE 2')
            ax.set_title(f'{title} Phase (steps {start}-{end})')
        
        plt.tight_layout()
        return fig
    
    def generate_summary_stats(self) -> dict[str, Any]:
        """Generate summary statistics."""
        stats = {
            'total_steps': 0,
            'total_deaths': 0,
            'avg_lifespan': 0,
            'avg_energy': 0,
            'food_efficiency': 0,
            'learning_improvement': 0,
        }
        
        if self.behavior_data:
            stats['total_steps'] = len(self.behavior_data.get('step', []))
            stats['avg_energy'] = np.mean(self.behavior_data.get('energy', [0]))
            stats['avg_lifespan'] = np.mean(
                self.behavior_data.get('current_lifespan', [0])
            )
        
        if self.learning_data:
            pred_loss = self.learning_data.get('prediction_loss', [])
            if len(pred_loss) > 10:
                initial = np.mean(pred_loss[:10])
                final = np.mean(pred_loss[-10:])
                if initial > 0:
                    stats['learning_improvement'] = 1 - (final / initial)
        
        return stats
    
    def generate_report(self) -> dict[str, str]:
        """Generate complete report with all figures.
        
        Returns:
            Dictionary mapping figure names to file paths
        """
        print("Loading data...")
        self.load_data()
        
        print("Generating learning curves...")
        fig1 = self.generate_learning_curves()
        path1 = self.analysis_dir / "learning_curves.png"
        fig1.savefig(str(path1), dpi=150, bbox_inches='tight')
        plt.close(fig1)
        
        print("Generating behavior analysis...")
        fig2 = self.generate_behavior_analysis()
        path2 = self.analysis_dir / "behavior_analysis.png"
        fig2.savefig(str(path2), dpi=150, bbox_inches='tight')
        plt.close(fig2)
        
        print("Generating t-SNE evolution...")
        fig3 = self.generate_tsne_evolution()
        if fig3 is not None:
            path3 = self.analysis_dir / "tsne_evolution.png"
            fig3.savefig(str(path3), dpi=150, bbox_inches='tight')
            plt.close(fig3)
        else:
            path3 = None
        
        print("Generating summary statistics...")
        stats = self.generate_summary_stats()
        stats_path = self.analysis_dir / "summary_stats.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        # Generate text report
        report_lines = [
            "=" * 60,
            "Cogito Baseline Analysis Report",
            "=" * 60,
            "",
            "Summary Statistics:",
            f"  Total steps analyzed: {stats['total_steps']:,}",
            f"  Average energy: {stats['avg_energy']:.1f}",
            f"  Average lifespan: {stats['avg_lifespan']:.1f}",
            f"  Learning improvement: {stats['learning_improvement']:.1%}",
            "",
            "Generated Figures:",
            f"  - learning_curves.png",
            f"  - behavior_analysis.png",
            f"  - tsne_evolution.png" if path3 else "  - tsne_evolution.png (skipped)",
            "",
            "Data Files:",
            f"  - summary_stats.json",
            "",
            "=" * 60,
        ]
        
        report_path = self.analysis_dir / "baseline_report.txt"
        with open(report_path, 'w') as f:
            f.write('\n'.join(report_lines))
        
        print('\n'.join(report_lines))
        
        return {
            'learning_curves': str(path1),
            'behavior_analysis': str(path2),
            'tsne_evolution': str(path3) if path3 else None,
            'summary_stats': str(stats_path),
            'text_report': str(report_path),
        }


def main():
    """Generate baseline report from command line."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate baseline analysis report")
    parser.add_argument(
        "--data-dir", type=str, default="data",
        help="Path to data directory"
    )
    
    args = parser.parse_args()
    
    report = BaselineReport(args.data_dir)
    report.generate_report()


if __name__ == "__main__":
    main()
