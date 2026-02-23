"""Real-time monitoring dashboard.

Multi-panel visualization showing:
    - World view
    - Behavior stats (energy, lifespan curves)
    - t-SNE plot (internal state clusters)
    - Learning curves
    - SVC detection status
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from cogito.config import Config

if TYPE_CHECKING:
    from cogito.agent.cogito_agent import CogitoAgent
    from cogito.monitoring.state_analyzer import AnalysisResult
    from cogito.monitoring.svc_detector import SVCReport
    from cogito.world.grid import CogitoWorld


class Dashboard:
    """Multi-panel real-time monitoring dashboard.

    Layout:
        [World View]     [Behavior Stats]
        [t-SNE Plot]     [Learning Curves]
        [SVC Status & Info]
    """

    def __init__(
        self,
        config: type[Config] | None = None,
        headless: bool = True,
    ):
        """Initialize the dashboard.

        Args:
            config: Configuration class.
            headless: If True, don't display window.
        """
        self.config = config or Config
        self.headless = headless

        # Setup figure
        if headless:
            plt.ioff()
        else:
            plt.ion()

        self.fig = plt.figure(figsize=(16, 10))
        self.fig.suptitle("Cogito Monitoring Dashboard", fontsize=14)

        # Create subplots
        gs = self.fig.add_gridspec(3, 2, height_ratios=[1, 1, 0.3], hspace=0.3, wspace=0.3)

        # World view
        self.ax_world = self.fig.add_subplot(gs[0, 0])
        self.ax_world.set_title("World View")

        # Behavior stats
        self.ax_behavior = self.fig.add_subplot(gs[0, 1])
        self.ax_behavior.set_title("Behavior Stats")

        # t-SNE plot
        self.ax_tsne = self.fig.add_subplot(gs[1, 0])
        self.ax_tsne.set_title("t-SNE (Internal States)")

        # Learning curves
        self.ax_learning = self.fig.add_subplot(gs[1, 1])
        self.ax_learning.set_title("Learning Curves")

        # Info panel
        self.ax_info = self.fig.add_subplot(gs[2, :])
        self.ax_info.axis("off")

        # Initialize plot elements
        self._init_plots()

    def _init_plots(self) -> None:
        """Initialize plot elements."""
        # World view placeholder
        self.world_image = self.ax_world.imshow(
            np.zeros((64, 64, 3)),
            origin="lower",
            aspect="equal",
        )
        self.ax_world.set_xticks([])
        self.ax_world.set_yticks([])

        # Agent marker placeholder
        self.agent_marker = None

        # Behavior stats placeholder
        self.energy_line, = self.ax_behavior.plot([], [], "b-", label="Energy")
        self.ax_behavior.set_xlabel("Step")
        self.ax_behavior.set_ylabel("Value")
        self.ax_behavior.legend(loc="upper right")

        # t-SNE placeholder
        self.tsne_scatter = None

        # Learning curves placeholder
        self.pred_loss_line, = self.ax_learning.plot([], [], "r-", label="Pred Loss")
        self.surv_loss_line, = self.ax_learning.plot([], [], "g-", label="Surv Loss")
        self.ax_learning.set_xlabel("Step")
        self.ax_learning.set_ylabel("Loss")
        self.ax_learning.legend(loc="upper right")
        self.ax_learning.set_yscale("log")

        # Info text
        self.info_text = self.ax_info.text(
            0.5, 0.5, "",
            transform=self.ax_info.transAxes,
            ha="center",
            va="center",
            fontsize=10,
            family="monospace",
        )

    def update(
        self,
        step: int,
        world: CogitoWorld | None,
        agent: CogitoAgent | None,
        analysis_result: AnalysisResult | None = None,
        svc_report: SVCReport | None = None,
        energy_history: list[float] | None = None,
        learning_data: dict | None = None,
    ) -> None:
        """Update all dashboard panels.

        Args:
            step: Current simulation step.
            world: World instance for rendering.
            agent: Agent instance for stats.
            analysis_result: Internal state analysis result.
            svc_report: SVC detection report.
            energy_history: List of energy values over time.
            learning_data: Dict with learning curve data.
        """
        # Update world view
        if world is not None:
            self._update_world(world, agent)

        # Update behavior stats
        if energy_history:
            self._update_behavior(energy_history)

        # Update t-SNE
        if analysis_result is not None:
            self._update_tsne(analysis_result)

        # Update learning curves
        if learning_data:
            self._update_learning(learning_data)

        # Update info panel
        self._update_info(step, agent, svc_report)

        # Draw
        if not self.headless:
            self.fig.canvas.draw_idle()
            plt.pause(0.01)

    def _update_world(
        self,
        world: CogitoWorld,
        agent: CogitoAgent | None,
    ) -> None:
        """Update world view panel."""
        # Build RGB image from grid
        from cogito.world.grid import EMPTY, WALL, FOOD, DANGER

        colors = {
            EMPTY: (1.0, 1.0, 1.0),
            WALL: (0.3, 0.3, 0.3),
            FOOD: (0.2, 0.8, 0.2),
            DANGER: (0.9, 0.2, 0.2),
        }

        image = np.ones((world.size, world.size, 3))
        for cell_type, color in colors.items():
            mask = world.grid == cell_type
            for c in range(3):
                image[:, :, c][mask] = color[c]

        # Flip for display
        image = np.flipud(image)
        self.world_image.set_data(image)

        # Update agent marker
        if agent is not None and hasattr(agent, "_last_pos"):
            ax, ay = agent._last_pos
            display_y = world.size - 1 - ay
            if self.agent_marker is not None:
                self.agent_marker.remove()
            self.agent_marker = self.ax_world.scatter(
                ax, display_y, s=200, c="black", marker="o",
                edgecolors="white", linewidths=2, zorder=10,
            )

    def _update_behavior(self, energy_history: list[float]) -> None:
        """Update behavior stats panel."""
        steps = list(range(len(energy_history)))
        self.energy_line.set_data(steps, energy_history)

        if len(steps) > 0:
            self.ax_behavior.set_xlim(0, max(steps))
            self.ax_behavior.set_ylim(0, max(energy_history) * 1.1 + 1)

    def _update_tsne(self, analysis_result: AnalysisResult) -> None:
        """Update t-SNE panel."""
        self.ax_tsne.clear()
        self.ax_tsne.set_title("t-SNE (Internal States)")

        coords = analysis_result.tsne_coords
        labels = analysis_result.cluster_labels

        # Color by cluster
        unique_labels = set(labels)
        colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_labels)))

        for label, color in zip(unique_labels, colors, strict=False):
            mask = labels == label
            self.ax_tsne.scatter(
                coords[mask, 0],
                coords[mask, 1],
                c=[color],
                label=f"C{label}" if label >= 0 else "Noise",
                s=10,
                alpha=0.6,
            )

        self.ax_tsne.legend(loc="upper right", fontsize=8)
        self.ax_tsne.set_xticks([])
        self.ax_tsne.set_yticks([])

    def _update_learning(self, learning_data: dict) -> None:
        """Update learning curves panel."""
        steps = learning_data.get("steps", [])
        pred_loss = learning_data.get("prediction_loss", [])
        surv_loss = learning_data.get("survival_loss", [])

        if steps and pred_loss:
            self.pred_loss_line.set_data(steps, pred_loss)
        if steps and surv_loss:
            self.surv_loss_line.set_data(steps, surv_loss)

        if steps:
            self.ax_learning.set_xlim(0, max(steps))
            # Set y limits dynamically
            all_losses = pred_loss + surv_loss
            if all_losses:
                valid_losses = [l for l in all_losses if l > 0]
                if valid_losses:
                    self.ax_learning.set_ylim(
                        min(valid_losses) * 0.9,
                        max(valid_losses) * 1.1 + 0.01,
                    )

    def _update_info(
        self,
        step: int,
        agent: CogitoAgent | None,
        svc_report: SVCReport | None,
    ) -> None:
        """Update info panel."""
        lines = [f"Step: {step}"]

        if agent:
            lines.append(f"Energy: {getattr(agent, '_current_energy', 0):.1f}")
            lines.append(f"Deaths: {getattr(agent, 'times_died', 0)}")

        if svc_report:
            svc_status = "DETECTED" if svc_report.is_detected else "Not detected"
            lines.append(f"SVC: {svc_status}")
            lines.append(f"Confidence: {svc_report.confidence:.2f}")
            if svc_report.candidate_clusters:
                lines.append(f"Candidates: {svc_report.candidate_clusters}")

        self.info_text.set_text(" | ".join(lines))

    def save_snapshot(self, filename: str) -> None:
        """Save current dashboard to file.

        Args:
            filename: Output file path.
        """
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(filename, dpi=100, bbox_inches="tight")

    def close(self) -> None:
        """Close the dashboard."""
        plt.close(self.fig)

    def __enter__(self) -> Dashboard:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
