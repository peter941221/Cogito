"""World visualization using matplotlib."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from cogito.config import Config
from cogito.world.grid import EMPTY, WALL, FOOD, DANGER, ECHO_ZONE, HIDDEN_INTERFACE

if TYPE_CHECKING:
    from cogito.world.grid import CogitoWorld

# Color scheme (RGBA tuples)
COLORS = {
    EMPTY: (1.0, 1.0, 1.0),      # White
    WALL: (0.3, 0.3, 0.3),       # Dark gray
    FOOD: (0.2, 0.8, 0.2),       # Green
    DANGER: (0.9, 0.2, 0.2),     # Red
    ECHO_ZONE: (1.0, 1.0, 0.2),  # Yellow (when active)
    HIDDEN_INTERFACE: (0.2, 0.4, 0.9),  # Blue (when active)
    "agent": (0.0, 0.0, 0.0),    # Black
}


class WorldRenderer:
    """Matplotlib-based world visualization.

    Supports:
        - Real-time rendering with configurable frequency
        - Headless mode (no window, for remote/batch runs)
        - Frame saving to files
    """

    def __init__(
        self,
        config: type[Config],
        headless: bool = True,
        render_interval: int = 100,
    ):
        """Initialize the renderer.

        Args:
            config: Configuration dataclass.
            headless: If True, don't display window (for remote runs).
            render_interval: Render every N steps.
        """
        self.config = config
        self.headless = headless
        self.render_interval = render_interval
        self.size = config.WORLD_SIZE

        # Setup matplotlib
        if headless:
            plt.ioff()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
        else:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))

        # Initialize image
        self.image_data = np.ones((self.size, self.size, 3))
        self.im = self.ax.imshow(self.image_data, origin="lower", aspect="equal")

        # Remove axis ticks
        self.ax.set_xticks([])
        self.ax.set_yticks([])

        # Text annotations
        self.energy_text = self.ax.text(
            0.02, 0.98, "", transform=self.ax.transAxes,
            fontsize=10, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

        # Agent marker
        self.agent_marker = None

        # Track step for render interval
        self.last_rendered_step = -1

    def render(
        self,
        world: CogitoWorld,
        agent_pos: tuple[int, int],
        agent_energy: float,
        step_count: int,
        force: bool = False,
    ) -> None:
        """Render current world state.

        Args:
            world: The CogitoWorld instance.
            agent_pos: Agent's (x, y) position.
            agent_energy: Current energy level.
            step_count: Current step number.
            force: If True, render regardless of interval.
        """
        # Check render interval
        if not force and step_count - self.last_rendered_step < self.render_interval:
            return

        self.last_rendered_step = step_count

        # Build RGB image from grid
        grid = world.grid
        image = np.ones((self.size, self.size, 3))

        for cell_type, color in COLORS.items():
            if isinstance(cell_type, int):
                mask = grid == cell_type
                for c in range(3):
                    image[:, :, c][mask] = color[c]

        # Update image data (flip y for correct orientation)
        self.image_data = np.flipud(image)
        self.im.set_data(self.image_data)

        # Update agent marker
        if self.agent_marker is not None:
            self.agent_marker.remove()

        # Agent position (y is flipped for display)
        ax, ay = agent_pos
        display_y = self.size - 1 - ay
        self.agent_marker = self.ax.scatter(
            ax, display_y, s=200, c="black", marker="o",
            edgecolors="white", linewidths=2, zorder=10,
        )

        # Update text
        self.energy_text.set_text(
            f"Step: {step_count} | Energy: {agent_energy:.0f}\n"
            f"Pos: ({ax}, {ay})"
        )

        # Draw
        if not self.headless:
            self.fig.canvas.draw_idle()
            plt.pause(0.001)

    def save_frame(self, filename: str) -> None:
        """Save current frame to file.

        Args:
            filename: Output file path (supports .png, .pdf, .svg, etc.).
        """
        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(filename, dpi=100, bbox_inches="tight")

    def close(self) -> None:
        """Close the figure."""
        plt.close(self.fig)

    def __enter__(self) -> WorldRenderer:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()


class BatchRenderer:
    """Renderer optimized for batch/headless operation.

    Saves frames at intervals without displaying.
    """

    def __init__(
        self,
        config: type[Config],
        output_dir: str = "data/frames",
        save_interval: int = 1000,
    ):
        """Initialize batch renderer.

        Args:
            config: Configuration dataclass.
            output_dir: Directory for saved frames.
            save_interval: Save every N steps.
        """
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_interval = save_interval
        self.frame_count = 0

        # Create a single renderer instance
        self.renderer = WorldRenderer(config, headless=True, render_interval=1)

    def render_and_save(
        self,
        world: CogitoWorld,
        agent_pos: tuple[int, int],
        agent_energy: float,
        step_count: int,
    ) -> str | None:
        """Render and save frame if at interval.

        Returns:
            Path to saved frame, or None if not saved.
        """
        if step_count % self.save_interval != 0:
            return None

        self.renderer.render(world, agent_pos, agent_energy, step_count, force=True)

        filename = self.output_dir / f"frame_{self.frame_count:06d}.png"
        self.renderer.save_frame(str(filename))
        self.frame_count += 1

        return str(filename)

    def close(self) -> None:
        """Close the renderer."""
        self.renderer.close()

    def __enter__(self) -> BatchRenderer:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit."""
        self.close()
