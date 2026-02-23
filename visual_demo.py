"""Enhanced visual demo of Cogito agent with monster avatar.

Features:
    - Monster avatar (cute creature)
    - Pause/Resume with spacebar
    - Movement trail
    - Energy bar
    - Real-time stats
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import matplotlib
# Use TkAgg backend for better interactivity on Windows
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Circle, Ellipse, Polygon, FancyBboxPatch
from matplotlib.animation import FuncAnimation
import numpy as np

from cogito.config import Config
from cogito.world.grid import CogitoWorld
from cogito.agent.cogito_agent import CogitoAgent


# Logs directory
LOGS_DIR = Path(__file__).parent / "logs"


def _get_next_log_number() -> int:
    """Get the next log file sequence number."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    existing = list(LOGS_DIR.glob("*.json"))
    if not existing:
        return 1
    numbers = []
    for f in existing:
        try:
            num = int(f.stem.split("_")[0])
            numbers.append(num)
        except (ValueError, IndexError):
            continue
    return max(numbers) + 1 if numbers else 1


# Color scheme
COLORS = {
    0: '#F5F5F5',  # Empty: light gray
    1: '#2C3E50',  # Wall: dark blue-gray
    2: '#27AE60',  # Food: green
    3: '#E74C3C',  # Danger: red
    4: '#F1C40F',  # Echo zone: yellow
    5: '#3498DB',  # Hidden interface: blue
    'trail': '#BDC3C7',  # Trail: light gray
}


class PixelTrail:
    """Draw pixel-art style trail."""
    
    @staticmethod
    def draw(ax, trail, current_energy, max_length=30):
        """Draw trail as pixel blocks with fading effect.
        
        Args:
            ax: Matplotlib axes
            trail: List of (y, x) positions
            current_energy: Agent's current energy for color
            max_length: Maximum trail length
        """
        if len(trail) < 2:
            return []
        
        artists = []
        trail_len = min(len(trail), max_length)
        
        # Trail color based on monster state
        if current_energy > 70:
            base_color = np.array([46, 204, 113])  # Green
        elif current_energy > 30:
            base_color = np.array([241, 196, 15])  # Yellow
        else:
            base_color = np.array([231, 76, 60])  # Red
        
        for i, pos in enumerate(trail[-trail_len:]):
            # Fade effect: older = more transparent
            alpha = (i + 1) / trail_len * 0.6
            
            # Size: newer = larger
            size = 0.3 + (i / trail_len) * 0.4
            
            # Color with fade
            color = base_color / 255
            
            y, x = pos
            
            # Draw pixel block (square with slight rounding)
            block = FancyBboxPatch(
                (x - size/2, y - size/2),
                size, size,
                boxstyle="round,pad=0.02,rounding_size=0.1",
                facecolor=color,
                edgecolor='none',
                alpha=alpha,
                zorder=2
            )
            ax.add_patch(block)
            artists.append(block)
            
            # Add subtle glow for newer blocks
            if i > trail_len * 0.7:
                glow = Circle(
                    (x, y), radius=size * 0.8,
                    facecolor=color,
                    edgecolor='none',
                    alpha=alpha * 0.3,
                    zorder=1
                )
                ax.add_patch(glow)
                artists.append(glow)
        
        return artists


class MonsterAvatar:
    """Draw a cute monster avatar."""
    
    @staticmethod
    def draw(ax, x, y, energy, scale=1.0):
        """Draw a monster at position (x, y).
        
        Color changes based on energy:
            - High (>70): Green monster
            - Medium (30-70): Yellow monster  
            - Low (<30): Red monster
        """
        # Determine color based on energy
        if energy > 70:
            body_color = '#2ECC71'  # Green
            eye_color = '#FFFFFF'
        elif energy > 30:
            body_color = '#F1C40F'  # Yellow
            eye_color = '#FFFFFF'
        else:
            body_color = '#E74C3C'  # Red
            eye_color = '#FFFFFF'
        
        s = scale
        artists = []
        
        # Body (round blob)
        body = Ellipse(
            (x, y), width=0.8*s, height=0.7*s,
            facecolor=body_color,
            edgecolor='#2C3E50',
            linewidth=2,
            zorder=10
        )
        ax.add_patch(body)
        artists.append(body)
        
        # Eyes (two big cute eyes)
        # Left eye
        left_eye_white = Circle(
            (x - 0.15*s, y + 0.1*s), radius=0.15*s,
            facecolor='white',
            edgecolor='#2C3E50',
            linewidth=1,
            zorder=11
        )
        ax.add_patch(left_eye_white)
        artists.append(left_eye_white)
        
        left_pupil = Circle(
            (x - 0.12*s, y + 0.12*s), radius=0.07*s,
            facecolor='#2C3E50',
            zorder=12
        )
        ax.add_patch(left_pupil)
        artists.append(left_pupil)
        
        # Right eye
        right_eye_white = Circle(
            (x + 0.15*s, y + 0.1*s), radius=0.15*s,
            facecolor='white',
            edgecolor='#2C3E50',
            linewidth=1,
            zorder=11
        )
        ax.add_patch(right_eye_white)
        artists.append(right_eye_white)
        
        right_pupil = Circle(
            (x + 0.18*s, y + 0.12*s), radius=0.07*s,
            facecolor='#2C3E50',
            zorder=12
        )
        ax.add_patch(right_pupil)
        artists.append(right_pupil)
        
        # Mouth (small smile)
        mouth_x = [x - 0.15*s, x, x + 0.15*s]
        mouth_y = [y - 0.1*s, y - 0.2*s, y - 0.1*s]
        mouth, = ax.plot(mouth_x, mouth_y, color='#2C3E50', linewidth=2, zorder=11)
        artists.append(mouth)
        
        # Little antenna/horn
        horn, = ax.plot(
            [x, x], [y + 0.35*s, y + 0.5*s],
            color=body_color, linewidth=3, zorder=9
        )
        artists.append(horn)
        
        horn_tip = Circle(
            (x, y + 0.52*s), radius=0.08*s,
            facecolor=body_color,
            edgecolor='#2C3E50',
            linewidth=1,
            zorder=10
        )
        ax.add_patch(horn_tip)
        artists.append(horn_tip)
        
        # Small feet
        left_foot = Ellipse(
            (x - 0.2*s, y - 0.35*s), width=0.25*s, height=0.12*s,
            facecolor=body_color,
            edgecolor='#2C3E50',
            linewidth=1,
            zorder=9
        )
        ax.add_patch(left_foot)
        artists.append(left_foot)
        
        right_foot = Ellipse(
            (x + 0.2*s, y - 0.35*s), width=0.25*s, height=0.12*s,
            facecolor=body_color,
            edgecolor='#2C3E50',
            linewidth=1,
            zorder=9
        )
        ax.add_patch(right_foot)
        artists.append(right_foot)
        
        return artists


class VisualDemo:
    """Enhanced visual demo with pause/resume and monster avatar."""
    
    def __init__(
        self,
        num_steps: int = 10000,
        speed: float = 0.05,
        trail_length: int = 30,
    ):
        self.num_steps = num_steps
        self.speed = speed
        self.trail_length = trail_length
        
        # World and agent
        self.world = CogitoWorld(Config)
        self.agent = CogitoAgent(Config)
        
        # State
        self.agent_pos = self.world.get_random_empty_position()
        self.agent_energy = float(Config.INITIAL_ENERGY)
        self.trail = []
        
        # Stats
        self.step_count = 0
        self.deaths = 0
        self.food_eaten = 0
        self.lifespan = 0
        self.max_lifespan = 0
        
        # Pause control
        self.paused = False
        
        # Monster artists (for clearing)
        self.monster_artists = []
        
        # Initialize log
        self._init_log()
        
        self.setup_visuals()
    
    def _init_log(self):
        """Initialize log data structure."""
        self.log_number = _get_next_log_number()
        self.log_data = {
            "run_info": {
                "log_number": self.log_number,
                "start_time": datetime.now().isoformat(),
                "max_steps": self.num_steps,
            },
            "config": {
                "world_size": Config.WORLD_SIZE,
                "initial_energy": Config.INITIAL_ENERGY,
                "step_cost": Config.STEP_COST,
                "food_energy": Config.FOOD_ENERGY,
                "danger_penalty": Config.DANGER_PENALTY,
            },
            "events": [],
            "summary": {},
        }
    
    def _log_event(self, event_type: str, data: dict):
        """Log an event."""
        event = {
            "step": self.step_count,
            "time": datetime.now().isoformat(),
            "type": event_type,
            "data": data,
        }
        self.log_data["events"].append(event)
    
    def _save_log(self):
        """Save log to JSON file."""
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        
        # Finalize summary
        self.log_data["run_info"]["end_time"] = datetime.now().isoformat()
        self.log_data["run_info"]["duration_seconds"] = time.time() - self._start_time if hasattr(self, '_start_time') else 0
        self.log_data["summary"] = {
            "total_steps": self.step_count,
            "deaths": self.deaths,
            "food_eaten": self.food_eaten,
            "max_lifespan": self.max_lifespan,
        }
        
        # Save to file
        filename = f"{self.log_number:04d}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = LOGS_DIR / filename
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.log_data, f, indent=2, ensure_ascii=False)
        
        return filepath
    
    def setup_visuals(self):
        """Setup the figure and axes."""
        plt.style.use('default')
        
        self.fig = plt.figure(figsize=(14, 9))
        self.fig.patch.set_facecolor('#2C3E50')
        
        # Main grid area (left side)
        self.ax_main = self.fig.add_axes([0.02, 0.05, 0.62, 0.90])
        self.ax_main.set_facecolor('#1a1a2e')
        
        # Right panel - single unified area
        self.ax_right = self.fig.add_axes([0.66, 0.05, 0.32, 0.90])
        self.ax_right.set_facecolor('#ECF0F1')
        self.ax_right.axis('off')
        
        # Grid image
        self.grid_image = self.ax_main.imshow(
            self._get_colored_grid(),
            interpolation='nearest',
            aspect='equal',
        )
        
        # Trail artists (for clearing)
        self.trail_artists = []
        
        self.ax_main.set_xlim(-0.5, Config.WORLD_SIZE - 0.5)
        self.ax_main.set_ylim(Config.WORLD_SIZE - 0.5, -0.5)
        self.ax_main.set_xticks([])
        self.ax_main.set_yticks([])
        for spine in self.ax_main.spines.values():
            spine.set_color('#3498DB')
            spine.set_linewidth(2)
        
        # Draw all right panel content
        self._draw_right_panel()
        
        # Legend on main grid
        legend_elements = [
            patches.Patch(facecolor=COLORS[2], label='Food (+20)', edgecolor='white'),
            patches.Patch(facecolor=COLORS[3], label='Danger (-10)', edgecolor='white'),
            patches.Patch(facecolor=COLORS[1], label='Wall', edgecolor='white'),
        ]
        self.ax_main.legend(
            handles=legend_elements,
            loc='upper right',
            fontsize=9,
            framealpha=0.9,
            facecolor='#2C3E50',
            edgecolor='#3498DB',
            labelcolor='white',
        )
        
        # Key press handler
        self.fig.canvas.mpl_connect('key_press_event', self._on_key)
        # Mouse click handler
        self.fig.canvas.mpl_connect('button_press_event', self._on_click)
    
    def _on_click(self, event):
        """Handle mouse click on pause button."""
        # Check if click is in right panel area
        if event.inaxes == self.ax_right and event.xdata is not None:
            # Button bounds in axes coords: x=[0.20, 0.80], y=[0.48, 0.56]
            if 0.15 <= event.xdata <= 0.85 and 0.45 <= event.ydata <= 0.60:
                self.paused = not self.paused
                self._draw_right_panel()
                self.fig.canvas.draw_idle()
    
    def _draw_right_panel(self):
        """Draw entire right panel content."""
        self.ax_right.clear()
        self.ax_right.axis('off')
        self.ax_right.set_xlim(0, 1)
        self.ax_right.set_ylim(0, 1)
        
        # ========== TITLE SECTION (top) ==========
        title_bg = FancyBboxPatch(
            (0.02, 0.90), 0.96, 0.08,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor='#2C3E50',
            edgecolor='#3498DB',
            linewidth=2,
            transform=self.ax_right.transAxes,
            zorder=1
        )
        self.ax_right.add_patch(title_bg)
        
        self.ax_right.text(0.5, 0.935, 'COGITO', 
            transform=self.ax_right.transAxes,
            fontsize=18, fontweight='bold', color='white',
            ha='center', va='center')
        self.ax_right.text(0.5, 0.905, 'Emergent Self-Awareness Experiment',
            transform=self.ax_right.transAxes,
            fontsize=8, color='#BDC3C7',
            ha='center', va='center')
        
        # ========== ENERGY SECTION (prominent) ==========
        energy_bg = FancyBboxPatch(
            (0.02, 0.76), 0.96, 0.12,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor='white',
            edgecolor='#BDC3C7',
            linewidth=1,
            transform=self.ax_right.transAxes,
            zorder=1
        )
        self.ax_right.add_patch(energy_bg)
        
        # Energy status
        if self.agent_energy > 70:
            status, status_color = "HEALTHY", "#27AE60"
        elif self.agent_energy > 30:
            status, status_color = "WARNING", "#F39C12"
        else:
            status, status_color = "CRITICAL", "#E74C3C"
        
        self.ax_right.text(0.06, 0.855, "ENERGY",
            transform=self.ax_right.transAxes,
            fontsize=9, fontweight='bold', color='#7F8C8D')
        self.ax_right.text(0.06, 0.82, f"{self.agent_energy:.0f}/100",
            transform=self.ax_right.transAxes,
            fontsize=16, fontweight='bold', color=status_color)
        self.ax_right.text(0.94, 0.835, status,
            transform=self.ax_right.transAxes,
            fontsize=10, fontweight='bold', color=status_color, ha='right', va='center')
        
        # Energy bar
        bar_bg = FancyBboxPatch(
            (0.06, 0.775), 0.88, 0.035,
            boxstyle="round,pad=0.003,rounding_size=0.01",
            facecolor='#E8E8E8',
            edgecolor='#CCCCCC',
            linewidth=1,
            transform=self.ax_right.transAxes,
            zorder=2
        )
        self.ax_right.add_patch(bar_bg)
        
        # Energy fill
        fill_width = 0.88 * (self.agent_energy / 100)
        if fill_width > 0:
            bar_fill = FancyBboxPatch(
                (0.06, 0.775), fill_width, 0.035,
                boxstyle="round,pad=0.003,rounding_size=0.01",
                facecolor=status_color,
                edgecolor='none',
                transform=self.ax_right.transAxes,
                zorder=3
            )
            self.ax_right.add_patch(bar_fill)
        
        # ========== STATS SECTION ==========
        stats_bg = FancyBboxPatch(
            (0.02, 0.58), 0.96, 0.16,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor='white',
            edgecolor='#BDC3C7',
            linewidth=1,
            transform=self.ax_right.transAxes,
            zorder=1
        )
        self.ax_right.add_patch(stats_bg)
        
        self.ax_right.text(0.06, 0.715, "STATISTICS",
            transform=self.ax_right.transAxes,
            fontsize=9, fontweight='bold', color='#7F8C8D')
        
        # Stats in 2 columns
        stats = [
            ("Steps", f"{self.step_count:,}"),
            ("Lifespan", f"{self.lifespan:,}"),
            ("Deaths", f"{self.deaths}"),
            ("Best Run", f"{self.max_lifespan:,}"),
            ("Food", f"{self.food_eaten}"),
            ("Pos", f"({self.agent_pos[0]},{self.agent_pos[1]})"),
        ]
        
        col1_x, col2_x = 0.06, 0.52
        for i, (label, value) in enumerate(stats):
            col = i % 2
            row = i // 2
            x = col1_x if col == 0 else col2_x
            y = 0.68 - row * 0.045
            self.ax_right.text(x, y, f"{label}:",
                transform=self.ax_right.transAxes,
                fontsize=8, color='#95A5A6')
            self.ax_right.text(x + 0.18, y, value,
                transform=self.ax_right.transAxes,
                fontsize=9, fontweight='bold', color='#2C3E50', fontfamily='monospace')
        
        # ========== PAUSE BUTTON ==========
        btn_color = '#27AE60' if self.paused else '#E74C3C'
        btn_text = 'RESUME' if self.paused else 'PAUSE'
        btn_symbol = '[>]' if self.paused else '[||]'
        
        button_bg = FancyBboxPatch(
            (0.20, 0.48), 0.60, 0.08,
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=btn_color,
            edgecolor='white',
            linewidth=2,
            transform=self.ax_right.transAxes,
            zorder=1
        )
        self.ax_right.add_patch(button_bg)
        
        self.ax_right.text(0.5, 0.525, btn_text,
            transform=self.ax_right.transAxes,
            fontsize=13, fontweight='bold', color='white',
            ha='center', va='center')
        self.ax_right.text(0.5, 0.485, '(press SPACE)',
            transform=self.ax_right.transAxes,
            fontsize=7, color='#FFFFFF', alpha=0.8,
            ha='center', va='center')
        
        # ========== RULES SECTION ==========
        rules_bg = FancyBboxPatch(
            (0.02, 0.02), 0.96, 0.44,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor='white',
            edgecolor='#BDC3C7',
            linewidth=1,
            transform=self.ax_right.transAxes,
            zorder=1
        )
        self.ax_right.add_patch(rules_bg)
        
        # PROJECT
        self.ax_right.text(0.06, 0.43, 'PROJECT',
            transform=self.ax_right.transAxes,
            fontsize=9, fontweight='bold', color='#3498DB')
        self.ax_right.text(0.06, 0.395,
            'An AI agent learns to survive in a 64x64 grid.',
            transform=self.ax_right.transAxes,
            fontsize=8, color='#2C3E50')
        
        # Separator line
        self.ax_right.plot([0.06, 0.94], [0.365, 0.365], 
            color='#E0E0E0', linewidth=1, transform=self.ax_right.transAxes)
        
        # RULES
        self.ax_right.text(0.06, 0.335, 'RULES',
            transform=self.ax_right.transAxes,
            fontsize=9, fontweight='bold', color='#3498DB')
        rules = [
            'Green tile = Food (+20 energy)',
            'Red tile = Danger (-10 energy)',
            'Each step costs -1 energy',
            'Energy <= 0 causes death',
        ]
        for i, rule in enumerate(rules):
            self.ax_right.text(0.08, 0.30 - i * 0.035, rule,
                transform=self.ax_right.transAxes,
                fontsize=8, color='#2C3E50')
        
        # Separator line
        self.ax_right.plot([0.06, 0.94], [0.16, 0.16],
            color='#E0E0E0', linewidth=1, transform=self.ax_right.transAxes)
        
        # CONTROLS
        self.ax_right.text(0.06, 0.13, 'CONTROLS',
            transform=self.ax_right.transAxes,
            fontsize=9, fontweight='bold', color='#3498DB')
        self.ax_right.text(0.06, 0.095, 'SPACE: Pause/Resume',
            transform=self.ax_right.transAxes,
            fontsize=8, color='#2C3E50')
        self.ax_right.text(0.06, 0.065, 'Q: Quit',
            transform=self.ax_right.transAxes,
            fontsize=8, color='#2C3E50')
    
    def _on_key(self, event):
        """Handle key press."""
        if event.key == ' ':
            self.paused = not self.paused
            self._draw_right_panel()
            self.fig.canvas.draw_idle()
        elif event.key == 'q':
            plt.close(self.fig)
    
    def _get_colored_grid(self) -> np.ndarray:
        """Convert grid to RGB image."""
        grid = self.world.grid
        rgb = np.zeros((Config.WORLD_SIZE, Config.WORLD_SIZE, 3), dtype=np.uint8)
        
        for i in range(Config.WORLD_SIZE):
            for j in range(Config.WORLD_SIZE):
                cell_type = int(grid[i, j])
                hex_color = COLORS.get(cell_type, '#FFFFFF')
                hex_color = hex_color.lstrip('#')
                rgb[i, j] = tuple(int(hex_color[k:k+2], 16) for k in (0, 2, 4))
        
        return rgb
    
    def _update_stats(self):
        """Update stats panel - redraw right panel."""
        pass  # Stats are now part of _draw_right_panel()
    
    def _update_energy_bar(self):
        """Update energy bar - redraw right panel."""
        pass  # Energy is now part of _draw_right_panel()
    
    def _clear_monster(self):
        """Remove previous monster drawing."""
        for artist in self.monster_artists:
            try:
                artist.remove()
            except:
                pass
        self.monster_artists.clear()
    
    def _clear_trail(self):
        """Clear old trail artists."""
        for artist in self.trail_artists:
            try:
                artist.remove()
            except Exception:
                pass
        self.trail_artists.clear()
    
    def _draw_monster(self):
        """Draw the monster avatar."""
        self._clear_monster()
        
        # Convert grid coords to plot coords (y is inverted)
        plot_x = self.agent_pos[1]
        plot_y = self.agent_pos[0]
        
        self.monster_artists = MonsterAvatar.draw(
            self.ax_main, plot_x, plot_y, self.agent_energy, scale=1.2
        )
    
    def step(self):
        """Execute one simulation step."""
        if self.paused:
            return
        
        # Get observation and act
        observation = self.world.get_observation(self.agent_pos)
        action, info = self.agent.act(observation, self.agent_energy)
        
        # Execute action
        new_pos, energy_change, is_dead = self.world.step(
            self.agent_pos, action, self.agent_energy
        )
        
        # Update trail
        self.trail.append(self.agent_pos)
        if len(self.trail) > self.trail_length:
            self.trail.pop(0)
        
        # Update state
        old_pos = self.agent_pos
        self.agent_pos = new_pos
        self.agent_energy = max(0, self.agent_energy + energy_change)
        
        # Track food and log
        if energy_change > Config.STEP_COST:
            self.food_eaten += 1
            self._log_event("food_eaten", {
                "position": list(new_pos),
                "energy_after": self.agent_energy,
            })
        
        self.step_count += 1
        self.lifespan += 1
        self.max_lifespan = max(self.max_lifespan, self.lifespan)
        
        # Update world
        self.world.update(self.step_count)
        
        # Handle death
        done = is_dead or self.agent_energy <= 0
        if done:
            cause = "danger" if is_dead else "starvation"
            self._log_event("death", {
                "cause": cause,
                "position": list(old_pos),
                "lifespan": self.lifespan,
            })
            self.deaths += 1
            self.lifespan = 0
            self.agent_pos = self.world.get_random_empty_position()
            self.agent_energy = float(Config.INITIAL_ENERGY)
            self.agent.reset_on_death()
            self.trail.clear()
    
    def update(self, frame):
        """Animation update function."""
        if frame >= self.num_steps:
            plt.close(self.fig)
            return []
        
        # Run step
        self.step()
        
        # Update visuals
        self.grid_image.set_data(self._get_colored_grid())
        
        # Clear old trail artists
        self._clear_trail()
        
        # Draw pixel-aligned trail
        self.trail_artists = PixelTrail.draw(self.ax_main, self.trail, self.agent_energy)
        
        # Draw monster
        self._draw_monster()
        
        # Update right panel
        self._draw_right_panel()
        
        return [self.grid_image]
    
    def run(self):
        """Run the visual demo."""
        self._start_time = time.time()
        
        ani = FuncAnimation(
            self.fig,
            self.update,
            frames=self.num_steps,
            interval=50,  # 50ms = 20 FPS
            blit=False,
            repeat=False,
        )
        
        plt.show()
        
        # Save log
        log_path = self._save_log()
        
        # Final stats
        print("\n" + "=" * 40)
        print("Demo Complete!")
        print("=" * 40)
        print(f"Steps: {self.step_count:,}")
        print(f"Deaths: {self.deaths}")
        print(f"Food eaten: {self.food_eaten}")
        print(f"Max lifespan: {self.max_lifespan}")
        print(f"\nLog saved: {log_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Visual demo with monster avatar")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--speed", type=float, default=0.05)
    
    args = parser.parse_args()
    
    demo = VisualDemo(num_steps=args.steps, speed=args.speed)
    demo.run()


if __name__ == "__main__":
    main()