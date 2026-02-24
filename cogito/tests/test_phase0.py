"""Phase 0 integration test - full world simulation with random agent.

This script runs 1000 steps with a random-action agent to verify
all basic mechanisms work correctly:
    - World creation and update
    - Observation vector generation
    - Energy system (food, danger, step cost)
    - Food respawning
    - Death and respawn
    - Rendering (optional)
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator

from cogito.config import Config
from cogito.world.grid import CogitoWorld
from cogito.world.renderer import WorldRenderer


def run_phase0_test(
    num_steps: int = 1000,
    seed: int = 42,
    render: bool = False,
    headless: bool = True,
    verbose: bool = True,
) -> dict:
    """Run Phase 0 integration test.

    Args:
        num_steps: Number of simulation steps.
        seed: Random seed for reproducibility.
        render: Whether to render the world.
        headless: If True, don't show window (for batch runs).
        verbose: Print progress every 100 steps.

    Returns:
        Dictionary with test statistics.
    """
    rng = np.random.default_rng(seed)
    world = CogitoWorld(Config, rng)

    # Agent state
    agent_pos = world.get_random_empty_position()
    agent_energy = Config.INITIAL_ENERGY
    is_alive = True

    # Statistics
    stats = {
        "total_steps": 0,
        "deaths": 0,
        "food_eaten": 0,
        "energy_history": [],
        "lifespan_history": [],
        "current_lifespan": 0,
        "observation_dims_correct": True,
        "observation_values_valid": True,
        "food_count_stable": True,
    }

    # Renderer
    renderer = None
    if render:
        renderer = WorldRenderer(Config, headless=headless, render_interval=100)

    try:
        for step in range(num_steps):
            # Check food count
            if world.count_food() != Config.NUM_FOOD:
                stats["food_count_stable"] = False

            # Respawn if dead
            if not is_alive:
                stats["lifespan_history"].append(stats["current_lifespan"])
                stats["current_lifespan"] = 0
                agent_pos = world.get_random_empty_position()
                agent_energy = Config.INITIAL_ENERGY
                is_alive = True

            # Get observation
            obs = world.get_observation(agent_pos)

            # Validate observation
            if obs.shape != (256,):
                stats["observation_dims_correct"] = False

            if np.any(obs[:98] < 0) or np.any(obs[:98] > 1):
                stats["observation_values_valid"] = False

            # Random action
            action = int(rng.integers(0, Config.NUM_ACTIONS))

            # If agent is on food, higher chance to eat
            if world.grid[agent_pos[0], agent_pos[1]] == 2:  # FOOD
                if rng.random() < 0.5:
                    action = 4  # eat

            # Track food count before action
            food_before = world.count_food()

            # Execute step
            new_pos, energy_change, is_dead = world.step(
                agent_pos, action, agent_energy
            )

            # Track food eaten by energy gain (food_energy - step_cost > 0)
            if energy_change > 0:  # Positive energy change means food was eaten
                stats["food_eaten"] += 1

            agent_pos = new_pos
            agent_energy = max(0, min(Config.MAX_ENERGY, agent_energy + energy_change))

            if is_dead:
                stats["deaths"] += 1
                is_alive = False

            # Update world (danger movement, etc.)
            world.update(step)

            # Update stats
            stats["total_steps"] = step + 1
            stats["current_lifespan"] += 1
            stats["energy_history"].append(agent_energy)

            # Render
            if renderer:
                renderer.render(world, agent_pos, agent_energy, step)

            # Progress print
            if verbose and (step + 1) % 100 == 0:
                avg_energy = np.mean(stats["energy_history"][-100:])
                avg_lifespan = (
                    np.mean(stats["lifespan_history"][-10:])
                    if stats["lifespan_history"]
                    else stats["current_lifespan"]
                )
                print(
                    f"Step {step + 1:5d} | "
                    f"Avg Energy: {avg_energy:5.1f} | "
                    f"Lifespan: {stats['current_lifespan']:4d} | "
                    f"Deaths: {stats['deaths']:3d} | "
                    f"Food: {world.count_food():2d}"
                )

        # Add final lifespan if agent is alive
        if stats["current_lifespan"] > 0:
            stats["lifespan_history"].append(stats["current_lifespan"])

    finally:
        if renderer:
            renderer.close()

    return stats


def validate_phase0(stats: dict) -> bool:
    """Validate Phase 0 acceptance criteria.

    Args:
        stats: Statistics from run_phase0_test.

    Returns:
        True if all checks pass.
    """
    checks = {
        "Script runs without crash": True,
        "Observation dims correct (256,)": stats["observation_dims_correct"],
        "Observation values in [0,1]": stats["observation_values_valid"],
        "Food count stable": stats["food_count_stable"],
        "Deaths occurred": stats["deaths"] > 0,
        "Food eaten": stats["food_eaten"] > 0,
        "Energy fluctuates": len(set(stats["energy_history"])) > 10,
    }

    all_passed = all(checks.values())

    print("\n=== Phase 0 Validation ===")
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {check}")

    print(f"\nStatistics:")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Deaths: {stats['deaths']}")
    print(f"  Food eaten: {stats['food_eaten']}")
    print(f"  Avg lifespan: {np.mean(stats['lifespan_history']):.1f}")
    print(f"  Avg energy: {np.mean(stats['energy_history']):.1f}")

    return all_passed


def main():
    """Run Phase 0 integration test."""
    print("=== Phase 0 Integration Test ===")
    print(f"Running {1000} steps with random actions...\n")

    stats = run_phase0_test(
        num_steps=1000,
        seed=42,
        render=False,
        headless=True,
        verbose=True,
    )

    all_passed = validate_phase0(stats)

    if all_passed:
        print("\n*** Phase 0 PASSED ***")
        return 0
    else:
        print("\n*** Phase 0 FAILED ***")
        return 1


if __name__ == "__main__":
    exit(main())
