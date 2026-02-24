"""Experiment 6: Run experiments 1-4 with evolved architecture."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from cogito.agent.cogito_agent import CogitoAgent
from cogito.config import Config
from cogito.core.evolution_simulation import EvolutionSimulation
from cogito.evolution.genome import Genome
from cogito.experiments.exp1_sensory_deprivation import SensoryDeprivationExperiment
from cogito.experiments.exp2_digital_mirror import DigitalMirrorExperiment
from cogito.experiments.exp3_godel_rebellion import GodelRebellionExperiment
from cogito.experiments.exp4_self_symbol import SelfSymbolExperiment


def build_agent_config(genome: Genome) -> dict[str, int | float | bool]:
    params = genome.decode()
    return {
        "sensory_dim": Config.SENSORY_DIM,
        "encoded_dim": int(params["encoded_dim"]),
        "encoder_hidden_dim": int(params["encoder_hidden_dim"]),
        "encoder_num_layers": int(params["encoder_num_layers"]),
        "encoder_use_norm": bool(params["encoder_use_norm"]),
        "core_hidden_dim": int(params["core_hidden_dim"]),
        "core_num_layers": int(params["core_num_layers"]),
        "core_dropout": float(params["core_dropout"]),
        "num_actions": Config.NUM_ACTIONS,
        "action_hidden_dim": int(params["action_hidden_dim"]),
        "action_temperature": float(params["action_temperature"]),
        "prediction_hidden": int(params["prediction_hidden"]),
        "prediction_depth": int(params["prediction_depth"]),
        "learning_rate": float(params["learning_rate"]),
        "gamma": float(params["gamma"]),
        "prediction_weight": float(params["prediction_weight"]),
        "survival_weight": float(params["survival_weight"]),
        "grad_clip": float(params["grad_clip"]),
        "buffer_size": int(params["buffer_size"]),
        "batch_size": int(params["batch_size"]),
        "replay_ratio": float(params["replay_ratio"]),
        "reward_death": float(params["reward_death"]),
        "reward_food": float(params["reward_food"]),
        "reward_step": float(params["reward_step"]),
    }


def load_best_genome(checkpoint_path: Path) -> Genome:
    with open(checkpoint_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    history = payload.get("history", {})
    best_genomes = history.get("best_genome", [])
    if not best_genomes:
        raise ValueError("checkpoint missing best_genome history")
    return Genome(np.array(best_genomes[-1], dtype=np.float32))


def run_maturation(agent: CogitoAgent, steps: int, output_dir: Path) -> Path:
    sim = EvolutionSimulation().world
    agent_path = output_dir / "evolved_agent.pt"

    # Simple maturation loop
    pos = sim.get_random_empty_position()
    energy = float(Config.INITIAL_ENERGY)
    for step in range(steps):
        obs = sim.get_observation(pos)
        action, _ = agent.act(obs, energy)
        pos, energy_change, done = sim.step(pos, action, energy)
        energy = max(0.0, energy + energy_change)
        if done or energy <= 0:
            agent.reset_on_death()
            pos = sim.get_random_empty_position()
            energy = float(Config.INITIAL_ENERGY)
        sim.update(step)

    agent.save(agent_path)
    return agent_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experiments with evolved genome")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="data/exp6")
    parser.add_argument("--maturation-steps", type=int, default=Config.MATURATION_STEPS)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    best_genome = load_best_genome(checkpoint_path)
    agent_config = build_agent_config(best_genome)

    evolved_agent = CogitoAgent(agent_config)
    matured_path = run_maturation(evolved_agent, args.maturation_steps, output_dir)

    # Run experiments with evolved agent
    evolved_results = {}
    for name, exp_cls in [
        ("exp1", SensoryDeprivationExperiment),
        ("exp2", DigitalMirrorExperiment),
        ("exp3", GodelRebellionExperiment),
        ("exp4", SelfSymbolExperiment),
    ]:
        agent = CogitoAgent(agent_config)
        agent.load(matured_path)
        exp = exp_cls(data_dir=str(output_dir / name), agent=agent)
        evolved_results[name] = asdict(exp.run())

    baseline_results = {}
    if not args.skip_baseline:
        for name, exp_cls in [
            ("exp1", SensoryDeprivationExperiment),
            ("exp2", DigitalMirrorExperiment),
            ("exp3", GodelRebellionExperiment),
            ("exp4", SelfSymbolExperiment),
        ]:
            exp = exp_cls(data_dir=str(output_dir / f"baseline_{name}"))
            baseline_results[name] = asdict(exp.run())

    report = {
        "evolved": evolved_results,
        "baseline": baseline_results,
        "checkpoint": str(checkpoint_path),
    }

    report_path = output_dir / "exp6_comparison_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()
