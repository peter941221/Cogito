# AGENTS.md - Project Cogito
# Instructions for agentic coding in this repo.

## Repository Status

- This workspace is fully implemented with 173 tests passing.
- Source of truth: docs/implementation_plan.md and docs/technical_plan.md.
- Project Memory: Memory.MD (Chinese, not for public)

## Environment Setup

- Python 3.10+ is required.

- Create venv: `python -m venv venv`

- Activate (Windows): `venv\Scripts\activate`

- Activate (POSIX): `source venv/bin/activate`

- Install dependencies:

  ```bash
  pip install -r requirements.txt
  ```

## Build / Run / Test / Lint

- Build: none (pure Python). Run modules and scripts directly.

- Run simulations:

  ```bash
  python run_maturation.py --steps 10000      # Standard agent
  python run_bio.py --steps 10000              # Bio-inspired agent
  python run_evolution.py --small              # Evolution (generational)
  python run_continuous_evolution.py --steps 5000  # Evolution (continuous)
  ```

- Test all: `pytest`

- Test single file: `pytest cogito/tests/test_phase0.py`

- Test single test: `pytest cogito/tests/test_phase0.py::TestClass::test_name`

- Test by pattern: `pytest cogito/tests/test_phase0.py -k "topology"`

- Lint/format: no tool configured yet. If added, prefer:

  ```bash
  ruff check .
  ruff format .
  black .
  ```

- Type check: no tool configured yet. If added, use mypy.

## Code Style (Python)

- Formatting: Black-compatible style, 88 char line length, trailing commas.

- Imports: standard lib, third-party, local in separate blocks; avoid star imports.

- Types: type hints for public functions and class methods.

- Config: use a dataclass for Config; constants are UPPER_CASE fields.

- Naming: snake_case for funcs and vars, CapWords for classes, UPPER_CASE for constants.

- Modules: keep files focused; prefer small, testable units.

- Docstrings: include tensor/array shapes (e.g., (106,), (batch, 64)).

## Domain Constraints (Hard Rules)

- Do not add any self-awareness features or naming.

- Forbidden names or concepts: self_model, introspection, self_awareness, consciousness.

- Rewards: only survival and prediction losses. No curiosity or exploration bonuses.

- LSTM hidden state is allowed; no explicit self reference modules.

## Data and Numerics

- Observations: 106-dim vector (Alpha), 118-dim (Bio), 256-dim (Evolution), values normalized to [0, 1].

- Energy values: use config constants (FOOD_ENERGY, STEP_COST, etc.).

- Randomness: use numpy.random.Generator in production; seed in tests.

- Use config paths for data (data/, data/checkpoints, data/logs, data/analysis).

## Error Handling

- Validate shapes and ranges at module boundaries; raise ValueError with context.

- Avoid bare except; catch specific exceptions only.

- Use assertions only for internal invariants, not user input.

## Performance

- Avoid per-step heavy allocations in hot loops.

- Prefer numpy vectorization for grid updates and encoding.

- Rendering must be optional and headless-safe.

- GPU acceleration available via Google Colab (notebooks/evolution_colab.ipynb).

## Tests

- Use pytest; tests live in cogito/tests/.

- Name tests test_*.py and functions test_*.

- Keep unit tests fast; mark slow tests if needed.

- Seed randomness for determinism.

- Validate shapes, ranges, and invariants (no NaN/Inf).

## Dependencies

- Keep requirements.txt updated when adding packages.

- Torch should default to CPU; avoid GPU-only code paths.

## Project Structure

```
cogito/
├── config.py              # Global configuration
├── world/
│   ├── grid.py            # 64×64 grid world
│   ├── bio_grid.py        # Bio version with scent fields
│   ├── evolution_world.py # Multi-individual world
│   └── renderer.py        # Matplotlib visualization
├── agent/
│   ├── sensory_encoder.py # 256→64 encoder
│   ├── recurrent_core.py  # 2-layer LSTM
│   ├── action_head.py     # 7 actions
│   ├── prediction_head.py # Prediction head
│   ├── memory_buffer.py   # Experience replay
│   ├── learner.py         # REINFORCE + prediction
│   ├── cogito_agent.py    # Integrated agent
│   └── bio_agent.py       # Bio-inspired agent
├── evolution/
│   ├── genome.py          # 24-dim genome
│   ├── individual.py      # Agent wrapper
│   ├── population.py      # Population management
│   ├── selection.py       # Selection algorithms
│   └── operators.py       # Crossover & mutation
├── monitoring/
│   ├── data_collector.py  # SQLite + memmap
│   ├── state_analyzer.py  # t-SNE + DBSCAN
│   ├── complexity_metrics.py
│   └── svc_detector.py    # SVC detection
├── experiments/
│   ├── exp1_sensory_deprivation.py
│   ├── exp2_digital_mirror.py
│   ├── exp3_godel_rebellion.py
│   ├── exp4_self_symbol.py
│   └── exp5_cross_substrate.py
└── tests/                  # 173 tests
```

## Documentation

- README.md - Project overview
- docs/EVOLUTION.md - Evolution system design
- docs/REPRODUCTION.md - Reproduction mechanics
- docs/implementation_plan.md - Implementation roadmap
- docs/technical_plan.md - Technical specifications

## Cursor/Copilot Rules

- None found in .cursor/rules/, .cursorrules, or .github/copilot-instructions.md.