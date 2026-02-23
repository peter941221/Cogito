# AGENTS.md - Project Genesis: Cogito
# Instructions for agentic coding in this repo.

## Repository status

- This workspace contains docs and initial scaffolding only.

- Source of truth: docs/implementation_plan.md and docs/technical_plan.md.

- Commands below are expected defaults until tooling exists.

## Environment setup (from docs)

- Python 3.10+ is required.

- Create venv: python -m venv venv

- Activate (Windows): venv\Scripts\activate

- Activate (POSIX): source venv/bin/activate

- Install core deps:

  pip install -r requirements.txt

## Build / run / test / lint

- Build: none (pure Python). Run modules and scripts directly.

- Run examples (when implemented):

  python cogito/core/simulation.py

  python run_maturation.py

- Test all: pytest

- Test single file: pytest cogito/tests/test_phase0.py

- Test single test: pytest cogito/tests/test_phase0.py::TestClass::test_name

- Test by pattern: pytest cogito/tests/test_phase0.py -k "topology"

- Lint/format: no tool configured yet. If added, prefer:

  ruff check .

  ruff format .

  black .

- Type check: no tool configured yet. If added, use mypy.

## Code style (Python)

- Formatting: Black-compatible style, 88 char line length, trailing commas.

- Imports: standard lib, third-party, local in separate blocks; avoid star imports.

- Types: type hints for public functions and class methods.

- Config: use a dataclass for Config; constants are UPPER_CASE fields.

- Naming: snake_case for funcs and vars, CapWords for classes, UPPER_CASE for constants.

- Modules: keep files focused; prefer small, testable units.

- Docstrings: include tensor/array shapes (e.g., (106,), (batch, 64)).

## Domain constraints (hard rules)

- Do not add any self-awareness features or naming.

- Forbidden names or concepts: self_model, introspection, self_awareness, consciousness.

- Rewards: only survival and prediction losses. No curiosity or exploration bonuses.

- LSTM hidden state is allowed; no explicit self reference modules.

## Data and numerics

- Observations: 106-dim vector, values normalized to [0, 1].

- Energy values: use config constants (FOOD_ENERGY, STEP_COST, etc.).

- Randomness: use numpy.random.Generator in production; seed in tests.

- Use config paths for data (data/, data/checkpoints, data/logs, data/analysis).

## Error handling

- Validate shapes and ranges at module boundaries; raise ValueError with context.

- Avoid bare except; catch specific exceptions only.

- Use assertions only for internal invariants, not user input.

## Performance

- Avoid per-step heavy allocations in hot loops.

- Prefer numpy vectorization for grid updates and encoding.

- Rendering must be optional and headless-safe.

## Tests

- Use pytest; tests live in tests/.

- Name tests test_*.py and functions test_*.

- Keep unit tests fast; mark slow tests if needed.

- Seed randomness for determinism.

- Validate shapes, ranges, and invariants (no NaN/Inf).

## Dependencies

- Keep requirements.txt updated when adding packages.

- Torch should default to CPU; avoid GPU-only code paths.

## Cursor/Copilot rules

- None found in .cursor/rules/, .cursorrules, or .github/copilot-instructions.md.

## Planned project structure (from docs)

+-- cogito/
|   +-- __init__.py
|   +-- config.py
|   +-- world/
|   |   +-- __init__.py
|   |   +-- grid.py
|   |   +-- entities.py
|   |   +-- echo_zone.py
|   |   +-- hidden_interface.py
|   |   +-- renderer.py
|   +-- agent/
|   |   +-- __init__.py
|   |   +-- genesis_alpha.py
|   |   +-- genesis_beta.py
|   |   +-- sensory_encoder.py
|   |   +-- recurrent_core.py
|   |   +-- action_head.py
|   |   +-- prediction_head.py
|   |   +-- memory_buffer.py
|   |   +-- learner.py
|   +-- monitoring/
|   |   +-- __init__.py
|   |   +-- data_collector.py
|   |   +-- state_analyzer.py
|   |   +-- svc_detector.py
|   |   +-- complexity_metrics.py
|   |   +-- dashboard.py
|   |   +-- storage.py
|   +-- experiments/
|   |   +-- __init__.py
|   |   +-- exp1_sensory_deprivation.py
|   |   +-- exp2_digital_mirror.py
|   |   +-- exp3_godel_rebellion.py
|   |   +-- exp4_self_symbol.py
|   |   +-- exp5_cross_substrate.py
|   |   +-- controls.py
|   +-- analysis/
|   |   +-- __init__.py
|   |   +-- exp1_analysis.py
|   |   +-- exp2_analysis.py
|   |   +-- exp3_analysis.py
|   |   +-- exp4_analysis.py
|   |   +-- exp5_analysis.py
|   |   +-- cross_experiment.py
|   |   +-- statistics.py
|   +-- core/
|   |   +-- __init__.py
|   |   +-- simulation.py
|   +-- tests/
|       +-- test_phase0.py
|       +-- test_phase2.py
