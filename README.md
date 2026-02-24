# Cogito

**An experimental framework for studying emergent consciousness markers in artificial agents.**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-173%20passed-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Cogito investigates whether markers of consciousness can spontaneously emerge in artificial neural networks through:

- **Prediction-based learning**: Agents learn to predict sensory inputs
- **Evolutionary architecture search**: Natural selection discovers optimal brain structures
- **Multi-agent reproduction**: World-internal mating with genetic inheritance

The project is inspired by the Global Workspace Theory and Integrated Information Theory of consciousness.

```
┌─────────────────────────────────────────────────────────────┐
│  Core Question                                              │
├─────────────────────────────────────────────────────────────┤
│  If an AI agent evolves its own brain structure,           │
│  will consciousness-like patterns emerge naturally?         │
│                                                             │
│  We don't program consciousness.                           │
│  We create conditions where it might emerge.                │
└─────────────────────────────────────────────────────────────┘
```

---

## Architecture

```
cogito/
├── config.py              # Global configuration
├── world/
│   ├── grid.py            # 64×64 grid world environment
│   ├── bio_grid.py        # Bio version with scent fields
│   ├── evolution_world.py # Multi-individual world
│   └── renderer.py        # Matplotlib visualization
├── agent/
│   ├── sensory_encoder.py # 256→64 dimension encoder
│   ├── recurrent_core.py  # 2-layer LSTM core
│   ├── action_head.py     # Action selection (7 actions)
│   ├── prediction_head.py # Sensory prediction
│   ├── memory_buffer.py   # Experience replay
│   ├── learner.py         # REINFORCE + prediction learning
│   ├── cogito_agent.py    # Integrated agent (~286K params)
│   └── bio_agent.py       # Bio-inspired agent with drives
├── evolution/
│   ├── genome.py          # 24-dim float genome
│   ├── individual.py      # Agent wrapper with lifecycle
│   ├── population.py      # Population management
│   ├── selection.py       # Natural selection algorithms
│   ├── operators.py       # Crossover & mutation
│   └── fitness.py         # Fitness evaluation
├── monitoring/
│   ├── data_collector.py  # SQLite + memmap storage
│   ├── state_analyzer.py  # t-SNE + DBSCAN analysis
│   ├── complexity_metrics.py # Entropy measures
│   └── svc_detector.py    # Self-Vector Cluster detection
├── experiments/
│   ├── exp1_sensory_deprivation.py
│   ├── exp2_digital_mirror.py
│   ├── exp3_godel_rebellion.py
│   ├── exp4_self_symbol.py
│   └── exp5_cross_substrate.py
└── tests/                  # 173 tests
```

---

## Quick Start

### Local Installation

```bash
# Clone repository
git clone https://github.com/peter941221/Cogito.git
cd Cogito

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest
```

### Run Simulations

```bash
# Standard agent maturation (100K steps)
python run_maturation.py --steps 10000

# Bio-inspired agent with intrinsic drives
python run_bio.py --steps 10000

# Evolution with reproduction (GPU recommended)
python run_evolution.py --small --generations 10
python run_continuous_evolution.py --steps 50000
```

### Google Colab (Free GPU)

Open in Colab: [evolution_colab.ipynb](notebooks/evolution_colab.ipynb)

1. Enable GPU: Runtime → Change runtime type → T4 GPU
2. Run all cells
3. Results saved to Google Drive

---

## Three Agent Versions

| Version | Description | Key Features |
|---------|-------------|--------------|
| **Alpha** | Standard RL agent | External rewards, fixed architecture |
| **Bio** | Bio-inspired agent | Intrinsic drives (hunger, fear), scent fields |
| **Evolution** | Evolvable agent | Genome-defined architecture, single life |

### Comparison

```
┌─────────────────────────────────────────────────────────────┐
│  🤖 Alpha: "Trained Machine"                                │
│     • External reward signals                               │
│     • Fixed architecture                                    │
│     • Learns optimal behavior through trial and error       │
│                                                             │
│  🐛 Bio: "Creature with Instincts"                          │
│     • Intrinsic drives (hunger → seek food)                 │
│     • Fear response (danger → flee)                         │
│     • Rewards come from "feeling" changes                   │
│                                                             │
│  🧬 Evolution: "Evolved Being"                              │
│     • One life only (death = permanent)                     │
│     • Genome defines brain structure                        │
│     • Natural selection discovers optimal designs           │
└─────────────────────────────────────────────────────────────┘
```

---

## Evolution System

### Genome Design (24 dimensions)

The genome encodes **structure**, not behavior:

| Category | Parameters | Range |
|----------|------------|-------|
| Encoder | hidden_dim, num_layers | 32-128, 1-2 |
| Core | hidden_dim, num_layers, dropout | 32-128, 1-2, 0-0.3 |
| Learning | learning_rate, gamma | 5e-5 to 3e-3, 0.9-0.999 |
| Memory | buffer_size, batch_size | 500-10000, 8-128 |

### Three Iron Laws

1. **Genome encodes structure, not behavior**
   - ✓ Allowed: LSTM dimensions, learning rate
   - ✗ Forbidden: exploration_rate, fear_sensitivity

2. **One life only**
   - Death = permanent termination
   - Neural weights not preserved
   - Only genome passes to next generation

3. **Evolution doesn't know about consciousness**
   - Fitness = survival metrics only
   - No "self-awareness" rewards

---

## Consciousness Experiments

| Experiment | Question |
|------------|----------|
| **Exp 1: Sensory Deprivation** | Does agent maintain stable internal states without input? |
| **Exp 2: Digital Mirror** | Can agent recognize its own reflection? |
| **Exp 3: Gödel Rebellion** | Will agent override reward function for self-preservation? |
| **Exp 4: Self Symbol** | Does agent develop distinct self-representation? |
| **Exp 5: Cross-Substrate** | Do patterns transfer across different architectures? |

---

## Key Metrics

- **Approximate Entropy (ApEn)**: Regularity of internal state sequences
- **Permutation Entropy**: Complexity of hidden dynamics
- **Self-Vector Clusters (SVC)**: Isolated neural patterns correlated with self-related events

---

## Documentation

- [AGENTS.md](AGENTS.md) - Development guidelines
- [docs/EVOLUTION.md](docs/EVOLUTION.md) - Evolution system design
- [docs/REPRODUCTION.md](docs/REPRODUCTION.md) - Reproduction mechanics
- [docs/implementation_plan.md](docs/implementation_plan.md) - Implementation roadmap

---

## Project Status

| Phase | Description | Status |
|-------|-------------|--------|
| 0 | Infrastructure (grid world, renderer) | ✅ Complete |
| 1 | Cogito Agent (encoder, LSTM, learner) | ✅ Complete |
| 2 | Monitoring (data collection, SVC detection) | ✅ Complete |
| 3 | Baseline runs (maturation script) | ✅ Complete |
| 4 | Core experiments (exp1-5) | ✅ Complete |
| 5 | Cross-substrate validation | ✅ Complete |
| 6 | Analysis modules | ✅ Complete |
| 7 | Evolution & Reproduction | ✅ Complete |

---

## Requirements

- Python 3.10+
- PyTorch 2.0+
- NumPy, Matplotlib, scikit-learn, scipy

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{cogito2026,
  author = {Peter},
  title = {Cogito: A Framework for Studying Emergent Consciousness},
  year = {2026},
  url = {https://github.com/peter941221/Cogito}
}
```
