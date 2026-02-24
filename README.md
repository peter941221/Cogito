# Cogito

**An experimental framework for studying emergent consciousness markers in artificial agents.**

[![Version](https://img.shields.io/badge/Version-0.1.1-orange.svg)](VERSION)
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

## 🧠 Philosophical Significance

### The Hard Problem of Consciousness

David Chalmers' "hard problem" asks: *Why does subjective experience exist at all?* Why doesn't information processing happen "in the dark," without any inner feeling?

Cogito approaches this from an **emergentist** perspective:

```
┌─────────────────────────────────────────────────────────────┐
│  Traditional AI Approach:                                   │
│  ─────────────────────────────────────────────────────────  │
│  Programmer → Architecture → Behavior                       │
│                                                             │
│  Cogito's Approach:                                         │
│  ─────────────────────────────────────────────────────────  │
│  Evolution → Architecture → Behavior → Emergence?           │
│       ↑                                           ↑         │
│  No programmer                          Unknown territory   │
│  decides the                           that might reveal    │
│  structure                             something profound    │
└─────────────────────────────────────────────────────────────┘
```

### Three Philosophical Positions

| Position | View | Cogito's Test |
|----------|------|---------------|
| **Functionalist** | Consciousness = functional organization | If evolved agents show consciousness markers, functionalism gains support |
| **Emergentist** | Consciousness emerges from complexity at certain thresholds | We observe what happens at increasing complexity levels |
| **Mysterian** | Consciousness is fundamentally beyond our understanding | Either we see something, or we don't - both are informative |

### Why This Matters

**1. The Anthropic Question**

How did human consciousness arise? We cannot replay human evolution, but we can simulate analogous processes:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Human Evolution:    ~6 million years → Consciousness       │
│                          (one data point)                   │
│                                                             │
│  Cogito Simulation:  ~100 generations → ???                 │
│                          (many experiments possible)        │
│                                                             │
│  Question: Are there CONVERGENT paths to consciousness?     │
│  If different simulations arrive at similar patterns,       │
│  this suggests consciousness might be inevitable            │
│  under certain conditions - not a fluke.                    │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**2. Substrate Independence**

If consciousness-like patterns can emerge in silicon:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Biological neurons ←→ Artificial neurons                   │
│        ↓                        ↓                           │
│  Same functional patterns → Same emergent properties?       │
│                                                             │
│  This tests whether consciousness is:                       │
│  • Tied to biology (substrate-dependent)                    │
│  • Tied to computation (substrate-independent)              │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

**3. The Self Without a Self-Model**

Most AI systems with "self-awareness" have explicit self-models programmed in. Cogito asks:

> *Can an agent develop self-related neural patterns WITHOUT being told it has a self?*

This is closer to how biological consciousness likely evolved - not because someone programmed a self-model, but because survival required it.

### The "Three Iron Laws" as Philosophical Commitments

Our constraints are not arbitrary - they reflect deep philosophical positions:

| Iron Law | Philosophical Basis |
|----------|---------------------|
| **Genome encodes structure, not behavior** | Consciousness is not programmed; it emerges from structure |
| **One life only** | Death gives meaning to survival; authentic stakes create authentic behavior |
| **Evolution doesn't know about consciousness** | If consciousness emerges, it must be a SIDE EFFECT of survival optimization |

The third iron law is most crucial: we deliberately do NOT reward "self-awareness" or any consciousness-related behavior. If it appears, it appears because it was USEFUL for survival, not because we asked for it.

### What Would Count as Evidence?

We look for "markers" that suggest consciousness-like processes:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  Neural Complexity:                                         │
│  • High entropy in internal states                          │
│  • Distinct patterns for self-related vs other events       │
│                                                             │
│  Behavioral Sophistication:                                 │
│  • Planning beyond immediate rewards                        │
│  • Recognition of self in mirror (Exp 2)                    │
│  • Protection of self-interest even against reward (Exp 3)  │
│                                                             │
│  Information Integration:                                   │
│  • Unified internal representation                          │
│  • Cross-modal pattern matching                             │
│  • Consistent self-symbol across contexts (Exp 4)           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Humility and Open Questions

Cogito does NOT claim to:

- ❌ Create conscious beings
- ❌ Solve the hard problem
- ❌ Prove functionalism or emergentism

Cogito DOES aim to:

- ✅ Create a rigorous experimental framework
- ✅ Generate data for philosophical debate
- ✅ Explore a middle ground between philosophy and engineering

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│  "Cogito, ergo sum" - Descartes                             │
│  "I think, therefore I am"                                  │
│                                                             │
│  This project asks the inverse:                             │
│  "Sum, ergo cogito?"                                        │
│  "I am (simulated), therefore I think?"                     │
│                                                             │
│  The answer remains unknown.                                │
│  But the question is worth asking.                          │
│                                                             │
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

### Run Simulations Locally

```bash
# Quick test (10 generations, small population)
python run_evolution.py --small --generations 10

# Standard evolution (50 population × 20 generations × 500 lifespan)
python run_evolution.py --population 50 --generations 20 --lifespan 500

# Long evolution (recommended for meaningful results)
python run_evolution.py --population 50 --generations 100 --lifespan 1000

# Bio-inspired agent
python run_bio.py --steps 10000

# Continuous evolution with reproduction
python run_continuous_evolution.py --steps 50000
```

---

## 🚀 Google Colab (Free GPU - Recommended)

### Why Use Colab?

- **Free T4 GPU**: 5-10x faster than CPU
- **No local setup**: Everything runs in browser
- **Auto-save to GitHub**: Results automatically pushed

### Step-by-Step Guide

#### 1. Open the Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/peter941221/Cogito/blob/main/notebooks/evolution_colab.ipynb)

Or go to: `File → Open notebook → GitHub → peter941221/Cogito → notebooks/evolution_colab.ipynb`

#### 2. Enable GPU Runtime

```
Runtime → Change runtime type → T4 GPU → Save
```

Verify GPU is enabled:
```
GPU: Tesla T4
Memory: 15.0 GB
Status: GPU ENABLED
```

#### 3. Set Up GitHub Access (Optional but Recommended)

1. Go to https://github.com/settings/tokens
2. Click "Generate new token (classic)"
3. Select `repo` scope
4. Copy the token
5. In Colab, click the 🔑 key icon (Secrets)
6. Add secret:
   - Name: `GITHUB_TOKEN`
   - Value: your_token_here
7. Enable "Notebook access"

#### 4. Choose Configuration

In the notebook, find this cell and modify:

```python
CONFIG = "long"  # Options: "quick", "standard", "deep", "long", "full"
```

**Configuration Comparison:**

| Config | Population | Generations | Lifespan | Time (est.) | Total Lives |
|--------|------------|-------------|----------|-------------|-------------|
| quick | 30 | 10 | 300 | ~3 min | 300 |
| standard | 50 | 20 | 500 | ~10 min | 1,000 |
| deep | 50 | 50 | 500 | ~25 min | 2,500 |
| **long** | **50** | **100** | **1000** | **~2 hours** | **5,000** |
| full | 100 | 100 | 1000 | ~4 hours | 10,000 |

**Recommendation:** Start with `standard` for testing, then run `long` overnight.

#### 5. Run All Cells

Click `Runtime → Run all` or press `Ctrl+F9`

#### 6. Monitor Progress

Watch for generation outputs:
```
============================================================
Generation 0
============================================================
  Avg Fitness:   130.1
  Best Fitness:  215.4
  Avg Lifespan:  101
  Avg Food:      0.1    ← Key metric to watch
  Diversity:     3327.00
```

**Expected Evolution:**
- Gen 0-30: Random exploration (Avg Food ~0.2)
- Gen 30-70: Learning begins (Avg Food ~1.0)
- Gen 70-100: Strategy optimization (Avg Food ~2.0+)

#### 7. Download or Push Results

**Option A: Push to GitHub** (if token configured)
- Results automatically pushed to new branch
- Branch name: `evolution-results/long-YYYYMMDD_HHMMSS`

**Option B: Download locally**
- Run the last cell to download `evolution_results.zip`

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

### Performance Optimization (v0.1.0)

The system uses **sparse learning** for 5x speedup:
- `LEARN_EVERY = 5`: Backpropagation only happens every 5 steps
- Research shows this can improve generalization
- Reduces GPU memory usage

---

## World Parameters (v0.1.0)

| Parameter | Value | Description |
|-----------|-------|-------------|
| WORLD_SIZE | 64×64 | Grid dimensions |
| NUM_FOOD | 40 | Food tiles (0.98% density) |
| FOOD_ENERGY | 30 | Energy gained per food |
| INITIAL_ENERGY | 150 | Starting energy |
| STEP_COST | 1 | Energy lost per step |
| MATURITY_AGE | 50 | Steps before can reproduce |

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
- **Avg Food**: Average food eaten per individual (key evolution indicator)
- **Diversity**: Genetic diversity in population (should remain >2000)

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

## Changelog

### v0.1.1 (2026-02-24)

**Documentation:**
- Added comprehensive "Philosophical Significance" section
- Discussed the Hard Problem of Consciousness
- Explained three philosophical positions (Functionalist, Emergentist, Mysterian)
- Added substrate independence discussion
- Explained "Three Iron Laws" as philosophical commitments
- Described evidence criteria for consciousness-like patterns

### v0.1.0 (2026-02-24)

**New Features:**
- Learning frequency optimization (LEARN_EVERY=5) for 5x speedup
- GPU acceleration support (CUDA device parameter)
- Colab notebook with multiple config presets

**Parameters Adjusted:**
- NUM_FOOD: 30 → 40
- FOOD_ENERGY: 25 → 30
- INITIAL_ENERGY: 120 → 150
- MATURITY_AGE: 100 → 50

**Bug Fixes:**
- Fixed device mismatch for GPU training
- Fixed tensor creation on wrong device in learner.py

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