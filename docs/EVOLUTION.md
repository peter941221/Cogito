# Evolution Module Technical Documentation

## Design Philosophy

### Role in Cogito

The evolution module does not replace the existing architecture—it adds a layer on top:

```
Phase 1-5: Single agent, fixed architecture, observe emergence
Phase 6-7: Multiple agents, evolved architecture, observe which structures emerge
```

**Core Question**: "If natural selection designed this lens, what shape would it create?"

### Three Iron Laws

#### Law 1: Genome Encodes Structure, Not Behavior

| Allowed | Forbidden |
|---------|-----------|
| LSTM hidden_dim = 128 | exploration_rate = 0.35 |
| learning_rate = 0.0005 | fear_sensitivity = 0.8 |
| buffer_size = 3000 | risk_aversion = 0.6 |

**Reason**: Behavior must emerge from structure + learning. If behavior is encoded in genes, we cannot distinguish "emergent consciousness" from "programmed reactions."

#### Law 2: One Life Only

- Individual death = permanent termination
- Neural network weights not preserved
- Only genome passes to next generation

**Reason**: True fear of death requires true finality. If weights persist forever, death has no meaning.

#### Law 3: Evolution Doesn't Know About Consciousness

- Fitness function = survival metrics only
- No "self-awareness" rewards
- No "introspection" indicators

**Reason**: If an architecture is evolutionarily preferred and happens to pass consciousness experiments, that's the strongest evidence that natural selection independently discovered "consciousness structure."

---

## Genome Definition

### 24-Dimensional Float Vector

| Index | Name | Range | Type |
|-------|------|-------|------|
| 0 | encoder_hidden_dim | 32-128 | int (×8) |
| 1 | encoder_num_layers | 1-2 | int |
| 2 | encoder_use_norm | 0-1 | bool |
| 3 | core_hidden_dim | 32-128 | int (×8) |
| 4 | core_num_layers | 1-2 | int |
| 5 | core_dropout | 0.0-0.3 | float |
| 6 | action_hidden_dim | 16-128 | int (×8) |
| 7 | action_temperature | 0.1-2.0 | float |
| 8 | prediction_hidden | 16-128 | int (×8) |
| 9 | prediction_depth | 1-3 | int |
| 10 | learning_rate | 5e-5 to 3e-3 | float |
| 11 | gamma | 0.9-0.999 | float |
| 12 | prediction_weight | 0.1-5.0 | float |
| 13 | survival_weight | 0.1-5.0 | float |
| 14 | grad_clip | 0.1-10.0 | float |
| 15 | buffer_size | 500-10000 | int |
| 16 | batch_size | 8-128 | int |
| 17 | replay_ratio | 0.0-1.0 | float |
| 18 | weight_init_scale | 0.01-1.0 | float |
| 19 | bias_init_scale | 0.0-0.5 | float |
| 20 | encoded_dim | 16-128 | int (×8) |
| 21 | reward_death | -20 to -1 | float |
| 22 | reward_food | 1-20 | float |
| 23 | reward_step | -1 to 0 | float |

### Gene Expression Flow

```
Genome (24 floats)
    ↓ decode()
Architecture Parameters (dict)
    ↓
Brain Construction (CogitoAgent)
    ↓
Epigenetic Marks (optional modifications)
    ↓
Living Individual
```

---

## Individual Lifecycle

```python
class Individual:
    # Core attributes
    id: int
    genome: Genome
    brain: CogitoAgent
    learner: OnlineLearner
    
    # Life state
    is_alive: bool = True
    energy: float
    position: tuple[int, int]
    age: int = 0
    
    # Reproduction
    sex: int  # 0 or 1
    mating_cooldown: int = 0
    
    # Statistics
    stats: LifeStats
```

### Fitness Function

```python
def compute_fitness(individual: Individual) -> float:
    """
    Fitness = weighted combination of:
    - Lifespan (primary)
    - Food eaten
    - Offspring produced
    """
    lifespan_score = individual.stats.lifespan / max_lifespan
    food_score = individual.stats.food_eaten / max_food
    offspring_score = individual.stats.offspring_produced / max_offspring
    
    return (0.5 * lifespan_score + 
            0.3 * food_score + 
            0.2 * offspring_score)
```

---

## Genetic Operators

### Mutation

```python
def mutate(genome: Genome, rate: float = 0.1) -> Genome:
    """
    Gaussian mutation with adaptive rate.
    Each gene has `rate` probability of mutation.
    Mutation magnitude = 10% of gene range.
    """
```

### Crossover

```python
def crossover(parent1: Genome, parent2: Genome) -> tuple[Genome, Genome]:
    """
    Uniform crossover with random gene selection.
    Each gene comes from either parent with 50% probability.
    """
```

---

## Selection Algorithms

### Tournament Selection

```python
def tournament_select(population: list[Individual], k: int = 3) -> Individual:
    """
    Select k random individuals, return the fittest.
    Selection pressure controlled by tournament size k.
    """
```

### Elitism

```python
def apply_elitism(population: list[Individual], n: int = 2) -> list[Individual]:
    """
    Preserve top n individuals unchanged to next generation.
    Ensures fitness never decreases.
    """
```

---

## Population Management

### Population Guard (Anti-Extinction)

```python
class PopulationGuard:
    """
    Prevents complete population extinction.
    Injects mutated offspring from surviving individuals.
    """
    
    MIN_POPULATION = 5
    
    def check_and_inject(self, world: EvolutionWorld) -> None:
        alive = world.get_alive_individuals()
        if len(alive) < self.MIN_POPULATION:
            # Sample from existing population
            parent = random.choice(alive)
            # Mutate genome
            new_genes = GeneticOperators.mutate(parent.genome, rate=0.15)
            # Create offspring
            child = Individual(genome=new_genes, ...)
            world.add_individual(child)
```

---

## Lineage Tracking

```python
class LineageTracker:
    """
    Tracks genealogical relationships across generations.
    """
    
    def record_birth(self, individual: Individual, step: int) -> None:
        """Record birth event with parent IDs."""
        
    def find_most_successful_ancestor(self) -> tuple[Genome, int]:
        """Find ancestor with most living descendants."""
```

---

## Run Modes

### Generational Evolution

```bash
python run_evolution.py --population 50 --generations 100 --lifespan 2000
```

Fixed lifespan per generation, fitness evaluated at death.

### Continuous Evolution

```bash
python run_continuous_evolution.py --steps 500000
```

Real-time reproduction, dynamic population, natural birth/death cycles.

---

## Performance Optimization

### Genome Range Reduction

Original ranges allowed networks up to 1.3M parameters. Reduced to:

- `encoder_hidden_dim`: 32-256 → 32-128
- `core_hidden_dim`: 32-256 → 32-128
- `core_num_layers`: 1-4 → 1-2

**Result**: Max parameters reduced from 1.3M to ~330K (75% reduction)

### GPU Acceleration

Use Google Colab with T4 GPU for 10x speedup:

```
CPU: ~5 steps/sec
GPU: ~50+ steps/sec
```

---

## Verification Checklist

- [ ] Genome creates valid architecture
- [ ] Individual can live and learn in world
- [ ] Mutation produces viable offspring
- [ ] Crossover combines parent traits
- [ ] Selection maintains diversity
- [ ] Population guard prevents extinction
- [ ] Lineage tracking works correctly
- [ ] Fitness correlates with survival
