# Reproduction Module Technical Documentation

## Design Philosophy

Reproduction is not just "a mechanism for evolution"—it's the most complex social behavior in the world.

For an agent to successfully reproduce, it must:

1. Stay alive (basic survival)
2. Eat enough food (sufficient energy)
3. Perceive other individuals (social sensing)
4. Navigate to a mate (navigation ability)
5. Choose mating over other actions (decision making)
6. Survive during mating period (risk management)

**All these capabilities must emerge from learning and evolution—none can be pre-programmed.**

---

## Agent Attributes for Reproduction

```python
class Individual:
    # Reproduction attributes
    sex: int              # 0 (male) or 1 (female)
    age: int              # Steps since birth
    mating_cooldown: int  # Steps until next mating possible
    is_mating: bool       # Currently in mating state
```

### Fertility Conditions

```python
@property
def is_fertile(self) -> bool:
    """Check if individual can mate."""
    if not self.is_alive:
        return False
    if self.age < Config.MATURITY_AGE:  # Default: 200 steps
        return False
    if self.energy < Config.MATING_ENERGY_THRESHOLD:  # Default: 40
        return False
    return self.mating_cooldown == 0
```

---

## Mating Process

### Conditions for Mating

Two individuals can mate if:

| Condition | Requirement |
|-----------|-------------|
| Both alive | `is_alive == True` |
| Both fertile | `is_fertile == True` |
| Opposite sex | `a.sex != b.sex` |
| Adjacent positions | Manhattan distance = 1 |
| Both consenting | Action 6 (interact) chosen |

### Mating Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    Mating Process                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Two fertile individuals become adjacent                 │
│                     ↓                                       │
│  2. Both choose action 6 (interact)                        │
│                     ↓                                       │
│  3. Mating check passes                                     │
│                     ↓                                       │
│  4. Both enter mating state (5 steps)                      │
│                     ↓                                       │
│  5. Energy consumed (10 each)                              │
│                     ↓                                       │
│  6. Offspring produced (1-2 children)                      │
│                     ↓                                       │
│  7. Cooldown begins (50 steps)                             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Sensory Extensions

### Social View (49 dimensions)

Agents perceive other individuals in a 7×7 area:

```python
def _get_social_view(self, individual: Individual) -> np.ndarray:
    """
    7x7 social view centered on agent.
    Each cell encodes:
    - 0: empty
    - 1: same sex individual
    - 2: opposite sex individual
    - 3: fertile opposite sex (potential mate)
    """
```

### Self-State Channel (4 dimensions)

```python
def get_sensory_self_state(self) -> np.ndarray:
    """
    [0] sex: 0 or 1
    [1] adult: 1 if age >= MATURITY_AGE
    [2] cooldown_ratio: mating_cooldown / MATING_COOLDOWN
    [3] energy_ok: 1 if energy >= MATING_ENERGY_THRESHOLD
    """
```

---

## Offspring Generation

### Genetic Crossover

```python
def produce_offspring(parent1: Individual, parent2: Individual) -> Individual:
    """Create child from two parents."""
    
    # Crossover genomes
    child_genome = GeneticOperators.crossover(parent1.genome, parent2.genome)
    
    # Mutate
    child_genome = GeneticOperators.mutate(child_genome, rate=0.1)
    
    # Create individual
    child = Individual(
        genome=child_genome,
        generation=max(parent1.generation, parent2.generation) + 1,
        parent_ids=(parent1.id, parent2.id)
    )
    
    return child
```

### Birth Position

```python
def get_birth_position(parents: list[Individual], world: EvolutionWorld) -> tuple:
    """
    Spawn offspring near parents.
    Priority:
    1. Adjacent to either parent
    2. Within 3 cells of parents
    3. Random empty position
    """
```

---

## Configuration Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MATURITY_AGE` | 200 | Steps before fertility |
| `MATING_ENERGY_THRESHOLD` | 40 | Minimum energy to mate |
| `MATING_COOLDOWN` | 50 | Steps between matings |
| `MATING_DURATION` | 5 | Steps for mating process |
| `MATING_ENERGY_COST` | 10 | Energy consumed per mating |
| `MAX_POPULATION` | 100 | Population cap |
| `OFFSPRING_COUNT` | 1-2 | Children per mating |

---

## Population Dynamics

### Birth Rate Factors

- Energy availability (food supply)
- Encounter rate (population density)
- Mating success rate (agent decision making)

### Death Rate Factors

- Energy depletion (starvation)
- Danger zones (environmental hazard)
- Old age (max lifespan)

### Equilibrium

```
Population stabilizes when:
  Birth Rate ≈ Death Rate

Can be influenced by:
  - Food spawn rate
  - Danger zone coverage
  - Mating energy cost
```

---

## Action Space Extension

Action 6 (Interact) added for reproduction:

| Action | Description |
|--------|-------------|
| 0 | Move up |
| 1 | Move down |
| 2 | Move left |
| 3 | Move right |
| 4 | Wait |
| 5 | Interact no-op |
| 6 | **Interact (mating attempt)** |

### Interact Action Behavior

```python
def process_interact(agent: Individual, world: EvolutionWorld) -> None:
    """
    Action 6: Attempt to interact with adjacent agent.
    
    If adjacent to fertile opposite-sex agent who also chose action 6:
        → Start mating
    
    Otherwise:
        → No effect (wasted action)
    """
```

---

## Edge Cases

### Mating Interruption

If one parent dies during mating:
- Other parent exits mating state
- No offspring produced
- Energy still consumed

### Population Cap

When population reaches MAX_POPULATION:
- New births are prevented
- Mating can still occur (energy consumed)
- No offspring spawned until population decreases

### Inbreeding

Currently not prevented. Could add:
- Parent-child mating prevention
- Sibling mating prevention
- Coefficient of relatedness tracking

---

## Metrics

```python
class ReproductionStats:
    total_matings: int
    successful_matings: int
    total_births: int
    total_deaths: int
    
    avg_generation: float
    avg_offspring_per_parent: float
    
    population_over_time: list[int]
    births_over_time: list[int]
    deaths_over_time: list[int]
```

---

## Testing Checklist

- [ ] Fertile agents can detect each other
- [ ] Mating requires both agents to choose action 6
- [ ] Offspring genome combines parent genomes
- [ ] Birth position near parents
- [ ] Cooldown prevents immediate re-mating
- [ ] Energy cost applied correctly
- [ ] Death during mating handled
- [ ] Population cap enforced
