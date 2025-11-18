import math
import random

# Objective function
def f(x):
    return x * math.sin(10 * math.pi * x) + 1

# GA parameters
POP_SIZE = 50
GENERATIONS = 50
MUTATION_RATE = 0.1
CROSSOVER_RATE = 0.8
DOMAIN_MIN = 0.0
DOMAIN_MAX = 1.0

# Generate initial population
def init_population():
    return [random.uniform(DOMAIN_MIN, DOMAIN_MAX) for _ in range(POP_SIZE)]

# Evaluate fitness
def fitness(x):
    return f(x)

# Tournament selection
def select(pop):
    a = random.choice(pop)
    b = random.choice(pop)
    return a if fitness(a) > fitness(b) else b

# Single point arithmetic crossover
def crossover(p1, p2):
    if random.random() < CROSSOVER_RATE:
        alpha = random.random()
        c1 = alpha * p1 + (1 - alpha) * p2
        c2 = alpha * p2 + (1 - alpha) * p1
        return clamp(c1), clamp(c2)
    return p1, p2

# Mutation
def mutate(x):
    if random.random() < MUTATION_RATE:
        x += random.uniform(-0.1, 0.1)
    return clamp(x)

# Keep x in [0,1]
def clamp(x):
    return max(DOMAIN_MIN, min(DOMAIN_MAX, x))

# Genetic Algorithm
def genetic_algorithm():
    pop = init_population()

    for gen in range(GENERATIONS):
        new_pop = []

        # Elitism: keep the best 1
        best = max(pop, key=fitness)
        new_pop.append(best)

        # Generate new offspring
        while len(new_pop) < POP_SIZE:
            p1 = select(pop)
            p2 = select(pop)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            new_pop.extend([c1, c2])

        pop = new_pop[:POP_SIZE]

        # Print progress
        print(f"Generation {gen+1}: Best x={best:.5f}, f(x)={fitness(best):.5f}")

    # Final best
    final_best = max(pop, key=fitness)
    return final_best, fitness(final_best)

# Run GA
best_x, best_val = genetic_algorithm()
print("\n=== FINAL RESULT ===")
print("Best x =", best_x)
print("Max f(x) =", best_val)
