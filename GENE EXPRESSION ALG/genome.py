import random
import numpy as np

# --------- STEP 1: Define the Problem ---------
def create_distance_matrix(n_points=10, seed=42):
    random.seed(seed)
    points = [(random.randint(0, 100), random.randint(0, 100)) for _ in range(n_points)]
    dist_matrix = np.zeros((n_points, n_points))
    for i in range(n_points):
        for j in range(n_points):
            dist_matrix[i][j] = np.linalg.norm(np.array(points[i]) - np.array(points[j]))
    return dist_matrix, points


# --------- STEP 2: Expression Function ---------
def express_genome(genome):
    """
    Gene expression: convert genome (list of integers) to route.
    For this example, genome is already a permutation of cities.
    """
    # If genome contains duplicates/missing cities, fix it:
    seen = set()
    route = []
    for g in genome:
        if g not in seen:
            route.append(g)
            seen.add(g)
    # Fill missing cities (if any)
    for i in range(len(genome)):
        if i not in seen:
            route.append(i)
    return route


# --------- STEP 3: Fitness Function ---------
def calculate_total_distance(route, dist_matrix):
    distance = 0
    for i in range(len(route) - 1):
        distance += dist_matrix[route[i]][route[i + 1]]
    distance += dist_matrix[route[-1]][route[0]]  # return to depot
    return distance


def fitness(genome, dist_matrix):
    route = express_genome(genome)
    return 1 / (calculate_total_distance(route, dist_matrix) + 1e-6)


# --------- STEP 4: Initialize Population ---------
def initialize_population(pop_size, n_points):
    return [random.sample(range(n_points), n_points) for _ in range(pop_size)]


# --------- STEP 5: Selection (Tournament) ---------
def selection(population, dist_matrix, k=3):
    selected = []
    for _ in range(2):  # pick two parents
        tournament = random.sample(population, k)
        tournament.sort(key=lambda g: calculate_total_distance(express_genome(g), dist_matrix))
        selected.append(tournament[0])
    return selected


# --------- STEP 6: Crossover (Partially Matched) ---------
def crossover(parent1, parent2):
    start, end = sorted(random.sample(range(len(parent1)), 2))
    child = [None] * len(parent1)
    child[start:end] = parent1[start:end]

    # Fill remaining genes in order from parent2
    pointer = 0
    for g in parent2:
        if g not in child:
            while child[pointer] is not None:
                pointer += 1
            child[pointer] = g
    return child


# --------- STEP 7: Mutation ---------
def mutate(genome, mutation_rate=0.1):
    for i in range(len(genome)):
        if random.random() < mutation_rate:
            j = random.randint(0, len(genome) - 1)
            genome[i], genome[j] = genome[j], genome[i]
    return genome


# --------- STEP 8: Main GEA Loop ---------
def gene_expression_algorithm(dist_matrix, pop_size=100, generations=300, mutation_rate=0.1):
    n_points = len(dist_matrix)
    population = initialize_population(pop_size, n_points)
    best_genome = min(population, key=lambda g: calculate_total_distance(express_genome(g), dist_matrix))
    best_distance = calculate_total_distance(express_genome(best_genome), dist_matrix)

    for gen in range(generations):
        new_population = []
        while len(new_population) < pop_size:
            parent1, parent2 = selection(population, dist_matrix)
            child1 = crossover(parent1, parent2)
            child2 = crossover(parent2, parent1)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.extend([child1, child2])

        population = new_population
        current_best = min(population, key=lambda g: calculate_total_distance(express_genome(g), dist_matrix))
        current_best_dist = calculate_total_distance(express_genome(current_best), dist_matrix)

        if current_best_dist < best_distance:
            best_distance = current_best_dist
            best_genome = current_best

        if gen % 50 == 0:
            print(f"Generation {gen}, Best Distance: {best_distance:.2f}")

    return express_genome(best_genome), best_distance


# --------- Run the Algorithm ---------
if __name__ == "__main__":
    dist_matrix, points = create_distance_matrix(n_points=10)
    best_route, best_distance = gene_expression_algorithm(dist_matrix)
    print("\nOptimal Route (Approx):", best_route)
    print("Total Distance:", round(best_distance, 2))