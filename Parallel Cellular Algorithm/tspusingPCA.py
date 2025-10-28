import numpy as np
import random
import math
import matplotlib.pyplot as plt

# ---------------------------
# 1. Cities (10-city example)
# ---------------------------
cities = {
    'A': (0, 0),
    'B': (10, 0),
    'C': (20, 0),
    'D': (30, 0),
    'E': (40, 0),
    'F': (40, 10),
    'G': (30, 10),
    'H': (20, 10),
    'I': (10, 10),
    'J': (0, 10)
}

city_labels = list(cities.keys())
city_coords = [cities[label] for label in city_labels]

num_cities = len(city_coords)

# ---------------------------
# 2. Distance Function
# ---------------------------
def euclidean_distance(city1, city2):
    return math.sqrt((city1[0] - city2[0])**2 + (city1[1] - city2[1])**2)

def total_distance(route, coords):
    dist = 0
    for i in range(len(route)):
        dist += euclidean_distance(coords[route[i]], coords[route[(i+1)%len(route)]])
    return dist

# ---------------------------
# 3. Genetic Operators
# ---------------------------
def order_crossover(parent1, parent2):
    size = len(parent1)
    a, b = sorted(random.sample(range(size), 2))
    child = [None]*size
    child[a:b] = parent1[a:b]
    fill = [city for city in parent2 if city not in child]
    j = 0
    for i in range(size):
        if child[i] is None:
            child[i] = fill[j]
            j += 1
    return child

def swap_mutation(route, mutation_rate=0.2):
    route = route.copy()
    if random.random() < mutation_rate:
        i, j = random.sample(range(len(route)), 2)
        route[i], route[j] = route[j], route[i]
    return route

# ---------------------------
# 4. Cellular Grid Neighborhood
# ---------------------------
def get_neighbors(grid, r, c):
    rows, cols = grid.shape
    neighbors = [
        grid[(r-1)%rows, c],
        grid[(r+1)%rows, c],
        grid[r, (c-1)%cols],
        grid[r, (c+1)%cols]
    ]
    return neighbors

# ---------------------------
# 5. Cellular Genetic Algorithm
# ---------------------------
def cellular_ga_tsp(coords, grid_size=(5,5), generations=200):
    rows, cols = grid_size

    # Initialize grid with random routes
    grid = np.empty((rows, cols), dtype=object)
    for r in range(rows):
        for c in range(cols):
            route = list(range(len(coords)))
            random.shuffle(route)
            grid[r,c] = route

    # Sample initial route for display
    sample_route = grid[random.randint(0, rows-1), random.randint(0, cols-1)]
    init_dist = total_distance(sample_route, coords)

    # ---- Plot Initial Random Route ----
    plt.figure(figsize=(6,6))
    xs = [coords[i][0] for i in sample_route] + [coords[sample_route[0]][0]]
    ys = [coords[i][1] for i in sample_route] + [coords[sample_route[0]][1]]
    plt.plot(xs, ys, '-o', color='orange')
    plt.title(f"Initial Random Route | Distance = {init_dist:.2f}")
    for i, label in enumerate(city_labels):
        plt.text(coords[i][0]+0.5, coords[i][1]+0.5, label)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    best_route = None
    best_dist = float('inf')
    best_distances = []

    for gen in range(generations):
        new_grid = np.empty_like(grid)
        gen_best_dist = float('inf')
        for r in range(rows):
            for c in range(cols):
                current = grid[r,c]
                current_fit = total_distance(current, coords)
                neighbors = get_neighbors(grid, r, c)
                best_neighbor = min(neighbors, key=lambda x: total_distance(x, coords))
                child = order_crossover(current, best_neighbor)
                child = swap_mutation(child)
                child_fit = total_distance(child, coords)
                if child_fit < current_fit:
                    new_grid[r,c] = child
                else:
                    new_grid[r,c] = current
                if child_fit < gen_best_dist:
                    gen_best_dist = child_fit
                    if child_fit < best_dist:
                        best_dist = child_fit
                        best_route = child
        grid = new_grid
        best_distances.append(gen_best_dist)
        if gen % 20 == 0 or gen == generations-1:
            print(f"Generation {gen+1:3d} | Best distance: {gen_best_dist:.2f}")

    # ---- Plot Final Optimized Route ----
    plt.figure(figsize=(6,6))
    xs = [coords[i][0] for i in best_route] + [coords[best_route[0]][0]]
    ys = [coords[i][1] for i in best_route] + [coords[best_route[0]][1]]
    plt.plot(xs, ys, '-o', color='green')
    plt.title(f"Final Optimized Route | Distance = {best_dist:.2f}")
    for i, label in enumerate(city_labels):
        plt.text(coords[i][0]+0.5, coords[i][1]+0.5, label)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()

    # ---- Plot Distance over Generations ----
    plt.figure(figsize=(6,4))
    plt.plot(best_distances, color='red')
    plt.title("Best Distance over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Distance")
    plt.grid(True)
    plt.show()

    return best_route, best_dist

# ---------------------------
# 6. Run the algorithm
# ---------------------------
if __name__ == "__main__":
    best_route, best_dist = cellular_ga_tsp(city_coords, grid_size=(5,5), generations=200)
    print("\n✅ Best route found (indices):", best_route)
    print("🏁 Corresponding cities:", [city_labels[i] for i in best_route])
    print(f"🏁 Final total distance: {best_dist:.2f}")
