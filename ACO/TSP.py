import matplotlib.pyplot as plt
import networkx as nx
import random

class ACO_TSP_Visualizer:
    def __init__(self, graph, pheromone, n_ants=5, n_iterations=10, alpha=1, beta=5, rho=0.5, Q=100):
        self.graph = graph
        self.pheromone = pheromone
        self.n = len(graph)
        self.n_ants = n_ants
        self.n_iterations = n_iterations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.best_path = None
        self.best_length = float('inf')

    def run(self):
        for it in range(self.n_iterations):
            all_paths = []
            all_lengths = []

            for ant in range(self.n_ants):
                path = self.construct_solution()
                length = self.path_length(path)
                all_paths.append(path)
                all_lengths.append(length)

                if length < self.best_length:
                    self.best_length = length
                    self.best_path = path

            self.update_pheromones(all_paths, all_lengths)

        print(f"Best Path after {self.n_iterations} iterations: {self.best_path}")
        print(f"Best Length after {self.n_iterations} iterations: {self.best_length:.2f}")

        return self.best_path, self.best_length

    def construct_solution(self):
        start = random.randint(0, self.n - 1)
        path = [start]
        visited = set(path)

        while len(path) < self.n:
            current = path[-1]
            next_city = self.choose_next_city(current, visited)
            path.append(next_city)
            visited.add(next_city)

        return path

    def choose_next_city(self, current, visited):
        probabilities = []
        denominator = 0

        for j in range(self.n):
            if j not in visited:
                tau = self.pheromone[current][j] ** self.alpha
                eta = (1 / self.graph[current][j]) ** self.beta if self.graph[current][j] > 0 else 0
                denominator += tau * eta

        for j in range(self.n):
            if j not in visited:
                tau = self.pheromone[current][j] ** self.alpha
                eta = (1 / self.graph[current][j]) ** self.beta if self.graph[current][j] > 0 else 0
                probabilities.append((j, (tau * eta) / denominator))

        r = random.random()
        cumulative = 0
        for city, prob in probabilities:
            cumulative += prob
            if r <= cumulative:
                return city

        return probabilities[-1][0]

    def path_length(self, path):
        length = 0
        for i in range(len(path) - 1):
            length += self.graph[path[i]][path[i + 1]]
        length += self.graph[path[-1]][path[0]]  # return to start
        return length

    def update_pheromones(self, all_paths, all_lengths):
        for i in range(self.n):
            for j in range(self.n):
                self.pheromone[i][j] *= (1 - self.rho)

        for path, length in zip(all_paths, all_lengths):
            deposit = self.Q / length
            for i in range(len(path) - 1):
                a, b = path[i], path[i + 1]
                self.pheromone[a][b] += deposit
                self.pheromone[b][a] += deposit
            self.pheromone[path[-1]][path[0]] += deposit
            self.pheromone[path[0]][path[-1]] += deposit

    def visualize_best_path(self):
        G = nx.Graph()
        for i in range(self.n):
            for j in range(i + 1, self.n):
                if self.graph[i][j] > 0:
                    G.add_edge(i, j, weight=self.graph[i][j])

        pos = nx.circular_layout(G)

        plt.figure(figsize=(7, 7))
        nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=12, font_weight="bold")
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=10)

        # Highlight best path
        path = self.best_path
        edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)] + [(path[-1], path[0])]
        nx.draw_networkx_edges(G, pos, edgelist=edges, edge_color="red", width=3)

        plt.title(f"Best Path Length: {self.best_length:.2f}")
        plt.show()


# ---------------- Example Usage ----------------
if __name__ == "__main__":
    graph = [
        [0, 2, 9, 10, 7],
        [2, 0, 6, 4, 3],
        [9, 6, 0, 8, 5],
        [10, 4, 8, 0, 6],
        [7, 3, 5, 6, 0]
    ]

    pheromone = [
        [0, 1, 1, 1, 1],
        [1, 0, 1, 1, 1],
        [1, 1, 0, 1, 1],
        [1, 1, 1, 0, 1],
        [1, 1, 1, 1, 0]
    ]

    aco = ACO_TSP_Visualizer(graph, pheromone, n_ants=10, n_iterations=20)
    best_path, best_length = aco.run()
    aco.visualize_best_path()
