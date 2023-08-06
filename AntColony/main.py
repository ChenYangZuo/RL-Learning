"""
@FileName：main.py
@Description：
@Author：ZuoChenyang
@Time：2023/8/2 15:18
@Copyright：©2023 ZuoChenyang
"""
import math
import random


class Graph(object):
    def __init__(self, n, pheromone=1.0, alpha=1.0, beta=1.0):
        self.n = n
        self.pheromone = [[pheromone for _ in range(n)] for _ in range(n)]
        self.alpha = alpha
        self.beta = beta
        self.distances = self._generate_distances()

    def _generate_distances(self):
        # randomly generate distances between cities
        distances = []
        for i in range(self.n):
            row = []
            for j in range(self.n):
                if i == j:
                    row.append(0)
                elif j > i:
                    row.append(random.uniform(1, 10))
                else:
                    row.append(distances[j][i])
            distances.append(row)
        return distances


class Ant(object):
    def __init__(self, graph, start_city):
        self.graph = graph
        self.start_city = start_city
        self.curr_city = start_city
        self.visited = [False for _ in range(graph.n)]
        self.visited[start_city] = True
        self.tour_length = 0
        self.tour = [start_city]

    def choose_next_city(self):
        # calculate probabilities of all cities
        probs = []
        total_prob = 0
        for i in range(self.graph.n):
            if not self.visited[i]:
                prob = self.graph.pheromone[self.curr_city][i] ** self.graph.alpha * (
                            1.0 / self.graph.distances[self.curr_city][i]) ** self.graph.beta
                probs.append((i, prob))
                total_prob += prob
        # select the next city randomly based on the probabilities
        r = random.uniform(0, total_prob)
        upto = 0
        for i, prob in probs:
            if upto + prob >= r:
                self.curr_city = i
                self.visited[i] = True
                self.tour.append(i)
                self.tour_length += self.graph.distances[self.tour[-2]][i]
                return
            upto += prob

    def tour_complete(self):
        return len(self.tour) == self.graph.n


def ant_colony(graph, num_ants, num_iterations, evaporation_rate=0.5, q=500):
    shortest_tour = None
    shortest_tour_length = float('inf')
    for i in range(num_iterations):
        # initialize ants
        ants = [Ant(graph, random.randint(0, graph.n - 1)) for _ in range(num_ants)]
        # have each ant choose a path
        for ant in ants:
            while not ant.tour_complete():
                ant.choose_next_city()
            ant.tour_length += graph.distances[ant.tour[-1]][ant.start_city]
            # check if this ant found a shorter tour
            if ant.tour_length < shortest_tour_length:
                shortest_tour_length = ant.tour_length
                shortest_tour = ant.tour[:]
        # update pheromones
        for i in range(graph.n):
            for j in range(graph.n):
                if i != j:
                    graph.pheromone[i][j] *= (1 - evaporation_rate)
                    for ant in ants:
                        if (i, j) in zip(ant.tour, ant.tour[1:]):
                            graph.pheromone[i][j] += q / ant.tour_length
    return shortest_tour, shortest_tour_length


# generate a 11-city TSP problem and solve it using the ant colony algorithm
graph = Graph(11)
m_shortest_tour, m_shortest_tour_length = ant_colony(graph, num_ants=10, num_iterations=50)

# print the shortest tour and its length
print("Shortest tour:", m_shortest_tour)
print("Shortest tour length:", m_shortest_tour_length)
