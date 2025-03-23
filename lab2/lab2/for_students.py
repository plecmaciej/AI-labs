from itertools import compress
import random
import time
import matplotlib.pyplot as plt
from data import *

def initial_population(individual_size, population_size):
    return [[random.choice([True, False]) for _ in range(individual_size)] for _ in range(population_size)]

def fitness(items, knapsack_max_capacity, individual):
    total_weight = sum(compress(items['Weight'], individual))
    if total_weight > knapsack_max_capacity:
        return 0
    return sum(compress(items['Value'], individual))

def population_best(items, knapsack_max_capacity, population):
    best_individual = None
    best_individual_fitness = -1
    for individual in population:
        individual_fitness = fitness(items, knapsack_max_capacity, individual)
        if individual_fitness > best_individual_fitness:
            best_individual = individual
            best_individual_fitness = individual_fitness
    return best_individual, best_individual_fitness

def roulette_selection(population, items, knapsack_max_capacity):
    total_fitness = sum(fitness(items, knapsack_max_capacity, individual) for individual in population)
    probabilities = [fitness(items, knapsack_max_capacity, individual) / total_fitness for individual in population]
    selected = random.choices(population, weights=probabilities, k=2)
    return selected

def single_point_crossover(parent1, parent2):
    crossover_point = len(parent1) // 2
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2

def mutate(individual, mutation_rate):
    for i in range(len(individual)):
        if random.random() < mutation_rate:
            individual[i] = not individual[i]
    return individual

items, knapsack_max_capacity = get_big()

population_size = 100
generations = 200
n_selection = 20
n_elite = 1
mutation_rate = 0.01

start_time = time.time()
best_solution = None
best_fitness = 0
population_history = []
best_history = []
population = initial_population(len(items), population_size)
for _ in range(generations):
    population_history.append(population)

    # Elitism: Select top individuals
    elites = sorted(population, key=lambda x: fitness(items, knapsack_max_capacity, x), reverse=True)[:n_elite]

    # Selection
    selected = [roulette_selection(population, items, knapsack_max_capacity) for _ in range(n_selection)]



    # Crossover
    children = []
    for _ in range(population_size // 2):  # 49 razy
        parent1 = random.choice(selected)[0]  # Losowy pierwszy rodzic z grupy wybranych
        parent2 = random.choice(population)  # Losowy drugi rodzic z całej populacji
        child1, child2 = single_point_crossover(parent1, parent2)
        children.append(child1)
        children.append(child2)

    # Jeśli populacja dzieci ma być równa 99, możemy usunąć jedno losowe dziecko
    if len(children) > population_size-n_elite:
        children.pop(random.randrange(len(children)))
    #print('chil: ', len(children))
    # Mutate
    mutated_children = [mutate(child, mutation_rate) for child in children]

    # Update population
    population = elites + mutated_children

    best_individual, best_individual_fitness = population_best(items, knapsack_max_capacity, population)
    if best_individual_fitness > best_fitness:
        best_solution = best_individual
        best_fitness = best_individual_fitness
    best_history.append(best_fitness)

end_time = time.time()
total_time = end_time - start_time
print('Best solution:', list(compress(items['Name'], best_solution)))
print('Best solution value:', best_fitness)
print('Time: ', total_time)

# plot generations
x = []
y = []
top_best = 10
for i, population in enumerate(population_history):
    plotted_individuals = min(len(population), top_best)
    x.extend([i] * plotted_individuals)
    population_fitnesses = [fitness(items, knapsack_max_capacity, individual) for individual in population]
    population_fitnesses.sort(reverse=True)
    y.extend(population_fitnesses[:plotted_individuals])
plt.scatter(x, y, marker='.')
plt.plot(best_history, 'r')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.show()
