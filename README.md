# AI Labs

## Overview
This repository, contains laboratory assignments completed during the 4th semester of Computer Science studies. The goal of these labs is to explore various machine learning and artificial intelligence techniques through hands-on coding exercises.

## Lab 1: Linear Regression
### Description
The first laboratory focuses on implementing a simple **Linear Regression** model to predict fuel efficiency (**MPG - Miles Per Gallon**) based on vehicle weight. The model is implemented using two different approaches:
1. **Closed-Form Solution** (Normal Equation)
2. **Batch Gradient Descent**

## Lab 2: Genetic Algorithm for the Knapsack Problem  
### Description
This laboratory implements a genetic algorithm to solve the Knapsack Problem, a classic combinatorial optimization problem. The goal is to determine the most valuable combination of items that can be placed in a knapsack without exceeding its maximum weight capacity.  

### Features

- Randomized Initial Population: Generates an initial set of candidate solutions.
- Fitness Function: Evaluates how good each solution is based on the total value of selected items while ensuring weight constraints.
- Roulette Selection: Selects parents for crossover based on their fitness proportionally.
- Single-Point Crossover: Produces offspring by combining genes from two parent solutions.
- Mutation: Introduces random variations to maintain diversity and avoid local optima.
- Elitism: Preserves the best solution from each generation to ensure progress.
- Performance Tracking: Records fitness history and plots the evolution of solutions over generations.
   
### Output

- Best solution found (list of selected item names)
- Total value of the best solution
- Execution time
- A graph showing the fitness progress over generations

### Visualization

The algorithm tracks and visualizes how the fitness of the population evolves. The red line represents the best solution in each generation, while scattered points represent different solutions across generations.
