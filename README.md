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


## Lab 3: Minimax Algorithm in connect4
### Overview
This project implements the Minimax algorithm with heuristic evaluation to make strategic decisions in the game Connect4. The agent simulates future game states to maximize its chances of winning while minimizing the opponent's advantage. This technique is commonly used in AI-driven game playing and is a fundamental concept in reinforcement learning and decision-making algorithms.

### Implementation Details
**MinimaxAgent** (Basic Minimax without Alpha-Beta Pruning)
- Uses a recursive minimax function to evaluate possible moves.

- Heuristic evaluation considers factors such as:

   - Number of tokens in a row.

   - Control of the center column.

   - Potential winning moves.

- Implements a normalized scoring system to adjust the evaluation range between -1 and 1.

**AlphaBetaAgent** (Optimized with Alpha-Beta Pruning)
- Extends the Minimax algorithm with Alpha-Beta pruning, significantly reducing computational complexity.

- Cuts off branches in the search tree that cannot affect the final decision, making it more efficient than standard Minimax.

### Heuristic Evaluation
The heuristic function assigns scores to different board states:

- Winning move: +1 (for the AI) or -1 (for the opponent).

- Three in a row with an open space: +100 (high priority for completion).

- Blocking the opponent's three in a row: -100 (defensive move).

- Control of the center column: Higher weight since it provides more connectivity opportunities.
