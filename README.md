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
### Description
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

## Lab 4: Classification

### Description
Implementation of a decision tree classifier and random forest ensemble for solving classification problems. The project demonstrates fundamental machine learning concepts including:
- Decision tree construction using Gini impurity
- Random forest with bootstrap aggregating (bagging)
- Feature subset selection for ensemble diversity

### Key Components

#### Decision Tree Implementation (`Node` class)
Core tree-building functionality:
- `gini_best_score()`: Calculates optimal data splits using Gini impurity
- `find_best_split()`: Identifies the best feature and value for node splitting
- `train()`: Recursively builds the decision tree with configurable depth
- `predict()`: Classifies samples by traversing the tree

#### Random Forest Implementation
Ensemble learning features:
- Bagging with replacement (bootstrap sampling)
- Parallel tree training with feature subset selection
- Majority voting for classification
- Accuracy evaluation method

## Hyperparameters
- `depth`: Maximum tree depth (controls overfitting)
- `ntrees`: Number of trees in the forest
- `feature_subset`: Features considered at each split (√n_features by default)

## Lab 5: K-Means Clustering  

### Description   

This projectfocus on clustering techniques using the K-Means algorithm. The implementation includes different centroid initialization strategies and evaluates clustering performance on the Iris dataset.

### Implementation Details

The project implements K-Means clustering with two initialization methods:   

- Forgy Initialization - Randomly selects k points from the dataset as initial centroids.   
  
- K-Means++ Initialization - Selects the first centroid randomly, then picks subsequent centroids based on the maximum distance from already chosen centroids to improve clustering performance.   

## Lab 6: Neural Networks for Classification

This labs contains implementations of neural networks for classification tasks using both NumPy and PyTorch. The project includes examples of simple perceptrons, two-layer neural networks, and deep multi-layer networks trained with gradient descent.

### Features

- **Single Neuron Classifier**: Implements a basic perceptron using NumPy, utilizing activation functions such as ReLU, Sigmoid, and Hardlim. The neuron is trained on linearly separable data.
- **Two-Layer Neural Network**: Builds a neural network with one hidden layer to classify non-linearly separable data. The network consists of a hidden layer using Hardlim activation and an output layer making binary predictions.
- **Multi-Layer Neural Network (PyTorch)**: Uses deep learning with PyTorch to classify spiral-shaped data. The model consists of multiple hidden layers with ReLU activation and an output layer with a Sigmoid activation for binary classification.
- **Visualization Utilities**: Tools to inspect and visualize datasets and decision boundaries. These include plotting decision boundaries, activation functions, and loss curves.

### Implementation Details

#### Single Neuron Model (NumPy)

- The model initializes weights randomly and applies an activation function to classify data.
- The function `zad1_single_neuron(student_id)` loads a linearly separable dataset and fits a single neuron model.
- The decision boundary is visualized to evaluate performance.

#### Two-Layer Neural Network (NumPy)

- The `DenseLayer` class represents a single layer of neurons.
- The `SimpleTwoLayerNetwork` class stacks two `DenseLayer`s to create a simple MLP.
- The model is tested on non-linearly separable data and visualized using decision boundaries.

#### Multi-Layer Neural Network (PyTorch)

- The `TorchMultiLayerNetwork` class constructs a deep neural network using `torch.nn.Module`.
- The model is trained using stochastic gradient descent and binary cross-entropy loss.
- The `training()` function optimizes weights, while `evaluate_model()` assesses accuracy.
