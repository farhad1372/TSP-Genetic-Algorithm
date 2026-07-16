# TSP Genetic Algorithm

This project solves the Traveling Salesman Problem (TSP) using a Genetic Algorithm.

The program works with two distance matrices:

1. A complete and symmetric distance matrix
2. An incomplete and asymmetric distance matrix

The algorithm is executed several times with different random seeds, and the final results are reported using the mean and standard deviation.

## Project Goal

The goal is to find a tour that:

- Starts from one city
- Visits every city exactly once
- Returns to the starting city
- Has the minimum possible total distance

Each chromosome represents one complete ordering of the cities.

## Main Features

- Population-based TSP optimization
- Tournament selection
- Ordered Crossover (OX)
- Three mutation operators:
  - Swap
  - Inversion
  - Scramble
- Elitism
- Comparison of crossover, mutation, and elitism rates
- Comparison of `mu + lambda` and `mu, lambda` replacement methods
- Support for incomplete and asymmetric distance matrices
- Fitness convergence plots
- CSV and JSON output files
- Multiple independent runs with different random seeds

## Requirements

Python 3.9 or newer is recommended.

Install the required packages with:

```bash
pip install numpy pandas matplotlib
```

The project also uses standard Python modules such as:

- argparse
- json
- multiprocessing
- pathlib
- dataclasses

These modules do not need separate installation.

## Project Files

Place the following files in the same directory:

```text
project/
|-- main.py
|-- distance_matrix_symmetric.csv
|-- distance_matrix_incomplete_asymmetric (1).csv
|-- README.md
`-- README_FA.md
```

## How to Run

First, run a small test:

```bash
python main.py --runs 2 --workers 1 --output test_results
```

For the final experiment with 100 independent runs:

```bash
python main.py --runs 100 --workers 1 --output final_results
```

To use multiple CPU cores:

```bash
python main.py --runs 100 --workers 4 --output final_results
```

On Windows, `py` can be used instead of `python`:

```bash
py main.py --runs 100 --workers 1 --output final_results
```

## Command-Line Arguments

- `--symmetric`: path to the complete symmetric matrix
- `--asymmetric`: path to the incomplete asymmetric matrix
- `--runs`: number of independent runs for each setting
- `--workers`: number of parallel processes
- `--output`: directory used to save the results

Example:

```bash
python main.py   --symmetric "distance_matrix_symmetric.csv"   --asymmetric "distance_matrix_incomplete_asymmetric (1).csv"   --runs 100   --workers 2   --output final_results
```

## Algorithm Settings

The default configuration is:

- Population size: 100
- Crossover rate: 90%
- Mutation rate: 5%
- Elitism rate: 5%
- Maximum generations: 200
- Early stopping patience: 40 generations
- Selection method: Tournament Selection
- Default mutation: Inversion

## How the Code Works

### Reading the Distance Matrix

The `load_distance_matrix` function reads a CSV file and converts it into a NumPy matrix. It also verifies that the matrix is square.

### Route Representation

Each route is stored as an array of city indices.

For example:

```text
[0, 4, 2, 1, 3]
```

The last city is automatically connected to the first city when the route cost is calculated.

### Fitness Calculation

The fitness value is the total route distance. Since this is a minimization problem, a smaller value means a better solution.

For the incomplete matrix, missing edges receive a very large penalty. This prevents invalid routes from being selected as good solutions.

### Initial Population

For the complete matrix, random permutations are generated.

For the incomplete matrix, the program tries to generate routes using available edges so that the initial population contains more valid tours.

### Selection

Tournament selection chooses a few chromosomes randomly and keeps the best one as a parent.

### Crossover

Ordered Crossover is used because it preserves a valid permutation and prevents cities from being duplicated or removed.

### Mutation

The code supports three mutation methods:

- Swap: exchanges two cities
- Inversion: reverses part of the route
- Scramble: randomly shuffles part of the route

Mutation maintains population diversity and reduces the chance of premature convergence.

### Replacement

The program supports:

- Generational replacement with elitism
- `mu + lambda`: selects the best solutions from parents and children
- `mu, lambda`: selects the next generation only from children

### Stopping Condition

The algorithm stops when:

- The maximum number of generations is reached, or
- No better solution is found for a fixed number of consecutive generations

## Output Files

The program creates files such as:

```text
all_results_summary.csv
raw_results_symmetric.csv
summary_symmetric.csv
best_routes_symmetric.json
raw_results_incomplete_asymmetric.csv
summary_incomplete_asymmetric.csv
best_routes_incomplete_asymmetric.json
```

It also creates fitness and comparison plots in PNG format.

## Checking the Results

A correct result should satisfy the following conditions:

- Every route contains all cities exactly once
- No city is repeated
- The final route cost is finite
- `valid_route_percent` should be close to or equal to 100
- The fitness curve should generally decrease and then become stable
- Repeating the experiment with the same seeds should produce the same results

## Notes

The Genetic Algorithm is stochastic, so one execution is not enough for reliable evaluation. For this reason, each setting is executed 100 times using different random seeds, and the mean and standard deviation are reported.
