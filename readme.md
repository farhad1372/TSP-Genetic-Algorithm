# TSP Genetic Algorithm

This project solves the Traveling Salesman Problem (TSP) using a Genetic Algorithm.

The program works with two distance matrices:

1. A complete and symmetric distance matrix
2. An incomplete and asymmetric distance matrix

The algorithm is executed several times with different random seeds. The final results are reported using the mean and standard deviation.

## Project Goal

The goal is to find a tour that:

- Starts from one city
- Visits every city exactly once
- Returns to the starting city
- Has the minimum possible total distance

Each chromosome represents one complete ordering of the cities.

## Main Features

- Genetic Algorithm for TSP
- Tournament selection
- Ordered Crossover (OX)
- Swap, Inversion, and Scramble mutation
- Elitism
- Comparison of mutation rates
- Comparison of crossover rates
- Comparison of elitism rates
- Comparison of `mu + lambda` and `mu, lambda` replacement
- Support for incomplete and asymmetric matrices
- Fitness convergence plots
- CSV and JSON output files
- Multiple runs with different random seeds

## Requirements

Python 3.9 or newer is recommended.

The required packages are listed in `requirements.txt`.

Install all dependencies with:

```bash
pip install -r requirements.txt
```

The main dependencies are:

- NumPy
- Pandas
- Matplotlib

Other modules used by the project are included in the Python standard library and do not require separate installation.

## Project Structure

Place the following files in the same directory:

```text
project/
|-- main.py
|-- requirements.txt
|-- distance_matrix_symmetric.csv
|-- distance_matrix_incomplete_asymmetric.csv
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
python main.py   --symmetric "distance_matrix_symmetric.csv"   --asymmetric "distance_matrix_incomplete_asymmetric.csv"   --runs 100   --workers 2   --output final_results
```

## Default Algorithm Settings

The default settings are:

- Population size: 100
- Crossover rate: 90%
- Mutation rate: 5%
- Elitism rate: 5%
- Maximum generations: 200
- Early stopping patience: 40 generations
- Selection method: Tournament Selection
- Default mutation operator: Inversion

## How the Code Works

### Reading the Distance Matrix

The `load_distance_matrix` function reads a CSV file and converts it into a NumPy matrix. It also verifies that the matrix is square.

### Route Representation

Each route is stored as an array of city indices.

Example:

```text
[0, 4, 2, 1, 3]
```

The last city is automatically connected to the first city when the route cost is calculated.

### Fitness Calculation

The fitness value is the total route distance. Since the problem is a minimization problem, a smaller value represents a better solution.

For the incomplete matrix, missing edges receive a large penalty. This prevents invalid routes from being considered good solutions.

### Initial Population

For the complete matrix, random permutations are generated.

For the incomplete matrix, the program tries to generate routes using available edges. This increases the number of valid tours in the initial population.

### Selection

Tournament selection chooses a small number of chromosomes randomly and selects the best one as a parent.

### Crossover

Ordered Crossover is used because it preserves a valid permutation and prevents cities from being duplicated or removed.

### Mutation

The program supports three mutation operators:

- Swap: exchanges two cities
- Inversion: reverses part of the route
- Scramble: randomly shuffles part of the route

Mutation helps maintain population diversity and reduces the chance of premature convergence.

### Replacement

The program supports three replacement methods:

- Generational replacement with elitism
- `mu + lambda`: the next population is selected from both parents and children
- `mu, lambda`: the next population is selected only from children

### Stopping Condition

The algorithm stops when:

- The maximum number of generations is reached, or
- No better solution is found for a fixed number of consecutive generations

## Experiments

The program compares:

- Mutation type
- Mutation rate
- Crossover rate
- Elitism rate
- Replacement method

Only one parameter is changed in each experiment while the other parameters remain fixed.

Because the Genetic Algorithm is stochastic, each setting is executed multiple times with different random seeds. The assignment requires 100 independent runs.

## Output Files

The program creates output files such as:

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

## Important Output Values

The summary files contain values such as:

- Mean route length
- Standard deviation
- Minimum route length
- Maximum route length
- Mean convergence generation
- Mean stopping generation
- Valid route percentage

## Checking the Results

A correct result should satisfy the following conditions:

- Every route contains all cities exactly once
- No city is repeated
- The route cost is not `inf` or `NaN`
- `valid_route_percent` should be close to or equal to 100
- The fitness curve should generally decrease and then become stable
- Running the program again with the same seeds should produce the same results

## Notes

A single execution is not enough to evaluate a Genetic Algorithm because selection, crossover, mutation, and population generation contain random operations.

For this reason, the program uses different random seeds and reports the mean and standard deviation of the results.
