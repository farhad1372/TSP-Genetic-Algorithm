from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import asdict, dataclass, replace
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# =========================================================
# تنظیمات الگوریتم ژنتیک
# =========================================================

@dataclass(frozen=True)
class GAConfig:
    population_size: int = 100
    crossover_rate: float = 0.90
    mutation_rate: float = 0.05
    elite_rate: float = 0.05

    mutation_type: str = "inversion"
    replacement_method: str = "generational"

    max_generations: int = 200
    patience: int = 40
    tournament_size: int = 3


# =========================================================
# خواندن ماتریس فاصله
# =========================================================

def load_distance_matrix(file_path):
    data = pd.read_csv(file_path, index_col=0)

    city_names = [str(name) for name in data.columns]
    distance_matrix = data.to_numpy(dtype=float)

    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("ماتریس فاصله باید مربعی باشد.")

    np.fill_diagonal(distance_matrix, 0.0)

    return city_names, distance_matrix


# =========================================================
# محاسبه طول تور
# =========================================================

def calculate_route_cost(route, distance_matrix, penalty):
    next_cities = np.roll(route, -1)
    edges = distance_matrix[route, next_cities]

    edges = np.where(np.isfinite(edges), edges, penalty)

    return float(np.sum(edges))


def evaluate_population(population, distance_matrix, penalty):
    next_cities = np.roll(population, -1, axis=1)
    edges = distance_matrix[population, next_cities]

    edges = np.where(np.isfinite(edges), edges, penalty)

    return np.sum(edges, axis=1)


def is_valid_route(route, distance_matrix):
    next_cities = np.roll(route, -1)
    edges = distance_matrix[route, next_cities]

    return bool(np.all(np.isfinite(edges)))


# =========================================================
# تولید جمعیت اولیه
# =========================================================

def create_random_valid_route(distance_matrix, rng):
    number_of_cities = distance_matrix.shape[0]
    finite_edges = np.isfinite(distance_matrix)

    for _ in range(300):
        start_city = int(rng.integers(number_of_cities))

        route = [start_city]
        unvisited = set(range(number_of_cities))
        unvisited.remove(start_city)

        valid = True

        while unvisited:
            current_city = route[-1]

            candidates = [
                city
                for city in unvisited
                if finite_edges[current_city, city]
            ]

            if len(unvisited) == 1:
                candidates = [
                    city
                    for city in candidates
                    if finite_edges[city, start_city]
                ]

            if not candidates:
                valid = False
                break

            candidates = np.asarray(candidates, dtype=int)
            candidate_distances = distance_matrix[current_city, candidates]

            number_of_choices = min(6, len(candidates))

            best_indices = np.argpartition(
                candidate_distances,
                number_of_choices - 1
            )[:number_of_choices]

            selected_candidates = candidates[best_indices]
            next_city = int(rng.choice(selected_candidates))

            route.append(next_city)
            unvisited.remove(next_city)

        if valid and finite_edges[route[-1], route[0]]:
            return np.asarray(route, dtype=np.int16)

    return rng.permutation(number_of_cities).astype(np.int16)


def create_initial_population(
    population_size,
    distance_matrix,
    rng
):
    number_of_cities = distance_matrix.shape[0]

    if np.all(np.isfinite(distance_matrix)):
        population = [
            rng.permutation(number_of_cities)
            for _ in range(population_size)
        ]
    else:
        population = [
            create_random_valid_route(distance_matrix, rng)
            for _ in range(population_size)
        ]

    return np.asarray(population, dtype=np.int16)


# =========================================================
# انتخاب تورنمنتی
# =========================================================

def tournament_selection(
    population,
    fitness,
    tournament_size,
    rng
):
    selected_indices = rng.integers(
        0,
        len(population),
        size=tournament_size
    )

    best_index = selected_indices[
        np.argmin(fitness[selected_indices])
    ]

    return population[best_index]


# =========================================================
# عملگر ترکیب Ordered Crossover
# =========================================================

def ordered_crossover(parent1, parent2, rng):
    number_of_cities = len(parent1)

    left, right = sorted(
        rng.choice(
            number_of_cities,
            size=2,
            replace=False
        )
    )

    right += 1

    child = np.full(
        number_of_cities,
        -1,
        dtype=np.int16
    )

    child[left:right] = parent1[left:right]

    used_cities = set(
        int(city)
        for city in child[left:right]
    )

    remaining_cities = [
        int(city)
        for city in parent2
        if int(city) not in used_cities
    ]

    empty_positions = (
        list(range(right, number_of_cities))
        + list(range(0, left))
    )

    child[empty_positions] = remaining_cities

    return child


# =========================================================
# عملگرهای جهش
# =========================================================

def swap_mutation(route, rng):
    i, j = rng.choice(
        len(route),
        size=2,
        replace=False
    )

    route[i], route[j] = route[j], route[i]


def inversion_mutation(route, rng):
    i, j = sorted(
        rng.choice(
            len(route),
            size=2,
            replace=False
        )
    )

    route[i:j + 1] = route[i:j + 1][::-1]


def scramble_mutation(route, rng):
    i, j = sorted(
        rng.choice(
            len(route),
            size=2,
            replace=False
        )
    )

    rng.shuffle(route[i:j + 1])


def mutate(route, mutation_type, rng):
    if mutation_type == "swap":
        swap_mutation(route, rng)

    elif mutation_type == "inversion":
        inversion_mutation(route, rng)

    elif mutation_type == "scramble":
        scramble_mutation(route, rng)

    else:
        raise ValueError(
            f"نوع جهش نامعتبر است: {mutation_type}"
        )


# =========================================================
# اجرای یک مرتبه الگوریتم ژنتیک
# =========================================================

def run_genetic_algorithm(
    distance_matrix,
    config,
    seed
):
    rng = np.random.default_rng(seed)

    number_of_cities = distance_matrix.shape[0]

    finite_values = distance_matrix[
        np.isfinite(distance_matrix)
    ]

    maximum_distance = np.max(finite_values)

    penalty = float(
        maximum_distance
        * number_of_cities
        * 20
    )

    population = create_initial_population(
        config.population_size,
        distance_matrix,
        rng
    )

    fitness = evaluate_population(
        population,
        distance_matrix,
        penalty
    )

    best_index = int(np.argmin(fitness))
    best_route = population[best_index].copy()
    best_cost = float(fitness[best_index])

    fitness_history = [best_cost]

    convergence_generation = 0
    no_improvement_count = 0

    elite_count = int(
        round(
            config.population_size
            * config.elite_rate
        )
    )

    elite_count = max(
        0,
        min(
            elite_count,
            config.population_size - 1
        )
    )

    generation = 0

    for generation in range(
        1,
        config.max_generations + 1
    ):
        children = []

        while len(children) < config.population_size:
            parent1 = tournament_selection(
                population,
                fitness,
                config.tournament_size,
                rng
            )

            parent2 = tournament_selection(
                population,
                fitness,
                config.tournament_size,
                rng
            )

            if rng.random() < config.crossover_rate:
                child = ordered_crossover(
                    parent1,
                    parent2,
                    rng
                )
            else:
                child = parent1.copy()

            if rng.random() < config.mutation_rate:
                mutate(
                    child,
                    config.mutation_type,
                    rng
                )

            children.append(child)

        children = np.asarray(
            children,
            dtype=np.int16
        )

        children_fitness = evaluate_population(
            children,
            distance_matrix,
            penalty
        )

        # روش جایگزینی نسلی همراه با نخبگی
        if config.replacement_method == "generational":
            if elite_count > 0:
                elite_indices = np.argsort(
                    fitness
                )[:elite_count]

                worst_children_indices = np.argsort(
                    children_fitness
                )[-elite_count:]

                children[
                    worst_children_indices
                ] = population[elite_indices]

                children_fitness[
                    worst_children_indices
                ] = fitness[elite_indices]

            population = children
            fitness = children_fitness

        # روش μ + λ
        elif config.replacement_method == "plus":
            combined_population = np.vstack(
                (population, children)
            )

            combined_fitness = np.concatenate(
                (fitness, children_fitness)
            )

            selected_indices = np.argsort(
                combined_fitness
            )[:config.population_size]

            population = combined_population[
                selected_indices
            ]

            fitness = combined_fitness[
                selected_indices
            ]

        # روش μ , λ
        elif config.replacement_method == "comma":
            selected_indices = np.argsort(
                children_fitness
            )[:config.population_size]

            population = children[
                selected_indices
            ]

            fitness = children_fitness[
                selected_indices
            ]

        else:
            raise ValueError(
                "روش جایگزینی نامعتبر است."
            )

        current_best_index = int(
            np.argmin(fitness)
        )

        current_best_cost = float(
            fitness[current_best_index]
        )

        if current_best_cost < best_cost - 1e-12:
            best_cost = current_best_cost
            best_route = population[
                current_best_index
            ].copy()

            convergence_generation = generation
            no_improvement_count = 0

        else:
            no_improvement_count += 1

        fitness_history.append(best_cost)

        if no_improvement_count >= config.patience:
            break

    route_is_valid = is_valid_route(
        best_route,
        distance_matrix
    )

    if route_is_valid:
        real_cost = calculate_route_cost(
            best_route,
            distance_matrix,
            math.inf
        )
    else:
        real_cost = math.inf

    return {
        "seed": seed,
        "best_cost": real_cost,
        "valid_route": route_is_valid,
        "stop_generation": generation,
        "convergence_generation": convergence_generation,
        "best_route": best_route.tolist(),
        "fitness_history": fitness_history
    }


# =========================================================
# تنظیم آزمایش‌ها
# =========================================================

def create_experiment_settings():
    base_config = GAConfig()

    settings = []

    # مقایسه نوع جهش با نرخ ثابت 5 درصد
    for mutation_type in [
        "swap",
        "inversion",
        "scramble"
    ]:
        settings.append(
            (
                f"mutation_type_{mutation_type}",
                replace(
                    base_config,
                    mutation_type=mutation_type,
                    mutation_rate=0.05
                ),
                "mutation_type"
            )
        )

    # مقایسه درصد جهش
    for mutation_rate in [
        0.02,
        0.05,
        0.10
    ]:
        settings.append(
            (
                f"mutation_rate_{int(mutation_rate * 100)}",
                replace(
                    base_config,
                    mutation_type="inversion",
                    mutation_rate=mutation_rate
                ),
                "mutation_rate"
            )
        )

    # مقایسه درصد ترکیب
    for crossover_rate in [
        0.70,
        0.90,
        0.95
    ]:
        settings.append(
            (
                f"crossover_rate_{int(crossover_rate * 100)}",
                replace(
                    base_config,
                    crossover_rate=crossover_rate
                ),
                "crossover_rate"
            )
        )

    # مقایسه درصد نخبگی
    for elite_rate in [
        0.00,
        0.05,
        0.10
    ]:
        settings.append(
            (
                f"elite_rate_{int(elite_rate * 100)}",
                replace(
                    base_config,
                    elite_rate=elite_rate
                ),
                "elite_rate"
            )
        )

    # مقایسه روش‌های جایگزینی
    settings.append(
        (
            "replacement_mu_plus_lambda",
            replace(
                base_config,
                replacement_method="plus",
                elite_rate=0.0
            ),
            "replacement_method"
        )
    )

    settings.append(
        (
            "replacement_mu_comma_lambda",
            replace(
                base_config,
                replacement_method="comma",
                elite_rate=0.0
            ),
            "replacement_method"
        )
    )

    return settings


# =========================================================
# توابع مربوط به اجرای موازی
# =========================================================

WORKER_DISTANCE_MATRIX = None


def initialize_worker(distance_matrix):
    global WORKER_DISTANCE_MATRIX
    WORKER_DISTANCE_MATRIX = distance_matrix


def run_worker(task):
    config_dictionary, seed, setting_name, group_name = task

    config = GAConfig(**config_dictionary)

    result = run_genetic_algorithm(
        WORKER_DISTANCE_MATRIX,
        config,
        seed
    )

    result["setting"] = setting_name
    result["group"] = group_name

    return result


# =========================================================
# اجرای 100 مرتبه برای هر حالت
# =========================================================

def run_experiments(
    distance_matrix,
    number_of_runs,
    number_of_workers
):
    settings = create_experiment_settings()

    tasks = []

    for setting_name, config, group_name in settings:
        for seed in range(number_of_runs):
            tasks.append(
                (
                    asdict(config),
                    seed,
                    setting_name,
                    group_name
                )
            )

    if number_of_workers == 1:
        initialize_worker(distance_matrix)
        results = [
            run_worker(task)
            for task in tasks
        ]
    else:
        with Pool(
            processes=number_of_workers,
            initializer=initialize_worker,
            initargs=(distance_matrix,)
        ) as pool:
            results = list(
                pool.imap_unordered(
                    run_worker,
                    tasks,
                    chunksize=5
                )
            )

    result_rows = []
    histories = {}
    best_routes = {}

    for result in results:
        row = {
            key: value
            for key, value in result.items()
            if key not in [
                "best_route",
                "fitness_history"
            ]
        }

        result_rows.append(row)

        setting_name = result["setting"]

        histories.setdefault(
            setting_name,
            []
        ).append(
            result["fitness_history"]
        )

        if (
            setting_name not in best_routes
            or result["best_cost"]
            < best_routes[setting_name]["best_cost"]
        ):
            best_routes[setting_name] = {
                "seed": result["seed"],
                "best_cost": result["best_cost"],
                "best_route": result["best_route"]
            }

    raw_results = pd.DataFrame(result_rows)

    summary = (
        raw_results
        .groupby(
            ["group", "setting"],
            as_index=False
        )
        .agg(
            mean_cost=("best_cost", "mean"),
            std_cost=("best_cost", "std"),
            min_cost=("best_cost", "min"),
            max_cost=("best_cost", "max"),
            mean_convergence_generation=(
                "convergence_generation",
                "mean"
            ),
            std_convergence_generation=(
                "convergence_generation",
                "std"
            ),
            mean_stop_generation=(
                "stop_generation",
                "mean"
            ),
            valid_route_percent=(
                "valid_route",
                "mean"
            )
        )
    )

    summary["valid_route_percent"] *= 100

    return (
        raw_results,
        summary,
        histories,
        best_routes
    )


# =========================================================
# هم‌اندازه کردن تاریخچه‌های برازندگی
# =========================================================

def prepare_fitness_histories(histories):
    maximum_length = max(
        len(history)
        for history in histories
    )

    result = np.empty(
        (
            len(histories),
            maximum_length
        ),
        dtype=float
    )

    for index, history in enumerate(histories):
        result[index, :len(history)] = history
        result[index, len(history):] = history[-1]

    return result


# =========================================================
# رسم نمودار برازندگی
# =========================================================

def plot_fitness_curve(
    histories,
    setting_name,
    matrix_name,
    output_path
):
    fitness_values = prepare_fitness_histories(
        histories[setting_name]
    )

    mean_fitness = np.mean(
        fitness_values,
        axis=0
    )

    std_fitness = np.std(
        fitness_values,
        axis=0,
        ddof=1
    )

    generations = np.arange(
        len(mean_fitness)
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        generations,
        mean_fitness,
        label="Mean best fitness"
    )

    plt.fill_between(
        generations,
        mean_fitness - std_fitness,
        mean_fitness + std_fitness,
        alpha=0.2,
        label="Mean ± standard deviation"
    )

    plt.xlabel("Generation")
    plt.ylabel("Best route length")
    plt.title(
        f"Fitness curve - {matrix_name}"
    )

    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=180
    )
    plt.close()


# =========================================================
# رسم نمودار مقایسه پارامترها
# =========================================================

def plot_comparison(
    summary,
    group_name,
    matrix_name,
    output_path
):
    selected_data = summary[
        summary["group"] == group_name
    ].copy()

    positions = np.arange(
        len(selected_data)
    )

    labels = selected_data[
        "setting"
    ].tolist()

    plt.figure(figsize=(10, 5))

    plt.errorbar(
        positions,
        selected_data["mean_cost"],
        yerr=selected_data["std_cost"],
        fmt="o",
        capsize=5
    )

    plt.xticks(
        positions,
        labels,
        rotation=30,
        ha="right"
    )

    plt.ylabel(
        "Route length (mean ± standard deviation)"
    )

    plt.title(
        f"{matrix_name} - {group_name}"
    )

    plt.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()
    plt.savefig(
        output_path,
        dpi=180
    )
    plt.close()


# =========================================================
# اجرای آزمایش روی یک ماتریس
# =========================================================

def process_matrix(
    matrix_key,
    matrix_file,
    output_directory,
    number_of_runs,
    number_of_workers
):
    city_names, distance_matrix = load_distance_matrix(
        matrix_file
    )

    print("\n" + "=" * 70)
    print(f"Matrix: {matrix_key}")
    print(f"Number of cities: {len(city_names)}")
    print("=" * 70)

    raw_results, summary, histories, best_routes = run_experiments(
        distance_matrix,
        number_of_runs,
        number_of_workers
    )

    raw_results.insert(
        0,
        "matrix",
        matrix_key
    )

    summary.insert(
        0,
        "matrix",
        matrix_key
    )

    raw_results.to_csv(
        output_directory
        / f"raw_results_{matrix_key}.csv",
        index=False
    )

    summary.to_csv(
        output_directory
        / f"summary_{matrix_key}.csv",
        index=False
    )

    with open(
        output_directory
        / f"best_routes_{matrix_key}.json",
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            {
                "city_names": city_names,
                "results": best_routes
            },
            file,
            ensure_ascii=False,
            indent=2
        )

    for group_name in [
        "mutation_type",
        "mutation_rate",
        "crossover_rate",
        "elite_rate",
        "replacement_method"
    ]:
        plot_comparison(
            summary,
            group_name,
            matrix_key,
            output_directory
            / f"{matrix_key}_{group_name}.png"
        )

    plot_fitness_curve(
        histories,
        "mutation_rate_5",
        matrix_key,
        output_directory
        / f"{matrix_key}_fitness_curve.png"
    )

    print(summary.to_string(index=False))

    return summary


# =========================================================
# تابع اصلی
# =========================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--symmetric",
        default="distance_matrix_symmetric.csv"
    )

    parser.add_argument(
        "--asymmetric",
        default="distance_matrix_incomplete_asymmetric.csv"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=100
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=max(
            1,
            min(
                8,
                os.cpu_count() or 1
            )
        )
    )

    parser.add_argument(
        "--output",
        default="results"
    )

    args = parser.parse_args()

    output_directory = Path(args.output)
    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    symmetric_summary = process_matrix(
        matrix_key="symmetric",
        matrix_file=args.symmetric,
        output_directory=output_directory,
        number_of_runs=args.runs,
        number_of_workers=args.workers
    )

    asymmetric_summary = process_matrix(
        matrix_key="incomplete_asymmetric",
        matrix_file=args.asymmetric,
        output_directory=output_directory,
        number_of_runs=args.runs,
        number_of_workers=args.workers
    )

    all_summaries = pd.concat(
        [
            symmetric_summary,
            asymmetric_summary
        ],
        ignore_index=True
    )

    all_summaries.to_csv(
        output_directory
        / "all_results_summary.csv",
        index=False
    )

    print("\nتمام آزمایش‌ها با موفقیت انجام شدند.")
    print(
        "نتایج در پوشه زیر ذخیره شدند:"
    )
    print(
        output_directory.resolve()
    )


if __name__ == "__main__":
    main()