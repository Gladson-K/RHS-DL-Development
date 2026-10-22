"""Evolutionary / Genetic Algorithm search strategy for RHS NAS.

Stage 3 Pareto & Evolutionary Search:
Optimizes population of architectures over competing objectives (rate, latency, params).
Produces Pareto frontier: Accuracy-optimal, Balanced, and Latency-optimal architectures.
"""

from __future__ import annotations

import copy
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from rhs_beamform_nas.nas.search_space.model_space import (
    DEPTH_CHOICES,
    HIDDEN_DIM_CHOICES,
    CONDITIONING_CHOICES,
    M_REPR_CHOICES,
    W_REPR_CHOICES,
)
from rhs_beamform_nas.nas.search_space.rhs_adaptive_cell import DEFAULT_CANDIDATES


class EvolutionaryStrategy:
    """Multi-objective evolutionary algorithm for RHS beamforming architectures.

    Supports mutation, crossover, selection, and Pareto frontier extraction.
    """

    def __init__(
        self,
        population_size: int = 20,
        mutation_rate: float = 0.25,
        crossover_rate: float = 0.5,
        num_generations: int = 10,
        seed: Optional[int] = None,
    ) -> None:
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.population: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        if seed is not None:
            random.seed(seed)

    def initialize_population(self) -> List[Dict[str, Any]]:
        """Initialize random population of candidate architectures."""
        from rhs_beamform_nas.nas.strategies.random_search import RandomSearchStrategy
        sampler = RandomSearchStrategy(num_trials=self.population_size)
        self.population = sampler.generate_trials()
        return self.population

    def mutate(self, individual: Dict[str, Any]) -> Dict[str, Any]:
        """Apply random mutation to an architecture individual."""
        child = copy.deepcopy(individual)
        mutation_type = random.choice([
            "hidden_dim", "depth", "operator", "conditioning", "representation"
        ])

        if mutation_type == "hidden_dim":
            child["hidden_dim"] = random.choice(HIDDEN_DIM_CHOICES)
        elif mutation_type == "depth":
            new_depth = random.choice(DEPTH_CHOICES)
            current_ops = child.get("spatial_operator", [])
            if len(current_ops) < new_depth:
                current_ops.extend(
                    [random.choice(DEFAULT_CANDIDATES) for _ in range(new_depth - len(current_ops))]
                )
            else:
                current_ops = current_ops[:new_depth]
            child["rhs_depth"] = new_depth
            child["spatial_operator"] = current_ops
        elif mutation_type == "operator":
            ops = child.get("spatial_operator", [])
            if ops:
                idx = random.randrange(len(ops))
                ops[idx] = random.choice(DEFAULT_CANDIDATES)
                child["spatial_operator"] = ops
        elif mutation_type == "conditioning":
            child["conditioning"] = random.choice(CONDITIONING_CHOICES)
        elif mutation_type == "representation":
            child["M_representation"] = random.choice(M_REPR_CHOICES)
            child["w_representation"] = random.choice(W_REPR_CHOICES)

        return child

    def crossover(self, parent1: Dict[str, Any], parent2: Dict[str, Any]) -> Dict[str, Any]:
        """Perform crossover between two parent architectures."""
        child: Dict[str, Any] = {}
        for key in ["hidden_dim", "conditioning", "M_representation", "w_representation"]:
            child[key] = random.choice([parent1.get(key), parent2.get(key)])

        # Depth & operator crossover
        p1_ops = parent1.get("spatial_operator", [])
        p2_ops = parent2.get("spatial_operator", [])
        chosen_ops = []
        max_len = max(len(p1_ops), len(p2_ops))
        for i in range(max_len):
            op1 = p1_ops[i] if i < len(p1_ops) else random.choice(DEFAULT_CANDIDATES)
            op2 = p2_ops[i] if i < len(p2_ops) else random.choice(DEFAULT_CANDIDATES)
            chosen_ops.append(random.choice([op1, op2]))

        child["rhs_depth"] = len(chosen_ops)
        child["spatial_operator"] = chosen_ops
        return child

    def step_generation(
        self,
        evaluate_fn: Callable[[Dict[str, Any]], float],
    ) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Run one generation of evaluation, selection, crossover, and mutation."""
        if not self.population:
            self.initialize_population()

        # Evaluate population
        scored = []
        for individual in self.population:
            score = evaluate_fn(individual)
            scored.append((score, individual))
            self.history.append({"architecture": individual, "score": score})

        # Sort by score descending (higher is better)
        scored.sort(key=lambda x: x[0], reverse=True)
        best_individual = scored[0][1]
        best_score = scored[0][0]

        # Elitism: keep top 20%
        num_elites = max(1, int(self.population_size * 0.2))
        next_population = [item[1] for item in scored[:num_elites]]

        # Fill remaining slots via selection, crossover, mutation
        while len(next_population) < self.population_size:
            p1 = random.choice(scored[: self.population_size // 2])[1]
            p2 = random.choice(scored[: self.population_size // 2])[1]
            if random.random() < self.crossover_rate:
                child = self.crossover(p1, p2)
            else:
                child = copy.deepcopy(p1)
            if random.random() < self.mutation_rate:
                child = self.mutate(child)
            next_population.append(child)

        self.population = next_population
        return self.population, {"best_score": best_score, "best_architecture": best_individual}

    def compute_pareto_frontier(
        self,
        evaluated_candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Compute non-dominated Pareto frontier candidates given metrics.

        Objectives: maximize normalized_rate, minimize latency_ms, minimize parameters.
        """
        pareto = []
        for candidate in evaluated_candidates:
            metrics = candidate.get("metrics", {})
            rate = metrics.get("normalized_rate", 0.0)
            latency = metrics.get("latency_ms", 1e6)
            params = metrics.get("parameters", 1e9)

            dominated = False
            for other in evaluated_candidates:
                if other == candidate:
                    continue
                o_metrics = other.get("metrics", {})
                o_rate = o_metrics.get("normalized_rate", 0.0)
                o_latency = o_metrics.get("latency_ms", 1e6)
                o_params = o_metrics.get("parameters", 1e9)

                # other dominates candidate if >= rate and <= latency and <= params, with at least one strictly better
                if (
                    o_rate >= rate
                    and o_latency <= latency
                    and o_params <= params
                    and (o_rate > rate or o_latency < latency or o_params < params)
                ):
                    dominated = True
                    break

            if not dominated:
                pareto.append(candidate)

        return pareto
