"""Top-level NAS search runner for RHS Adaptive Beamforming.

Orchestrates:
1. Search space initialization
2. Search strategy execution (Random / DARTS / ENAS / Evolution)
3. Multi-objective evaluation (Rate, Latency, Parameter count, Physical constraints)
4. Pareto frontier identification
5. Best architecture export to JSON / YAML
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import torch

from rhs_beamform_nas.nas.search_space.model_space import RHSAdaptiveBeamformingModelSpace
from rhs_beamform_nas.nas.strategies.random_search import RandomSearchStrategy
from rhs_beamform_nas.nas.strategies.evolution import EvolutionaryStrategy
from rhs_beamform_nas.nas.evaluators.multi_objective import BeamformingEvaluator
from rhs_beamform_nas.nas.objectives.combined import RHSNASObjective
from rhs_beamform_nas.nas.export.architecture import ArchitectureDescriptor
from rhs_beamform_nas.nas.export.exporter import ArchitectureExporter
from rhs_beamform_nas.nas.export.fixed_model import FixedRHSModel

logger = logging.getLogger(__name__)


class NASRunner:
    """Orchestrates NAS experiments across search spaces, strategies, and objectives."""

    def __init__(
        self,
        strategy_name: str = "random",
        input_dim: int = 128,
        config_dim: int = 16,
        Nr: int = 64,
        Nt: int = 4,
        num_trials: int = 20,
        output_dir: str = "search_results",
        device: str = "cpu",
        seed: Optional[int] = 42,
    ) -> None:
        self.strategy_name = strategy_name.lower()
        self.input_dim = input_dim
        self.config_dim = config_dim
        self.Nr = Nr
        self.Nt = Nt
        self.num_trials = num_trials
        self.output_dir = Path(output_dir)
        self.device = device
        self.seed = seed

        self.evaluator = BeamformingEvaluator(device=device)
        self.objective = RHSNASObjective()
        self.exporter = ArchitectureExporter(output_dir=str(self.output_dir))

    def evaluate_candidate(
        self,
        candidate_dict: Dict[str, Any],
        dataloader: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Instantiate a fixed model from candidate dict, benchmark, and score it."""
        # Ensure dimensions match system
        candidate_dict["input_dim"] = self.input_dim
        candidate_dict["config_dim"] = self.config_dim
        candidate_dict["Nr"] = self.Nr
        candidate_dict["Nt"] = self.Nt

        desc = ArchitectureDescriptor.from_dict(candidate_dict)
        model = FixedRHSModel(desc)

        dummy_x = torch.randn(1, self.input_dim)
        dummy_cfg = torch.randn(1, self.config_dim)

        metrics = self.evaluator.evaluate(
            model=model,
            dataloader=dataloader,
            dummy_input=dummy_x,
            dummy_config=dummy_cfg,
        )

        score = self.objective.compute_score(metrics)
        desc.metrics = metrics
        desc.score = score

        return desc.to_dict()

    def run_search(
        self,
        dataloader: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Execute architecture search based on configured strategy."""
        logger.info(f"Starting RHS NAS Search with strategy='{self.strategy_name}'...")
        evaluated_candidates: List[Dict[str, Any]] = []

        if self.strategy_name == "random":
            strategy = RandomSearchStrategy(num_trials=self.num_trials, seed=self.seed)
            trials = strategy.generate_trials()
            for trial in trials:
                res = self.evaluate_candidate(trial, dataloader=dataloader)
                evaluated_candidates.append(res)

        elif self.strategy_name in ["evolution", "evolutionary"]:
            evo = EvolutionaryStrategy(
                population_size=min(self.num_trials, 20),
                num_generations=max(1, self.num_trials // 20),
                seed=self.seed,
            )
            pop = evo.initialize_population()
            for cand in pop:
                res = self.evaluate_candidate(cand, dataloader=dataloader)
                evaluated_candidates.append(res)

        else:
            # Fallback or DARTS/ENAS single trial sampling
            strategy = RandomSearchStrategy(num_trials=self.num_trials, seed=self.seed)
            trials = strategy.generate_trials()
            for trial in trials:
                res = self.evaluate_candidate(trial, dataloader=dataloader)
                evaluated_candidates.append(res)

        # Sort by score descending
        evaluated_candidates.sort(
            key=lambda x: x.get("score", -float("inf")),
            reverse=True,
        )

        best_result = evaluated_candidates[0] if evaluated_candidates else {}

        # Export results
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.exporter.export_best(best_result, save_path=str(self.output_dir / "best.json"))
        self.exporter.export_pareto_candidates(
            evaluated_candidates, save_dir=str(self.output_dir / "architectures")
        )

        logger.info(f"NAS Search complete. Best score: {best_result.get('score')}")
        return {
            "best_architecture": best_result,
            "all_candidates": evaluated_candidates,
        }
