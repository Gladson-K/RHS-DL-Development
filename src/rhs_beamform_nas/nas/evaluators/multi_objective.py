class BeamformingEvaluator:

    def evaluate(self, model, dataloader):

        rate = evaluate_rate(
            model,
            dataloader
        )

        latency = benchmark_latency(
            model,
            dataloader
        )

        parameters = count_parameters(
            model
        )

        constraint_violation = evaluate_constraints(
            model,
            dataloader
        )

        return {
            "normalized_rate": rate,
            "latency_ms": latency,
            "parameters": parameters,
            "constraint_violation": constraint_violation,
        }