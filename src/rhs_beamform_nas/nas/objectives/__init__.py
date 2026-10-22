"""Objectives package — individual NAS objective functions."""

from rhs_beamform_nas.nas.objectives.accuracy import AccuracyObjective
from rhs_beamform_nas.nas.objectives.rate import RateObjective
from rhs_beamform_nas.nas.objectives.latency import LatencyObjective
from rhs_beamform_nas.nas.objectives.model_size import ModelSizeObjective
from rhs_beamform_nas.nas.objectives.combined import RHSNASObjective

__all__ = [
    "AccuracyObjective",
    "RateObjective",
    "LatencyObjective",
    "ModelSizeObjective",
    "RHSNASObjective",
]
