"""
Пакет с модулями расчетной схемы Пушкиной Т.В.
"""

from .phase_permeability import PhasePermeabilityModel
from .regression_model import RegressionModel
from .pressure_calculation import PressureCalculationModel
from .pressure_recovery import PressureRecoveryModel
from .skin_curve import SkinCurveModel
from .filter_reduction import FilterReductionModel
from .fracture_length import FractureLengthModel
from .production_wells import ProductionWellsModel

__all__ = [
    'PhasePermeabilityModel',
    'RegressionModel',
    'PressureCalculationModel',
    'PressureRecoveryModel',
    'SkinCurveModel',
    'FilterReductionModel',
    'FractureLengthModel',
    'ProductionWellsModel',
]