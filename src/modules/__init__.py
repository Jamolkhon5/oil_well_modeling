"""
Пакет с модулями расчетной схемы Пушкиной Т.В.
"""

# ДОБАВЛЯЕМ ИМПОРТ МОДУЛЕЙ ОПРЕДЕЛЕНИЯ СТАТУСОВ СКВАЖИН И СОСТОЯНИЯ ПРОЦЕССОВ
from .well_status import WellStatusModel
from .well_process_state import WellProcessStateModel
from .phase_permeability import PhasePermeabilityModel
from .regression_model import RegressionModel
from .pressure_calculation import PressureCalculationModel
from .pressure_recovery import PressureRecoveryModel
from .skin_curve import SkinCurveModel
from .filter_reduction import FilterReductionModel
from .fracture_length import FractureLengthModel
from .production_wells import ProductionWellsModel

__all__ = [
    'WellStatusModel',      # ДОБАВЛЕНО В СПИСОК ЭКСПОРТА
    'WellProcessStateModel',  # ДОБАВЛЕНО В СПИСОК ЭКСПОРТА
    'PhasePermeabilityModel',
    'RegressionModel',
    'PressureCalculationModel',
    'PressureRecoveryModel',
    'SkinCurveModel',
    'FilterReductionModel',
    'FractureLengthModel',
    'ProductionWellsModel',
]