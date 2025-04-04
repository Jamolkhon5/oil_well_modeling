"""
Модули для визуализации результатов расчетов.
"""

from .plotting import (
    plot_permeability_curves,
    plot_pressure_changes,
    plot_recovery_times,
    plot_skin_curve,
    plot_filter_reduction_curve,
    plot_fracture_length_curve,
    plot_production_profiles,
    plot_cumulative_production,
    create_summary_report
)

__all__ = [
    'plot_permeability_curves',
    'plot_pressure_changes',
    'plot_recovery_times',
    'plot_skin_curve',
    'plot_filter_reduction_curve',
    'plot_fracture_length_curve',
    'plot_production_profiles',
    'plot_cumulative_production',
    'create_summary_report'
]