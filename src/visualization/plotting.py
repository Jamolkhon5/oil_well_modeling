"""
Модуль для визуализации результатов расчетов.

Содержит функции для построения различных графиков
и создания сводных отчетов.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages
import logging

logger = logging.getLogger(__name__)


def plot_permeability_curves(model, output_path=None):
    """
    Построение графиков относительных фазовых проницаемостей.

    Args:
        model (PhasePermeabilityModel): Модель относительных фазовых проницаемостей
        output_path (str, optional): Путь для сохранения графика

    Returns:
        plt.Figure: Объект фигуры matplotlib
    """
    return model.plot_permeability_curves(output_path)


def plot_pressure_changes(model, output_path=None):
    """
    Построение графиков изменения пластового давления.

    Args:
        model (PressureCalculationModel): Модель расчета пластового давления
        output_path (str, optional): Путь для сохранения графика

    Returns:
        plt.Figure: Объект фигуры matplotlib
    """
    return model.plot_pressure_changes(output_path)


def plot_recovery_times(model, output_path=None):
    """
    Построение графиков времени восстановления давления.

    Args:
        model (PressureRecoveryModel): Модель расчета времени восстановления давления
        output_path (str, optional): Путь для сохранения графика

    Returns:
        plt.Figure: Объект фигуры matplotlib
    """
    return model.plot_recovery_times(output_path)


def plot_skin_curve(model, output_path=None):
    """
    Построение графика изменения скин-фактора.

    Args:
        model (SkinCurveModel): Модель расчета скин-фактора
        output_path (str, optional): Путь для сохранения графика

    Returns:
        plt.Figure: Объект фигуры matplotlib
    """
    return model.plot_skin_curve(output_path)


def plot_filter_reduction_curve(model, output_path=None):
    """
    Построение графика уменьшения работающей части фильтра.

    Args:
        model (FilterReductionModel): Модель расчета коэффициента уменьшения фильтра
        output_path (str, optional): Путь для сохранения графика

    Returns:
        plt.Figure: Объект фигуры matplotlib
    """
    return model.plot_coefficient_curve(output_path)


def plot_fracture_length_curve(model, output_path=None, log_scale=False):
    """
    Построение графика зависимости полудлины трещины от объема закачки.

    Args:
        model (FractureLengthModel): Модель расчета полудлин трещин
        output_path (str, optional): Путь для сохранения графика
        log_scale (bool): Использовать логарифмический масштаб

    Returns:
        plt.Figure: Объект фигуры matplotlib
    """
    if log_scale:
        return model.plot_log_log_curve(output_path)
    else:
        return model.plot_fracture_length_curve(output_path)


def plot_production_profiles(model, output_path=None):
    """
    Построение графиков профилей добычи.

    Args:
        model (ProductionWellsModel): Модель расчета добывающих скважин
        output_path (str, optional): Путь для сохранения графика

    Returns:
        plt.Figure: Объект фигуры matplotlib
    """
    return model.plot_production_profiles(output_path)


def plot_cumulative_production(model, output_path=None):
    """
    Построение графика накопленной добычи.

    Args:
        model (ProductionWellsModel): Модель расчета добывающих скважин
        output_path (str, optional): Путь для сохранения графика

    Returns:
        plt.Figure: Объект фигуры matplotlib
    """
    return model.plot_cumulative_production(output_path)


def create_summary_dashboard(models, output_path):
    """
    Создание сводной панели с графиками для всех моделей.

    Args:
        models (dict): Словарь с объектами моделей
        output_path (str): Путь для сохранения графика

    Returns:
        plt.Figure: Объект фигуры matplotlib
    """
    # Создаем фигуру с сеткой графиков
    fig = plt.figure(figsize=(20, 24))
    gs = gridspec.GridSpec(4, 2, figure=fig)

    # Графики для каждой модели
    if 'phase_permeability' in models:
        ax1 = fig.add_subplot(gs[0, 0])
        models['phase_permeability'].plot_permeability_curves()
        plt.sca(ax1)
        plt.title("Относительные фазовые проницаемости")

    if 'pressure_calculation' in models:
        ax2 = fig.add_subplot(gs[0, 1])
        models['pressure_calculation'].plot_pressure_changes()
        plt.sca(ax2)
        plt.title("Пластовое давление")

    if 'pressure_recovery' in models:
        ax3 = fig.add_subplot(gs[1, 0])
        models['pressure_recovery'].plot_recovery_times()
        plt.sca(ax3)
        plt.title("Время восстановления давления")

    if 'skin_curve' in models:
        ax4 = fig.add_subplot(gs[1, 1])
        models['skin_curve'].plot_skin_curve()
        plt.sca(ax4)
        plt.title("Изменение скин-фактора")

    if 'filter_reduction' in models:
        ax5 = fig.add_subplot(gs[2, 0])
        models['filter_reduction'].plot_coefficient_curve()
        plt.sca(ax5)
        plt.title("Уменьшение работающей части фильтра")

    if 'fracture_length' in models:
        ax6 = fig.add_subplot(gs[2, 1])
        models['fracture_length'].plot_fracture_length_curve()
        plt.sca(ax6)
        plt.title("Полудлина трещины")

    if 'production_wells' in models:
        ax7 = fig.add_subplot(gs[3, 0])
        models['production_wells'].plot_production_profiles()
        plt.sca(ax7)
        plt.title("Профили добычи")

        ax8 = fig.add_subplot(gs[3, 1])
        models['production_wells'].plot_cumulative_production()
        plt.sca(ax8)
        plt.title("Накопленная добыча")

    plt.tight_layout()

    # Сохранение графика
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Сводная панель сохранена в {output_path}")

    return fig


def create_summary_report(models, output_path):
    """
    Создание PDF-отчета со всеми графиками и таблицами результатов.

    Args:
        models (dict): Словарь с объектами моделей
        output_path (str): Путь для сохранения отчета
    """
    try:
        with PdfPages(output_path) as pdf:
            # Титульная страница
            fig = plt.figure(figsize=(8.5, 11))
            fig.suptitle('Отчет о результатах расчета параметров нефтяных скважин', fontsize=16)
            plt.figtext(0.5, 0.5, 'Расчетная схема Пушкиной Т.В.',
                        fontsize=14, ha='center')
            plt.figtext(0.5, 0.4, f'Дата создания: {pd.Timestamp.now().strftime("%Y-%m-%d")}',
                        fontsize=12, ha='center')
            plt.axis('off')
            pdf.savefig(fig)
            plt.close(fig)

            # График для каждой модели
            for model_name, model in models.items():
                if model_name == 'phase_permeability':
                    fig = model.plot_permeability_curves()
                    pdf.savefig(fig)
                    plt.close(fig)

                elif model_name == 'pressure_calculation':
                    fig = model.plot_pressure_changes()
                    pdf.savefig(fig)
                    plt.close(fig)

                elif model_name == 'pressure_recovery':
                    fig = model.plot_recovery_times()
                    pdf.savefig(fig)
                    plt.close(fig)

                    fig = model.plot_parameters_influence()
                    pdf.savefig(fig)
                    plt.close(fig)

                elif model_name == 'skin_curve':
                    fig = model.plot_skin_curve()
                    pdf.savefig(fig)
                    plt.close(fig)

                elif model_name == 'filter_reduction':
                    fig = model.plot_coefficient_curve()
                    pdf.savefig(fig)
                    plt.close(fig)

                elif model_name == 'fracture_length':
                    fig = model.plot_fracture_length_curve()
                    pdf.savefig(fig)
                    plt.close(fig)

                    fig = model.plot_log_log_curve()
                    pdf.savefig(fig)
                    plt.close(fig)

                elif model_name == 'production_wells':
                    fig = model.plot_production_profiles()
                    pdf.savefig(fig)
                    plt.close(fig)

                    fig = model.plot_cumulative_production()
                    pdf.savefig(fig)
                    plt.close(fig)

            # Текстовый отчет для каждой модели
            fig = plt.figure(figsize=(8.5, 11))
            plt.axis('off')

            y_pos = 0.95
            for model_name, model in models.items():
                plt.figtext(0.1, y_pos, f"Отчет по модулю: {model_name}", fontsize=12, weight='bold')
                y_pos -= 0.05

                report_text = model.report()
                report_lines = report_text.split('\n')

                for line in report_lines:
                    plt.figtext(0.1, y_pos, line, fontsize=10)
                    y_pos -= 0.02

                    if y_pos < 0.1:  # Если достигли конца страницы
                        pdf.savefig(fig)
                        plt.close(fig)

                        # Создаем новую страницу
                        fig = plt.figure(figsize=(8.5, 11))
                        plt.axis('off')
                        y_pos = 0.95

                # Добавляем разделитель между отчетами
                plt.figtext(0.1, y_pos, "---", fontsize=10)
                y_pos -= 0.05

                if y_pos < 0.2:  # Если мало места до конца страницы
                    pdf.savefig(fig)
                    plt.close(fig)

                    # Создаем новую страницу
                    fig = plt.figure(figsize=(8.5, 11))
                    plt.axis('off')
                    y_pos = 0.95

            # Сохраняем последнюю страницу, если на ней есть контент
            pdf.savefig(fig)
            plt.close(fig)

        logger.info(f"Отчет успешно создан и сохранен в {output_path}")

    except Exception as e:
        logger.error(f"Ошибка при создании отчета: {str(e)}")