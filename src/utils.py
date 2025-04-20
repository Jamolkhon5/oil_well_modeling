"""
Модуль с вспомогательными функциями для проекта.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize
import logging

logger = logging.getLogger(__name__)


def ensure_directory_exists(directory):
    """
    Проверяет существование директории и создает ее при необходимости.

    Args:
        directory (str): Путь к директории
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Создана директория: {directory}")


def corey_water_function(Sw, Swo, Swk, krwk, nw):
    """
    Функция Кори для определения относительной проницаемости по воде.

    Args:
        Sw (float): Водонасыщенность
        Swo (float): Остаточная водонасыщенность
        Swk (float): Водонасыщенность при остаточной нефтенасыщенности
        krwk (float): Конечное значение относительной водопроницаемости
        nw (float): Показатель степени

    Returns:
        float: Относительная проницаемость по воде
    """
    if Sw <= Swo:
        return 0.0
    elif Sw >= Swk:
        return krwk
    else:
        normalized_sw = (Sw - Swo) / (Swk - Swo)
        return krwk * (normalized_sw ** nw)


def corey_oil_function(Sw, Swo, Swk, krok, no):
    """
    Функция Кори для определения относительной проницаемости по нефти.

    Args:
        Sw (float): Водонасыщенность
        Swo (float): Остаточная водонасыщенность
        Swk (float): Водонасыщенность при остаточной нефтенасыщенности
        krok (float): Конечное значение относительной нефтепроницаемости
        no (float): Показатель степени

    Returns:
        float: Относительная проницаемость по нефти
    """
    if Sw <= Swo:
        return krok
    elif Sw >= Swk:
        return 0.0
    else:
        normalized_sw = (Sw - Swo) / (Swk - Swo)
        return krok * ((1 - normalized_sw) ** no)


def calculate_fracture_half_length(volume, coeff_a, coeff_b):
    """
    Расчет полудлины трещины на основе объема закачки.

    Args:
        volume (float): Объем закачки воды, м³
        coeff_a (float): Коэффициент a
        coeff_b (float): Коэффициент b

    Returns:
        float: Полудлина трещины, м
    """
    # Формула из документа (раздел 7)
    return coeff_a * (volume ** coeff_b)


def calculate_skin_factor_after_fracking(days, initial_skin, max_skin, growth_rate):
    """
    Расчет динамики изменения скин-фактора после ГРП.

    Args:
        days (float): Время после проведения ГРП, сут
        initial_skin (float): Начальный скин-фактор сразу после ГРП
        max_skin (float): Максимальное значение скин-фактора
        growth_rate (float): Скорость роста скин-фактора

    Returns:
        float: Скин-фактор на заданное время
    """
    # Формула из документа (раздел 5)
    return initial_skin + (max_skin - initial_skin) * (1 - np.exp(-growth_rate * days))


def calculate_filter_reduction_coefficient(days, initial_coeff, min_coeff, reduction_rate):
    """
    Расчет коэффициента уменьшения работающей части фильтра.

    Args:
        days (float): Время после запуска скважины, сут
        initial_coeff (float): Начальный коэффициент
        min_coeff (float): Минимальный коэффициент
        reduction_rate (float): Скорость уменьшения

    Returns:
        float: Коэффициент уменьшения работающей части фильтра
    """
    # Формула из документа (раздел 6)
    return min_coeff + (initial_coeff - min_coeff) * np.exp(-reduction_rate * days)


def calculate_pressure_recovery_time(permeability, porosity, viscosity, skin_factor, well_radius):
    """
    Расчет времени восстановления давления в остановленных скважинах.

    Args:
        permeability (float): Проницаемость, мД
        porosity (float): Пористость, д.ед.
        viscosity (float): Вязкость флюида, сПз
        skin_factor (float): Скин-фактор
        well_radius (float): Радиус скважины, м

    Returns:
        float: Время восстановления давления, сут
    """
    # Формула из документа (раздел 4)
    # Константы из документа
    c1 = 0.000295  # Коэффициент из документа
    c2 = 0.5  # Показатель степени из документа

    # Коэффициент пьезопроводности
    piezoconductivity = permeability / (porosity * viscosity)

    # Расчет времени восстановления
    return c1 * ((skin_factor + 1) / piezoconductivity) ** c2


def fit_relative_permeability_curves(sw_data, krw_data, kro_data, initial_params):
    """
    Подбор параметров функций Кори для относительных проницаемостей.

    Args:
        sw_data (array): Массив значений водонасыщенности
        krw_data (array): Массив значений относительной проницаемости по воде
        kro_data (array): Массив значений относительной проницаемости по нефти
        initial_params (dict): Начальные значения параметров

    Returns:
        dict: Оптимальные значения параметров
    """


    # Функция ошибки, которую нужно минимизировать
    def error_function(params):
        Swo, Swk, krwk, krok, nw, no = params

        # Расчет проницаемостей по моделям
        krw_model = np.array([corey_water_function(sw, Swo, Swk, krwk, nw) for sw in sw_data])
        kro_model = np.array([corey_oil_function(sw, Swo, Swk, krok, no) for sw in sw_data])

        # Среднеквадратичная ошибка
        krw_error = np.sum((krw_model - krw_data) ** 2)
        kro_error = np.sum((kro_model - kro_data) ** 2)

        return krw_error + kro_error

    # Начальные значения параметров
    initial_values = [
        initial_params['Swo'],
        initial_params['Swk'],
        initial_params['krwk'],
        initial_params['krok'],
        initial_params['krw_min'],
        initial_params['kro_min']
    ]

    # Диапазоны параметров (bounds)
    bounds = [
        (initial_params['Swo'] * 0.5, initial_params['Swo'] * 1.5),
        (initial_params['Swk'] * 0.9, initial_params['Swk'] * 1.1),
        (initial_params['krwk'] * 0.5, initial_params['krwk'] * 1.5),
        (initial_params['krok'] * 0.9, initial_params['krok'] * 1.1),
        (initial_params['krw_min'] * 0.5, initial_params['krw_max'] * 1.5),
        (initial_params['kro_min'] * 0.5, initial_params['kro_max'] * 1.5),
    ]

    # Минимизация функции ошибки
    result = minimize(error_function, initial_values, bounds=bounds, method='L-BFGS-B')

    # Преобразование результата обратно в словарь
    optimized_params = {
        'Swo': result.x[0],
        'Swk': result.x[1],
        'krwk': result.x[2],
        'krok': result.x[3],
        'nw': result.x[4],
        'no': result.x[5]
    }

    return optimized_params


def plot_well_distribution(status_counts, title, output_path=None):
    """
    Построение круговой диаграммы распределения скважин.

    Args:
        status_counts (pd.Series): Серия с количеством скважин по категориям
        title (str): Заголовок диаграммы
        output_path (str, optional): Путь для сохранения графика

    Returns:
        plt.Figure: Объект фигуры matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 8))

    # Задаем цвета для разных категорий
    colors = plt.cm.Set3(np.linspace(0, 1, len(status_counts)))

    wedges, texts, autotexts = ax.pie(
        status_counts,
        labels=status_counts.index,
        autopct='%1.1f%%',
        textprops={'fontsize': 10},
        colors=colors,
        shadow=True,
        startangle=90
    )

    # Улучшение читаемости
    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title(title)

    # Добавление легенды
    ax.legend(
        wedges,
        [f"{label} ({count})" for label, count in zip(status_counts.index, status_counts)],
        title="Категории",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1)
    )

    plt.tight_layout()

    # Сохранение графика, если указан путь
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"График сохранен в {output_path}")

    return fig


def plot_relative_permeability_curves(Swo, Swk, krwk, krok, nw, no, output_path=None):
    """
    Построение графиков относительных проницаемостей.

    Args:
        Swo (float): Остаточная водонасыщенность
        Swk (float): Водонасыщенность при остаточной нефтенасыщенности
        krwk (float): Конечное значение относительной водопроницаемости
        krok (float): Конечное значение относительной нефтепроницаемости
        nw (float): Показатель степени для воды
        no (float): Показатель степени для нефти
        output_path (str, optional): Путь для сохранения графика

    Returns:
        plt.Figure: Объект фигуры matplotlib
    """
    # Создание массива значений водонасыщенности
    sw_values = np.linspace(0.0, 1.0, 100)

    # Расчет проницаемостей
    krw_values = np.array([corey_water_function(sw, Swo, Swk, krwk, nw) for sw in sw_values])
    kro_values = np.array([corey_oil_function(sw, Swo, Swk, krok, no) for sw in sw_values])

    # Построение графика
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(sw_values, krw_values, 'b-', linewidth=2, label='Вода (krw)')
    ax.plot(sw_values, kro_values, 'r-', linewidth=2, label='Нефть (kro)')

    # Добавление вертикальных линий для Swo и Swk
    ax.axvline(x=Swo, color='gray', linestyle='--', label=f'Swo = {Swo:.3f}')
    ax.axvline(x=Swk, color='gray', linestyle='-.', label=f'Swk = {Swk:.3f}')

    # Настройка графика
    ax.set_xlabel('Водонасыщенность (Sw), д.ед.')
    ax.set_ylabel('Относительная проницаемость, д.ед.')
    ax.set_title('Относительные фазовые проницаемости')
    ax.legend()
    ax.grid(True)

    # Сохранение графика, если указан путь
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"График сохранен в {output_path}")

    return fig