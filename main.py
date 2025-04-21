#!/usr/bin/env python3
"""
Главный файл для запуска расчетной схемы Пушкиной Т.В.

Этот скрипт загружает данные и выполняет расчеты всех
модулей схемы, включая определение статусов скважин и состояния процессов.
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Добавляем директорию проекта в sys.path для корректного импорта модулей
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from scripts.export_results import main as export_results

from src.config import OFP_PARAMS, OUTPUT_DIR
from src.data_loader import DataLoader
from src.utils import ensure_directory_exists

# Импорт модулей расчетной схемы
from src.modules.phase_permeability import PhasePermeabilityModel
from src.modules.regression_model import RegressionModel
from src.modules.pressure_calculation import PressureCalculationModel
from src.modules.pressure_recovery import PressureRecoveryModel
from src.modules.skin_curve import SkinCurveModel
from src.modules.filter_reduction import FilterReductionModel
from src.modules.fracture_length import FractureLengthModel
from src.modules.production_wells import ProductionWellsModel

# ДОБАВЛЯЕМ ИМПОРТ МОДУЛЕЙ ОПРЕДЕЛЕНИЯ СТАТУСОВ СКВАЖИН И СОСТОЯНИЯ ПРОЦЕССОВ
from src.modules.well_status import WellStatusModel
from src.modules.well_process_state import WellProcessStateModel

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("oil_well_model.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Основная функция для запуска расчетной схемы.
    """
    # Создаем директорию для выходных данных, если она не существует
    ensure_directory_exists(OUTPUT_DIR)

    logger.info("Запуск расчетной схемы Пушкиной Т.В.")

    # Создаем загрузчик данных
    data_loader = DataLoader()

    try:
        # Загружаем все данные
        data_loader.load_all_data()

        # ДОБАВЛЯЕМ ИСПОЛЬЗОВАНИЕ get_column_statistics
        logger.info("\n=== Анализ статистики загруженных данных ===")

        # Анализируем статистику по ключевым столбцам
        datasets_to_analyze = {
            'pzab': ['Давление', 'Pressure'],
            'ppl': ['Давление', 'Pressure'],
            'arcgis': ['Скважина', 'Well', 'Статус', 'Status']
        }

        for dataset_name, columns in datasets_to_analyze.items():
            logger.info(f"\nСтатистика для {dataset_name}:")
            for column in columns:
                stats = data_loader.get_column_statistics(dataset_name, column)
                if stats:
                    logger.info(f"Столбец '{column}':")
                    if 'mean' in stats:  # Числовой столбец
                        logger.info(f"  - Среднее: {stats.get('mean', 'N/A'):.2f}")
                        logger.info(f"  - Минимум: {stats.get('min', 'N/A'):.2f}")
                        logger.info(f"  - Максимум: {stats.get('max', 'N/A'):.2f}")
                    else:  # Нечисловой столбец
                        logger.info(f"  - Уникальных значений: {stats.get('unique_values', 'N/A')}")
                        logger.info(f"  - Наиболее частое: {stats.get('top_value', 'N/A')}")

        # Выводим примеры загруженных данных
        logger.info("Примеры загруженных данных:")
        for dataset_name in ['pzab', 'ppl', 'arcgis', 'gdi_vnr', 'gdi_reint', 'gtp', 'nnt_ngt']:
            sample = data_loader.view_data_sample(dataset_name, n_rows=2)
            if sample is not None:
                if isinstance(sample, dict):
                    for sheet_name, sheet_data in sample.items():
                        logger.info(f"Датасет {dataset_name}, лист {sheet_name}:")
                        logger.info(f"\n{sheet_data}")
                else:
                    logger.info(f"Датасет {dataset_name}:")
                    logger.info(f"\n{sample}")

        # Словарь для хранения всех моделей
        models = {}

        # ДОБАВЛЯЕМ ЭТАП ОПРЕДЕЛЕНИЯ СТАТУСОВ СКВАЖИН
        logger.info("\n=== Этап 0.1: Определение статусов скважин ===")

        # Создаем модель определения статусов скважин
        well_status_model = WellStatusModel()

        # Инициализируем модель данными
        well_status_data = {
            'arcgis_data': data_loader.arcgis_data,
            'pzab_data': data_loader.pzab_data,
            'ppl_data': data_loader.ppl_data
        }

        if well_status_model.initialize_from_data(well_status_data):
            # Определяем статусы скважин
            if well_status_model.determine_well_statuses():
                # Строим и сохраняем графики распределения статусов
                well_status_model.plot_status_distribution(
                    os.path.join(OUTPUT_DIR, "well_status_distribution.png")
                )

                well_status_model.plot_status_category_distribution(
                    os.path.join(OUTPUT_DIR, "well_status_category_distribution.png")
                )

                # Выводим отчет
                logger.info("Результаты определения статусов скважин:")
                logger.info(well_status_model.report())

                # Добавляем модель в словарь
                models['well_status'] = well_status_model

                # Передаем результаты статусов скважин в data_loader для использования в последующих модулях
                if hasattr(data_loader, 'arcgis_data') and data_loader.arcgis_data is not None:
                    well_status_results = well_status_model.get_results()

                    # Объединяем данные о статусах с существующими данными arcgis
                    if isinstance(data_loader.arcgis_data, pd.DataFrame):
                        # Если данные arcgis - датафрейм, добавляем информацию о статусах
                        data_loader.arcgis_data = pd.merge(
                            data_loader.arcgis_data,
                            well_status_results[['Well', 'Status_Code', 'Status_Description', 'Status_Category']],
                            left_on='Well',  # Нужно будет найти правильное имя колонки
                            right_on='Well',
                            how='left'
                        )

                    logger.info("Статусы скважин интегрированы в общие данные")
            else:
                logger.error("Не удалось определить статусы скважин")
        else:
            logger.error("Не удалось инициализировать модель определения статусов скважин данными")

        # ДОБАВЛЯЕМ ЭТАП ОПРЕДЕЛЕНИЯ СОСТОЯНИЯ ПРОЦЕССОВ СКВАЖИН
        logger.info("\n=== Этап 0.2: Определение состояния процессов скважин ===")

        # Создаем модель определения состояния процессов скважин
        well_process_state_model = WellProcessStateModel()

        # Инициализируем модель данными
        well_process_state_data = {
            'arcgis_data': data_loader.arcgis_data,  # Теперь уже обогащенная данными о статусах
            'pzab_data': data_loader.pzab_data,
            'gtp_data': data_loader.gtp_data
        }

        if well_process_state_model.initialize_from_data(well_process_state_data):
            # Определяем состояние процессов скважин
            if well_process_state_model.determine_process_states():
                # Строим и сохраняем графики распределения состояний процессов
                well_process_state_model.plot_process_state_distribution(
                    os.path.join(OUTPUT_DIR, "well_process_state_distribution.png")
                )

                well_process_state_model.plot_days_in_state(
                    os.path.join(OUTPUT_DIR, "well_process_state_days.png")
                )

                # Выводим отчет
                logger.info("Результаты определения состояния процессов скважин:")
                logger.info(well_process_state_model.report())

                # Добавляем модель в словарь
                models['well_process_state'] = well_process_state_model

                # Передаем результаты состояния процессов в data_loader для использования в последующих модулях
                if hasattr(data_loader, 'arcgis_data') and data_loader.arcgis_data is not None:
                    well_process_state_results = well_process_state_model.get_results()

                    # Объединяем данные о состоянии процессов с существующими данными arcgis
                    if isinstance(data_loader.arcgis_data, pd.DataFrame):
                        # Если данные arcgis - датафрейм, добавляем информацию о состоянии процессов
                        data_loader.arcgis_data = pd.merge(
                            data_loader.arcgis_data,
                            well_process_state_results[['Well', 'Process_State_Code', 'Process_State_Description',
                                                        'Process_Details', 'Days_In_Current_State']],
                            on='Well',
                            how='left'
                        )

                    logger.info("Состояния процессов скважин интегрированы в общие данные")
            else:
                logger.error("Не удалось определить состояния процессов скважин")
        else:
            logger.error("Не удалось инициализировать модель определения состояния процессов скважин данными")

        # Этап 1: Подбор относительных фазовых проницаемостей
        logger.info("\n=== Этап 1: Подбор относительных фазовых проницаемостей ===")

        # Получаем данные для первого модуля
        phase_perm_data = data_loader.get_data_for_phase_permeability()

        # Создаем модель
        phase_perm_model = PhasePermeabilityModel(OFP_PARAMS)

        # Инициализируем модель данными
        if phase_perm_model.initialize_from_data(phase_perm_data):
            # Подбираем параметры модели
            if phase_perm_model.fit_model():
                # Строим и сохраняем графики
                phase_perm_model.plot_permeability_curves(
                    os.path.join(OUTPUT_DIR, "phase_permeability_curves.png")
                )

                # Выводим отчет
                logger.info("Результаты подбора относительных фазовых проницаемостей:")
                logger.info(phase_perm_model.report())

                # Добавляем модель в словарь
                models['phase_permeability'] = phase_perm_model
            else:
                logger.error("Не удалось подобрать параметры для модели фазовых проницаемостей")
        else:
            logger.error("Не удалось инициализировать модель фазовых проницаемостей данными")

        # Этап 2: Подбор итеративной регрессионной моделью
        logger.info("\n=== Этап 2: Подбор итеративной регрессионной моделью ===")

        # Получаем данные для второго модуля
        regression_data = data_loader.get_data_for_regression_model()

        # Создаем модель
        regression_model = RegressionModel()

        # Инициализируем модель данными
        if regression_model.initialize_from_data(regression_data):
            # Подбираем параметры модели
            if regression_model.fit_model():

                # Адаптируем модель на историю добычи
                logger.info("\n=== Адаптация модели на историю добычи ===")

                # Получаем данные для адаптации
                production_data = {
                    'production_wells': data_loader.get_data_for_production_wells()
                }

                # Параметры пласта (из данных или значения по умолчанию)
                reservoir_params = {
                    'permeability': 50.0,  # мД
                    'porosity': 0.2,  # д.ед.
                    'total_compressibility': 1e-5,  # 1/атм
                    'mu_w': 0.45,  # сПз
                    'mu_o': 1.3,  # сПз
                    'mu_g': 0.02  # сПз
                }

                # Выполняем адаптацию на историю добычи
                if regression_model.adapt_to_production_history(production_data, reservoir_params):
                    logger.info("Адаптация на историю добычи выполнена успешно")

                    # Строим графики относительных проницаемостей после адаптации
                    regression_model.plot_relative_permeability_curves(
                        os.path.join(OUTPUT_DIR, "relative_permeability_curves_adapted.png")
                    )
                else:
                    logger.warning("Не удалось адаптировать модель на историю добычи")

                # Выводим отчет
                logger.info("Результаты итеративного подбора регрессионной моделью:")
                logger.info(regression_model.report())

                # Добавляем модель в словарь
                models['regression_model'] = regression_model
            else:
                logger.error("Не удалось подобрать параметры для регрессионной модели")
        else:
            logger.error("Не удалось инициализировать регрессионную модель данными")

        # Этап 3: Граничные условия изменения при автоматическом расчете Рпл в нефтяных скважинах
        logger.info("\n=== Этап 3: Граничные условия изменения при автоматическом расчете Рпл ===")

        # Получаем данные для третьего модуля
        pressure_calc_data = data_loader.get_data_for_pressure_calculation()

        # Создаем модель
        pressure_calc_model = PressureCalculationModel()

        # Инициализируем модель данными
        if pressure_calc_model.initialize_from_data(pressure_calc_data):
            # Выполняем расчет пластовых давлений
            if pressure_calc_model.calculate_pressures():
                # Применяем граничные условия
                if pressure_calc_model.apply_boundary_conditions():
                    # Строим и сохраняем график
                    pressure_calc_model.plot_pressure_changes(
                        os.path.join(OUTPUT_DIR, "pressure_changes.png")
                    )

                    # Выводим отчет
                    logger.info("Результаты расчета пластовых давлений:")
                    logger.info(pressure_calc_model.report())

                    # Добавляем модель в словарь
                    models['pressure_calculation'] = pressure_calc_model
                else:
                    logger.error("Не удалось применить граничные условия")
            else:
                logger.error("Не удалось выполнить расчет пластовых давлений")
        else:
            logger.error("Не удалось инициализировать модель расчета пластовых давлений данными")

        # Этап 4: Подбор времени восстановления давления в остановленных скважинах
        logger.info("\n=== Этап 4: Подбор времени восстановления давления в остановленных скважинах ===")

        # Получаем данные для четвертого модуля
        pressure_recovery_data = data_loader.get_data_for_pressure_recovery()

        # Создаем модель
        pressure_recovery_model = PressureRecoveryModel()

        # Инициализируем модель данными
        if pressure_recovery_model.initialize_from_data(pressure_recovery_data):
            # Рассчитываем время восстановления давления
            if pressure_recovery_model.calculate_recovery_times():
                # Строим и сохраняем графики
                pressure_recovery_model.plot_recovery_times(
                    os.path.join(OUTPUT_DIR, "recovery_times.png")
                )
                pressure_recovery_model.plot_parameters_influence(
                    os.path.join(OUTPUT_DIR, "recovery_parameters_influence.png")
                )

                # Выводим отчет
                logger.info("Результаты расчета времени восстановления давления:")
                logger.info(pressure_recovery_model.report())

                # Добавляем модель в словарь
                models['pressure_recovery'] = pressure_recovery_model
            else:
                logger.error("Не удалось рассчитать время восстановления давления")
        else:
            logger.error("Не удалось инициализировать модель времени восстановления давления данными")

        # Этап 5: Подбор кривой увеличения SKIN для нефтяных скважин с течением времени после ГРП
        logger.info("\n=== Этап 5: Подбор кривой увеличения SKIN после ГРП ===")

        # Получаем данные для пятого модуля
        skin_curve_data = data_loader.get_data_for_skin_curve()

        # Создаем модель
        skin_curve_model = SkinCurveModel()

        # Инициализируем модель данными
        if skin_curve_model.initialize_from_data(skin_curve_data):
            # Подбираем параметры модели
            if skin_curve_model.fit_model():
                # Строим и сохраняем график
                skin_curve_model.plot_skin_curve(
                    os.path.join(OUTPUT_DIR, "skin_curve.png")
                )

                # Выводим отчет
                logger.info("Результаты подбора кривой увеличения SKIN:")
                logger.info(skin_curve_model.report())

                # Добавляем модель в словарь
                models['skin_curve'] = skin_curve_model
            else:
                logger.error("Не удалось подобрать параметры для кривой увеличения SKIN")
        else:
            logger.error("Не удалось инициализировать модель кривой увеличения SKIN данными")

        # Этап 6: Подбор к-та уменьшения работающей части фильтра горизонтальной нефтяной скважины
        logger.info("\n=== Этап 6: Подбор к-та уменьшения работающей части фильтра ===")

        # Получаем данные для шестого модуля
        filter_reduction_data = data_loader.get_data_for_filter_reduction()

        # Создаем модель
        filter_reduction_model = FilterReductionModel()

        # Инициализируем модель данными
        if filter_reduction_model.initialize_from_data(filter_reduction_data):
            # Подбираем параметры модели
            if filter_reduction_model.fit_model():
                # Строим и сохраняем график
                filter_reduction_model.plot_coefficient_curve(
                    os.path.join(OUTPUT_DIR, "filter_reduction_curve.png")
                )

                # Выводим отчет
                logger.info("Результаты подбора к-та уменьшения работающей части фильтра:")
                logger.info(filter_reduction_model.report())

                # Добавляем модель в словарь
                models['filter_reduction'] = filter_reduction_model
            else:
                logger.error("Не удалось подобрать параметры для к-та уменьшения работающей части фильтра")
        else:
            logger.error("Не удалось инициализировать модель к-та уменьшения работающей части фильтра данными")

        # Этап 7: Подбор коэффициентов для расчета полудлин трещин при закачке разных объемов воды
        logger.info("\n=== Этап 7: Подбор коэффициентов для расчета полудлин трещин ===")

        # Получаем данные для седьмого модуля
        fracture_length_data = data_loader.get_data_for_fracture_length()

        # Создаем модель
        fracture_length_model = FractureLengthModel()

        # Инициализируем модель данными
        if fracture_length_model.initialize_from_data(fracture_length_data):
            # Подбираем параметры модели
            if fracture_length_model.fit_model():
                # Строим и сохраняем графики
                fracture_length_model.plot_fracture_length_curve(
                    os.path.join(OUTPUT_DIR, "fracture_length_curve.png")
                )
                fracture_length_model.plot_log_log_curve(
                    os.path.join(OUTPUT_DIR, "fracture_length_loglog.png")
                )

                # Выводим отчет
                logger.info("Результаты подбора коэффициентов для расчета полудлин трещин:")
                logger.info(fracture_length_model.report())

                # Добавляем модель в словарь
                models['fracture_length'] = fracture_length_model
            else:
                logger.error("Не удалось подобрать коэффициенты для расчета полудлин трещин")
        else:
            logger.error("Не удалось инициализировать модель расчета полудлин трещин данными")

        # Этап 8: РАСЧЁТ ДОБЫВАЮЩИХ СКВАЖИН
        logger.info("\n=== Этап 8: РАСЧЁТ ДОБЫВАЮЩИХ СКВАЖИН ===")

        # Получаем данные для восьмого модуля
        production_wells_data = data_loader.get_data_for_production_wells()

        # Создаем модель добывающих скважин, передавая ей предыдущие модели
        production_wells_model = ProductionWellsModel(models)

        # Инициализируем модель данными
        if production_wells_model.initialize_from_data(production_wells_data):
            # Рассчитываем параметры добывающих скважин
            if production_wells_model.calculate_well_parameters():
                # Выполняем прогнозирование добычи на 365 дней
                if production_wells_model.forecast_production(forecast_period=365):
                    # Строим и сохраняем графики
                    production_wells_model.plot_production_profiles(
                        os.path.join(OUTPUT_DIR, "production_profiles.png")
                    )
                    production_wells_model.plot_cumulative_production(
                        os.path.join(OUTPUT_DIR, "cumulative_production.png")
                    )

                    # Выводим отчет
                    logger.info("Результаты расчета добывающих скважин:")
                    logger.info(production_wells_model.report())

                    # Добавляем модель в словарь
                    models['production_wells'] = production_wells_model
                else:
                    logger.error("Не удалось выполнить прогнозирование добычи")
            else:
                logger.error("Не удалось рассчитать параметры добывающих скважин")
        else:
            logger.error("Не удалось инициализировать модель добывающих скважин данными")

        # Экспорт результатов всех моделей
        logger.info("\n=== Экспорт результатов всех модулей ===")
        try:
            # Вызываем функцию экспорта результатов из модуля export_results
            export_results(models)
            logger.info("Экспорт результатов успешно выполнен")
        except Exception as export_error:
            logger.error(f"Ошибка при экспорте результатов: {str(export_error)}")

        logger.info("Расчетная схема Пушкиной Т.В. успешно выполнена")

    except Exception as e:
        logger.error(f"Произошла ошибка при выполнении расчетной схемы: {str(e)}", exc_info=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())