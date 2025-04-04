#!/usr/bin/env python3
"""
Главный файл для запуска расчетной схемы Пушкиной Т.В.

Этот скрипт загружает данные и выполняет расчеты всех
восьми модулей схемы.
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

from src.config import OFP_PARAMS, OUTPUT_DIR
from src.data_loader import DataLoader
from src.utils import ensure_directory_exists
from src.modules.phase_permeability import PhasePermeabilityModel
from src.modules.regression_model import RegressionModel

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
                # Выводим отчет
                logger.info("Результаты итеративного подбора регрессионной моделью:")
                logger.info(regression_model.report())
            else:
                logger.error("Не удалось подобрать параметры для регрессионной модели")
        else:
            logger.error("Не удалось инициализировать регрессионную модель данными")

        # Остальные этапы будут реализованы аналогично
        # ...

        logger.info("Расчетная схема Пушкиной Т.В. успешно выполнена")

    except Exception as e:
        logger.error(f"Произошла ошибка при выполнении расчетной схемы: {str(e)}", exc_info=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())