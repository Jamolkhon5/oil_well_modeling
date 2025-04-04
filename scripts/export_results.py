#!/usr/bin/env python3
"""
Скрипт для экспорта результатов расчетов в различные форматы.

Экспортирует результаты моделирования в форматы CSV, Excel, JSON
и создает сводные отчеты.
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Добавляем родительскую директорию в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import OUTPUT_DIR
from src.visualization.plotting import create_summary_report

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("export_results.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def export_to_csv(data, output_file):
    """
    Экспорт данных в формат CSV.

    Args:
        data (pd.DataFrame): Данные для экспорта
        output_file (str): Путь к выходному файлу
    """
    try:
        data.to_csv(output_file, index=False, sep=';', encoding='utf-8')
        logger.info(f"Данные успешно экспортированы в CSV: {output_file}")
    except Exception as e:
        logger.error(f"Ошибка при экспорте в CSV: {str(e)}")


def export_to_excel(data, output_file, sheet_name="Results"):
    """
    Экспорт данных в формат Excel.

    Args:
        data (pd.DataFrame или dict): Данные для экспорта
        output_file (str): Путь к выходному файлу
        sheet_name (str): Имя листа (если data - DataFrame)
    """
    try:
        if isinstance(data, pd.DataFrame):
            data.to_excel(output_file, sheet_name=sheet_name, index=False)
        elif isinstance(data, dict):
            with pd.ExcelWriter(output_file) as writer:
                for name, df in data.items():
                    if isinstance(df, pd.DataFrame):
                        df.to_excel(writer, sheet_name=name, index=False)
                    else:
                        pd.DataFrame([df] if isinstance(df, dict) else df).to_excel(
                            writer, sheet_name=name, index=False
                        )
        else:
            pd.DataFrame([data] if isinstance(data, dict) else data).to_excel(
                output_file, sheet_name=sheet_name, index=False
            )

        logger.info(f"Данные успешно экспортированы в Excel: {output_file}")
    except Exception as e:
        logger.error(f"Ошибка при экспорте в Excel: {str(e)}")


def export_to_json(data, output_file):
    """
    Экспорт данных в формат JSON.

    Args:
        data (dict, DataFrame, или list): Данные для экспорта
        output_file (str): Путь к выходному файлу
    """
    try:
        if isinstance(data, pd.DataFrame):
            data_json = data.to_dict(orient='records')
        elif isinstance(data, dict):
            data_json = {
                key: (value.to_dict(orient='records') if isinstance(value, pd.DataFrame) else value)
                for key, value in data.items()
            }
        else:
            data_json = data

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(data_json, f, ensure_ascii=False, indent=4)

        logger.info(f"Данные успешно экспортированы в JSON: {output_file}")
    except Exception as e:
        logger.error(f"Ошибка при экспорте в JSON: {str(e)}")


def export_model_parameters(models, output_file):
    """
    Экспорт параметров всех моделей в Excel-файл.

    Args:
        models (dict): Словарь с объектами моделей
        output_file (str): Путь к выходному файлу
    """
    try:
        model_params = {}

        for model_name, model in models.items():
            if hasattr(model, 'get_parameters'):
                model_params[model_name] = model.get_parameters()

        # Создаем DataFrame для каждой модели
        sheets_data = {}

        for model_name, params in model_params.items():
            if isinstance(params, dict):
                # Преобразуем словарь параметров в DataFrame
                params_df = pd.DataFrame(
                    {'Parameter': list(params.keys()),
                     'Value': list(params.values())}
                )
                sheets_data[model_name] = params_df

        # Экспортируем в Excel
        with pd.ExcelWriter(output_file) as writer:
            for sheet_name, df in sheets_data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)

        logger.info(f"Параметры моделей успешно экспортированы в: {output_file}")
    except Exception as e:
        logger.error(f"Ошибка при экспорте параметров моделей: {str(e)}")


def export_model_results(models, output_dir):
    """
    Экспорт результатов всех моделей в отдельные файлы.

    Args:
        models (dict): Словарь с объектами моделей
        output_dir (str): Директория для сохранения результатов
    """
    try:
        # Создаем директорию, если она не существует
        os.makedirs(output_dir, exist_ok=True)

        # Экспортируем результаты каждой модели
        for model_name, model in models.items():
            model_dir = os.path.join(output_dir, model_name)
            os.makedirs(model_dir, exist_ok=True)

            # Получаем результаты модели
            if hasattr(model, 'get_results'):
                results = model.get_results()

                if results is not None:
                    # Экспортируем в разные форматы
                    export_to_csv(results, os.path.join(model_dir, f"{model_name}_results.csv"))
                    export_to_excel(results, os.path.join(model_dir, f"{model_name}_results.xlsx"))
                    export_to_json(results, os.path.join(model_dir, f"{model_name}_results.json"))

            # Для модели добывающих скважин также экспортируем прогнозные результаты
            if model_name == 'production_wells' and hasattr(model, 'get_forecast_results'):
                forecast = model.get_forecast_results()

                if forecast is not None:
                    export_to_csv(forecast, os.path.join(model_dir, "forecast_results.csv"))
                    export_to_excel(forecast, os.path.join(model_dir, "forecast_results.xlsx"))
                    export_to_json(forecast, os.path.join(model_dir, "forecast_results.json"))

        logger.info(f"Результаты моделей успешно экспортированы в директорию: {output_dir}")
    except Exception as e:
        logger.error(f"Ошибка при экспорте результатов моделей: {str(e)}")


def create_report(models, output_dir):
    """
    Создание отчета о результатах расчетов.

    Args:
        models (dict): Словарь с объектами моделей
        output_dir (str): Директория для сохранения отчета
    """
    try:
        # Создаем директорию, если она не существует
        os.makedirs(output_dir, exist_ok=True)

        # Создаем имя файла с текущей датой
        now = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(output_dir, f"report_{now}.pdf")

        # Создаем отчет
        create_summary_report(models, report_file)

        logger.info(f"Отчет успешно создан и сохранен в: {report_file}")
    except Exception as e:
        logger.error(f"Ошибка при создании отчета: {str(e)}")


def main(models):
    """
    Основная функция для экспорта результатов.

    Args:
        models (dict): Словарь с объектами моделей
    """
    logger.info("Начинаем экспорт результатов")

    # Создаем директорию для результатов
    results_dir = os.path.join(OUTPUT_DIR, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Экспортируем параметры моделей
    params_file = os.path.join(results_dir, "model_parameters.xlsx")
    export_model_parameters(models, params_file)

    # Экспортируем результаты моделей
    export_model_results(models, results_dir)

    # Создаем отчет
    create_report(models, OUTPUT_DIR)

    logger.info("Экспорт результатов успешно завершен")


if __name__ == "__main__":
    # Этот скрипт обычно запускается из main.py
    # и получает модели в качестве аргумента.
    # Здесь приведен пример для тестирования

    # Создаем фиктивные модели для тестирования
    models = {
        'phase_permeability': None,
        'regression_model': None,
        'pressure_calculation': None,
        'pressure_recovery': None,
        'skin_curve': None,
        'filter_reduction': None,
        'fracture_length': None,
        'production_wells': None
    }

    main(models)