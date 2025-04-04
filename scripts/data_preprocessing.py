#!/usr/bin/env python3
"""
Скрипт для предварительной обработки исходных данных.

Выполняет очистку, преобразование и подготовку данных
для использования в расчетной схеме.
"""

import os
import sys
import pandas as pd
import numpy as np
import logging
from pathlib import Path

# Добавляем родительскую директорию в sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    PZAB_FILE, PPL_FILE, ARCGIS_FILE, GDI_VNR_FILE,
    GDI_REINT_FILE, GTP_FILE, NNT_NGT_FILE,
    DATA_DIR
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("data_preprocessing.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def preprocess_pzab_data(input_file, output_file=None):
    """
    Предобработка данных по забойному давлению.

    Args:
        input_file (str): Путь к исходному файлу
        output_file (str, optional): Путь для сохранения обработанных данных

    Returns:
        pd.DataFrame: Обработанные данные
    """
    logger.info(f"Начинаем обработку файла {os.path.basename(input_file)}")

    try:
        # Загрузка данных
        df = pd.read_csv(
            input_file,
            encoding='cp1252',
            delimiter=';',
            decimal=',',
            low_memory=False
        )

        logger.info(f"Исходные данные: {df.shape[0]} строк, {df.shape[1]} столбцов")

        # Обработка данных
        # 1. Удаление дубликатов
        df.drop_duplicates(inplace=True)

        # 2. Обработка пропущенных значений
        # Заменяем пропущенные значения на NaN
        df.replace(['', 'н/д', '-'], np.nan, inplace=True)

        # 3. Преобразование типов данных
        # Преобразуем числовые столбцы
        numeric_columns = df.select_dtypes(include=['object']).columns
        for col in numeric_columns:
            try:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            except:
                pass

        # 4. Фильтрация данных
        # Удаляем строки с пропущенными важными значениями
        # (здесь нужно указать конкретные столбцы в зависимости от структуры данных)

        logger.info(f"После обработки: {df.shape[0]} строк, {df.shape[1]} столбцов")

        # Сохранение результатов
        if output_file:
            df.to_csv(output_file, index=False, sep=';', encoding='utf-8')
            logger.info(f"Обработанные данные сохранены в {output_file}")

        return df

    except Exception as e:
        logger.error(f"Ошибка при обработке файла {os.path.basename(input_file)}: {str(e)}")
        return None


def preprocess_ppl_data(input_file, output_file=None):
    """
    Предобработка данных по пластовому давлению.

    Args:
        input_file (str): Путь к исходному файлу
        output_file (str, optional): Путь для сохранения обработанных данных

    Returns:
        pd.DataFrame or dict: Обработанные данные
    """
    logger.info(f"Начинаем обработку файла {os.path.basename(input_file)}")

    try:
        # Проверка наличия нескольких листов
        xls = pd.ExcelFile(input_file)
        sheets = xls.sheet_names

        result = {}

        for sheet in sheets:
            logger.info(f"Обработка листа: {sheet}")

            # Загрузка данных с листа
            df = pd.read_excel(input_file, sheet_name=sheet)

            logger.info(f"Исходные данные листа {sheet}: {df.shape[0]} строк, {df.shape[1]} столбцов")

            # Обработка данных
            # 1. Удаление дубликатов
            df.drop_duplicates(inplace=True)

            # 2. Обработка пропущенных значений
            df.replace(['', 'н/д', '-'], np.nan, inplace=True)

            # 3. Фильтрация данных
            # (здесь нужно указать конкретные условия в зависимости от структуры данных)

            logger.info(f"После обработки листа {sheet}: {df.shape[0]} строк, {df.shape[1]} столбцов")

            result[sheet] = df

        # Сохранение результатов
        if output_file:
            with pd.ExcelWriter(output_file) as writer:
                for sheet, df in result.items():
                    df.to_excel(writer, sheet_name=sheet, index=False)

            logger.info(f"Обработанные данные сохранены в {output_file}")

        return result if len(sheets) > 1 else result[sheets[0]]

    except Exception as e:
        logger.error(f"Ошибка при обработке файла {os.path.basename(input_file)}: {str(e)}")
        return None


def preprocess_arcgis_data(input_file, output_file=None):
    """
    Предобработка данных о фонде скважин для ArcGIS.

    Args:
        input_file (str): Путь к исходному файлу
        output_file (str, optional): Путь для сохранения обработанных данных

    Returns:
        pd.DataFrame: Обработанные данные
    """
    logger.info(f"Начинаем обработку файла {os.path.basename(input_file)}")

    try:
        # Загрузка данных
        df = pd.read_csv(
            input_file,
            encoding='cp1252',
            delimiter=';',
            low_memory=False
        )

        logger.info(f"Исходные данные: {df.shape[0]} строк, {df.shape[1]} столбцов")

        # Обработка данных
        # Аналогичная обработка, как для предыдущих файлов
        # ...

        logger.info(f"После обработки: {df.shape[0]} строк, {df.shape[1]} столбцов")

        # Сохранение результатов
        if output_file:
            df.to_csv(output_file, index=False, sep=';', encoding='utf-8')
            logger.info(f"Обработанные данные сохранены в {output_file}")

        return df

    except Exception as e:
        logger.error(f"Ошибка при обработке файла {os.path.basename(input_file)}: {str(e)}")
        return None


def preprocess_gdi_data(input_file, output_file=None):
    """
    Предобработка данных ГДИ.

    Args:
        input_file (str): Путь к исходному файлу
        output_file (str, optional): Путь для сохранения обработанных данных

    Returns:
        pd.DataFrame or dict: Обработанные данные
    """
    logger.info(f"Начинаем обработку файла {os.path.basename(input_file)}")

    try:
        # Проверка наличия нескольких листов
        xls = pd.ExcelFile(input_file)
        sheets = xls.sheet_names

        result = {}

        for sheet in sheets:
            logger.info(f"Обработка листа: {sheet}")

            # Загрузка данных с листа
            df = pd.read_excel(input_file, sheet_name=sheet)

            logger.info(f"Исходные данные листа {sheet}: {df.shape[0]} строк, {df.shape[1]} столбцов")

            # Обработка данных
            # 1. Удаление дубликатов
            df.drop_duplicates(inplace=True)

            # 2. Обработка пропущенных значений
            df.replace(['', 'н/д', '-'], np.nan, inplace=True)

            # 3. Фильтрация данных
            # Удаляем строки с пропущенными важными значениями
            # (здесь нужно указать конкретные столбцы в зависимости от структуры данных)

            logger.info(f"После обработки листа {sheet}: {df.shape[0]} строк, {df.shape[1]} столбцов")

            result[sheet] = df

        # Сохранение результатов
        if output_file:
            with pd.ExcelWriter(output_file) as writer:
                for sheet, df in result.items():
                    df.to_excel(writer, sheet_name=sheet, index=False)

            logger.info(f"Обработанные данные сохранены в {output_file}")

        return result if len(sheets) > 1 else result[sheets[0]]

    except Exception as e:
        logger.error(f"Ошибка при обработке файла {os.path.basename(input_file)}: {str(e)}")
        return None


def preprocess_gtp_data(input_file, output_file=None):
    """
    Предобработка данных ГТР.

    Args:
        input_file (str): Путь к исходному файлу
        output_file (str, optional): Путь для сохранения обработанных данных

    Returns:
        pd.DataFrame or dict: Обработанные данные
    """
    logger.info(f"Начинаем обработку файла {os.path.basename(input_file)}")

    try:
        # Проверяем расширение файла
        if input_file.lower().endswith('.xls'):
            # Для формата XLS используем xlrd
            try:
                import xlrd
                # Загрузка Excel файла с xlrd
                workbook = xlrd.open_workbook(input_file)
                sheet_names = workbook.sheet_names()

                result = {}
                for sheet_name in sheet_names:
                    # Используем безопасную обработку имен листов для логирования
                    safe_sheet_name = sheet_name.encode('utf-8', errors='replace').decode('utf-8')
                    try:
                        logger.info(f"Обработка листа: {safe_sheet_name}")

                        sheet = workbook.sheet_by_name(sheet_name)
                        data = []
                        for i in range(sheet.nrows):
                            data.append(sheet.row_values(i))

                        if data:
                            # Используем первую строку как заголовки
                            columns = data[0]
                            df = pd.DataFrame(data[1:], columns=columns)
                        else:
                            df = pd.DataFrame()

                        logger.info(
                            f"Исходные данные листа {safe_sheet_name}: {df.shape[0]} строк, {df.shape[1]} столбцов")

                        # Обработка данных
                        # 1. Удаление дубликатов
                        df.drop_duplicates(inplace=True)

                        # 2. Обработка пропущенных значений
                        df.replace(['', 'н/д', '-'], np.nan, inplace=True)

                        logger.info(
                            f"После обработки листа {safe_sheet_name}: {df.shape[0]} строк, {df.shape[1]} столбцов")

                        result[sheet_name] = df
                    except Exception as e:
                        logger.error(f"Ошибка при обработке листа {safe_sheet_name}: {str(e)}")
                        result[sheet_name] = pd.DataFrame()

                # Сохранение результатов
                if output_file and result:
                    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                        for sheet_name, df in result.items():
                            df.to_excel(writer, sheet_name=sheet_name, index=False)
                    logger.info(f"Обработанные данные сохранены в {output_file}")

                return result

            except ImportError:
                logger.error("Библиотека xlrd не установлена. Попробуйте выполнить: pip install xlrd>=2.0.1")
                return None
            except Exception as e:
                logger.error(f"Ошибка при обработке XLS файла: {str(e)}")
                return None
        else:
            # Для формата XLSX используем openpyxl
            # Проверка наличия нескольких листов
            xls = pd.ExcelFile(input_file, engine='openpyxl')
            sheets = xls.sheet_names

            result = {}

            for sheet in sheets:
                logger.info(f"Обработка листа: {sheet}")

                # Загрузка данных с листа
                df = pd.read_excel(input_file, sheet_name=sheet, engine='openpyxl')

                logger.info(f"Исходные данные листа {sheet}: {df.shape[0]} строк, {df.shape[1]} столбцов")

                # Обработка данных
                # 1. Удаление дубликатов
                df.drop_duplicates(inplace=True)

                # 2. Обработка пропущенных значений
                df.replace(['', 'н/д', '-'], np.nan, inplace=True)

                logger.info(f"После обработки листа {sheet}: {df.shape[0]} строк, {df.shape[1]} столбцов")

                result[sheet] = df

            # Сохранение результатов
            if output_file:
                with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
                    for sheet, df in result.items():
                        df.to_excel(writer, sheet_name=sheet, index=False)

                logger.info(f"Обработанные данные сохранены в {output_file}")

            return result if len(sheets) > 1 else result[sheets[0]]

    except Exception as e:
        logger.error(f"Ошибка при обработке файла {os.path.basename(input_file)}: {str(e)}")
        return None


def preprocess_nnt_ngt_data(input_file, output_file=None):
    """
    Предобработка данных о нефтенасыщенных толщинах.

    Args:
        input_file (str): Путь к исходному файлу
        output_file (str, optional): Путь для сохранения обработанных данных

    Returns:
        pd.DataFrame or dict: Обработанные данные
    """
    # Аналогичная реализация, как для preprocess_gdi_data
    logger.info(f"Начинаем обработку файла {os.path.basename(input_file)}")

    try:
        # Проверка наличия нескольких листов
        xls = pd.ExcelFile(input_file)
        sheets = xls.sheet_names

        result = {}

        for sheet in sheets:
            logger.info(f"Обработка листа: {sheet}")

            # Загрузка данных с листа
            df = pd.read_excel(input_file, sheet_name=sheet)

            logger.info(f"Исходные данные листа {sheet}: {df.shape[0]} строк, {df.shape[1]} столбцов")

            # Обработка данных
            # 1. Удаление дубликатов
            df.drop_duplicates(inplace=True)

            # 2. Обработка пропущенных значений
            df.replace(['', 'н/д', '-'], np.nan, inplace=True)

            # 3. Фильтрация данных
            # (здесь нужно указать конкретные условия в зависимости от структуры данных)

            logger.info(f"После обработки листа {sheet}: {df.shape[0]} строк, {df.shape[1]} столбцов")

            result[sheet] = df

        # Сохранение результатов
        if output_file:
            with pd.ExcelWriter(output_file) as writer:
                for sheet, df in result.items():
                    df.to_excel(writer, sheet_name=sheet, index=False)

            logger.info(f"Обработанные данные сохранены в {output_file}")

        return result if len(sheets) > 1 else result[sheets[0]]

    except Exception as e:
        logger.error(f"Ошибка при обработке файла {os.path.basename(input_file)}: {str(e)}")
        return None


def main():
    """
    Основная функция для запуска предобработки всех файлов.
    """
    logger.info("Начинаем предобработку всех файлов данных")

    # Создаем директорию для обработанных данных
    processed_dir = os.path.join(DATA_DIR, 'processed')
    os.makedirs(processed_dir, exist_ok=True)

    # Обработка файла забойного давления
    pzab_output = os.path.join(processed_dir, 'processed_pzab.csv')
    preprocess_pzab_data(PZAB_FILE, pzab_output)

    # Обработка файла пластового давления
    ppl_output = os.path.join(processed_dir, 'processed_ppl.xlsx')
    preprocess_ppl_data(PPL_FILE, ppl_output)

    # Обработка файла фонда скважин для ArcGIS
    arcgis_output = os.path.join(processed_dir, 'processed_arcgis.csv')
    preprocess_arcgis_data(ARCGIS_FILE, arcgis_output)

    # Обработка файла ГДИ ВНР
    gdi_vnr_output = os.path.join(processed_dir, 'processed_gdi_vnr.xlsx')
    preprocess_gdi_data(GDI_VNR_FILE, gdi_vnr_output)

    # Обработка файла переинтерпретации ГДИ
    gdi_reint_output = os.path.join(processed_dir, 'processed_gdi_reint.xlsx')
    preprocess_gdi_data(GDI_REINT_FILE, gdi_reint_output)

    # Обработка файла ГТР
    gtp_output = os.path.join(processed_dir, 'processed_gtp.xlsx')
    preprocess_gtp_data(GTP_FILE, gtp_output)

    # Обработка файла nnt_NGT
    nnt_ngt_output = os.path.join(processed_dir, 'processed_nnt_ngt.xlsx')
    preprocess_nnt_ngt_data(NNT_NGT_FILE, nnt_ngt_output)

    logger.info("Предобработка всех файлов данных успешно завершена")


if __name__ == "__main__":
    main()