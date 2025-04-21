"""
Модуль для загрузки и предварительной обработки данных из файлов.
"""

import os
import pandas as pd
import numpy as np
import logging
from src.config import (
    PZAB_FILE, PPL_FILE, ARCGIS_FILE, GDI_VNR_FILE,
    GDI_REINT_FILE, GTP_FILE, NNT_NGT_FILE,
    PZAB_ENCODING, ARCGIS_ENCODING
)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DataLoader:
    """Класс для загрузки и подготовки данных из файлов."""

    def __init__(self):
        """Инициализация объекта загрузчика данных."""
        self.pzab_data = None
        self.ppl_data = None
        self.arcgis_data = None
        self.gdi_vnr_data = None
        self.gdi_reint_data = None
        self.gtp_data = None
        self.nnt_ngt_data = None

    def load_all_data(self):
        """Загрузка всех файлов данных."""
        logger.info("Начинаем загрузку всех файлов данных")

        self._load_pzab_data()
        self._load_ppl_data()
        self._load_arcgis_data()
        self._load_gdi_vnr_data()
        self._load_gdi_reint_data()
        self._load_gtp_data()
        self._load_nnt_ngt_data()

        logger.info("Все файлы данных успешно загружены")

    def _load_pzab_data(self):
        """Загрузка данных по забойному давлению."""
        logger.info(f"Загрузка файла {os.path.basename(PZAB_FILE)}")

        try:
            # Попытка загрузить CSV файл с разными кодировками
            encodings = ['utf-8', 'cp1251', 'latin1', 'cp1252', 'iso-8859-1']

            for encoding in encodings:
                try:
                    self.pzab_data = pd.read_csv(
                        PZAB_FILE,
                        encoding=encoding,
                        delimiter=';',  # Распространенный разделитель для русских CSV
                        decimal=',',  # Русский формат десятичных чисел
                        low_memory=False
                    )
                    logger.info(f"Файл {os.path.basename(PZAB_FILE)} успешно загружен с кодировкой {encoding}")
                    logger.info(f"Размер данных: {self.pzab_data.shape}")
                    return
                except UnicodeDecodeError:
                    continue
                except Exception as e:
                    logger.warning(f"Не удалось загрузить с кодировкой {encoding}: {str(e)}")
                    continue

            # Если ни одна кодировка не подошла, попробуем другой подход
            try:
                self.pzab_data = pd.read_csv(
                    PZAB_FILE,
                    encoding='latin1',  # Обычно работает с большинством файлов
                    sep=None,  # Автоопределение разделителя
                    engine='python',  # Python engine поддерживает автоопределение разделителя
                    decimal=',',
                )
                logger.info(f"Файл {os.path.basename(PZAB_FILE)} успешно загружен альтернативным способом")
                logger.info(f"Размер данных: {self.pzab_data.shape}")
            except Exception as e2:
                logger.error(f"Повторная ошибка при загрузке файла {os.path.basename(PZAB_FILE)}: {str(e2)}")
                self.pzab_data = None

        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {os.path.basename(PZAB_FILE)}: {str(e)}")
            self.pzab_data = None

    def _load_ppl_data(self):
        """Загрузка данных по пластовому давлению."""
        logger.info(f"Загрузка файла {os.path.basename(PPL_FILE)}")

        try:
            # Загрузка Excel файла
            self.ppl_data = pd.read_excel(PPL_FILE, engine='openpyxl')

            logger.info(f"Файл {os.path.basename(PPL_FILE)} успешно загружен")
            logger.info(f"Размер данных: {self.ppl_data.shape}")

            # ДОБАВЛЯЕМ ИСПОЛЬЗОВАНИЕ get_column_statistics
            # Выводим статистику по давлению, если есть такой столбец
            pressure_columns = [col for col in self.ppl_data.columns
                                if 'давлен' in col.lower() or 'pressure' in col.lower()]

            if pressure_columns:
                for col in pressure_columns[:2]:  # Статистика для первых двух колонок
                    stats = self.get_column_statistics('ppl', col)
                    if stats:
                        logger.info(f"Статистика столбца '{col}':")
                        logger.info(f"  - Среднее: {stats.get('mean', 'N/A')}")
                        logger.info(f"  - Минимум: {stats.get('min', 'N/A')}")
                        logger.info(f"  - Максимум: {stats.get('max', 'N/A')}")

            # Проверка на наличие нескольких листов в файле
            xls = pd.ExcelFile(PPL_FILE)
            if len(xls.sheet_names) > 1:
                logger.info(f"Файл {os.path.basename(PPL_FILE)} содержит несколько листов: {xls.sheet_names}")

                # Загрузим данные со всех листов в словарь
                sheets_data = {}
                for sheet_name in xls.sheet_names:
                    sheets_data[sheet_name] = pd.read_excel(PPL_FILE, sheet_name=sheet_name)

                self.ppl_data = sheets_data
                logger.info(f"Все листы загружены в словарь")

        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {os.path.basename(PPL_FILE)}: {str(e)}")
            self.ppl_data = None

    def _load_arcgis_data(self):
        """Загрузка данных о фонде скважин для ArcGIS."""
        logger.info(f"Загрузка файла {os.path.basename(ARCGIS_FILE)}")

        try:
            # Учитывая информацию из CSV, что в нем одна колонка с множеством полей через точку с запятой
            self.arcgis_data = pd.read_csv(
                ARCGIS_FILE,
                encoding=ARCGIS_ENCODING,
                delimiter=';',
                low_memory=False
            )

            logger.info(f"Файл {os.path.basename(ARCGIS_FILE)} успешно загружен")
            logger.info(f"Размер данных: {self.arcgis_data.shape}")

        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {os.path.basename(ARCGIS_FILE)}: {str(e)}")
            # Попробуем альтернативный подход
            try:
                self.arcgis_data = pd.read_csv(
                    ARCGIS_FILE,
                    encoding=ARCGIS_ENCODING,
                    delimiter=None,  # Автоопределение разделителя
                    engine='python',
                    low_memory=False
                )
                logger.info(f"Файл {os.path.basename(ARCGIS_FILE)} успешно загружен альтернативным способом")
            except Exception as e2:
                logger.error(f"Повторная ошибка при загрузке файла {os.path.basename(ARCGIS_FILE)}: {str(e2)}")
                self.arcgis_data = None

    def _load_gdi_vnr_data(self):
        """Загрузка данных ГДИ ВНР."""
        logger.info(f"Загрузка файла {os.path.basename(GDI_VNR_FILE)}")

        try:
            # Проверяем расширение файла
            if GDI_VNR_FILE.lower().endswith('.xls'):
                # Для формата XLS используем xlrd
                try:
                    import xlrd
                    # Загрузка Excel файла с xlrd
                    workbook = xlrd.open_workbook(GDI_VNR_FILE)
                    sheet_names = workbook.sheet_names()

                    # Если есть только один лист
                    if len(sheet_names) == 1:
                        sheet = workbook.sheet_by_index(0)
                        # Преобразуем лист в DataFrame
                        data = []
                        for i in range(sheet.nrows):
                            data.append(sheet.row_values(i))

                        if data:
                            # Используем первую строку как заголовки
                            columns = data[0]
                            self.gdi_vnr_data = pd.DataFrame(data[1:], columns=columns)
                        else:
                            self.gdi_vnr_data = pd.DataFrame()
                    else:
                        # Если несколько листов, загружаем их в словарь
                        sheets_data = {}
                        for sheet_name in sheet_names:
                            sheet = workbook.sheet_by_name(sheet_name)
                            data = []
                            for i in range(sheet.nrows):
                                data.append(sheet.row_values(i))

                            if data:
                                # Используем первую строку как заголовки
                                columns = data[0]
                                sheets_data[sheet_name] = pd.DataFrame(data[1:], columns=columns)
                            else:
                                sheets_data[sheet_name] = pd.DataFrame()

                        self.gdi_vnr_data = sheets_data

                    logger.info(f"Файл {os.path.basename(GDI_VNR_FILE)} успешно загружен с xlrd")
                    if isinstance(self.gdi_vnr_data, dict):
                        logger.info(f"Загружено {len(self.gdi_vnr_data)} листов")
                    else:
                        logger.info(f"Размер данных: {self.gdi_vnr_data.shape}")

                except ImportError:
                    logger.error("Библиотека xlrd не установлена. Попробуйте выполнить: pip install xlrd>=2.0.1")
                    self.gdi_vnr_data = None
                except Exception as e:
                    logger.error(f"Ошибка при загрузке XLS файла: {str(e)}")
                    self.gdi_vnr_data = None
            else:
                # Для формата XLSX используем openpyxl
                self.gdi_vnr_data = pd.read_excel(GDI_VNR_FILE, engine='openpyxl')

                logger.info(f"Файл {os.path.basename(GDI_VNR_FILE)} успешно загружен")
                logger.info(f"Размер данных: {self.gdi_vnr_data.shape}")

                # Проверка на наличие нескольких листов в файле
                xls = pd.ExcelFile(GDI_VNR_FILE, engine='openpyxl')
                if len(xls.sheet_names) > 1:
                    logger.info(f"Файл {os.path.basename(GDI_VNR_FILE)} содержит несколько листов: {xls.sheet_names}")

                    # Загрузим данные со всех листов в словарь
                    sheets_data = {}
                    for sheet_name in xls.sheet_names:
                        sheets_data[sheet_name] = pd.read_excel(GDI_VNR_FILE, sheet_name=sheet_name, engine='openpyxl')

                    self.gdi_vnr_data = sheets_data
                    logger.info(f"Все листы загружены в словарь")

        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {os.path.basename(GDI_VNR_FILE)}: {str(e)}")
            self.gdi_vnr_data = None

    def _load_gdi_reint_data(self):
        """Загрузка данных переинтерпретации ГДИ."""
        logger.info(f"Загрузка файла {os.path.basename(GDI_REINT_FILE)}")

        try:
            # Загрузка Excel файла
            self.gdi_reint_data = pd.read_excel(GDI_REINT_FILE, engine='openpyxl')

            logger.info(f"Файл {os.path.basename(GDI_REINT_FILE)} успешно загружен")
            logger.info(f"Размер данных: {self.gdi_reint_data.shape}")

            # Проверка на наличие нескольких листов в файле
            xls = pd.ExcelFile(GDI_REINT_FILE)
            if len(xls.sheet_names) > 1:
                logger.info(f"Файл {os.path.basename(GDI_REINT_FILE)} содержит несколько листов: {xls.sheet_names}")

                # Загрузим данные со всех листов в словарь
                sheets_data = {}
                for sheet_name in xls.sheet_names:
                    sheets_data[sheet_name] = pd.read_excel(GDI_REINT_FILE, sheet_name=sheet_name)

                self.gdi_reint_data = sheets_data
                logger.info(f"Все листы загружены в словарь")

        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {os.path.basename(GDI_REINT_FILE)}: {str(e)}")
            self.gdi_reint_data = None

    def _load_gtp_data(self):
        """Загрузка данных ГТР."""
        logger.info(f"Загрузка файла {os.path.basename(GTP_FILE)}")

        try:
            # Загрузка Excel файла
            self.gtp_data = pd.read_excel(GTP_FILE, engine='openpyxl')

            logger.info(f"Файл {os.path.basename(GTP_FILE)} успешно загружен")
            logger.info(f"Размер данных: {self.gtp_data.shape}")

            # Проверка на наличие нескольких листов в файле
            xls = pd.ExcelFile(GTP_FILE)
            if len(xls.sheet_names) > 1:
                logger.info(f"Файл {os.path.basename(GTP_FILE)} содержит несколько листов: {xls.sheet_names}")

                # Загрузим данные со всех листов в словарь
                sheets_data = {}
                for sheet_name in xls.sheet_names:
                    sheets_data[sheet_name] = pd.read_excel(GTP_FILE, sheet_name=sheet_name)

                self.gtp_data = sheets_data
                logger.info(f"Все листы загружены в словарь")

        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {os.path.basename(GTP_FILE)}: {str(e)}")
            self.gtp_data = None

    def _load_nnt_ngt_data(self):
        """Загрузка данных о нефтенасыщенных толщинах."""
        logger.info(f"Загрузка файла {os.path.basename(NNT_NGT_FILE)}")

        try:
            # Загрузка Excel файла
            self.nnt_ngt_data = pd.read_excel(NNT_NGT_FILE, engine='openpyxl')

            logger.info(f"Файл {os.path.basename(NNT_NGT_FILE)} успешно загружен")
            logger.info(f"Размер данных: {self.nnt_ngt_data.shape}")

            # Проверка на наличие нескольких листов в файле
            xls = pd.ExcelFile(NNT_NGT_FILE)
            if len(xls.sheet_names) > 1:
                logger.info(f"Файл {os.path.basename(NNT_NGT_FILE)} содержит несколько листов: {xls.sheet_names}")

                # Загрузим данные со всех листов в словарь
                sheets_data = {}
                for sheet_name in xls.sheet_names:
                    sheets_data[sheet_name] = pd.read_excel(NNT_NGT_FILE, sheet_name=sheet_name)

                self.nnt_ngt_data = sheets_data
                logger.info(f"Все листы загружены в словарь")

        except Exception as e:
            logger.error(f"Ошибка при загрузке файла {os.path.basename(NNT_NGT_FILE)}: {str(e)}")
            self.nnt_ngt_data = None

    def get_data_for_phase_permeability(self):
        """
        Подготовка данных для модуля 1 (Подбор относительных фазовых проницаемостей).

        Returns:
            dict: Данные для модуля 1
        """
        logger.info("Подготовка данных для модуля 1 (относительные фазовые проницаемости)")

        try:
            # Извлекаем нужные данные из загруженных файлов
            # В частности, нам нужны данные о фазовых проницаемостях из файлов ГДИ

            result = {
                'ppl_data': self.ppl_data,
                'gdi_data': self.gdi_reint_data,
            }

            # Дополнительная предобработка данных, если требуется
            # Например, из данных ГДИ нужно извлечь информацию о проницаемостях и насыщенностях

            if isinstance(self.gdi_reint_data, dict):
                # Если данные в виде словаря с листами, выбираем нужный лист
                # или объединяем данные из разных листов
                gdi_sheets = list(self.gdi_reint_data.keys())
                result['gdi_main_data'] = self.gdi_reint_data.get(gdi_sheets[0], pd.DataFrame())

                # Попытка найти лист с фазовыми проницаемостями
                for sheet_name in gdi_sheets:
                    if 'проницаемост' in sheet_name.lower() or 'фаз' in sheet_name.lower():
                        result['gdi_perm_data'] = self.gdi_reint_data[sheet_name]
                        logger.info(f"Найдены данные о фазовых проницаемостях в листе {sheet_name}")
                        break

            # Также нам могут потребоваться данные о свойствах флюидов из других файлов
            if self.nnt_ngt_data is not None:
                result['nnt_ngt_data'] = self.nnt_ngt_data

            logger.info("Данные для модуля 1 успешно подготовлены")
            return result

        except Exception as e:
            logger.error(f"Ошибка при подготовке данных для модуля 1: {str(e)}")
            # Возвращаем минимальный набор данных, чтобы модуль мог работать
            return {
                'ppl_data': self.ppl_data,
                'gdi_data': self.gdi_reint_data
            }

    def get_data_for_regression_model(self):
        """
        Подготовка данных для модуля 2 (Подбор итеративной регрессионной моделью).

        Returns:
            dict: Данные для модуля 2
        """
        logger.info("Подготовка данных для модуля 2 (итеративная регрессионная модель)")

        try:
            # Для регрессионной модели нам нужны данные о разработке системы пласта
            # Основным источником будут данные о пластовом давлении и гидродинамических исследованиях

            result = {
                'ppl_data': self.ppl_data,
                'gdi_data': self.gdi_reint_data,
                'pzab_data': self.pzab_data,
            }

            # Если есть данные о фонде скважин, то также добавляем их
            if self.arcgis_data is not None:
                result['arcgis_data'] = self.arcgis_data

            # Дополнительная предобработка данных, если требуется
            # Например, извлечение истории разработки системы пласта

            # Предобработка данных о пластовом давлении
            if isinstance(self.ppl_data, pd.DataFrame):
                # Если есть временные столбцы, отсортируем по времени
                time_columns = [col for col in self.ppl_data.columns if 'дата' in col.lower()
                             or 'time' in col.lower() or 'date' in col.lower()]

                if time_columns and len(time_columns) > 0:
                    time_col = time_columns[0]
                    result['ppl_data_sorted'] = self.ppl_data.sort_values(by=time_col)
                    logger.info(f"Данные о пластовом давлении отсортированы по столбцу {time_col}")

            logger.info("Данные для модуля 2 успешно подготовлены")
            return result

        except Exception as e:
            logger.error(f"Ошибка при подготовке данных для модуля 2: {str(e)}")
            # Возвращаем минимальный набор данных
            return {
                'ppl_data': self.ppl_data,
                'gdi_data': self.gdi_reint_data
            }

    def get_data_for_pressure_calculation(self):
        """
        Подготовка данных для модуля 3 (Расчет Рпл в нефтяных скважинах).

        Returns:
            dict: Данные для модуля 3
        """
        logger.info("Подготовка данных для модуля 3 (расчет пластового давления)")

        try:
            # Для расчета пластового давления нам нужны данные о забойном давлении,
            # а также граничные условия и параметры моделей из предыдущих модулей

            result = {
                'pzab_data': self.pzab_data,
                'ppl_data': self.ppl_data,
            }

            # Если есть данные о фонде скважин, также добавляем их
            if self.arcgis_data is not None:
                result['arcgis_data'] = self.arcgis_data

            # Если есть данные о гидродинамических исследованиях, добавляем и их
            if self.gdi_vnr_data is not None:
                result['gdi_vnr_data'] = self.gdi_vnr_data

            # Предобработка данных о забойном давлении
            if isinstance(self.pzab_data, pd.DataFrame):
                # Выделяем столбцы с забойным давлением
                # Предположим, что они содержат 'заб' и 'давл' в названии
                pressure_columns = [col for col in self.pzab_data.columns
                                   if ('заб' in col.lower() and 'давл' in col.lower())
                                   or 'pzab' in col.lower()]

                if pressure_columns and len(pressure_columns) > 0:
                    # Выбираем нужные столбцы + идентификатор скважины
                    well_id_columns = [col for col in self.pzab_data.columns
                                      if 'скв' in col.lower() or 'well' in col.lower()
                                      or 'id' in col.lower()]

                    if well_id_columns and len(well_id_columns) > 0:
                        selected_columns = well_id_columns + pressure_columns
                        result['pzab_selected'] = self.pzab_data[selected_columns]
                        logger.info(f"Выбраны данные о забойном давлении: {selected_columns}")

            logger.info("Данные для модуля 3 успешно подготовлены")
            return result

        except Exception as e:
            logger.error(f"Ошибка при подготовке данных для модуля 3: {str(e)}")
            # Возвращаем минимальный набор данных
            return {
                'pzab_data': self.pzab_data,
                'ppl_data': self.ppl_data
            }

    def get_data_for_pressure_recovery(self):
        """
        Подготовка данных для модуля 4 (Подбор времени восстановления давления).

        Returns:
            dict: Данные для модуля 4
        """
        logger.info("Подготовка данных для модуля 4 (время восстановления давления)")

        try:
            # Для расчета времени восстановления давления нам нужны данные о ГДИ,
            # параметрах пласта и скважин

            result = {
                'gdi_vnr_data': self.gdi_vnr_data,
                'gdi_reint_data': self.gdi_reint_data,
                'arcgis_data': self.arcgis_data
            }

            # Предобработка данных ГДИ ВНР
            if isinstance(self.gdi_vnr_data, dict):
                # Если данные в виде словаря с листами, выбираем нужный лист
                # или объединяем данные из разных листов
                gdi_sheets = list(self.gdi_vnr_data.keys())

                # Попытка найти лист с данными о восстановлении давления
                for sheet_name in gdi_sheets:
                    if 'восстановлен' in sheet_name.lower() or 'внр' in sheet_name.lower():
                        result['gdi_recovery_data'] = self.gdi_vnr_data[sheet_name]
                        logger.info(f"Найдены данные о восстановлении давления в листе {sheet_name}")
                        break

            # Предобработка данных переинтерпретации ГДИ
            if isinstance(self.gdi_reint_data, dict):
                # Попытка найти лист с данными о проницаемости, пористости и скин-факторе
                gdi_reint_sheets = list(self.gdi_reint_data.keys())

                for sheet_name in gdi_reint_sheets:
                    if 'параметр' in sheet_name.lower() or 'проницаем' in sheet_name.lower():
                        result['gdi_params_data'] = self.gdi_reint_data[sheet_name]
                        logger.info(f"Найдены данные о параметрах пласта в листе {sheet_name}")
                        break

            logger.info("Данные для модуля 4 успешно подготовлены")
            return result

        except Exception as e:
            logger.error(f"Ошибка при подготовке данных для модуля 4: {str(e)}")
            # Возвращаем минимальный набор данных
            return {
                'gdi_vnr_data': self.gdi_vnr_data,
                'gdi_reint_data': self.gdi_reint_data
            }

    def get_data_for_skin_curve(self):
        """
        Подготовка данных для модуля 5 (Подбор кривой увеличения SKIN).

        Returns:
            dict: Данные для модуля 5
        """
        logger.info("Подготовка данных для модуля 5 (кривая увеличения SKIN)")

        try:
            # Для подбора кривой увеличения SKIN нам нужны данные о ГРП и
            # динамике изменения скин-фактора со временем

            result = {
                'gdi_reint_data': self.gdi_reint_data,
                'gtp_data': self.gtp_data  # Данные о ГТР могут содержать информацию о ГРП
            }

            # Предобработка данных переинтерпретации ГДИ
            if isinstance(self.gdi_reint_data, dict):
                # Попытка найти лист с данными о скин-факторе
                gdi_reint_sheets = list(self.gdi_reint_data.keys())

                for sheet_name in gdi_reint_sheets:
                    if 'скин' in sheet_name.lower() or 'skin' in sheet_name.lower():
                        result['skin_data'] = self.gdi_reint_data[sheet_name]
                        logger.info(f"Найдены данные о скин-факторе в листе {sheet_name}")
                        break

            # Предобработка данных ГТР
            if isinstance(self.gtp_data, dict):
                # Попытка найти лист с данными о ГРП
                gtp_sheets = list(self.gtp_data.keys())

                for sheet_name in gtp_sheets:
                    if 'грп' in sheet_name.lower() or 'гидроразрыв' in sheet_name.lower():
                        result['fracking_data'] = self.gtp_data[sheet_name]
                        logger.info(f"Найдены данные о ГРП в листе {sheet_name}")
                        break

            logger.info("Данные для модуля 5 успешно подготовлены")
            return result

        except Exception as e:
            logger.error(f"Ошибка при подготовке данных для модуля 5: {str(e)}")
            # Возвращаем минимальный набор данных
            return {
                'gdi_reint_data': self.gdi_reint_data,
                'gtp_data': self.gtp_data
            }

    def get_data_for_filter_reduction(self):
        """
        Подготовка данных для модуля 6 (Подбор к-та уменьшения работающей части фильтра).

        Returns:
            dict: Данные для модуля 6
        """
        logger.info("Подготовка данных для модуля 6 (уменьшение работающей части фильтра)")

        try:
            # Для подбора коэффициента уменьшения работающей части фильтра нам нужны данные
            # о горизонтальных скважинах, эффективной проходке ствола и скин-факторе

            result = {
                'gdi_reint_data': self.gdi_reint_data,
                'arcgis_data': self.arcgis_data  # Данные о фонде скважин (для определения горизонтальных)
            }

            # Предобработка данных фонда скважин для выделения горизонтальных скважин
            if isinstance(self.arcgis_data, pd.DataFrame):
                # Ищем столбцы, которые могут содержать информацию о типе скважины
                type_columns = [col for col in self.arcgis_data.columns
                                if 'тип' in col.lower() or 'type' in col.lower()
                                or 'конструкц' in col.lower()]

                if type_columns and len(type_columns) > 0:
                    # Попытка выделить горизонтальные скважины
                    horizontal_wells = []

                    for col in type_columns:
                        if 'гор' in self.arcgis_data[col].astype(str).str.lower().values:
                            # Нашли столбец с информацией о горизонтальных скважинах
                            horizontal_mask = self.arcgis_data[col].astype(str).str.lower().str.contains('гор')
                            horizontal_wells = self.arcgis_data[horizontal_mask]
                            result['horizontal_wells'] = horizontal_wells
                            logger.info(f"Выделено {len(horizontal_wells)} горизонтальных скважин")
                            break

                # Предобработка данных ГДИ для выделения информации о фильтрах
                if isinstance(self.gdi_reint_data, dict):
                    # Попытка найти лист с данными о фильтрах или эффективной проходке
                    gdi_reint_sheets = list(self.gdi_reint_data.keys())

                    for sheet_name in gdi_reint_sheets:
                        if 'фильтр' in sheet_name.lower() or 'проходк' in sheet_name.lower():
                            result['filter_data'] = self.gdi_reint_data[sheet_name]
                            logger.info(f"Найдены данные о фильтрах скважин в листе {sheet_name}")
                            break

            logger.info("Данные для модуля 6 успешно подготовлены")
            return result

        except Exception as e:
            logger.error(f"Ошибка при подготовке данных для модуля 6: {str(e)}")
            # Возвращаем минимальный набор данных
            return {
                'gdi_reint_data': self.gdi_reint_data,
                'arcgis_data': self.arcgis_data
            }

    def get_data_for_fracture_length(self):
        """
        Подготовка данных для модуля 7 (Подбор коэффициентов для расчета полудлин трещин).

        Returns:
            dict: Данные для модуля 7
        """
        logger.info("Подготовка данных для модуля 7 (расчет полудлин трещин)")

        try:
            # Для подбора коэффициентов расчета полудлин трещин нам нужны данные
            # о закачках воды и параметрах трещин ГРП

            result = {
                'gtp_data': self.gtp_data,  # Данные о ГТР (могут содержать информацию о ГРП)
                'gdi_reint_data': self.gdi_reint_data  # Данные о параметрах трещин
            }

            # Предобработка данных ГТР
            if isinstance(self.gtp_data, dict):
                # Попытка найти лист с данными о ГРП или закачках воды
                gtp_sheets = list(self.gtp_data.keys())

                for sheet_name in gtp_sheets:
                    # Ищем информацию о ГРП
                    if 'грп' in sheet_name.lower() or 'гидроразрыв' in sheet_name.lower():
                        result['fracking_data'] = self.gtp_data[sheet_name]
                        logger.info(f"Найдены данные о ГРП в листе {sheet_name}")
                        break

                    # Или информацию о закачках
                    if 'закачк' in sheet_name.lower() or 'объем' in sheet_name.lower():
                        result['injection_data'] = self.gtp_data[sheet_name]
                        logger.info(f"Найдены данные о закачках в листе {sheet_name}")
                        break

            # Предобработка данных ГДИ
            if isinstance(self.gdi_reint_data, dict):
                # Попытка найти лист с данными о трещинах
                gdi_reint_sheets = list(self.gdi_reint_data.keys())

                for sheet_name in gdi_reint_sheets:
                    if 'трещин' in sheet_name.lower() or 'fracture' in sheet_name.lower():
                        result['fracture_data'] = self.gdi_reint_data[sheet_name]
                        logger.info(f"Найдены данные о трещинах в листе {sheet_name}")
                        break

            logger.info("Данные для модуля 7 успешно подготовлены")
            return result

        except Exception as e:
            logger.error(f"Ошибка при подготовке данных для модуля 7: {str(e)}")
            # Возвращаем минимальный набор данных
            return {
                'gtp_data': self.gtp_data,
                'gdi_reint_data': self.gdi_reint_data
            }

    def get_data_for_production_wells(self):
        """
        Подготовка данных для модуля 8 (Расчет добывающих скважин).

        Returns:
            dict: Данные для модуля 8
        """
        logger.info("Подготовка данных для модуля 8 (расчет добывающих скважин)")

        try:
            # Для комплексного расчета параметров добывающих скважин нам нужны
            # все доступные данные о скважинах

            result = {
                'pzab_data': self.pzab_data,  # Данные о забойном давлении
                'ppl_data': self.ppl_data,  # Данные о пластовом давлении
                'arcgis_data': self.arcgis_data,  # Данные о фонде скважин
                'gdi_vnr_data': self.gdi_vnr_data,  # Данные о ГДИ ВНР
                'gdi_reint_data': self.gdi_reint_data,  # Данные о переинтерпретации ГДИ
                'gtp_data': self.gtp_data,  # Данные о ГТР
                'nnt_ngt_data': self.nnt_ngt_data  # Данные о нефтенасыщенных толщинах
            }

            # Предобработка данных о фонде скважин для выделения добывающих скважин
            if isinstance(self.arcgis_data, pd.DataFrame):
                # Ищем столбцы, которые могут содержать информацию о назначении скважины
                purpose_columns = [col for col in self.arcgis_data.columns
                                   if 'назнач' in col.lower() or 'purpose' in col.lower()
                                   or 'тип' in col.lower()]

                if purpose_columns and len(purpose_columns) > 0:
                    # Попытка выделить добывающие скважины
                    production_wells = []

                    for col in purpose_columns:
                        if 'добыв' in ' '.join(self.arcgis_data[col].astype(str).values).lower():
                            # Нашли столбец с информацией о добывающих скважинах
                            production_mask = self.arcgis_data[col].astype(str).str.lower().str.contains(
                                'добыв')
                            production_wells = self.arcgis_data[production_mask]
                            result['production_wells'] = production_wells
                            logger.info(f"Выделено {len(production_wells)} добывающих скважин")
                            break

            # Предобработка данных о дебитах (могут содержаться в разных файлах)
            # Сначала проверяем данные из ArcGIS
            if isinstance(self.arcgis_data, pd.DataFrame):
                # Ищем столбцы с информацией о дебитах
                flow_columns = [col for col in self.arcgis_data.columns
                                if 'дебит' in col.lower() or 'flow' in col.lower()
                                or 'rate' in col.lower()]

                if flow_columns and len(flow_columns) > 0:
                    result['flow_rate_data'] = self.arcgis_data[flow_columns]
                    logger.info(f"Найдены данные о дебитах в файле ArcGIS: {flow_columns}")

            # Также проверяем данные ГДИ
            if isinstance(self.gdi_reint_data, dict):
                # Попытка найти лист с данными о дебитах
                gdi_reint_sheets = list(self.gdi_reint_data.keys())

                for sheet_name in gdi_reint_sheets:
                    if 'дебит' in sheet_name.lower() or 'добыч' in sheet_name.lower():
                        result['flow_rate_gdi_data'] = self.gdi_reint_data[sheet_name]
                        logger.info(f"Найдены данные о дебитах в листе {sheet_name} файла ГДИ")
                        break

            logger.info("Данные для модуля 8 успешно подготовлены")
            return result

        except Exception as e:
            logger.error(f"Ошибка при подготовке данных для модуля 8: {str(e)}")
            # Возвращаем минимальный набор данных
            return {
                'pzab_data': self.pzab_data,
                'ppl_data': self.ppl_data,
                'arcgis_data': self.arcgis_data,
                'gdi_reint_data': self.gdi_reint_data
            }

    def view_data_sample(self, dataset_name, n_rows=5):
        """
        Просмотр образца данных из загруженного датасета.

        Args:
            dataset_name (str): Название датасета ('pzab', 'ppl', ...)
            n_rows (int): Количество строк для отображения

        Returns:
            pd.DataFrame: Образец данных
        """
        datasets = {
            'pzab': self.pzab_data,
            'ppl': self.ppl_data,
            'arcgis': self.arcgis_data,
            'gdi_vnr': self.gdi_vnr_data,
            'gdi_reint': self.gdi_reint_data,
            'gtp': self.gtp_data,
            'nnt_ngt': self.nnt_ngt_data
        }

        if dataset_name not in datasets:
            logger.error(f"Датасет '{dataset_name}' не найден")
            return None

        data = datasets[dataset_name]

        if data is None:
            logger.warning(f"Датасет '{dataset_name}' не загружен")
            return None

        # Если данные представлены словарем (несколько листов Excel)
        if isinstance(data, dict):
            result = {}
            for sheet_name, sheet_data in data.items():
                result[sheet_name] = sheet_data.head(n_rows)
            return result

        return data.head(n_rows)

    def get_column_statistics(self, dataset_name, column_name):
        """
        Получение статистики по заданному столбцу данных.

        Args:
            dataset_name (str): Название датасета ('pzab', 'ppl', ...)
            column_name (str): Название столбца

        Returns:
            dict: Статистика по столбцу (среднее, мин, макс и т.д.)
        """
        datasets = {
            'pzab': self.pzab_data,
            'ppl': self.ppl_data,
            'arcgis': self.arcgis_data,
            'gdi_vnr': self.gdi_vnr_data,
            'gdi_reint': self.gdi_reint_data,
            'gtp': self.gtp_data,
            'nnt_ngt': self.nnt_ngt_data
        }

        if dataset_name not in datasets:
            logger.error(f"Датасет '{dataset_name}' не найден")
            return None

        data = datasets[dataset_name]

        if data is None:
            logger.warning(f"Датасет '{dataset_name}' не загружен")
            return None

        # Если данные представлены словарем (несколько листов Excel)
        if isinstance(data, dict):
            # Ищем столбец в каждом листе
            for sheet_name, sheet_data in data.items():
                if column_name in sheet_data.columns:
                    # Вычисляем статистику для числовых данных
                    if np.issubdtype(sheet_data[column_name].dtype, np.number):
                        stats = {
                            'sheet': sheet_name,
                            'mean': sheet_data[column_name].mean(),
                            'median': sheet_data[column_name].median(),
                            'min': sheet_data[column_name].min(),
                            'max': sheet_data[column_name].max(),
                            'std': sheet_data[column_name].std(),
                            'null_count': sheet_data[column_name].isnull().sum(),
                            'non_null_count': sheet_data[column_name].count()
                        }
                        return stats
                    else:
                        # Статистика для нечисловых данных
                        value_counts = sheet_data[column_name].value_counts().to_dict()
                        stats = {
                            'sheet': sheet_name,
                            'unique_values': len(value_counts),
                            'top_value': sheet_data[column_name].mode()[0] if not sheet_data[
                                column_name].empty else None,
                            'null_count': sheet_data[column_name].isnull().sum(),
                            'non_null_count': sheet_data[column_name].count(),
                            'value_counts': value_counts
                        }
                        return stats

            logger.error(f"Столбец '{column_name}' не найден ни в одном листе датасета '{dataset_name}'")
            return None
        else:
            # Проверяем наличие столбца в данных
            if column_name not in data.columns:
                logger.error(f"Столбец '{column_name}' не найден в датасете '{dataset_name}'")
                return None

            # Вычисляем статистику для числовых данных
            if np.issubdtype(data[column_name].dtype, np.number):
                stats = {
                    'mean': data[column_name].mean(),
                    'median': data[column_name].median(),
                    'min': data[column_name].min(),
                    'max': data[column_name].max(),
                    'std': data[column_name].std(),
                    'null_count': data[column_name].isnull().sum(),
                    'non_null_count': data[column_name].count()
                }
                return stats
            else:
                # Статистика для нечисловых данных
                value_counts = data[column_name].value_counts().to_dict()
                stats = {
                    'unique_values': len(value_counts),
                    'top_value': data[column_name].mode()[0] if not data[column_name].empty else None,
                    'null_count': data[column_name].isnull().sum(),
                    'non_null_count': data[column_name].count(),
                    'value_counts': value_counts
                }
                return stats

    def find_well_data(self, well_id):
        """
        Поиск данных по конкретной скважине во всех датасетах.

        Args:
            well_id (str): Идентификатор скважины

        Returns:
            dict: Данные по скважине из разных источников
        """
        logger.info(f"Поиск данных для скважины {well_id}")

        result = {}

        # Перебираем все датасеты и ищем данные о скважине
        # Сначала проверяем PZAB
        if isinstance(self.pzab_data, pd.DataFrame):
            # Ищем столбцы с идентификаторами скважин
            well_id_columns = [col for col in self.pzab_data.columns
                               if 'скв' in col.lower() or 'well' in col.lower()
                               or 'id' in col.lower()]

            for col in well_id_columns:
                # Проверяем, содержится ли идентификатор скважины в столбце
                if str(well_id) in self.pzab_data[col].astype(str).values:
                    # Нашли скважину в датасете
                    well_mask = self.pzab_data[col].astype(str) == str(well_id)
                    result['pzab_data'] = self.pzab_data[well_mask]
                    logger.info(f"Найдены данные о скважине {well_id} в датасете PZAB")
                    break

        # Проверяем PPL
        if isinstance(self.ppl_data, pd.DataFrame):
            # Аналогичный поиск
            well_id_columns = [col for col in self.ppl_data.columns
                               if 'скв' in col.lower() or 'well' in col.lower()
                               or 'id' in col.lower()]

            for col in well_id_columns:
                if str(well_id) in self.ppl_data[col].astype(str).values:
                    well_mask = self.ppl_data[col].astype(str) == str(well_id)
                    result['ppl_data'] = self.ppl_data[well_mask]
                    logger.info(f"Найдены данные о скважине {well_id} в датасете PPL")
                    break
        elif isinstance(self.ppl_data, dict):
            # Если данные в виде словаря с листами, проверяем каждый лист
            for sheet_name, sheet_data in self.ppl_data.items():
                well_id_columns = [col for col in sheet_data.columns
                                   if 'скв' in col.lower() or 'well' in col.lower()
                                   or 'id' in col.lower()]

                for col in well_id_columns:
                    if str(well_id) in sheet_data[col].astype(str).values:
                        well_mask = sheet_data[col].astype(str) == str(well_id)
                        if 'ppl_data' not in result:
                            result['ppl_data'] = {}
                        result['ppl_data'][sheet_name] = sheet_data[well_mask]
                        logger.info(f"Найдены данные о скважине {well_id} в листе {sheet_name} датасета PPL")
                        break

        # Аналогично проверяем остальные датасеты
        # ...

        # Проверяем ArcGIS
        if isinstance(self.arcgis_data, pd.DataFrame):
            well_id_columns = [col for col in self.arcgis_data.columns
                               if 'скв' in col.lower() or 'well' in col.lower()
                               or 'id' in col.lower()]

            for col in well_id_columns:
                if str(well_id) in self.arcgis_data[col].astype(str).values:
                    well_mask = self.arcgis_data[col].astype(str) == str(well_id)
                    result['arcgis_data'] = self.arcgis_data[well_mask]
                    logger.info(f"Найдены данные о скважине {well_id} в датасете ArcGIS")
                    break

        # И так далее для других датасетов...

        if not result:
            logger.warning(f"Данные о скважине {well_id} не найдены ни в одном датасете")

        return result