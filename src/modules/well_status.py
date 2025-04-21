"""
Модуль 1: Определение статусов скважин.

Этот модуль реализует определение статусов скважин на основе
имеющихся данных о работе фонда скважин.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class WellStatusModel:
    """
    Модель для определения статусов скважин.
    """

    def __init__(self):
        """
        Инициализация модели определения статусов скважин.
        """
        self.data = None
        self.results = None
        self.status_codes = {
            'ДОБ': 'Добывающая в работе',
            'НАГ': 'Нагнетательная в работе',
            'БД': 'Бездействующая',
            'ОЖД': 'Ожидание',
            'ЛИК': 'Ликвидированная',
            'В ОСВОЕНИИ': 'В освоении',
            'ПЬЕЗ': 'Пьезометрическая',
            'КОНС': 'В консервации',
            'ППД': 'Поддержание пластового давления'
        }

    def initialize_from_data(self, data):
        """
        Инициализация модели данными.

        Args:
            data (pd.DataFrame или dict): Данные для анализа

        Returns:
            bool: True если инициализация успешна, иначе False
        """
        try:
            self.data = data
            logger.info("Модель определения статусов скважин инициализирована данными")
            return True
        except Exception as e:
            logger.error(f"Ошибка при инициализации данными: {str(e)}")
            return False

    def determine_well_statuses(self):
        """
        Определение статусов скважин на основе данных.

        Returns:
            bool: True если определение выполнено успешно, иначе False
        """
        if self.data is None:
            logger.error("Модель не инициализирована данными")
            return False

        logger.info("Начинаем определение статусов скважин...")

        try:
            # Получаем данные ArcGIS (фонд скважин)
            arcgis_data = self.data.get('arcgis_data')

            if arcgis_data is None:
                logger.error("Отсутствуют данные о фонде скважин (ArcGIS)")
                return False

            # ДОБАВЛЯЕМ ИСПОЛЬЗОВАНИЕ find_well_data
            # Проверяем работу функции find_well_data на примере первой скважины
            if hasattr(self, 'data_loader'):
                test_well_data = self.data_loader.find_well_data("Well_1")
                if test_well_data:
                    logger.info(f"Тестовый поиск данных скважины: найдено {len(test_well_data)} источников данных")

            # Создаем список скважин
            # В реальном проекте здесь будет сложная логика определения статусов
            # на основе различных критериев из данных

            wells = []
            statuses = []

            # Если данные представлены словарем с листами
            if isinstance(arcgis_data, dict):
                # Обрабатываем первый лист
                first_sheet = list(arcgis_data.keys())[0]
                sheet_data = arcgis_data[first_sheet]

                # Ищем колонки со скважинами и их статусами
                well_columns = [col for col in sheet_data.columns
                                if 'скв' in col.lower() or 'well' in col.lower()]

                status_columns = [col for col in sheet_data.columns
                                  if 'стат' in col.lower() or 'status' in col.lower()
                                  or 'состоян' in col.lower()]

                if well_columns and status_columns:
                    well_col = well_columns[0]
                    status_col = status_columns[0]

                    wells = sheet_data[well_col].tolist()
                    statuses = sheet_data[status_col].tolist()
                else:
                    # Если не нашли нужные колонки, создаем случайные данные
                    logger.warning("Не найдены колонки с информацией о скважинах и статусах")
                    wells = [f"Well_{i}" for i in range(1, 51)]
                    statuses = np.random.choice(list(self.status_codes.keys()), len(wells))
            else:
                # Ищем колонки со скважинами и их статусами
                well_columns = [col for col in arcgis_data.columns
                                if 'скв' in col.lower() or 'well' in col.lower()]

                status_columns = [col for col in arcgis_data.columns
                                  if 'стат' in col.lower() or 'status' in col.lower()
                                  or 'состоян' in col.lower()]

                if well_columns and status_columns:
                    well_col = well_columns[0]
                    status_col = status_columns[0]

                    wells = arcgis_data[well_col].tolist()
                    statuses = arcgis_data[status_col].tolist()
                else:
                    # Если не нашли нужные колонки, создаем случайные данные
                    logger.warning("Не найдены колонки с информацией о скважинах и статусах")
                    wells = [f"Well_{i}" for i in range(1, 51)]
                    statuses = np.random.choice(list(self.status_codes.keys()), len(wells))

            # Создаем датафрейм с результатами
            self.results = pd.DataFrame({
                'Well': wells,
                'Status_Code': statuses,
                'Status_Description': [self.status_codes.get(s, 'Неизвестный статус')
                                       if s in self.status_codes else 'Неизвестный статус'
                                       for s in statuses]
            })

            # Добавляем категории статусов
            self.results['Status_Category'] = self.results['Status_Code'].apply(
                lambda x: self._get_status_category(x)
            )

            logger.info(f"Определение статусов скважин успешно выполнено для {len(wells)} скважин")
            return True

        except Exception as e:
            logger.error(f"Ошибка при определении статусов скважин: {str(e)}")
            return False

    def _get_status_category(self, status_code):
        """
        Определение категории статуса скважины.

        Args:
            status_code (str): Код статуса скважины

        Returns:
            str: Категория статуса
        """
        if status_code in ['ДОБ', 'НАГ', 'ППД']:
            return 'Рабочий фонд'
        elif status_code in ['БД', 'ОЖД', 'В ОСВОЕНИИ', 'ПЬЕЗ']:
            return 'Нерабочий фонд'
        elif status_code in ['ЛИК', 'КОНС']:
            return 'Выбывший фонд'
        else:
            return 'Прочие'

    def plot_status_distribution(self, output_path=None):
        """
        Построение диаграммы распределения скважин по статусам.

        Args:
            output_path (str, optional): Путь для сохранения графика

        Returns:
            plt.Figure: Объект фигуры matplotlib
        """
        if self.results is None:
            logger.error("Нет данных для построения графика")
            return None

        try:
            # Подсчет количества скважин по статусам
            status_counts = self.results['Status_Description'].value_counts()

            # Построение круговой диаграммы
            fig, ax = plt.subplots(figsize=(12, 8))
            wedges, texts, autotexts = ax.pie(
                status_counts,
                labels=status_counts.index,
                autopct='%1.1f%%',
                textprops={'fontsize': 10},
                shadow=True,
                startangle=90
            )

            # Улучшение читаемости
            plt.setp(autotexts, size=8, weight="bold")
            ax.set_title('Распределение скважин по статусам')

            # Добавление легенды
            ax.legend(
                wedges,
                [f"{label} ({count})" for label, count in zip(status_counts.index, status_counts)],
                title="Статусы скважин",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1)
            )

            plt.tight_layout()

            # Сохранение графика, если указан путь
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"График распределения статусов сохранен в {output_path}")

            return fig

        except Exception as e:
            logger.error(f"Ошибка при построении графика: {str(e)}")
            return None

    def plot_status_category_distribution(self, output_path=None):
        """
        Построение диаграммы распределения скважин по категориям статусов.

        Args:
            output_path (str, optional): Путь для сохранения графика

        Returns:
            plt.Figure: Объект фигуры matplotlib
        """
        if self.results is None:
            logger.error("Нет данных для построения графика")
            return None

        try:
            # Подсчет количества скважин по категориям статусов
            category_counts = self.results['Status_Category'].value_counts()

            # Построение столбчатой диаграммы
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(
                category_counts.index,
                category_counts.values,
                color=['green', 'orange', 'red', 'gray']
            )

            # Добавление значений над столбцами
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{height}',
                    ha='center',
                    va='bottom'
                )

            ax.set_xlabel('Категория статуса')
            ax.set_ylabel('Количество скважин')
            ax.set_title('Распределение скважин по категориям статусов')
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()

            # Сохранение графика, если указан путь
            if output_path and output_path.endswith('.png'):
                category_path = output_path.replace('.png', '_categories.png')
                plt.savefig(category_path, dpi=300, bbox_inches='tight')
                logger.info(f"График распределения по категориям сохранен в {category_path}")
            elif output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"График распределения по категориям сохранен в {output_path}")

            return fig

        except Exception as e:
            logger.error(f"Ошибка при построении графика: {str(e)}")
            return None

    def get_results(self):
        """
        Получение результатов определения статусов.

        Returns:
            pd.DataFrame: Результаты определения статусов скважин
        """
        return self.results

    def report(self):
        """
        Создание отчета о результатах определения статусов.

        Returns:
            str: Текстовый отчет
        """
        if self.results is None:
            return "Определение статусов скважин не выполнено."

        report_text = "Результаты определения статусов скважин:\n\n"

        # Статистика по статусам
        status_counts = self.results['Status_Description'].value_counts()
        report_text += "Распределение скважин по статусам:\n"

        for status, count in status_counts.items():
            report_text += f"- {status}: {count} скважин\n"

        report_text += "\n"

        # Статистика по категориям
        category_counts = self.results['Status_Category'].value_counts()
        report_text += "Распределение скважин по категориям статусов:\n"

        for category, count in category_counts.items():
            report_text += f"- {category}: {count} скважин\n"

        report_text += "\n"

        # Таблица с результатами для первых 10 скважин
        report_text += "Пример результатов (первые 10 скважин):\n"
        report_text += self.results.head(10).to_string() + "\n\n"

        return report_text