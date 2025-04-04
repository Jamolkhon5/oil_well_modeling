"""
Модуль 3: Граничные условия измения при автоматическом расчете Рпл в нефтяных скважинах.

Этот модуль реализует расчет пластового давления с учетом граничных условий.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class PressureCalculationModel:
    """
    Модель для расчета пластового давления в нефтяных скважинах.
    """

    def __init__(self, initial_params=None):
        """
        Инициализация модели с начальными параметрами.

        Args:
            initial_params (dict, optional): Начальные параметры модели
        """
        self.params = initial_params if initial_params else {}
        self.data = None
        self.results = None

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
            logger.info("Модель расчета пластового давления инициализирована данными")
            return True
        except Exception as e:
            logger.error(f"Ошибка при инициализации данными: {str(e)}")
            return False

    def calculate_pressures(self):
        """
        Расчет пластовых давлений с учетом граничных условий.

        Returns:
            bool: True если расчет выполнен успешно, иначе False
        """
        if self.data is None:
            logger.error("Модель не инициализирована данными")
            return False

        logger.info("Начинаем расчет пластовых давлений...")

        try:
            # Здесь будет код для расчета пластовых давлений
            # в соответствии с методикой из документа

            # Для примера создадим случайные данные
            # В реальном проекте здесь будет сложный расчет по методике

            # Создаем фиктивный результат для демонстрации
            wells = [f"Well_{i}" for i in range(1, 11)]
            pressure_initial = np.random.uniform(200, 250, len(wells))
            pressure_calculated = np.random.uniform(180, 230, len(wells))

            self.results = pd.DataFrame({
                'Well': wells,
                'Initial_Pressure': pressure_initial,
                'Calculated_Pressure': pressure_calculated,
                'Difference': pressure_initial - pressure_calculated
            })

            logger.info("Расчет пластовых давлений успешно выполнен")
            return True

        except Exception as e:
            logger.error(f"Ошибка при расчете пластовых давлений: {str(e)}")
            return False

    def apply_boundary_conditions(self):
        """
        Применение граничных условий к расчетным значениям давления.

        Returns:
            bool: True если применение выполнено успешно, иначе False
        """
        if self.results is None:
            logger.error("Сначала выполните расчет давлений")
            return False

        logger.info("Применяем граничные условия к расчетным значениям давления...")

        try:
            # Здесь будет код для применения граничных условий
            # в соответствии с методикой из документа

            # Для примера просто скорректируем некоторые значения
            # В реальном проекте здесь будет логика из документа

            # Создаем новый столбец для скорректированных давлений
            self.results['Adjusted_Pressure'] = self.results['Calculated_Pressure'].copy()

            # Пример применения граничного условия:
            # если разница между начальным и расчетным давлением больше 15,
            # то корректируем расчетное давление
            mask = self.results['Difference'] > 15
            self.results.loc[mask, 'Adjusted_Pressure'] = (
                    self.results.loc[mask, 'Initial_Pressure'] - 15
            )

            # Добавляем информацию о примененных ограничениях
            self.results['Boundary_Applied'] = mask

            logger.info("Граничные условия успешно применены")
            return True

        except Exception as e:
            logger.error(f"Ошибка при применении граничных условий: {str(e)}")
            return False

    def plot_pressure_changes(self, output_path=None):
        """
        Построение графика изменения давлений.

        Args:
            output_path (str, optional): Путь для сохранения графика

        Returns:
            plt.Figure: Объект фигуры matplotlib
        """
        if self.results is None:
            logger.error("Нет данных для построения графика")
            return None

        try:
            # Построение графика
            fig, ax = plt.subplots(figsize=(12, 6))

            wells = self.results['Well']
            initial = self.results['Initial_Pressure']
            calculated = self.results['Calculated_Pressure']
            adjusted = self.results['Adjusted_Pressure']

            x = np.arange(len(wells))
            width = 0.25

            ax.bar(x - width, initial, width, label='Начальное давление')
            ax.bar(x, calculated, width, label='Рассчитанное давление')
            ax.bar(x + width, adjusted, width, label='Скорректированное давление')

            ax.set_xlabel('Скважины')
            ax.set_ylabel('Давление, атм')
            ax.set_title('Изменение пластового давления в скважинах')
            ax.set_xticks(x)
            ax.set_xticklabels(wells, rotation=45)
            ax.legend()

            plt.tight_layout()

            # Сохранение графика, если указан путь
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"График сохранен в {output_path}")

            return fig

        except Exception as e:
            logger.error(f"Ошибка при построении графика: {str(e)}")
            return None

    def get_results(self):
        """
        Получение результатов расчета.

        Returns:
            pd.DataFrame: Результаты расчета пластовых давлений
        """
        return self.results

    def report(self):
        """
        Создание отчета о результатах расчета.

        Returns:
            str: Текстовый отчет
        """
        if self.results is None:
            return "Расчет пластовых давлений не выполнен."

        report_text = "Результаты расчета пластовых давлений с учетом граничных условий:\n\n"

        # Статистика по всем скважинам
        report_text += "Общая статистика:\n"
        report_text += f"- Количество скважин: {len(self.results)}\n"
        report_text += f"- Среднее начальное давление: {self.results['Initial_Pressure'].mean():.2f} атм\n"
        report_text += f"- Среднее рассчитанное давление: {self.results['Calculated_Pressure'].mean():.2f} атм\n"
        report_text += f"- Среднее скорректированное давление: {self.results['Adjusted_Pressure'].mean():.2f} атм\n"
        report_text += f"- Количество скважин с примененными граничными условиями: {self.results['Boundary_Applied'].sum()}\n\n"

        # Таблица с результатами для первых 5 скважин
        report_text += "Пример результатов (первые 5 скважин):\n"
        report_text += self.results.head().to_string() + "\n\n"

        return report_text