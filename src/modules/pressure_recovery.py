"""
Модуль 4: Подбор времени восстановления давления в остановленных скважинах.

Этот модуль реализует расчет времени восстановления давления
в остановленных скважинах.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from src.utils import calculate_pressure_recovery_time

logger = logging.getLogger(__name__)


class PressureRecoveryModel:
    """
    Модель для расчета времени восстановления давления в остановленных скважинах.
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
            logger.info("Модель расчета времени восстановления давления инициализирована данными")
            return True
        except Exception as e:
            logger.error(f"Ошибка при инициализации данными: {str(e)}")
            return False

    def calculate_recovery_times(self):
        """
        Расчет времени восстановления давления в остановленных скважинах.

        Returns:
            bool: True если расчет выполнен успешно, иначе False
        """
        if self.data is None:
            logger.error("Модель не инициализирована данными")
            return False

        logger.info("Начинаем расчет времени восстановления давления...")

        try:
            # Здесь будет код для расчета времени восстановления давления
            # в соответствии с методикой из документа

            # Для примера создадим случайные данные
            # В реальном проекте здесь будет сложный расчет по методике

            # Создаем фиктивный набор данных для скважин
            wells = [f"Well_{i}" for i in range(1, 11)]

            # Параметры скважин
            permeability = np.random.uniform(10, 100, len(wells))  # Проницаемость, мД
            porosity = np.random.uniform(0.1, 0.3, len(wells))  # Пористость, д.ед.
            viscosity = np.random.uniform(1, 10, len(wells))  # Вязкость, сПз
            skin_factor = np.random.uniform(-2, 5, len(wells))  # Скин-фактор
            well_radius = np.ones(len(wells)) * 0.1  # Радиус скважины, м

            # Расчет времени восстановления для каждой скважины
            recovery_times = np.array([
                calculate_pressure_recovery_time(
                    permeability[i],
                    porosity[i],
                    viscosity[i],
                    skin_factor[i],
                    well_radius[i]
                )
                for i in range(len(wells))
            ])

            # Создаем датафрейм с результатами
            self.results = pd.DataFrame({
                'Well': wells,
                'Permeability': permeability,
                'Porosity': porosity,
                'Viscosity': viscosity,
                'Skin_Factor': skin_factor,
                'Well_Radius': well_radius,
                'Recovery_Time': recovery_times
            })

            logger.info("Расчет времени восстановления давления успешно выполнен")
            return True

        except Exception as e:
            logger.error(f"Ошибка при расчете времени восстановления давления: {str(e)}")
            return False

    def plot_recovery_times(self, output_path=None):
        """
        Построение графика времени восстановления давления.

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
            recovery_times = self.results['Recovery_Time']

            # Сортируем скважины по времени восстановления
            sorted_indices = np.argsort(recovery_times)
            sorted_wells = [wells[i] for i in sorted_indices]
            sorted_times = [recovery_times[i] for i in sorted_indices]

            # Строим гистограмму
            ax.bar(range(len(sorted_wells)), sorted_times, color='skyblue')

            ax.set_xlabel('Скважины')
            ax.set_ylabel('Время восстановления, сут.')
            ax.set_title('Время восстановления давления в остановленных скважинах')
            ax.set_xticks(range(len(sorted_wells)))
            ax.set_xticklabels(sorted_wells, rotation=45)

            # Добавляем значения над столбцами
            for i, v in enumerate(sorted_times):
                ax.text(i, v + 0.1, f"{v:.1f}", ha='center')

            plt.tight_layout()

            # Сохранение графика, если указан путь
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"График сохранен в {output_path}")

            return fig

        except Exception as e:
            logger.error(f"Ошибка при построении графика: {str(e)}")
            return None

    def plot_parameters_influence(self, output_path=None):
        """
        Построение графиков влияния параметров на время восстановления.

        Args:
            output_path (str, optional): Путь для сохранения графика

        Returns:
            plt.Figure: Объект фигуры matplotlib
        """
        if self.results is None:
            logger.error("Нет данных для построения графика")
            return None

        try:
            # Создаем графики зависимостей
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))

            # Зависимость от проницаемости
            axs[0, 0].scatter(self.results['Permeability'], self.results['Recovery_Time'])
            axs[0, 0].set_xlabel('Проницаемость, мД')
            axs[0, 0].set_ylabel('Время восстановления, сут.')
            axs[0, 0].set_title('Зависимость от проницаемости')
            axs[0, 0].grid(True)

            # Зависимость от пористости
            axs[0, 1].scatter(self.results['Porosity'], self.results['Recovery_Time'])
            axs[0, 1].set_xlabel('Пористость, д.ед.')
            axs[0, 1].set_ylabel('Время восстановления, сут.')
            axs[0, 1].set_title('Зависимость от пористости')
            axs[0, 1].grid(True)

            # Зависимость от вязкости
            axs[1, 0].scatter(self.results['Viscosity'], self.results['Recovery_Time'])
            axs[1, 0].set_xlabel('Вязкость, сПз')
            axs[1, 0].set_ylabel('Время восстановления, сут.')
            axs[1, 0].set_title('Зависимость от вязкости')
            axs[1, 0].grid(True)

            # Зависимость от скин-фактора
            axs[1, 1].scatter(self.results['Skin_Factor'], self.results['Recovery_Time'])
            axs[1, 1].set_xlabel('Скин-фактор')
            axs[1, 1].set_ylabel('Время восстановления, сут.')
            axs[1, 1].set_title('Зависимость от скин-фактора')
            axs[1, 1].grid(True)

            plt.tight_layout()

            # Сохранение графика, если указан путь
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"График зависимостей сохранен в {output_path}")

            return fig

        except Exception as e:
            logger.error(f"Ошибка при построении графиков влияния параметров: {str(e)}")
            return None

    def get_results(self):
        """
        Получение результатов расчета.

        Returns:
            pd.DataFrame: Результаты расчета времени восстановления давления
        """
        return self.results

    def report(self):
        """
        Создание отчета о результатах расчета.

        Returns:
            str: Текстовый отчет
        """
        if self.results is None:
            return "Расчет времени восстановления давления не выполнен."

        report_text = "Результаты расчета времени восстановления давления:\n\n"

        # Статистика по всем скважинам
        report_text += "Общая статистика:\n"
        report_text += f"- Количество скважин: {len(self.results)}\n"
        report_text += f"- Минимальное время восстановления: {self.results['Recovery_Time'].min():.2f} сут.\n"
        report_text += f"- Максимальное время восстановления: {self.results['Recovery_Time'].max():.2f} сут.\n"
        report_text += f"- Среднее время восстановления: {self.results['Recovery_Time'].mean():.2f} сут.\n\n"

        # Таблица с результатами для первых 5 скважин
        report_text += "Пример результатов (первые 5 скважин):\n"
        columns_to_show = ['Well', 'Permeability', 'Porosity', 'Viscosity', 'Skin_Factor', 'Recovery_Time']
        report_text += self.results[columns_to_show].head().to_string() + "\n\n"

        # Рекомендации по интерпретации результатов
        report_text += "Интерпретация результатов:\n"
        report_text += "- Время восстановления давления зависит от проницаемости, пористости, вязкости флюида и скин-фактора.\n"
        report_text += "- Скважины с высоким скин-фактором требуют больше времени для восстановления давления.\n"
        report_text += "- Скважины с низкой проницаемостью также требуют больше времени для восстановления давления.\n"

        return report_text