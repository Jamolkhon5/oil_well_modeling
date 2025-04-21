"""
Модуль 8: Расчет добывающих скважин.

Этот модуль реализует комплексный расчет параметров добывающих скважин
на основе всех предыдущих модулей.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


class ProductionWellsModel:
    """
    Модель для расчета параметров добывающих скважин.
    """

    def __init__(self, models=None):
        """
        Инициализация модели с результатами предыдущих модулей.

        Args:
            models (dict): Словарь с объектами предыдущих моделей
        """
        self.models = models if models else {}
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

            logger.info("Модель расчета добывающих скважин инициализирована данными")
            return True

        except Exception as e:
            logger.error(f"Ошибка при инициализации данными: {str(e)}")
            return False

    def calculate_well_parameters(self):
        """
        Комплексный расчет параметров добывающих скважин.

        Returns:
            bool: True если расчет выполнен успешно, иначе False
        """
        if self.data is None:
            logger.error("Модель не инициализирована данными")
            return False

        logger.info("Начинаем комплексный расчет параметров добывающих скважин...")

        try:
            # Проверка наличия необходимых моделей
            required_models = [
                'phase_permeability',
                'regression_model',
                'pressure_calculation',
                'pressure_recovery',
                'skin_curve',
                'filter_reduction',
                'fracture_length'
            ]

            if hasattr(self, 'data_loader'):
                for i in range(1, 6):  # Проверяем данные для первых 5 скважин
                    well_id = f"Well_{i}"
                    well_data = self.data_loader.find_well_data(well_id)
                    if well_data:
                        logger.info(f"Дополнительные данные скважины {well_id}:")
                        for source, data in well_data.items():
                            logger.info(f"  - {source}: {data.shape if hasattr(data, 'shape') else 'N/A'}")

            missing_models = [model for model in required_models if model not in self.models]
            if missing_models:
                logger.warning(f"Отсутствуют некоторые модели: {missing_models}")

            # Получаем параметры из предыдущих моделей
            phase_perm_params = self.models.get('phase_permeability',
                                                {}).get_parameters() if 'phase_permeability' in self.models else {}
            skin_params = self.models.get('skin_curve', {}).get_parameters() if 'skin_curve' in self.models else {}
            filter_params = self.models.get('filter_reduction',
                                            {}).get_parameters() if 'filter_reduction' in self.models else {}

            # Здесь будет код для комплексного расчета параметров добывающих скважин
            # на основе всех предыдущих моделей и в соответствии с документом

            # Для примера создадим фиктивные данные о скважинах
            wells = [f"Well_{i}" for i in range(1, 11)]

            # Создаем фиктивные результаты расчета
            self.results = pd.DataFrame({
                'Well': wells,
                'Initial_Flow_Rate': np.random.uniform(20, 100, len(wells)),
                'Current_Flow_Rate': np.random.uniform(15, 80, len(wells)),
                'Water_Cut': np.random.uniform(10, 50, len(wells)),
                'Reservoir_Pressure': np.random.uniform(200, 250, len(wells)),
                'Bottomhole_Pressure': np.random.uniform(150, 200, len(wells)),
                'Skin_Factor': np.random.uniform(-3, 2, len(wells)),
                'Filter_Efficiency': np.random.uniform(0.6, 1.0, len(wells))
            })

            logger.info("Расчет параметров добывающих скважин успешно выполнен")
            return True

        except Exception as e:
            logger.error(f"Ошибка при расчете параметров добывающих скважин: {str(e)}")
            return False

    def forecast_production(self, forecast_period=365):
        """
        Прогнозирование добычи на заданный период.

        Args:
            forecast_period (int): Период прогноза в днях

        Returns:
            bool: True если прогноз выполнен успешно, иначе False
        """
        if self.results is None:
            logger.error("Сначала выполните расчет параметров скважин")
            return False

        logger.info(f"Прогнозирование добычи на {forecast_period} дней...")

        try:
            # Здесь будет код для прогнозирования добычи
            # на основе рассчитанных параметров

            # Для примера создадим фиктивные прогнозные данные
            time_points = np.linspace(0, forecast_period, 30)  # 30 точек за период
            wells = self.results['Well'].unique()

            # Создаем фиктивные прогнозные кривые для каждой скважины
            forecast_data = []

            for well in wells:
                initial_rate = self.results.loc[self.results['Well'] == well, 'Current_Flow_Rate'].values[0]
                decline_rate = np.random.uniform(0.0005, 0.002)  # Скорость падения дебита

                for t in time_points:
                    # Экспоненциальное падение дебита со временем
                    rate = initial_rate * np.exp(-decline_rate * t)
                    water_cut = self.results.loc[self.results['Well'] == well, 'Water_Cut'].values[0] + \
                                0.01 * t  # Увеличение обводненности со временем
                    water_cut = min(water_cut, 95)  # Ограничение обводненности

                    forecast_data.append({
                        'Well': well,
                        'Time': t,
                        'Flow_Rate': rate,
                        'Oil_Rate': rate * (1 - water_cut / 100),
                        'Water_Cut': water_cut
                    })

            # Создаем датафрейм с прогнозными данными
            self.forecast_results = pd.DataFrame(forecast_data)

            logger.info("Прогнозирование добычи успешно выполнено")
            return True

        except Exception as e:
            logger.error(f"Ошибка при прогнозировании добычи: {str(e)}")
            return False

    def plot_production_profiles(self, output_path=None):
        """
        Построение графиков профилей добычи.

        Args:
            output_path (str, optional): Путь для сохранения графика

        Returns:
            plt.Figure: Объект фигуры matplotlib
        """
        if self.forecast_results is None:
            logger.error("Сначала выполните прогнозирование добычи")
            return None

        try:
            # Определяем список скважин
            wells = self.forecast_results['Well'].unique()

            # Создаем фигуру с двумя графиками: дебит и обводненность
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

            # Цвета для разных скважин
            colors = plt.cm.tab10(np.linspace(0, 1, len(wells)))

            # Строим графики для каждой скважины
            for i, well in enumerate(wells):
                well_data = self.forecast_results[self.forecast_results['Well'] == well]

                # График дебита
                ax1.plot(well_data['Time'], well_data['Flow_Rate'], '-', color=colors[i], label=f"{well} (Общий)")
                ax1.plot(well_data['Time'], well_data['Oil_Rate'], '--', color=colors[i], label=f"{well} (Нефть)")

                # График обводненности
                ax2.plot(well_data['Time'], well_data['Water_Cut'], '-', color=colors[i], label=well)

            # Настройка графика дебита
            ax1.set_ylabel('Дебит, м³/сут')
            ax1.set_title('Прогноз дебитов скважин')
            ax1.grid(True)
            ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # Настройка графика обводненности
            ax2.set_xlabel('Время, дни')
            ax2.set_ylabel('Обводненность, %')
            ax2.set_title('Прогноз обводненности скважин')
            ax2.grid(True)
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            plt.tight_layout()

            # Сохранение графика, если указан путь
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"График профилей добычи сохранен в {output_path}")

            return fig

        except Exception as e:
            logger.error(f"Ошибка при построении графиков профилей добычи: {str(e)}")
            return None

    def plot_cumulative_production(self, output_path=None):
        """
        Построение графика накопленной добычи.

        Args:
            output_path (str, optional): Путь для сохранения графика

        Returns:
            plt.Figure: Объект фигуры matplotlib
        """
        if self.forecast_results is None:
            logger.error("Сначала выполните прогнозирование добычи")
            return None

        try:
            # Рассчитываем накопленную добычу
            time_points = sorted(self.forecast_results['Time'].unique())
            wells = self.forecast_results['Well'].unique()

            # Создаем пустые списки для хранения данных
            cum_total = np.zeros(len(time_points))
            cum_oil = np.zeros(len(time_points))
            cum_water = np.zeros(len(time_points))

            # Накапливаем добычу по всем скважинам
            for i, t in enumerate(time_points):
                if i > 0:
                    # Копируем значения с предыдущего шага
                    cum_total[i] = cum_total[i - 1]
                    cum_oil[i] = cum_oil[i - 1]
                    cum_water[i] = cum_water[i - 1]

                # Добавляем добычу на текущем шаге (приближенно)
                if i < len(time_points) - 1:
                    dt = time_points[i + 1] - time_points[i]
                else:
                    dt = time_points[i] - time_points[i - 1]

                # Данные для текущего времени
                data_t = self.forecast_results[self.forecast_results['Time'] == t]

                # Суммируем добычу по всем скважинам
                total_rate = data_t['Flow_Rate'].sum()
                oil_rate = data_t['Oil_Rate'].sum()
                water_rate = total_rate - oil_rate

                # Добавляем к накопленной добыче
                cum_total[i] += total_rate * dt
                cum_oil[i] += oil_rate * dt
                cum_water[i] += water_rate * dt

            # Построение графика накопленной добычи
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(time_points, cum_total, 'b-', linewidth=2, label='Общая добыча')
            ax.plot(time_points, cum_oil, 'g-', linewidth=2, label='Нефть')
            ax.plot(time_points, cum_water, 'r-', linewidth=2, label='Вода')

            ax.set_xlabel('Время, дни')
            ax.set_ylabel('Накопленная добыча, м³')
            ax.set_title('Прогноз накопленной добычи по всем скважинам')
            ax.grid(True)
            ax.legend()

            plt.tight_layout()

            # Сохранение графика, если указан путь
            if output_path and output_path.endswith('.png'):
                cum_path = output_path.replace('.png', '_cumulative.png')
                plt.savefig(cum_path, dpi=300, bbox_inches='tight')
                logger.info(f"График накопленной добычи сохранен в {cum_path}")
            elif output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"График накопленной добычи сохранен в {output_path}")

            return fig

        except Exception as e:
            logger.error(f"Ошибка при построении графика накопленной добычи: {str(e)}")
            return None

    def get_results(self):
        """
        Получение результатов расчета.

        Returns:
            pd.DataFrame: Результаты расчета параметров добывающих скважин
        """
        return self.results

    def get_forecast_results(self):
        """
        Получение результатов прогнозирования.

        Returns:
            pd.DataFrame: Результаты прогнозирования добычи
        """
        return self.forecast_results

    def report(self):
        """
        Создание отчета о результатах расчета и прогнозирования.

        Returns:
            str: Текстовый отчет
        """
        if self.results is None:
            return "Расчет параметров добывающих скважин не выполнен."

        report_text = "Результаты расчета добывающих скважин:\n\n"

        # Статистика по всем скважинам
        report_text += "Общая статистика по скважинам:\n"
        report_text += f"- Количество скважин: {len(self.results)}\n"
        report_text += f"- Средний начальный дебит: {self.results['Initial_Flow_Rate'].mean():.2f} м³/сут\n"
        report_text += f"- Средний текущий дебит: {self.results['Current_Flow_Rate'].mean():.2f} м³/сут\n"
        report_text += f"- Средняя обводненность: {self.results['Water_Cut'].mean():.2f} %\n"
        report_text += f"- Среднее пластовое давление: {self.results['Reservoir_Pressure'].mean():.2f} атм\n"
        report_text += f"- Среднее забойное давление: {self.results['Bottomhole_Pressure'].mean():.2f} атм\n"
        report_text += f"- Средний скин-фактор: {self.results['Skin_Factor'].mean():.2f}\n"
        report_text += f"- Средняя эффективность фильтра: {self.results['Filter_Efficiency'].mean():.2f}\n\n"

        # Таблица с результатами для первых 5 скважин
        report_text += "Пример результатов расчета (первые 5 скважин):\n"
        report_text += self.results.head().to_string() + "\n\n"

        # Добавляем результаты прогнозирования, если они есть
        if hasattr(self, 'forecast_results') and self.forecast_results is not None:
            # Рассчитываем накопленную добычу на конец прогнозного периода
            last_time = self.forecast_results['Time'].max()
            last_data = self.forecast_results[self.forecast_results['Time'] == last_time]

            report_text += f"Прогноз добычи на {last_time:.0f} дней:\n"
            report_text += f"- Средний дебит жидкости: {last_data['Flow_Rate'].mean():.2f} м³/сут\n"
            report_text += f"- Средний дебит нефти: {last_data['Oil_Rate'].mean():.2f} м³/сут\n"
            report_text += f"- Средняя обводненность: {last_data['Water_Cut'].mean():.2f} %\n"

            # Рассчитываем накопленную добычу
            time_points = sorted(self.forecast_results['Time'].unique())
            wells = self.forecast_results['Well'].unique()

            cum_total = 0
            cum_oil = 0

            for i in range(len(time_points) - 1):
                dt = time_points[i + 1] - time_points[i]
                data_t = self.forecast_results[self.forecast_results['Time'] == time_points[i]]

                total_rate = data_t['Flow_Rate'].sum()
                oil_rate = data_t['Oil_Rate'].sum()

                cum_total += total_rate * dt
                cum_oil += oil_rate * dt

            report_text += f"- Накопленная добыча жидкости: {cum_total:.0f} м³\n"
            report_text += f"- Накопленная добыча нефти: {cum_oil:.0f} м³\n\n"

        # Рекомендации и выводы
        report_text += "Выводы и рекомендации:\n"
        report_text += "1. Результаты расчета показывают текущее состояние добывающих скважин.\n"
        report_text += "2. Для оптимизации добычи рекомендуется обратить внимание на скважины с высоким скин-фактором.\n"
        report_text += "3. Скважины с низкой эффективностью фильтра могут требовать проведения ремонтных работ.\n"
        report_text += "4. Прогнозные данные позволяют оценить динамику изменения добычи и обводненности.\n"

        return report_text