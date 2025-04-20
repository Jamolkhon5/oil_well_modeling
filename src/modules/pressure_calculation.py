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

    def apply_geological_boundaries(self):
        """
        Применение граничных условий по геологии пласта.

        Returns:
            bool: True если применение выполнено успешно, иначе False
        """
        if self.results is None:
            logger.error("Сначала выполните расчет давлений")
            return False

        logger.info("Применяем граничные условия по геологии пласта...")

        try:
            # Получаем геологические данные
            geo_data = self.data.get('nnt_ngt_data')

            if geo_data is None:
                logger.warning("Отсутствуют геологические данные, создаем синтетические границы")

                # Создаем синтетические геологические ограничения
                # В реальном проекте эти данные будут извлекаться из геологических моделей

                # Определяем верхнюю и нижнюю границы по геологии
                initial_pressure = self.results['Initial_Pressure'].mean()
                std_dev = self.results['Initial_Pressure'].std()

                upper_geo_limit = initial_pressure * 1.2  # 20% выше среднего
                lower_geo_limit = initial_pressure * 0.8  # 20% ниже среднего

                # В реальности здесь будет сложная логика определения граничных условий
                # на основе геологических данных пласта

            else:
                # Извлекаем границы из геологических данных
                # В зависимости от структуры данных это может быть реализовано по-разному

                if isinstance(geo_data, dict):
                    # Если данные представлены словарем с листами
                    # Ищем лист с данными о пластовом давлении
                    pressure_sheets = [sheet for sheet in geo_data.keys()
                                       if 'давлен' in sheet.lower() or 'pressure' in sheet.lower()]

                    if pressure_sheets:
                        pressure_sheet = pressure_sheets[0]
                        pressure_data = geo_data[pressure_sheet]

                        # Ищем колонки с граничными значениями
                        min_columns = [col for col in pressure_data.columns
                                       if 'min' in col.lower() or 'нижн' in col.lower()]

                        max_columns = [col for col in pressure_data.columns
                                       if 'max' in col.lower() or 'верхн' in col.lower()]

                        if min_columns and max_columns:
                            lower_geo_limit = pressure_data[min_columns[0]].mean()
                            upper_geo_limit = pressure_data[max_columns[0]].mean()
                        else:
                            # Если не нашли нужные колонки, используем статистику
                            logger.warning("Не найдены колонки с граничными значениями давления")
                            lower_geo_limit = pressure_data.iloc[:, 1].quantile(0.1)  # 10-й перцентиль
                            upper_geo_limit = pressure_data.iloc[:, 1].quantile(0.9)  # 90-й перцентиль
                    else:
                        # Если не нашли нужный лист, используем общую статистику
                        logger.warning("Не найден лист с данными о пластовом давлении")
                        initial_pressure = self.results['Initial_Pressure'].mean()
                        std_dev = self.results['Initial_Pressure'].std()

                        lower_geo_limit = initial_pressure - 2 * std_dev
                        upper_geo_limit = initial_pressure + 2 * std_dev
                else:
                    # Если данные представлены одним датафреймом
                    # Ищем колонки с граничными значениями
                    min_columns = [col for col in geo_data.columns
                                   if 'min' in col.lower() or 'нижн' in col.lower()]

                    max_columns = [col for col in geo_data.columns
                                   if 'max' in col.lower() or 'верхн' in col.lower()]

                    if min_columns and max_columns:
                        lower_geo_limit = geo_data[min_columns[0]].mean()
                        upper_geo_limit = geo_data[max_columns[0]].mean()
                    else:
                        # Если не нашли нужные колонки, используем статистику
                        logger.warning("Не найдены колонки с граничными значениями давления")
                        pressure_columns = [col for col in geo_data.columns
                                            if 'давлен' in col.lower() or 'pressure' in col.lower()]

                        if pressure_columns:
                            lower_geo_limit = geo_data[pressure_columns[0]].quantile(0.1)  # 10-й перцентиль
                            upper_geo_limit = geo_data[pressure_columns[0]].quantile(0.9)  # 90-й перцентиль
                        else:
                            # Если не нашли нужные колонки, используем общую статистику
                            initial_pressure = self.results['Initial_Pressure'].mean()
                            std_dev = self.results['Initial_Pressure'].std()

                            lower_geo_limit = initial_pressure - 2 * std_dev
                            upper_geo_limit = initial_pressure + 2 * std_dev

            # Применяем граничные условия к расчетным значениям давления
            # Создаем новый столбец для скорректированных давлений с учетом геологии
            if 'Adjusted_Pressure' not in self.results.columns:
                self.results['Adjusted_Pressure'] = self.results['Calculated_Pressure'].copy()

            # Применяем верхнюю границу
            upper_mask = self.results['Adjusted_Pressure'] > upper_geo_limit
            self.results.loc[upper_mask, 'Adjusted_Pressure'] = upper_geo_limit

            # Применяем нижнюю границу
            lower_mask = self.results['Adjusted_Pressure'] < lower_geo_limit
            self.results.loc[lower_mask, 'Adjusted_Pressure'] = lower_geo_limit

            # Добавляем информацию о примененных геологических ограничениях
            self.results['Geo_Boundary_Applied'] = upper_mask | lower_mask
            self.results['Geo_Upper_Limit'] = upper_geo_limit
            self.results['Geo_Lower_Limit'] = lower_geo_limit

            # Подсчитываем количество скважин с примененными ограничениями
            upper_count = sum(upper_mask)
            lower_count = sum(lower_mask)
            total_count = sum(upper_mask | lower_mask)

            logger.info(f"Применены геологические ограничения для {total_count} скважин")
            logger.info(f"- Верхняя граница ({upper_geo_limit:.2f} атм) применена для {upper_count} скважин")
            logger.info(f"- Нижняя граница ({lower_geo_limit:.2f} атм) применена для {lower_count} скважин")

            return True

        except Exception as e:
            logger.error(f"Ошибка при применении геологических граничных условий: {str(e)}")
            return False

    def plot_pressure_geology_boundaries(self, output_path=None):
        """
        Построение графика давлений с геологическими границами.

        Args:
            output_path (str, optional): Путь для сохранения графика

        Returns:
            plt.Figure: Объект фигуры matplotlib
        """
        if self.results is None or 'Geo_Upper_Limit' not in self.results.columns:
            logger.error("Сначала примените геологические граничные условия")
            return None

        try:
            # Получаем данные
            wells = self.results['Well']
            calculated = self.results['Calculated_Pressure']
            adjusted = self.results['Adjusted_Pressure']
            upper_limit = self.results['Geo_Upper_Limit'].iloc[0]  # Одинаковое для всех скважин
            lower_limit = self.results['Geo_Lower_Limit'].iloc[0]  # Одинаковое для всех скважин

            # Сортируем скважины по расчетному давлению для лучшей визуализации
            sorted_indices = np.argsort(calculated.values)
            wells = wells.values[sorted_indices]
            calculated = calculated.values[sorted_indices]
            adjusted = adjusted.values[sorted_indices]

            # Построение графика
            fig, ax = plt.subplots(figsize=(14, 8))

            # Расчетные и скорректированные давления
            ax.plot(wells, calculated, 'bo-', label='Расчетное давление')
            ax.plot(wells, adjusted, 'ro-', label='Скорректированное давление')

            # Геологические границы
            ax.axhline(y=upper_limit, color='g', linestyle='--', label=f'Верхняя граница ({upper_limit:.2f} атм)')
            ax.axhline(y=lower_limit, color='orange', linestyle='--', label=f'Нижняя граница ({lower_limit:.2f} атм)')

            # Заливка областей за пределами границ
            ax.fill_between(wells, upper_limit, max(calculated) * 1.1, alpha=0.2, color='g')
            ax.fill_between(wells, min(calculated) * 0.9, lower_limit, alpha=0.2, color='orange')

            # Настройка графика
            ax.set_xlabel('Скважины')
            ax.set_ylabel('Давление, атм')
            ax.set_title('Расчетные давления с геологическими границами')
            ax.legend()

            # Поворот подписей оси X для лучшей читаемости
            plt.xticks(rotation=90)

            plt.tight_layout()

            # Сохранение графика, если указан путь
            if output_path and output_path.endswith('.png'):
                geo_path = output_path.replace('.png', '_geology.png')
                plt.savefig(geo_path, dpi=300, bbox_inches='tight')
                logger.info(f"График давлений с геологическими границами сохранен в {geo_path}")
            elif output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"График давлений с геологическими границами сохранен в {output_path}")

            return fig

        except Exception as e:
            logger.error(f"Ошибка при построении графика давлений с геологическими границами: {str(e)}")
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