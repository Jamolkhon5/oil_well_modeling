"""
Модуль 5: Подбор кривой увеличения SKIN для нефтяных скважин с течением времени после ГРП.

Этот модуль реализует подбор параметров для расчета динамики
изменения скин-фактора после гидроразрыва пласта (ГРП).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import logging
from src.utils import calculate_skin_factor_after_fracking

logger = logging.getLogger(__name__)


class SkinCurveModel:
    """
    Модель для подбора кривой увеличения SKIN после ГРП.
    """

    def __init__(self, initial_params=None):
        """
        Инициализация модели с начальными параметрами.

        Args:
            initial_params (dict, optional): Начальные параметры модели
        """
        # Параметры по умолчанию
        default_params = {
            'initial_skin': -3.0,  # Начальный скин-фактор сразу после ГРП
            'max_skin': 0.0,  # Максимальное значение скин-фактора
            'growth_rate': 0.01  # Скорость роста скин-фактора
        }

        self.params = initial_params if initial_params else default_params
        self.data = None
        self.fitted_params = None
        self.time_points = None
        self.skin_values = None

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

            # Здесь будет код для извлечения данных о SKIN-факторе из данных ГДИС
            # В реальном проекте нужно будет извлечь соответствующие колонки из данных

            # Для примера создадим синтетические данные
            # В реальном проекте эти данные будут получены из файлов ГДИС

            # Создаем временные точки (дни после ГРП)
            self.time_points = np.array([0, 10, 30, 60, 90, 180, 365])

            # Генерируем значения скин-фактора с небольшим шумом
            true_skin_values = np.array([
                calculate_skin_factor_after_fracking(
                    t,
                    self.params['initial_skin'],
                    self.params['max_skin'],
                    self.params['growth_rate']
                ) for t in self.time_points
            ])

            # Добавляем шум для имитации реальных данных
            noise = np.random.normal(0, 0.2, len(true_skin_values))
            self.skin_values = true_skin_values + noise

            logger.info("Модель кривой увеличения SKIN инициализирована данными")
            return True

        except Exception as e:
            logger.error(f"Ошибка при инициализации данными: {str(e)}")
            return False

    def _skin_curve_function(self, t, initial_skin, max_skin, growth_rate):
        """
        Функция для аппроксимации изменения скин-фактора со временем.

        Args:
            t (float): Время после ГРП, сут
            initial_skin (float): Начальный скин-фактор сразу после ГРП
            max_skin (float): Максимальное значение скин-фактора
            growth_rate (float): Скорость роста скин-фактора

        Returns:
            float: Значение скин-фактора в момент времени t
        """
        return calculate_skin_factor_after_fracking(t, initial_skin, max_skin, growth_rate)

    def fit_model(self):
        """
        Подбор параметров модели на основе данных.

        Returns:
            bool: True если подбор выполнен успешно, иначе False
        """
        if self.time_points is None or self.skin_values is None:
            logger.error("Модель не инициализирована данными")
            return False

        logger.info("Начинаем подбор параметров модели кривой SKIN...")

        try:
            # Начальные приближения для параметров
            p0 = [
                self.params['initial_skin'],
                self.params['max_skin'],
                self.params['growth_rate']
            ]

            # Границы для параметров
            bounds = (
                [-5.0, -5.0, 0.001],  # Нижние границы
                [2.0, 5.0, 0.1]  # Верхние границы
            )

            # Подбор параметров с помощью метода наименьших квадратов
            popt, pcov = curve_fit(
                self._skin_curve_function,
                self.time_points,
                self.skin_values,
                p0=p0,
                bounds=bounds
            )

            # Сохраняем оптимальные параметры
            self.fitted_params = {
                'initial_skin': popt[0],
                'max_skin': popt[1],
                'growth_rate': popt[2]
            }

            # Обновляем параметры модели
            self.params.update(self.fitted_params)

            logger.info("Параметры модели кривой SKIN успешно подобраны")
            logger.info(f"Оптимальные параметры: {self.fitted_params}")

            return True

        except Exception as e:
            logger.error(f"Ошибка при подборе параметров кривой SKIN: {str(e)}")
            return False

    def predict_skin(self, time_points):
        """
        Прогнозирование значений скин-фактора для заданных временных точек.

        Args:
            time_points (array): Массив временных точек, сут

        Returns:
            array: Прогнозные значения скин-фактора
        """
        if self.fitted_params is None:
            logger.warning("Модель не подобрана, используются начальные параметры")
            params = self.params
        else:
            params = self.fitted_params

        # Расчет скин-фактора для каждой временной точки
        skin_values = np.array([
            calculate_skin_factor_after_fracking(
                t,
                params['initial_skin'],
                params['max_skin'],
                params['growth_rate']
            ) for t in time_points
        ])

        return skin_values

    def plot_skin_curve(self, output_path=None):
        """
        Построение графика изменения скин-фактора со временем.

        Args:
            output_path (str, optional): Путь для сохранения графика

        Returns:
            plt.Figure: Объект фигуры matplotlib
        """
        if self.time_points is None or self.skin_values is None:
            logger.error("Нет данных для построения графика")
            return None

        try:
            # Создаем временные точки для прогноза (более детальный график)
            t_pred = np.linspace(0, 2 * max(self.time_points), 100)

            # Прогнозируем значения скин-фактора
            skin_pred = self.predict_skin(t_pred)

            # Построение графика
            fig, ax = plt.subplots(figsize=(10, 6))

            # Исходные данные
            ax.scatter(self.time_points, self.skin_values, color='blue',
                       label='Исходные данные', s=50)

            # Аппроксимирующая кривая
            ax.plot(t_pred, skin_pred, 'r-', linewidth=2,
                    label='Аппроксимирующая кривая')

            # Добавление подписей и заголовка
            ax.set_xlabel('Время после ГРП, сут')
            ax.set_ylabel('Скин-фактор')
            ax.set_title('Изменение скин-фактора после ГРП')

            # Добавление информации о параметрах модели
            if self.fitted_params:
                params_text = (
                    f"Начальный скин: {self.fitted_params['initial_skin']:.2f}\n"
                    f"Максимальный скин: {self.fitted_params['max_skin']:.2f}\n"
                    f"Скорость роста: {self.fitted_params['growth_rate']:.4f}"
                )
                ax.text(0.02, 0.98, params_text, transform=ax.transAxes,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.grid(True)
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

    def get_parameters(self):
        """
        Получение текущих параметров модели.

        Returns:
            dict: Параметры модели
        """
        return self.params if self.fitted_params is None else self.fitted_params

    def report(self):
        """
        Создание отчета о результатах подбора параметров.

        Returns:
            str: Текстовый отчет
        """
        if self.fitted_params is None:
            return "Модель кривой SKIN не подобрана. Запустите метод fit_model()."

        report_text = "Результаты подбора кривой увеличения SKIN после ГРП:\n\n"

        report_text += "Оптимальные параметры модели:\n"
        report_text += f"- Начальный скин-фактор: {self.fitted_params['initial_skin']:.4f}\n"
        report_text += f"- Максимальный скин-фактор: {self.fitted_params['max_skin']:.4f}\n"
        report_text += f"- Скорость роста скин-фактора: {self.fitted_params['growth_rate']:.6f}\n\n"

        # Прогноз значений для некоторых временных точек
        forecast_times = [0, 30, 90, 180, 365, 730]
        forecast_values = self.predict_skin(forecast_times)

        report_text += "Прогноз изменения скин-фактора:\n"
        for t, s in zip(forecast_times, forecast_values):
            report_text += f"- Через {t} дней: {s:.4f}\n"

        return report_text