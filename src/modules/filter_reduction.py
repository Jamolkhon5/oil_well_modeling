"""
Модуль 6: Подбор к-та уменьшения работающей части фильтра горизонтальной нефтяной скважины.

Этот модуль реализует расчет коэффициента уменьшения работающей части фильтра
горизонтальной нефтяной скважины в течение времени после запуска.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import logging
from src.utils import calculate_filter_reduction_coefficient

logger = logging.getLogger(__name__)


class FilterReductionModel:
    """
    Модель для подбора коэффициента уменьшения работающей части фильтра.
    """

    def __init__(self, initial_params=None):
        """
        Инициализация модели с начальными параметрами.

        Args:
            initial_params (dict, optional): Начальные параметры модели
        """
        # Параметры по умолчанию
        default_params = {
            'initial_coeff': 1.0,  # Начальный коэффициент
            'min_coeff': 0.5,  # Минимальный коэффициент
            'reduction_rate': 0.003  # Скорость уменьшения
        }

        self.params = initial_params if initial_params else default_params
        self.data = None
        self.fitted_params = None
        self.time_points = None
        self.coeff_values = None

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

            # Здесь будет код для извлечения данных о коэффициенте из данных
            # В реальном проекте нужно будет извлечь соответствующие колонки из данных

            # Для примера создадим синтетические данные
            # В реальном проекте эти данные будут получены из файлов

            # Создаем временные точки (дни после запуска)
            self.time_points = np.array([0, 30, 90, 180, 365, 730, 1095])

            # Генерируем значения коэффициента с небольшим шумом
            true_coeff_values = np.array([
                calculate_filter_reduction_coefficient(
                    t,
                    self.params['initial_coeff'],
                    self.params['min_coeff'],
                    self.params['reduction_rate']
                ) for t in self.time_points
            ])

            # Добавляем шум для имитации реальных данных
            noise = np.random.normal(0, 0.03, len(true_coeff_values))
            self.coeff_values = true_coeff_values + noise

            # Обеспечиваем, чтобы значения не выходили за допустимые границы
            self.coeff_values = np.clip(
                self.coeff_values,
                self.params['min_coeff'] * 0.9,
                self.params['initial_coeff'] * 1.1
            )

            logger.info("Модель коэффициента уменьшения работающей части фильтра инициализирована данными")
            return True

        except Exception as e:
            logger.error(f"Ошибка при инициализации данными: {str(e)}")
            return False

    def _filter_reduction_function(self, t, initial_coeff, min_coeff, reduction_rate):
        """
        Функция для аппроксимации изменения коэффициента со временем.

        Args:
            t (float): Время после запуска, сут
            initial_coeff (float): Начальный коэффициент
            min_coeff (float): Минимальный коэффициент
            reduction_rate (float): Скорость уменьшения

        Returns:
            float: Значение коэффициента в момент времени t
        """
        return calculate_filter_reduction_coefficient(t, initial_coeff, min_coeff, reduction_rate)

    def fit_model(self):
        """
        Подбор параметров модели на основе данных.

        Returns:
            bool: True если подбор выполнен успешно, иначе False
        """
        if self.time_points is None or self.coeff_values is None:
            logger.error("Модель не инициализирована данными")
            return False

        logger.info("Начинаем подбор параметров модели коэффициента уменьшения...")

        try:
            # Начальные приближения для параметров
            p0 = [
                self.params['initial_coeff'],
                self.params['min_coeff'],
                self.params['reduction_rate']
            ]

            # Границы для параметров
            bounds = (
                [0.9, 0.1, 0.0001],  # Нижние границы
                [1.1, 0.9, 0.01]  # Верхние границы
            )

            # Подбор параметров с помощью метода наименьших квадратов
            popt, pcov = curve_fit(
                self._filter_reduction_function,
                self.time_points,
                self.coeff_values,
                p0=p0,
                bounds=bounds
            )

            # Сохраняем оптимальные параметры
            self.fitted_params = {
                'initial_coeff': popt[0],
                'min_coeff': popt[1],
                'reduction_rate': popt[2]
            }

            # Обновляем параметры модели
            self.params.update(self.fitted_params)

            logger.info("Параметры модели коэффициента уменьшения успешно подобраны")
            logger.info(f"Оптимальные параметры: {self.fitted_params}")

            return True

        except Exception as e:
            logger.error(f"Ошибка при подборе параметров коэффициента уменьшения: {str(e)}")
            return False

    def predict_coefficient(self, time_points):
        """
        Прогнозирование значений коэффициента для заданных временных точек.

        Args:
            time_points (array): Массив временных точек, сут

        Returns:
            array: Прогнозные значения коэффициента
        """
        if self.fitted_params is None:
            logger.warning("Модель не подобрана, используются начальные параметры")
            params = self.params
        else:
            params = self.fitted_params

        # Расчет коэффициента для каждой временной точки
        coeff_values = np.array([
            calculate_filter_reduction_coefficient(
                t,
                params['initial_coeff'],
                params['min_coeff'],
                params['reduction_rate']
            ) for t in time_points
        ])

        return coeff_values

    def plot_coefficient_curve(self, output_path=None):
        """
        Построение графика изменения коэффициента со временем.

        Args:
            output_path (str, optional): Путь для сохранения графика

        Returns:
            plt.Figure: Объект фигуры matplotlib
        """
        if self.time_points is None or self.coeff_values is None:
            logger.error("Нет данных для построения графика")
            return None

        try:
            # Создаем временные точки для прогноза (более детальный график)
            t_pred = np.linspace(0, 2 * max(self.time_points), 100)

            # Прогнозируем значения коэффициента
            coeff_pred = self.predict_coefficient(t_pred)

            # Построение графика
            fig, ax = plt.subplots(figsize=(10, 6))

            # Исходные данные
            ax.scatter(self.time_points, self.coeff_values, color='blue',
                       label='Исходные данные', s=50)

            # Аппроксимирующая кривая
            ax.plot(t_pred, coeff_pred, 'r-', linewidth=2,
                    label='Аппроксимирующая кривая')

            # Добавление подписей и заголовка
            ax.set_xlabel('Время после запуска, сут')
            ax.set_ylabel('Коэффициент работающей части фильтра')
            ax.set_title('Изменение коэффициента работающей части фильтра со временем')

            # Добавление информации о параметрах модели
            if self.fitted_params:
                params_text = (
                    f"Начальный коэффициент: {self.fitted_params['initial_coeff']:.2f}\n"
                    f"Минимальный коэффициент: {self.fitted_params['min_coeff']:.2f}\n"
                    f"Скорость уменьшения: {self.fitted_params['reduction_rate']:.6f}"
                )
                ax.text(0.02, 0.02, params_text, transform=ax.transAxes,
                        verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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
            return "Модель коэффициента уменьшения не подобрана. Запустите метод fit_model()."

        report_text = "Результаты подбора коэффициента уменьшения работающей части фильтра:\n\n"

        report_text += "Оптимальные параметры модели:\n"
        report_text += f"- Начальный коэффициент: {self.fitted_params['initial_coeff']:.4f}\n"
        report_text += f"- Минимальный коэффициент: {self.fitted_params['min_coeff']:.4f}\n"
        report_text += f"- Скорость уменьшения: {self.fitted_params['reduction_rate']:.6f}\n\n"

        # Прогноз значений для некоторых временных точек
        forecast_times = [0, 90, 180, 365, 730, 1095, 1825]
        forecast_values = self.predict_coefficient(forecast_times)

        report_text += "Прогноз изменения коэффициента работающей части фильтра:\n"
        for t, c in zip(forecast_times, forecast_values):
            report_text += f"- Через {t} дней: {c:.4f}\n"

        # Физическая интерпретация
        report_text += "\nФизическая интерпретация:\n"
        report_text += "- Начальное значение коэффициента близко к 1.0, что соответствует полностью работающему фильтру.\n"
        report_text += f"- Минимальное значение {self.fitted_params['min_coeff']:.2f} означает, что со временем эффективная длина фильтра\n"
        report_text += f"  уменьшается до {self.fitted_params['min_coeff'] * 100:.1f}% от начальной длины.\n"
        report_text += f"- При текущей скорости уменьшения через 1 год коэффициент составит {forecast_values[3]:.2f},\n"
        report_text += f"  а через 5 лет - {forecast_values[6]:.2f}.\n"

        return report_text