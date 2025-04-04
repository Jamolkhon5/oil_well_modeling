"""
Модуль 1: Подбор относительных фазовых проницаемостей.

Этот модуль реализует подбор параметров для расчета относительных фазовых
проницаемостей нефти и воды по функциям Кори.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from src.utils import (
    corey_water_function,
    corey_oil_function,
    fit_relative_permeability_curves,
    plot_relative_permeability_curves
)

logger = logging.getLogger(__name__)


class PhasePermeabilityModel:
    """
    Модель для подбора относительных фазовых проницаемостей.
    """

    def __init__(self, initial_params):
        """
        Инициализация модели с начальными параметрами.

        Args:
            initial_params (dict): Начальные параметры модели
        """
        self.params = initial_params
        self.optimized = False
        self.sw_values = None
        self.krw_values = None
        self.kro_values = None

    def initialize_from_data(self, data):
        """
        Инициализация модели данными из файлов.

        Args:
            data (dict): Словарь с данными из файлов
        """
        # Проверка наличия необходимых данных
        if 'ppl_data' not in data or 'gdi_data' not in data:
            logger.error("Отсутствуют необходимые данные для модели")
            return False

        # Здесь будет код для извлечения релевантных данных из файлов
        # и их предварительной обработки
        # ...

        # Для примера возьмем теоретические данные
        # В реальном коде здесь будет обработка данных из файлов

        # Создаем теоретические данные для демонстрации
        self.sw_values = np.linspace(self.params['Swo'], self.params['Swk'], 20)

        # Теоретические значения krw с небольшим шумом
        self.krw_values = np.array([
            corey_water_function(sw, self.params['Swo'], self.params['Swk'],
                                 self.params['krwk'], self.params['krw_min'])
            for sw in self.sw_values
        ])
        # Добавляем шум для имитации реальных данных
        self.krw_values += np.random.normal(0, 0.01, self.krw_values.shape)

        # Теоретические значения kro с небольшим шумом
        self.kro_values = np.array([
            corey_oil_function(sw, self.params['Swo'], self.params['Swk'],
                               self.params['krok'], self.params['kro_min'])
            for sw in self.sw_values
        ])
        # Добавляем шум для имитации реальных данных
        self.kro_values += np.random.normal(0, 0.01, self.kro_values.shape)

        logger.info("Модель инициализирована данными")
        return True

    def fit_model(self):
        """
        Подбор оптимальных параметров модели.

        Returns:
            bool: True если подбор выполнен успешно, иначе False
        """
        if self.sw_values is None or self.krw_values is None or self.kro_values is None:
            logger.error("Модель не инициализирована данными")
            return False

        logger.info("Начинаем подбор параметров модели...")

        try:
            # Подбор параметров с использованием функции fit_relative_permeability_curves
            optimized_params = fit_relative_permeability_curves(
                self.sw_values, self.krw_values, self.kro_values, self.params
            )

            # Обновляем параметры модели оптимальными значениями
            self.params.update(optimized_params)
            self.optimized = True

            logger.info("Параметры модели успешно подобраны")
            logger.info(f"Оптимальные параметры: {optimized_params}")

            return True

        except Exception as e:
            logger.error(f"Ошибка при подборе параметров: {str(e)}")
            return False

    def calculate_permeability(self, sw):
        """
        Расчет относительных проницаемостей для заданной водонасыщенности.

        Args:
            sw (float или array): Водонасыщенность, д.ед.

        Returns:
            tuple: (krw, kro) - относительные проницаемости по воде и нефти
        """
        # Проверка, что модель оптимизирована
        if not self.optimized:
            logger.warning("Модель не оптимизирована, используются начальные параметры")

        # Расчет krw
        krw = corey_water_function(
            sw, self.params['Swo'], self.params['Swk'],
            self.params['krwk'], self.params.get('nw', self.params['krw_min'])
        )

        # Расчет kro
        kro = corey_oil_function(
            sw, self.params['Swo'], self.params['Swk'],
            self.params['krok'], self.params.get('no', self.params['kro_min'])
        )

        return krw, kro

    def plot_permeability_curves(self, output_path=None):
        """
        Построение графиков относительных проницаемостей.

        Args:
            output_path (str, optional): Путь для сохранения графика

        Returns:
            plt.Figure: Объект фигуры matplotlib
        """
        # Проверка, что модель оптимизирована
        if not self.optimized:
            logger.warning("Модель не оптимизирована, используются начальные параметры")

        # Использование функции из utils для построения графика
        fig = plot_relative_permeability_curves(
            self.params['Swo'],
            self.params['Swk'],
            self.params['krwk'],
            self.params['krok'],
            self.params.get('nw', self.params['krw_min']),
            self.params.get('no', self.params['kro_min']),
            output_path
        )

        return fig

    def get_parameters(self):
        """
        Получение текущих параметров модели.

        Returns:
            dict: Параметры модели
        """
        return self.params

    def report(self):
        """
        Создание отчета о результатах подбора параметров.

        Returns:
            str: Текстовый отчет
        """
        if not self.optimized:
            report_text = "ПРЕДУПРЕЖДЕНИЕ: Модель не оптимизирована, используются начальные параметры.\n\n"
        else:
            report_text = "Модель успешно оптимизирована.\n\n"

        report_text += "Параметры относительных фазовых проницаемостей:\n"
        report_text += f"- Остаточная водонасыщенность (Swo): {self.params['Swo']:.4f}\n"
        report_text += f"- Водонасыщенность при остаточной нефтенасыщенности (Swk): {self.params['Swk']:.4f}\n"
        report_text += f"- Конечное значение относительной водопронецаемости (krwk): {self.params['krwk']:.4f}\n"
        report_text += f"- Конечное значение относительной нефтепронецаемости (krok): {self.params['krok']:.4f}\n"

        if 'nw' in self.params:
            report_text += f"- Показатель степени для воды (nw): {self.params['nw']:.4f}\n"
        else:
            report_text += f"- Показатель степени для воды (nw): {self.params['krw_min']:.4f}\n"

        if 'no' in self.params:
            report_text += f"- Показатель степени для нефти (no): {self.params['no']:.4f}\n"
        else:
            report_text += f"- Показатель степени для нефти (no): {self.params['kro_min']:.4f}\n"

        return report_text