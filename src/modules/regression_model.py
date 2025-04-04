"""
Модуль 2: Подбор итеративной регрессионной моделью.

Этот модуль реализует итеративный подбор параметров модели
с использованием регрессионного анализа.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import logging
from src.config import TOLERANCE

logger = logging.getLogger(__name__)


class RegressionModel:
    """
    Модель для итеративного подбора параметров с помощью регрессии.
    """

    def __init__(self, initial_params=None):
        """
        Инициализация модели с начальными параметрами.

        Args:
            initial_params (dict, optional): Начальные параметры модели
        """
        self.params = initial_params if initial_params else {}
        self.data = None
        self.optimized = False
        self.result = None
        self.rmse = None

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
            logger.info("Модель инициализирована данными")
            return True
        except Exception as e:
            logger.error(f"Ошибка при инициализации данными: {str(e)}")
            return False

    def _regression_model(self, X, params):
        """
        Регрессионная модель для прогнозирования.

        Args:
            X (np.array): Матрица признаков
            params (np.array): Параметры модели

        Returns:
            np.array: Прогнозные значения
        """
        # Здесь будет ваша регрессионная модель, например:
        # Srw, krw_max, Sro, kro_max = params
        #
        # Для примера используем линейную модель, но в реальности
        # это будет более сложная модель в соответствии с документом
        return np.dot(X, params)

    def _error_function(self, params):
        """
        Функция ошибки для минимизации.

        Args:
            params (np.array): Параметры модели

        Returns:
            float: Значение функции ошибки (RMSE)
        """
        # Пример для демонстрации, в реальности будет более сложная функция
        # в соответствии с документом

        # Проверяем ограничения из системы уравнений
        Srw, krw_max, Sro, kro_max = params

        # Проверка ограничений из документа
        if not (0 <= Srw <= 1 and 0 <= krw_max <= 1 and 0 <= Sro <= 1 and 0 <= kro_max <= 1):
            return np.inf

        # Наказание за нарушение условий системы уравнений
        # Здесь должны быть проверки условий из документа

        # Пример для демонстрации
        X = np.random.rand(100, 4)  # Матрица признаков
        y = np.random.rand(100)  # Целевые значения

        # Расчет прогнозных значений
        y_pred = self._regression_model(X, params)

        # Расчет RMSE
        rmse = np.sqrt(np.mean((y - y_pred) ** 2))

        return rmse

    def fit_model(self, max_iterations=1000):
        """
        Итеративный подбор параметров модели.

        Args:
            max_iterations (int): Максимальное число итераций

        Returns:
            bool: True если подбор выполнен успешно, иначе False
        """
        if self.data is None:
            logger.error("Модель не инициализирована данными")
            return False

        logger.info("Начинаем итеративный подбор параметров...")

        # Начальные параметры
        initial_params = np.array([0.4, 0.5, 0.3, 0.9])  # Пример значений

        # Границы параметров
        bounds = [(0, 1), (0, 1), (0, 1), (0, 1)]  # Все параметры от 0 до 1

        best_rmse = float('inf')
        best_params = None

        for iteration in range(max_iterations):
            try:
                # Минимизация функции ошибки
                result = minimize(
                    self._error_function,
                    initial_params,
                    bounds=bounds,
                    method='L-BFGS-B'
                )

                # Проверка сходимости
                if result.success:
                    current_rmse = result.fun

                    # Если текущий результат лучше предыдущего
                    if current_rmse < best_rmse:
                        best_rmse = current_rmse
                        best_params = result.x

                        logger.info(f"Итерация {iteration + 1}: RMSE = {best_rmse:.6f}")

                        # Если достигнута требуемая точность
                        if best_rmse < TOLERANCE:
                            logger.info(f"Достигнута требуемая точность: RMSE = {best_rmse:.6f}")
                            break

                    # Обновляем начальные параметры для следующей итерации
                    initial_params = result.x + np.random.normal(0, 0.01, len(result.x))

                    # Убеждаемся, что параметры остаются в допустимых пределах
                    for i in range(len(initial_params)):
                        initial_params[i] = max(bounds[i][0], min(bounds[i][1], initial_params[i]))
                else:
                    logger.warning(f"Итерация {iteration + 1}: Оптимизация не сошлась")
                    # Пробуем новые начальные значения
                    initial_params = np.random.uniform(0, 1, len(initial_params))

            except Exception as e:
                logger.error(f"Ошибка в итерации {iteration + 1}: {str(e)}")
                # Пробуем новые начальные значения
                initial_params = np.random.uniform(0, 1, len(initial_params))

        # Проверка результатов
        if best_params is not None:
            self.params = {
                'Srw': best_params[0],
                'krw_max': best_params[1],
                'Sro': best_params[2],
                'kro_max': best_params[3]
            }
            self.optimized = True
            self.rmse = best_rmse

            logger.info("Параметры модели успешно подобраны")
            logger.info(f"Оптимальные параметры: {self.params}")
            logger.info(f"Итоговая RMSE: {self.rmse:.6f}")

            return True
        else:
            logger.error("Не удалось подобрать параметры модели")
            return False

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
            return "Модель не оптимизирована. Запустите метод fit_model()."

        report_text = "Результаты итеративного подбора регрессионной моделью:\n\n"
        report_text += f"- Среднеквадратичная ошибка (RMSE): {self.rmse:.6f}\n"
        report_text += f"- Достигнута требуемая точность (<{TOLERANCE}): {'Да' if self.rmse < TOLERANCE else 'Нет'}\n\n"

        report_text += "Оптимальные параметры:\n"
        for param_name, param_value in self.params.items():
            report_text += f"- {param_name}: {param_value:.6f}\n"

        return report_text