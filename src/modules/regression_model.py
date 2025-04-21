"""
Модуль 2: Подбор итеративной регрессионной моделью.

Этот модуль реализует итеративный подбор параметров модели
с использованием регрессионного анализа и адаптацию на историю добычи.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
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
        self.history_matched = False

        # Начальные параметры для относительных фазовых проницаемостей
        self.ofp_params = {
            'Swo': 0.42,    # Остаточная водонасыщенность
            'Swk': 0.677,   # Водонасыщенность при остаточной нефтенасыщенности
            'Srw': 0.42,    # Остаточная водонасыщенность
            'Sro': 0.323,   # Остаточная нефтенасыщенность
            'Srg': 0.0,     # Остаточная газонасыщенность
            'nw': 1.33,     # Показатель степени для воды (подгоночная степень для функции Кори)
            'no': 3.34,     # Показатель степени для нефти (подгоночная степень для функции Кори)
            'krwk': 0.135,  # Конечное значение относительной водопроницаемости
            'krok': 1.0,    # Конечное значение относительной нефтепроницаемости
            'krgk': 1.0     # Конечное значение относительной газопроницаемости
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

    def _calculate_relative_permeability(self, Sw, params):
        """
        Расчет относительных фазовых проницаемостей по функциям Кори.

        Args:
            Sw (float): Водонасыщенность
            params (dict): Параметры модели

        Returns:
            tuple: (krw, kro, krg) - относительные проницаемости
        """
        Swo = params['Swo']
        Swk = params['Swk']
        Srw = params['Srw']
        Sro = params['Sro']
        Srg = params['Srg']

        # Для нефти-воды (из Image 1 и Image 3)
        if Sw <= Swo:
            krw = 0.0
            kro = params['krok']
        elif Sw >= Swk:
            krw = params['krwk']
            kro = 0.0
        else:
            # Функции Кори для нефти и воды
            Sw_norm = (Sw - Swo) / (1 - Swo)
            krw = params['krwk'] * (Sw_norm ** params['nw'])
            kro = params['krok'] * ((1 - Sw_norm) ** params['no'])

        # Для газа (если есть)
        Sg = 1 - Sw - Sro  # Газонасыщенность
        if Sg <= Srg:
            krg = 0.0
        else:
            Sg_norm = (Sg - Srg) / (1 - Srw - Srg)
            krg = params['krgk'] * (Sg_norm ** params.get('ng', 2.0))

        return krw, kro, krg

    def _calculate_total_mobility(self, Sw, params, fluid_props):
        """
        Расчет суммарной подвижности флюидов.

        Args:
            Sw (float): Водонасыщенность
            params (dict): Параметры модели
            fluid_props (dict): Свойства флюидов (вязкости)

        Returns:
            float: Суммарная подвижность
        """
        krw, kro, krg = self._calculate_relative_permeability(Sw, params)

        # Вязкости флюидов
        mu_w = fluid_props.get('mu_w', 0.45)  # сПз
        mu_o = fluid_props.get('mu_o', 1.3)   # сПз
        mu_g = fluid_props.get('mu_g', 0.02)  # сПз

        # Подвижности
        lambda_w = krw / mu_w if mu_w > 0 else 0
        lambda_o = kro / mu_o if mu_o > 0 else 0
        lambda_g = krg / mu_g if mu_g > 0 else 0

        return lambda_w + lambda_o + lambda_g

    def adapt_to_production_history(self, production_data, reservoir_params):
        """
        Адаптация модели на историю добычи.

        Args:
            production_data (pd.DataFrame): Историческиеданные добычи
            reservoir_params (dict): Параметры пласта (проницаемость, пористость и т.д.)

        Returns:
            bool: True если адаптация успешна, иначе False
        """
        logger.info("Начинаем адаптацию модели на историю добычи...")

        try:
            # Извлекаем необходимые данные
            if isinstance(production_data, dict):
                if 'production_wells' in production_data:
                    prod_wells_data = production_data['production_wells']
                else:
                    logger.error("Отсутствуют данные о добывающих скважинах")
                    return False
            else:
                prod_wells_data = production_data

            # Параметры пласта
            k = reservoir_params.get('permeability', 50.0)  # мД
            phi = reservoir_params.get('porosity', 0.2)     # д.ед.
            ct = reservoir_params.get('total_compressibility', 1e-5)  # 1/атм

            # Функция для минимизации ошибки между историческими и расчетными данными
            def objective_function(x):
                """
                Целевая функция для оптимизации.
                x - вектор параметров [nw, no, krwk, krok, Swo, Swk]
                """
                # Распаковываем параметры
                nw, no, krwk, krok, Swo, Swk = x

                # Обновляем параметры модели
                temp_params = self.ofp_params.copy()
                temp_params.update({
                    'nw': nw, 'no': no, 'krwk': krwk, 'krok': krok,
                    'Swo': Swo, 'Swk': Swk, 'Srw': Swo, 'Sro': 1 - Swk
                })

                # Расчет модельных значений для всех скважин
                mse = 0.0
                n_points = 0

                for _, well in prod_wells_data.iterrows():
                    if 'Water_Cut' in well and 'Flow_Rate' in well:
                        Sw = well['Water_Cut'] / 100.0 + self.ofp_params['Swo'] * (1 - well['Water_Cut'] / 100.0)
                        krw, kro, _ = self._calculate_relative_permeability(Sw, temp_params)

                        # Расчет дебита по формуле из документа
                        mu_w = reservoir_params.get('mu_w', 0.45)
                        mu_o = reservoir_params.get('mu_o', 1.3)

                        # Суммарная подвижность
                        total_mobility = krw / mu_w + kro / mu_o

                        # Модельный дебит (упрощенная формула)
                        model_rate = total_mobility * k * (well.get('Pressure_Drawdown', 50.0))

                        # Ошибка
                        mse += (well['Flow_Rate'] - model_rate) ** 2
                        n_points += 1

                if n_points > 0:
                    mse /= n_points
                else:
                    mse = np.inf

                return mse

            # Начальные значения параметров для оптимизации
            x0 = [
                self.ofp_params['nw'],
                self.ofp_params['no'],
                self.ofp_params['krwk'],
                self.ofp_params['krok'],
                self.ofp_params['Swo'],
                self.ofp_params['Swk']
            ]

            # Границы параметров
            bounds = [
                (0.45, 1.5),   # nw
                (0.68, 11.2),  # no
                (0.06, 0.88),  # krwk
                (0.9, 1.0),    # krok
                (0.025, 0.6),  # Swo
                (0.54, 0.9)    # Swk
            ]

            # Оптимизация
            result = minimize(
                objective_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000}
            )

            if result.success:
                # Обновляем параметры модели
                self.ofp_params.update({
                    'nw': result.x[0],
                    'no': result.x[1],
                    'krwk': result.x[2],
                    'krok': result.x[3],
                    'Swo': result.x[4],
                    'Swk': result.x[5],
                    'Srw': result.x[4],
                    'Sro': 1 - result.x[5]
                })

                self.params.update(self.ofp_params)
                self.history_matched = True

                logger.info("Адаптация на историю добычи успешно выполнена")
                logger.info(f"Оптимальные параметры: {self.ofp_params}")
                return True
            else:
                logger.error("Оптимизация не сошлась")
                return False

        except Exception as e:
            logger.error(f"Ошибка при адаптации на историю добычи: {str(e)}")
            return False

    def plot_relative_permeability_curves(self, output_path=None):
        """
        Построение графиков относительных проницаемостей после адаптации.

        Args:
            output_path (str, optional): Путь для сохранения графика

        Returns:
            plt.Figure: Объект фигуры matplotlib
        """
        try:
            # Создание массива водонасыщенностей
            Sw_values = np.linspace(0, 1, 100)
            krw_values = []
            kro_values = []

            # Расчет относительных проницаемостей
            for Sw in Sw_values:
                krw, kro, _ = self._calculate_relative_permeability(Sw, self.ofp_params)
                krw_values.append(krw)
                kro_values.append(kro)

            # Построение графика
            fig, ax = plt.subplots(figsize=(10, 6))

            ax.plot(Sw_values, krw_values, 'b-', linewidth=2, label='Вода')
            ax.plot(Sw_values, kro_values, 'r-', linewidth=2, label='Нефть')

            # Добавление вертикальных линий для Swo и Swk
            ax.axvline(x=self.ofp_params['Swo'], color='gray', linestyle='--',
                      label=f"Swo = {self.ofp_params['Swo']:.3f}")
            ax.axvline(x=self.ofp_params['Swk'], color='gray', linestyle='-.',
                      label=f"Swk = {self.ofp_params['Swk']:.3f}")

            ax.set_xlabel('Водонасыщенность, д.ед.')
            ax.set_ylabel('Относительная проницаемость, д.ед.')
            ax.set_title('Относительные фазовые проницаемости (после адаптации)')
            ax.legend()
            ax.grid(True)

            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"График сохранен в {output_path}")

            return fig

        except Exception as e:
            logger.error(f"Ошибка при построении графика: {str(e)}")
            return None

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

        # Начальные параметры для ОФП
        initial_params = np.array([
            self.ofp_params['Srw'],
            self.ofp_params['krwk'],
            self.ofp_params['Sro'],
            self.ofp_params['krok']
        ])

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
        if not self.optimized and not self.history_matched:
            return "Модель не оптимизирована. Запустите метод fit_model() или adapt_to_production_history()."

        report_text = "Результаты итеративного подбора регрессионной моделью:\n\n"

        if self.optimized:
            report_text += f"- Среднеквадратичная ошибка (RMSE): {self.rmse:.6f}\n"
            report_text += f"- Достигнута требуемая точность (<{TOLERANCE}): {'Да' if self.rmse < TOLERANCE else 'Нет'}\n\n"

        if self.history_matched:
            report_text += "Адаптация на историю добычи выполнена успешно.\n\n"
            report_text += "Оптимальные параметры относительных фазовых проницаемостей:\n"
            for param_name, param_value in self.ofp_params.items():
                report_text += f"- {param_name}: {param_value:.6f}\n"

            report_text += "\nФормулы относительных фазовых проницаемостей:\n"
            report_text += f"krw = {self.ofp_params['krwk']:.3f} * ((Sw - {self.ofp_params['Swo']:.3f}) / (1 - {self.ofp_params['Swo']:.3f}))^{self.ofp_params['nw']:.3f}\n"
            report_text += f"kro = {self.ofp_params['krok']:.3f} * (1 - (Sw - {self.ofp_params['Swo']:.3f}) / (1 - {self.ofp_params['Swo']:.3f}))^{self.ofp_params['no']:.3f}\n"
        else:
            report_text += "Адаптация на историю добычи не выполнена.\n\n"
            report_text += "Оптимальные параметры:\n"
            for param_name, param_value in self.params.items():
                report_text += f"- {param_name}: {param_value:.6f}\n"

        return report_text