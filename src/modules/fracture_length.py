"""
Модуль 7: Подбор коэффициентов для расчета полудлин трещин.

Этот модуль реализует подбор коэффициентов для расчета
полудлин трещин при закачке разных объемов воды.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import logging
from src.utils import calculate_fracture_half_length

logger = logging.getLogger(__name__)


class FractureLengthModel:
    """
    Модель для подбора коэффициентов расчета полудлин трещин.
    """

    def __init__(self, initial_params=None):
        """
        Инициализация модели с начальными параметрами.

        Args:
            initial_params (dict, optional): Начальные параметры модели
        """
        # Параметры по умолчанию
        default_params = {
            'coeff_a': 3.5,  # Коэффициент a
            'coeff_b': 0.33  # Коэффициент b (показатель степени)
        }

        self.params = initial_params if initial_params else default_params
        self.data = None
        self.fitted_params = None
        self.volumes = None
        self.lengths = None

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

            # Здесь будет код для извлечения данных о объемах закачки и полудлинах трещин
            # В реальном проекте нужно будет извлечь соответствующие колонки из данных

            # Для примера создадим синтетические данные
            # В реальном проекте эти данные будут получены из файлов

            # Создаем объемы закачки воды (м³)
            self.volumes = np.array([50, 100, 200, 300, 400, 500, 700, 1000])

            # Генерируем значения полудлин трещин с небольшим шумом
            true_lengths = np.array([
                calculate_fracture_half_length(
                    v,
                    self.params['coeff_a'],
                    self.params['coeff_b']
                ) for v in self.volumes
            ])

            # Добавляем шум для имитации реальных данных
            noise = np.random.normal(0, true_lengths * 0.05, len(true_lengths))  # 5% шума
            self.lengths = true_lengths + noise

            logger.info("Модель расчета полудлин трещин инициализирована данными")
            return True

        except Exception as e:
            logger.error(f"Ошибка при инициализации данными: {str(e)}")
            return False

    def _fracture_length_function(self, volume, coeff_a, coeff_b):
        """
        Функция для расчета полудлины трещины в зависимости от объема закачки.

        Args:
            volume (float): Объем закачки воды, м³
            coeff_a (float): Коэффициент a
            coeff_b (float): Коэффициент b (показатель степени)

        Returns:
            float: Полудлина трещины, м
        """
        return calculate_fracture_half_length(volume, coeff_a, coeff_b)

    def fit_model(self):
        """
        Подбор параметров модели на основе данных.

        Returns:
            bool: True если подбор выполнен успешно, иначе False
        """
        if self.volumes is None or self.lengths is None:
            logger.error("Модель не инициализирована данными")
            return False

        logger.info("Начинаем подбор коэффициентов для расчета полудлин трещин...")

        try:
            # Начальные приближения для параметров
            p0 = [
                self.params['coeff_a'],
                self.params['coeff_b']
            ]

            # Границы для параметров
            bounds = (
                [1.0, 0.1],  # Нижние границы
                [10.0, 0.5]  # Верхние границы
            )

            # Подбор параметров с помощью метода наименьших квадратов
            popt, pcov = curve_fit(
                self._fracture_length_function,
                self.volumes,
                self.lengths,
                p0=p0,
                bounds=bounds
            )

            # Сохраняем оптимальные параметры
            self.fitted_params = {
                'coeff_a': popt[0],
                'coeff_b': popt[1]
            }

            # Вычисляем стандартные отклонения параметров
            perr = np.sqrt(np.diag(pcov))
            self.fitted_params['coeff_a_std'] = perr[0]
            self.fitted_params['coeff_b_std'] = perr[1]

            # Обновляем параметры модели
            self.params.update({
                'coeff_a': self.fitted_params['coeff_a'],
                'coeff_b': self.fitted_params['coeff_b']
            })

            logger.info("Коэффициенты для расчета полудлин трещин успешно подобраны")
            logger.info(
                f"Оптимальные коэффициенты: a = {self.fitted_params['coeff_a']:.4f}, b = {self.fitted_params['coeff_b']:.4f}")

            return True

        except Exception as e:
            logger.error(f"Ошибка при подборе коэффициентов: {str(e)}")
            return False

    def predict_length(self, volumes):
        """
        Прогнозирование полудлин трещин для заданных объемов закачки.

        Args:
            volumes (array): Массив объемов закачки воды, м³

        Returns:
            array: Прогнозные значения полудлин трещин, м
        """
        if self.fitted_params is None:
            logger.warning("Модель не подобрана, используются начальные параметры")
            params = self.params
        else:
            params = {
                'coeff_a': self.fitted_params['coeff_a'],
                'coeff_b': self.fitted_params['coeff_b']
            }

        # Расчет полудлин трещин для каждого объема закачки
        lengths = np.array([
            calculate_fracture_half_length(
                v,
                params['coeff_a'],
                params['coeff_b']
            ) for v in volumes
        ])

        return lengths

    def plot_fracture_length_curve(self, output_path=None):
        """
        Построение графика зависимости полудлины трещины от объема закачки.

        Args:
            output_path (str, optional): Путь для сохранения графика

        Returns:
            plt.Figure: Объект фигуры matplotlib
        """
        if self.volumes is None or self.lengths is None:
            logger.error("Нет данных для построения графика")
            return None

        try:
            # Создаем точки объемов закачки для прогноза (более детальный график)
            v_pred = np.linspace(min(self.volumes) * 0.8, max(self.volumes) * 1.2, 100)

            # Прогнозируем значения полудлин трещин
            lengths_pred = self.predict_length(v_pred)

            # Построение графика
            fig, ax = plt.subplots(figsize=(10, 6))

            # Исходные данные
            ax.scatter(self.volumes, self.lengths, color='blue',
                       label='Исходные данные', s=50)

            # Аппроксимирующая кривая
            ax.plot(v_pred, lengths_pred, 'r-', linewidth=2,
                    label='Аппроксимирующая кривая')

            # Добавление подписей и заголовка
            ax.set_xlabel('Объем закачки воды, м³')
            ax.set_ylabel('Полудлина трещины, м')
            ax.set_title('Зависимость полудлины трещины от объема закачки')

            # Добавление информации о параметрах модели
            if self.fitted_params:
                formula_text = f"L = {self.fitted_params['coeff_a']:.2f} * V^{self.fitted_params['coeff_b']:.3f}"
                ax.text(0.98, 0.02, formula_text, transform=ax.transAxes,
                        horizontalalignment='right', verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

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

    def plot_log_log_curve(self, output_path=None):
        """
        Построение графика в логарифмических координатах.

        Args:
            output_path (str, optional): Путь для сохранения графика

        Returns:
            plt.Figure: Объект фигуры matplotlib
        """
        if self.volumes is None or self.lengths is None:
            logger.error("Нет данных для построения графика")
            return None

        try:
            # Создаем точки объемов закачки для прогноза (более детальный график)
            v_pred = np.logspace(
                np.log10(min(self.volumes)),
                np.log10(max(self.volumes)),
                100
            )

            # Прогнозируем значения полудлин трещин
            lengths_pred = self.predict_length(v_pred)

            # Построение графика в логарифмических координатах
            fig, ax = plt.subplots(figsize=(10, 6))

            # Исходные данные
            ax.loglog(self.volumes, self.lengths, 'bo', label='Исходные данные', markersize=8)

            # Аппроксимирующая прямая
            ax.loglog(v_pred, lengths_pred, 'r-', linewidth=2, label='Аппроксимирующая прямая')

            # Добавление подписей и заголовка
            ax.set_xlabel('Объем закачки воды, м³ (логарифмическая шкала)')
            ax.set_ylabel('Полудлина трещины, м (логарифмическая шкала)')
            ax.set_title('Зависимость полудлины трещины от объема закачки (логарифмические координаты)')

            # Добавление информации о параметрах модели
            if self.fitted_params:
                formula_text = f"L = {self.fitted_params['coeff_a']:.2f} * V^{self.fitted_params['coeff_b']:.3f}"
                ax.text(0.98, 0.02, formula_text, transform=ax.transAxes,
                        horizontalalignment='right', verticalalignment='bottom',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.grid(True, which="both", ls="-")
            ax.legend()

            plt.tight_layout()

            # Сохранение графика, если указан путь
            if output_path and output_path.endswith('.png'):
                log_log_path = output_path.replace('.png', '_loglog.png')
                plt.savefig(log_log_path, dpi=300, bbox_inches='tight')
                logger.info(f"График в логарифмических координатах сохранен в {log_log_path}")
            elif output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"График в логарифмических координатах сохранен в {output_path}")

            return fig

        except Exception as e:
            logger.error(f"Ошибка при построении графика в логарифмических координатах: {str(e)}")
            return None

    def get_parameters(self):
        """
        Получение текущих параметров модели.

        Returns:
            dict: Параметры модели
        """
        if self.fitted_params is None:
            return self.params
        else:
            return {
                'coeff_a': self.fitted_params['coeff_a'],
                'coeff_b': self.fitted_params['coeff_b'],
                'coeff_a_std': self.fitted_params.get('coeff_a_std'),
                'coeff_b_std': self.fitted_params.get('coeff_b_std')
            }

    def report(self):
        """
        Создание отчета о результатах подбора параметров.

        Returns:
            str: Текстовый отчет
        """
        if self.fitted_params is None:
            return "Коэффициенты для расчета полудлин трещин не подобраны. Запустите метод fit_model()."

        report_text = "Результаты подбора коэффициентов для расчета полудлин трещин:\n\n"

        report_text += "Оптимальные коэффициенты модели:\n"
        report_text += f"- Коэффициент a: {self.fitted_params['coeff_a']:.4f} ± {self.fitted_params.get('coeff_a_std', 0):.4f}\n"
        report_text += f"- Коэффициент b: {self.fitted_params['coeff_b']:.4f} ± {self.fitted_params.get('coeff_b_std', 0):.4f}\n\n"

        report_text += f"Формула для расчета полудлины трещины:\n"
        report_text += f"L = {self.fitted_params['coeff_a']:.4f} * V^{self.fitted_params['coeff_b']:.4f}\n\n"
        report_text += f"где L - полудлина трещины [м], V - объем закачки воды [м³]\n\n"

        # Прогноз значений для некоторых объемов закачки
        forecast_volumes = [100, 200, 500, 1000, 2000, 5000]
        forecast_lengths = self.predict_length(forecast_volumes)

        report_text += "Прогноз полудлин трещин для различных объемов закачки:\n"
        for v, l in zip(forecast_volumes, forecast_lengths):
            report_text += f"- Объем {v} м³: полудлина {l:.1f} м\n"

        # Примечание из документа
        report_text += "\nПримечание:\n"
        report_text += "Трещина авто ГРП имеет иную физику формирования -- описанная методика может\n"
        report_text += "использоваться для приблизительного подсчёта эффекта, но будет иметь\n"
        report_text += "отклонения для низкодебитных скважин.\n"

        return report_text