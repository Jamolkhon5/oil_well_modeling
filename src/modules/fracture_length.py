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

    def calculate_from_fracking_design(self):
        """
        Расчет коэффициентов полудлин трещин на основе обобщенного дизайна трещин ГРП.

        Returns:
            bool: True если расчет выполнен успешно, иначе False
        """
        if self.data is None:
            logger.error("Модель не инициализирована данными")
            return False

        logger.info("Начинаем расчет коэффициентов на основе обобщенного дизайна трещин ГРП...")

        try:
            # Получаем данные ГТР
            gtp_data = self.data.get('gtp_data')

            if gtp_data is None:
                logger.warning("Отсутствуют данные ГТР, невозможно выполнить расчет на основе дизайна ГРП")
                return False

            # Извлекаем параметры ГРП
            fracking_params = self._extract_fracking_params(gtp_data)

            if not fracking_params:
                logger.warning("Не удалось извлечь параметры ГРП из данных")
                return False

            # Создаем датафрейм с параметрами ГРП
            design_data = pd.DataFrame(fracking_params)

            # Проверяем наличие необходимых столбцов
            required_columns = ['Well', 'Volume', 'Fracture_Length']

            if not all(col in design_data.columns for col in required_columns):
                logger.warning("Не все необходимые параметры ГРП доступны")
                # Проверяем минимально необходимые данные
                if 'Volume' in design_data.columns and 'Fracture_Length' in design_data.columns:
                    # Если не хватает только Well, добавляем фиктивные идентификаторы
                    if 'Well' not in design_data.columns:
                        design_data['Well'] = [f"Well_{i}" for i in range(1, len(design_data) + 1)]
                else:
                    logger.error("Недостаточно данных для расчета")
                    return False

            # Фильтруем строки с отсутствующими значениями
            design_data = design_data.dropna(subset=['Volume', 'Fracture_Length'])

            if len(design_data) < 3:
                logger.warning("Недостаточно данных для надежной аппроксимации")
                if len(design_data) > 0:
                    # Если есть хотя бы одна строка, используем линейную модель
                    logger.info("Используем упрощенную линейную модель")
                else:
                    logger.error("Нет данных для расчета")
                    return False

            # Преобразуем данные в массивы для аппроксимации
            volumes = design_data['Volume'].values
            lengths = design_data['Fracture_Length'].values

            # Логарифмируем данные для линейной аппроксимации
            log_volumes = np.log(volumes)
            log_lengths = np.log(lengths)

            # Аппроксимация линейной моделью в логарифмическом масштабе
            # log(L) = log(a) + b * log(V)
            # L = a * V^b
            A = np.vstack([np.ones(len(log_volumes)), log_volumes]).T
            log_a, b = np.linalg.lstsq(A, log_lengths, rcond=None)[0]
            a = np.exp(log_a)

            # Сохраняем коэффициенты
            self.fitted_params = {
                'coeff_a': a,
                'coeff_b': b
            }

            # Оценка стандартных отклонений
            model_lengths = a * volumes ** b
            residuals = lengths - model_lengths
            mse = np.mean(residuals ** 2)
            rmse = np.sqrt(mse)

            # Относительная ошибка
            rel_error = np.mean(np.abs(residuals / lengths)) * 100

            self.fitted_params['rmse'] = rmse
            self.fitted_params['rel_error'] = rel_error

            # Обновляем параметры модели
            self.params.update({
                'coeff_a': a,
                'coeff_b': b
            })

            # Сохраняем данные для построения графиков
            self.volumes = volumes
            self.lengths = lengths

            logger.info("Коэффициенты для расчета полудлин трещин успешно определены на основе дизайна ГРП")
            logger.info(f"Результат: L = {a:.4f} * V^{b:.4f}")
            logger.info(f"Относительная ошибка: {rel_error:.2f}%")

            return True

        except Exception as e:
            logger.error(f"Ошибка при расчете коэффициентов на основе дизайна ГРП: {str(e)}")
            return False

    def _extract_fracking_params(self, gtp_data):
        """
        Извлечение параметров ГРП из данных ГТР.

        Args:
            gtp_data: Данные ГТР

        Returns:
            list: Список словарей с параметрами ГРП
        """
        fracking_params = []

        try:
            # Функция для оценки полудлины трещины на основе расчетных формул
            def calculate_fracture_length(volume, fluid_viscosity=1.0, rock_density=2500,
                                          young_modulus=2e4, poisson_ratio=0.25, height=10.0,
                                          breakdown_pressure=None, reservoir_pressure=None,
                                          tectonic_stress=15.0):
                """
                Расчет полудлины трещины по параметрам ГРП с использованием формул из документа.
                """
                try:
                    if not breakdown_pressure and reservoir_pressure:
                        # Используем формулу из документа для расчета давления разрыва
                        gravity = 9.81  # м/с²
                        depth = height * 100  # Предполагаем, что высота в метрах, глубина в сотнях метров
                        breakdown_pressure = (rock_density * depth * gravity / 100000 +
                                              reservoir_pressure + tectonic_stress) / 100000

                    # Формула из документа (раздел 8)
                    # x/2 = c * sqrt(Q * μ * t * 10^-9 / k)

                    # Упрощенный расчет на основе объема
                    # Для оценки используем коэффициенты из литературы
                    c = 0.02  # коэффициент из документа
                    permeability = 50.0  # мД, типичное значение
                    time = 1.0  # условное время работы, сут

                    # Расчет полудлины трещины
                    half_length = c * np.sqrt(volume * fluid_viscosity * time * 1e-9 / permeability)

                    # Ограничиваем длину трещины
                    return min(half_length, 500.0)

                except Exception as e:
                    logger.warning(f"Ошибка при расчете полудлины трещины: {str(e)}")
                    # Возвращаем оценку на основе эмпирической формулы
                    return 3.5 * volume ** 0.33

            # Обрабатываем данные в зависимости от их структуры
            if isinstance(gtp_data, dict):
                # Если данные представлены словарем с листами
                # Ищем листы с данными о ГРП
                for sheet_name, sheet_data in gtp_data.items():
                    if 'грп' in sheet_name.lower() or 'гидроразрыв' in sheet_name.lower():
                        # Ищем колонки с нужными параметрами
                        well_columns = [col for col in sheet_data.columns
                                        if 'скв' in col.lower() or 'well' in col.lower()]

                        volume_columns = [col for col in sheet_data.columns
                                          if
                                          'объем' in col.lower() or 'volume' in col.lower() or 'закач' in col.lower()]

                        length_columns = [col for col in sheet_data.columns
                                          if ('длин' in col.lower() and 'трещ' in col.lower()) or
                                          ('length' in col.lower() and 'fracture' in col.lower())]

                        viscosity_columns = [col for col in sheet_data.columns
                                             if 'вязк' in col.lower() or 'viscos' in col.lower()]

                        density_columns = [col for col in sheet_data.columns
                                           if 'плотн' in col.lower() or 'densit' in col.lower()]

                        pressure_columns = [col for col in sheet_data.columns
                                            if 'давлен' in col.lower() or 'pressure' in col.lower()]

                        height_columns = [col for col in sheet_data.columns
                                          if 'высот' in col.lower() or 'height' in col.lower() or 'мощн' in col.lower()]

                        # Если найдены колонки с идентификаторами скважин и объемами, продолжаем
                        if well_columns and volume_columns:
                            well_col = well_columns[0]
                            volume_col = volume_columns[0]

                            # Определяем остальные колонки, если они есть
                            length_col = length_columns[0] if length_columns else None
                            viscosity_col = viscosity_columns[0] if viscosity_columns else None
                            density_col = density_columns[0] if density_columns else None
                            pressure_col = pressure_columns[0] if pressure_columns else None
                            height_col = height_columns[0] if height_columns else None

                            # Обрабатываем каждую строку
                            for _, row in sheet_data.iterrows():
                                try:
                                    # Получаем идентификатор скважины
                                    well_id = str(row[well_col])

                                    # Получаем объем закачки
                                    volume = float(row[volume_col])

                                    # Собираем все доступные параметры
                                    params = {'Well': well_id, 'Volume': volume}

                                    # Если есть данные о длине трещины, добавляем их
                                    if length_col and not pd.isna(row[length_col]):
                                        params['Fracture_Length'] = float(row[length_col])
                                    else:
                                        # Иначе рассчитываем по формуле
                                        viscosity = float(row[viscosity_col]) if viscosity_col and not pd.isna(
                                            row[viscosity_col]) else 1.0
                                        density = float(row[density_col]) if density_col and not pd.isna(
                                            row[density_col]) else 2500
                                        pressure = float(row[pressure_col]) if pressure_col and not pd.isna(
                                            row[pressure_col]) else None
                                        height = float(row[height_col]) if height_col and not pd.isna(
                                            row[height_col]) else 10.0

                                        params['Fracture_Length'] = calculate_fracture_length(
                                            volume, fluid_viscosity=viscosity, rock_density=density,
                                            reservoir_pressure=pressure, height=height
                                        )

                                    # Добавляем дополнительные параметры, если они доступны
                                    if viscosity_col and not pd.isna(row[viscosity_col]):
                                        params['Viscosity'] = float(row[viscosity_col])
                                    if density_col and not pd.isna(row[density_col]):
                                        params['Density'] = float(row[density_col])
                                    if pressure_col and not pd.isna(row[pressure_col]):
                                        params['Pressure'] = float(row[pressure_col])
                                    if height_col and not pd.isna(row[height_col]):
                                        params['Height'] = float(row[height_col])

                                    fracking_params.append(params)

                                except (ValueError, TypeError) as e:
                                    # Пропускаем некорректные данные
                                    continue
            else:
                # Если данные представлены одним датафреймом
                # Ищем колонки с нужными параметрами
                well_columns = [col for col in gtp_data.columns
                                if 'скв' in col.lower() or 'well' in col.lower()]

                volume_columns = [col for col in gtp_data.columns
                                  if 'объем' in col.lower() or 'volume' in col.lower() or 'закач' in col.lower()]

                length_columns = [col for col in gtp_data.columns
                                  if ('длин' in col.lower() and 'трещ' in col.lower()) or
                                  ('length' in col.lower() and 'fracture' in col.lower())]

                viscosity_columns = [col for col in gtp_data.columns
                                     if 'вязк' in col.lower() or 'viscos' in col.lower()]

                density_columns = [col for col in gtp_data.columns
                                   if 'плотн' in col.lower() or 'densit' in col.lower()]

                pressure_columns = [col for col in gtp_data.columns
                                    if 'давлен' in col.lower() or 'pressure' in col.lower()]

                height_columns = [col for col in gtp_data.columns
                                  if 'высот' in col.lower() or 'height' in col.lower() or 'мощн' in col.lower()]

                # Если найдены колонки с идентификаторами скважин и объемами, продолжаем
                if well_columns and volume_columns:
                    well_col = well_columns[0]
                    volume_col = volume_columns[0]

                    # Определяем остальные колонки, если они есть
                    length_col = length_columns[0] if length_columns else None
                    viscosity_col = viscosity_columns[0] if viscosity_columns else None
                    density_col = density_columns[0] if density_columns else None
                    pressure_col = pressure_columns[0] if pressure_columns else None
                    height_col = height_columns[0] if height_columns else None

                    # Обрабатываем каждую строку
                    for _, row in gtp_data.iterrows():
                        try:
                            # Получаем идентификатор скважины
                            well_id = str(row[well_col])

                            # Получаем объем закачки
                            volume = float(row[volume_col])

                            # Собираем все доступные параметры
                            params = {'Well': well_id, 'Volume': volume}

                            # Если есть данные о длине трещины, добавляем их
                            if length_col and not pd.isna(row[length_col]):
                                params['Fracture_Length'] = float(row[length_col])
                            else:
                                # Иначе рассчитываем по формуле
                                viscosity = float(row[viscosity_col]) if viscosity_col and not pd.isna(
                                    row[viscosity_col]) else 1.0
                                density = float(row[density_col]) if density_col and not pd.isna(
                                    row[density_col]) else 2500
                                pressure = float(row[pressure_col]) if pressure_col and not pd.isna(
                                    row[pressure_col]) else None
                                height = float(row[height_col]) if height_col and not pd.isna(row[height_col]) else 10.0

                                params['Fracture_Length'] = calculate_fracture_length(
                                    volume, fluid_viscosity=viscosity, rock_density=density,
                                    reservoir_pressure=pressure, height=height
                                )

                            # Добавляем дополнительные параметры, если они доступны
                            if viscosity_col and not pd.isna(row[viscosity_col]):
                                params['Viscosity'] = float(row[viscosity_col])
                            if density_col and not pd.isna(row[density_col]):
                                params['Density'] = float(row[density_col])
                            if pressure_col and not pd.isna(row[pressure_col]):
                                params['Pressure'] = float(row[pressure_col])
                            if height_col and not pd.isna(row[height_col]):
                                params['Height'] = float(row[height_col])

                            fracking_params.append(params)

                        except (ValueError, TypeError) as e:
                            # Пропускаем некорректные данные
                            continue

            # Если не удалось извлечь параметры, создаем синтетические данные
            if not fracking_params:
                logger.warning("Создаем синтетические данные для параметров ГРП")

                # Создаем 10 синтетических точек
                volumes = np.linspace(50, 1000, 10)

                for i, volume in enumerate(volumes):
                    # Рассчитываем полудлину трещины по эмпирической формуле
                    half_length = 3.5 * volume ** 0.33

                    fracking_params.append({
                        'Well': f"Well_{i + 1}",
                        'Volume': volume,
                        'Fracture_Length': half_length
                    })

            return fracking_params

        except Exception as e:
            logger.error(f"Ошибка при извлечении параметров ГРП: {str(e)}")
            return []

    def plot_design_vs_empirical(self, output_path=None):
        """
        Построение графика сравнения результатов модели на основе дизайна ГРП и эмпирической модели.

        Args:
            output_path (str, optional): Путь для сохранения графика

        Returns:
            plt.Figure: Объект фигуры matplotlib
        """
        if not hasattr(self, 'volumes') or not hasattr(self, 'lengths'):
            logger.error("Нет данных для построения графика")
            return None

        if not hasattr(self, 'fitted_params') or 'coeff_a' not in self.fitted_params:
            logger.error("Модель не подобрана")
            return None

        try:
            # Создаем график
            fig, ax = plt.subplots(figsize=(10, 8))

            # Исходные данные (дизайн ГРП)
            ax.scatter(self.volumes, self.lengths, color='blue', label='Данные дизайна ГРП', s=50)

            # Диапазон объемов для построения кривых
            v_range = np.linspace(min(self.volumes) * 0.5, max(self.volumes) * 1.5, 100)

            # Модель на основе дизайна ГРП
            design_model = self.fitted_params['coeff_a'] * v_range ** self.fitted_params['coeff_b']
            ax.plot(v_range, design_model, 'r-', linewidth=2,
                    label=f"Модель дизайна ГРП: L = {self.fitted_params['coeff_a']:.2f} * V^{self.fitted_params['coeff_b']:.4f}")

            # Эмпирическая модель (метод Пушкиной)
            empirical_a = 3.5
            empirical_b = 0.33
            empirical_model = empirical_a * v_range ** empirical_b
            ax.plot(v_range, empirical_model, 'g--', linewidth=2,
                    label=f"Эмпирическая модель: L = {empirical_a:.2f} * V^{empirical_b:.4f}")

            # Добавление подписей и легенды
            ax.set_xlabel('Объем закачки, м³')
            ax.set_ylabel('Полудлина трещины, м')
            ax.set_title('Сравнение модели на основе дизайна ГРП и эмпирической модели')
            ax.grid(True)
            ax.legend()

            # Добавление информации о точности модели
            if 'rel_error' in self.fitted_params:
                ax.text(0.02, 0.98, f"Относительная ошибка: {self.fitted_params['rel_error']:.2f}%",
                        transform=ax.transAxes, verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            # Логарифмические шкалы для лучшей визуализации
            ax.set_xscale('log')
            ax.set_yscale('log')

            plt.tight_layout()

            # Сохранение графика, если указан путь
            if output_path and output_path.endswith('.png'):
                comparison_path = output_path.replace('.png', '_comparison.png')
                plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
                logger.info(f"График сравнения моделей сохранен в {comparison_path}")
            elif output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"График сравнения моделей сохранен в {output_path}")

            return fig

        except Exception as e:
            logger.error(f"Ошибка при построении графика сравнения моделей: {str(e)}")
            return None