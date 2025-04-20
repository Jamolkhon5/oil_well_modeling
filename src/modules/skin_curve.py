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

    def calculate_skin_from_gdis(self):
        """
        Расчет параметров изменения скин-фактора на основе данных ГДИС.

        Returns:
            bool: True если расчет выполнен успешно, иначе False
        """
        if self.data is None:
            logger.error("Модель не инициализирована данными")
            return False

        logger.info("Начинаем расчет параметров изменения скин-фактора на основе данных ГДИС...")

        try:
            # Получаем данные ГДИС и ГРП
            gdi_data = self.data.get('gdi_reint_data')
            gtp_data = self.data.get('gtp_data')

            if gdi_data is None:
                logger.warning("Отсутствуют данные ГДИС, создаем синтетические данные")
                # Если данных нет, возвращаемся к базовому методу
                return False

            # Извлекаем данные о скин-факторе и ГРП для скважин
            skin_data = self._extract_skin_data_from_gdis(gdi_data, gtp_data)

            if not skin_data:
                logger.warning("Не удалось извлечь данные о скин-факторе из ГДИС")
                return False

            # Выполняем аппроксимацию для каждой скважины
            fitted_params = {}
            time_points_all = []
            skin_values_all = []

            for well, data in skin_data.items():
                # Если для скважины есть несколько измерений скин-фактора
                if len(data['time']) >= 2:
                    # Нормализуем время (дни после ГРП)
                    t = np.array(data['time'])

                    # Значения скин-фактора
                    skin = np.array(data['skin'])

                    try:
                        # Аппроксимация функцией из базового метода
                        popt, pcov = curve_fit(
                            self._skin_curve_function,
                            t,
                            skin,
                            bounds=([-5.0, -5.0, 0.001], [2.0, 5.0, 0.1])
                        )

                        fitted_params[well] = {
                            'initial_skin': popt[0],
                            'max_skin': popt[1],
                            'growth_rate': popt[2]
                        }

                        # Добавляем данные для общей модели
                        time_points_all.extend(t)
                        skin_values_all.extend(skin)
                    except Exception as e:
                        logger.warning(f"Не удалось выполнить аппроксимацию для скважины {well}: {str(e)}")

            # Если удалось получить параметры хотя бы для одной скважины
            if fitted_params:
                # Рассчитываем средние значения параметров
                avg_initial_skin = np.mean([p['initial_skin'] for p in fitted_params.values()])
                avg_max_skin = np.mean([p['max_skin'] for p in fitted_params.values()])
                avg_growth_rate = np.mean([p['growth_rate'] for p in fitted_params.values()])

                # Если есть достаточно данных для общей модели, выполняем еще одну аппроксимацию
                if len(time_points_all) >= 5:
                    try:
                        # Аппроксимация по всем данным
                        popt, pcov = curve_fit(
                            self._skin_curve_function,
                            np.array(time_points_all),
                            np.array(skin_values_all),
                            p0=[avg_initial_skin, avg_max_skin, avg_growth_rate],
                            bounds=([-5.0, -5.0, 0.001], [2.0, 5.0, 0.1])
                        )

                        # Сохраняем параметры модели
                        self.fitted_params = {
                            'initial_skin': popt[0],
                            'max_skin': popt[1],
                            'growth_rate': popt[2]
                        }

                        # Обновляем параметры модели
                        self.params.update(self.fitted_params)

                        # Сохраняем данные для построения графиков
                        self.time_points = np.array(time_points_all)
                        self.skin_values = np.array(skin_values_all)

                        logger.info("Параметры изменения скин-фактора успешно рассчитаны на основе данных ГДИС")
                        logger.info(
                            f"Общая модель: initial_skin={popt[0]:.2f}, max_skin={popt[1]:.2f}, growth_rate={popt[2]:.6f}")

                        # Сохраняем также данные по отдельным скважинам
                        self.well_fitted_params = fitted_params

                        return True
                    except Exception as e:
                        logger.warning(f"Не удалось выполнить общую аппроксимацию: {str(e)}")

                # Если общая аппроксимация не удалась, используем средние значения
                self.fitted_params = {
                    'initial_skin': avg_initial_skin,
                    'max_skin': avg_max_skin,
                    'growth_rate': avg_growth_rate
                }

                # Обновляем параметры модели
                self.params.update(self.fitted_params)

                logger.info("Параметры изменения скин-фактора рассчитаны как средние по скважинам")
                logger.info(
                    f"Средние значения: initial_skin={avg_initial_skin:.2f}, max_skin={avg_max_skin:.2f}, growth_rate={avg_growth_rate:.6f}")

                # Сохраняем данные по отдельным скважинам
                self.well_fitted_params = fitted_params

                return True
            else:
                logger.warning("Не удалось рассчитать параметры ни для одной скважины")
                return False

        except Exception as e:
            logger.error(f"Ошибка при расчете параметров на основе данных ГДИС: {str(e)}")
            return False

    def _extract_skin_data_from_gdis(self, gdi_data, gtp_data=None):
        """
        Извлечение данных о скин-факторе из данных ГДИС.

        Args:
            gdi_data: Данные ГДИС
            gtp_data: Данные ГТР (опционально)

        Returns:
            dict: Словарь с данными о скин-факторе для каждой скважины
        """
        skin_data = {}

        try:
            # Функция для определения даты ГРП
            def find_fracking_date(well_id, gtp_data):
                if gtp_data is None:
                    return None

                # Ищем дату ГРП в данных ГТР
                if isinstance(gtp_data, dict):
                    # Если данные представлены словарем с листами
                    fracking_dates = {}

                    for sheet_name, sheet_data in gtp_data.items():
                        if 'грп' in sheet_name.lower() or 'гидроразрыв' in sheet_name.lower():
                            # Ищем колонки с идентификаторами скважин и датами
                            well_columns = [col for col in sheet_data.columns
                                            if 'скв' in col.lower() or 'well' in col.lower()]

                            date_columns = [col for col in sheet_data.columns
                                            if 'дата' in col.lower() or 'date' in col.lower()]

                            if well_columns and date_columns:
                                well_col = well_columns[0]
                                date_col = date_columns[0]

                                # Ищем скважину в данных
                                well_rows = sheet_data[sheet_data[well_col].astype(str) == str(well_id)]

                                if not well_rows.empty:
                                    # Берем самую последнюю дату ГРП
                                    latest_date = pd.to_datetime(well_rows[date_col], errors='coerce').max()
                                    fracking_dates[sheet_name] = latest_date

                    # Если найдено несколько дат, берем самую последнюю
                    if fracking_dates:
                        return max(fracking_dates.values())
                    else:
                        return None
                else:
                    # Если данные представлены одним датафреймом
                    # Ищем колонки с идентификаторами скважин и датами
                    well_columns = [col for col in gtp_data.columns
                                    if 'скв' in col.lower() or 'well' in col.lower()]

                    date_columns = [col for col in gtp_data.columns
                                    if 'дата' in col.lower() or 'date' in col.lower()]

                    if well_columns and date_columns:
                        well_col = well_columns[0]
                        date_col = date_columns[0]

                        # Ищем скважину в данных
                        well_rows = gtp_data[gtp_data[well_col].astype(str) == str(well_id)]

                        if not well_rows.empty:
                            # Берем самую последнюю дату ГРП
                            return pd.to_datetime(well_rows[date_col], errors='coerce').max()

                return None

            # Обрабатываем данные ГДИС
            if isinstance(gdi_data, dict):
                # Если данные представлены словарем с листами
                for sheet_name, sheet_data in gdi_data.items():
                    if ('скин' in sheet_name.lower() or 'skin' in sheet_name.lower() or
                            'гди' in sheet_name.lower() or 'исслед' in sheet_name.lower()):
                        # Ищем колонки с идентификаторами скважин, скин-фактором и датами
                        well_columns = [col for col in sheet_data.columns
                                        if 'скв' in col.lower() or 'well' in col.lower()]

                        skin_columns = [col for col in sheet_data.columns
                                        if 'скин' in col.lower() or 'skin' in col.lower()]

                        date_columns = [col for col in sheet_data.columns
                                        if 'дата' in col.lower() or 'date' in col.lower()]

                        if well_columns and skin_columns and date_columns:
                            well_col = well_columns[0]
                            skin_col = skin_columns[0]
                            date_col = date_columns[0]

                            # Для каждой скважины извлекаем данные о скин-факторе
                            for well_id in sheet_data[well_col].unique():
                                if pd.isna(well_id):
                                    continue

                                well_id_str = str(well_id)

                                # Ищем дату ГРП для скважины
                                fracking_date = find_fracking_date(well_id_str, gtp_data)

                                # Выбираем исследования для скважины
                                well_rows = sheet_data[sheet_data[well_col].astype(str) == well_id_str]

                                # Пропускаем, если нет данных
                                if well_rows.empty:
                                    continue

                                # Инициализируем данные для скважины
                                if well_id_str not in skin_data:
                                    skin_data[well_id_str] = {'time': [], 'skin': []}

                                # Обрабатываем каждое исследование
                                for _, row in well_rows.iterrows():
                                    try:
                                        # Получаем дату исследования
                                        test_date = pd.to_datetime(row[date_col], errors='coerce')

                                        # Пропускаем, если дата не определена
                                        if pd.isna(test_date):
                                            continue

                                        # Получаем значение скин-фактора
                                        skin_value = float(row[skin_col])

                                        # Если дата ГРП известна, рассчитываем время после ГРП в днях
                                        if fracking_date is not None:
                                            # Пропускаем исследования до ГРП
                                            if test_date < fracking_date:
                                                continue

                                            time_days = (test_date - fracking_date).days
                                        else:
                                            # Если дата ГРП неизвестна, используем условное время
                                            # (номер исследования в хронологическом порядке)
                                            if not skin_data[well_id_str]['time']:
                                                time_days = 0
                                            else:
                                                # Предполагаем, что исследования идут с интервалом в 30 дней
                                                time_days = max(skin_data[well_id_str]['time']) + 30

                                        # Добавляем данные
                                        skin_data[well_id_str]['time'].append(time_days)
                                        skin_data[well_id_str]['skin'].append(skin_value)

                                    except (ValueError, TypeError) as e:
                                        # Пропускаем некорректные данные
                                        continue
            else:
                # Если данные представлены одним датафреймом
                # Ищем колонки с идентификаторами скважин, скин-фактором и датами
                well_columns = [col for col in gdi_data.columns
                                if 'скв' in col.lower() or 'well' in col.lower()]

                skin_columns = [col for col in gdi_data.columns
                                if 'скин' in col.lower() or 'skin' in col.lower()]

                date_columns = [col for col in gdi_data.columns
                                if 'дата' in col.lower() or 'date' in col.lower()]

                if well_columns and skin_columns and date_columns:
                    well_col = well_columns[0]
                    skin_col = skin_columns[0]
                    date_col = date_columns[0]

                    # Для каждой скважины извлекаем данные о скин-факторе
                    for well_id in gdi_data[well_col].unique():
                        if pd.isna(well_id):
                            continue

                        well_id_str = str(well_id)

                        # Ищем дату ГРП для скважины
                        fracking_date = find_fracking_date(well_id_str, gtp_data)

                        # Выбираем исследования для скважины
                        well_rows = gdi_data[gdi_data[well_col].astype(str) == well_id_str]

                        # Пропускаем, если нет данных
                        if well_rows.empty:
                            continue

                        # Инициализируем данные для скважины
                        if well_id_str not in skin_data:
                            skin_data[well_id_str] = {'time': [], 'skin': []}

                        # Обрабатываем каждое исследование
                        for _, row in well_rows.iterrows():
                            try:
                                # Получаем дату исследования
                                test_date = pd.to_datetime(row[date_col], errors='coerce')

                                # Пропускаем, если дата не определена
                                if pd.isna(test_date):
                                    continue

                                # Получаем значение скин-фактора
                                skin_value = float(row[skin_col])

                                # Если дата ГРП известна, рассчитываем время после ГРП в днях
                                if fracking_date is not None:
                                    # Пропускаем исследования до ГРП
                                    if test_date < fracking_date:
                                        continue

                                    time_days = (test_date - fracking_date).days
                                else:
                                    # Если дата ГРП неизвестна, используем условное время
                                    # (номер исследования в хронологическом порядке)
                                    if not skin_data[well_id_str]['time']:
                                        time_days = 0
                                    else:
                                        # Предполагаем, что исследования идут с интервалом в 30 дней
                                        time_days = max(skin_data[well_id_str]['time']) + 30

                                # Добавляем данные
                                skin_data[well_id_str]['time'].append(time_days)
                                skin_data[well_id_str]['skin'].append(skin_value)

                            except (ValueError, TypeError) as e:
                                # Пропускаем некорректные данные
                                continue

            # Сортируем данные по времени для каждой скважины
            for well_id, data in skin_data.items():
                if data['time'] and data['skin']:
                    # Сортируем данные по времени
                    sorted_indices = np.argsort(data['time'])
                    data['time'] = [data['time'][i] for i in sorted_indices]
                    data['skin'] = [data['skin'][i] for i in sorted_indices]

            # Удаляем скважины с недостаточным количеством данных
            return {well_id: data for well_id, data in skin_data.items() if len(data['time']) >= 2}

        except Exception as e:
            logger.error(f"Ошибка при извлечении данных о скин-факторе из ГДИС: {str(e)}")
            return {}

    def plot_well_skin_curves(self, output_path=None):
        """
        Построение графиков изменения скин-фактора для отдельных скважин.

        Args:
            output_path (str, optional): Путь для сохранения графика

        Returns:
            plt.Figure: Объект фигуры matplotlib
        """
        if not hasattr(self, 'well_fitted_params') or not self.well_fitted_params:
            logger.error("Нет данных по отдельным скважинам")
            return None

        try:
            # Определяем количество скважин для отображения
            well_ids = list(self.well_fitted_params.keys())
            n_wells = min(len(well_ids), 6)  # Максимум 6 скважин на графике

            # Создаем график
            fig, axs = plt.subplots(n_wells, 1, figsize=(10, 3 * n_wells), sharex=True)

            # Если только одна скважина, превращаем axs в список
            if n_wells == 1:
                axs = [axs]

            # Строим графики для каждой скважины
            for i in range(n_wells):
                well_id = well_ids[i]
                params = self.well_fitted_params[well_id]

                # Временные точки для построения кривой
                t_pred = np.linspace(0, 365 * 2, 100)  # 2 года

                # Расчет значений скин-фактора
                skin_pred = np.array([
                    calculate_skin_factor_after_fracking(
                        t,
                        params['initial_skin'],
                        params['max_skin'],
                        params['growth_rate']
                    ) for t in t_pred
                ])

                # Построение графика для скважины
                axs[i].plot(t_pred, skin_pred, 'r-', linewidth=2)

                # Добавление информации о параметрах
                params_text = (
                    f"Скважина {well_id}\n"
                    f"Начальный скин: {params['initial_skin']:.2f}\n"
                    f"Максимальный скин: {params['max_skin']:.2f}\n"
                    f"Скорость роста: {params['growth_rate']:.6f}"
                )
                axs[i].text(0.02, 0.98, params_text, transform=axs[i].transAxes,
                            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                axs[i].set_ylabel('Скин-фактор')
                axs[i].grid(True)

            # Общие подписи
            axs[-1].set_xlabel('Время после ГРП, сут')
            fig.suptitle('Изменение скин-фактора после ГРП для отдельных скважин')

            plt.tight_layout()
            plt.subplots_adjust(top=0.95)

            # Сохранение графика, если указан путь
            if output_path and output_path.endswith('.png'):
                wells_path = output_path.replace('.png', '_wells.png')
                plt.savefig(wells_path, dpi=300, bbox_inches='tight')
                logger.info(f"График скин-фактора для отдельных скважин сохранен в {wells_path}")
            elif output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"График скин-фактора для отдельных скважин сохранен в {output_path}")

            return fig

        except Exception as e:
            logger.error(f"Ошибка при построении графиков для отдельных скважин: {str(e)}")
            return None