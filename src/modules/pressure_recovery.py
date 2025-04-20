"""
Модуль 4: Подбор времени восстановления давления в остановленных скважинах.

Этот модуль реализует расчет времени восстановления давления
в остановленных скважинах.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from src.utils import calculate_pressure_recovery_time

logger = logging.getLogger(__name__)


class PressureRecoveryModel:
    """
    Модель для расчета времени восстановления давления в остановленных скважинах.
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
            logger.info("Модель расчета времени восстановления давления инициализирована данными")
            return True
        except Exception as e:
            logger.error(f"Ошибка при инициализации данными: {str(e)}")
            return False

    def calculate_recovery_times(self):
        """
        Расчет времени восстановления давления в остановленных скважинах.

        Returns:
            bool: True если расчет выполнен успешно, иначе False
        """
        if self.data is None:
            logger.error("Модель не инициализирована данными")
            return False

        logger.info("Начинаем расчет времени восстановления давления...")

        try:
            # Здесь будет код для расчета времени восстановления давления
            # в соответствии с методикой из документа

            # Для примера создадим случайные данные
            # В реальном проекте здесь будет сложный расчет по методике

            # Создаем фиктивный набор данных для скважин
            wells = [f"Well_{i}" for i in range(1, 11)]

            # Параметры скважин
            permeability = np.random.uniform(10, 100, len(wells))  # Проницаемость, мД
            porosity = np.random.uniform(0.1, 0.3, len(wells))  # Пористость, д.ед.
            viscosity = np.random.uniform(1, 10, len(wells))  # Вязкость, сПз
            skin_factor = np.random.uniform(-2, 5, len(wells))  # Скин-фактор
            well_radius = np.ones(len(wells)) * 0.1  # Радиус скважины, м

            # Расчет времени восстановления для каждой скважины
            recovery_times = np.array([
                calculate_pressure_recovery_time(
                    permeability[i],
                    porosity[i],
                    viscosity[i],
                    skin_factor[i],
                    well_radius[i]
                )
                for i in range(len(wells))
            ])

            # Создаем датафрейм с результатами
            self.results = pd.DataFrame({
                'Well': wells,
                'Permeability': permeability,
                'Porosity': porosity,
                'Viscosity': viscosity,
                'Skin_Factor': skin_factor,
                'Well_Radius': well_radius,
                'Recovery_Time': recovery_times
            })

            logger.info("Расчет времени восстановления давления успешно выполнен")
            return True

        except Exception as e:
            logger.error(f"Ошибка при расчете времени восстановления давления: {str(e)}")
            return False

    def plot_recovery_times(self, output_path=None):
        """
        Построение графика времени восстановления давления.

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
            recovery_times = self.results['Recovery_Time']

            # Сортируем скважины по времени восстановления
            sorted_indices = np.argsort(recovery_times)
            sorted_wells = [wells[i] for i in sorted_indices]
            sorted_times = [recovery_times[i] for i in sorted_indices]

            # Строим гистограмму
            ax.bar(range(len(sorted_wells)), sorted_times, color='skyblue')

            ax.set_xlabel('Скважины')
            ax.set_ylabel('Время восстановления, сут.')
            ax.set_title('Время восстановления давления в остановленных скважинах')
            ax.set_xticks(range(len(sorted_wells)))
            ax.set_xticklabels(sorted_wells, rotation=45)

            # Добавляем значения над столбцами
            for i, v in enumerate(sorted_times):
                ax.text(i, v + 0.1, f"{v:.1f}", ha='center')

            plt.tight_layout()

            # Сохранение графика, если указан путь
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"График сохранен в {output_path}")

            return fig

        except Exception as e:
            logger.error(f"Ошибка при построении графика: {str(e)}")
            return None

    def plot_parameters_influence(self, output_path=None):
        """
        Построение графиков влияния параметров на время восстановления.

        Args:
            output_path (str, optional): Путь для сохранения графика

        Returns:
            plt.Figure: Объект фигуры matplotlib
        """
        if self.results is None:
            logger.error("Нет данных для построения графика")
            return None

        try:
            # Создаем графики зависимостей
            fig, axs = plt.subplots(2, 2, figsize=(14, 10))

            # Зависимость от проницаемости
            axs[0, 0].scatter(self.results['Permeability'], self.results['Recovery_Time'])
            axs[0, 0].set_xlabel('Проницаемость, мД')
            axs[0, 0].set_ylabel('Время восстановления, сут.')
            axs[0, 0].set_title('Зависимость от проницаемости')
            axs[0, 0].grid(True)

            # Зависимость от пористости
            axs[0, 1].scatter(self.results['Porosity'], self.results['Recovery_Time'])
            axs[0, 1].set_xlabel('Пористость, д.ед.')
            axs[0, 1].set_ylabel('Время восстановления, сут.')
            axs[0, 1].set_title('Зависимость от пористости')
            axs[0, 1].grid(True)

            # Зависимость от вязкости
            axs[1, 0].scatter(self.results['Viscosity'], self.results['Recovery_Time'])
            axs[1, 0].set_xlabel('Вязкость, сПз')
            axs[1, 0].set_ylabel('Время восстановления, сут.')
            axs[1, 0].set_title('Зависимость от вязкости')
            axs[1, 0].grid(True)

            # Зависимость от скин-фактора
            axs[1, 1].scatter(self.results['Skin_Factor'], self.results['Recovery_Time'])
            axs[1, 1].set_xlabel('Скин-фактор')
            axs[1, 1].set_ylabel('Время восстановления, сут.')
            axs[1, 1].set_title('Зависимость от скин-фактора')
            axs[1, 1].grid(True)

            plt.tight_layout()

            # Сохранение графика, если указан путь
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"График зависимостей сохранен в {output_path}")

            return fig

        except Exception as e:
            logger.error(f"Ошибка при построении графиков влияния параметров: {str(e)}")
            return None

    def calculate_linear_flow_recovery(self):
        """
        Расчет времени восстановления давления по уравнению линейного стока и данным ГДИС.

        Returns:
            bool: True если расчет выполнен успешно, иначе False
        """
        if self.data is None:
            logger.error("Модель не инициализирована данными")
            return False

        logger.info("Начинаем расчет времени восстановления давления по уравнению линейного стока...")

        try:
            # Получаем данные ГДИС
            gdi_data = self.data.get('gdi_reint_data')

            if gdi_data is None:
                logger.warning("Отсутствуют данные ГДИС, используем только базовый расчет")
                return False

            # Обрабатываем данные ГДИС для получения параметров пласта
            well_params = self._extract_well_parameters_from_gdi(gdi_data)

            if not well_params:
                logger.warning("Не удалось извлечь параметры скважин из данных ГДИС")
                return False

            # Обновляем результаты с учетом расчета по уравнению линейного стока
            if self.results is None:
                # Если результаты еще не созданы, создаем их
                self.results = pd.DataFrame({
                    'Well': list(well_params.keys()),
                    'Permeability': [params.get('permeability', np.nan) for params in well_params.values()],
                    'Porosity': [params.get('porosity', np.nan) for params in well_params.values()],
                    'Viscosity': [params.get('viscosity', np.nan) for params in well_params.values()],
                    'Skin_Factor': [params.get('skin_factor', np.nan) for params in well_params.values()],
                    'Well_Radius': [params.get('well_radius', 0.1) for params in well_params.values()],
                    'Compressibility': [params.get('compressibility', 1e-5) for params in well_params.values()],
                    'Reservoir_Height': [params.get('height', 10.0) for params in well_params.values()],
                    'Initial_Flow_Rate': [params.get('flow_rate', 50.0) for params in well_params.values()]
                })
            else:
                # Если результаты уже существуют, добавляем недостающие параметры
                for well, params in well_params.items():
                    idx = self.results.index[self.results['Well'] == well].tolist()
                    if idx:
                        for param_name, param_value in params.items():
                            if param_name == 'permeability':
                                self.results.loc[idx[0], 'Permeability'] = param_value
                            elif param_name == 'porosity':
                                self.results.loc[idx[0], 'Porosity'] = param_value
                            elif param_name == 'viscosity':
                                self.results.loc[idx[0], 'Viscosity'] = param_value
                            elif param_name == 'skin_factor':
                                self.results.loc[idx[0], 'Skin_Factor'] = param_value
                            elif param_name == 'well_radius':
                                self.results.loc[idx[0], 'Well_Radius'] = param_value
                            elif param_name == 'compressibility':
                                self.results.loc[idx[0], 'Compressibility'] = param_value
                            elif param_name == 'height':
                                self.results.loc[idx[0], 'Reservoir_Height'] = param_value
                            elif param_name == 'flow_rate':
                                self.results.loc[idx[0], 'Initial_Flow_Rate'] = param_value

            # Рассчитываем время восстановления по уравнению линейного стока
            linear_flow_recovery_times = []

            for _, row in self.results.iterrows():
                # Проверяем наличие всех необходимых параметров
                if (np.isnan(row['Permeability']) or np.isnan(row['Porosity']) or
                        np.isnan(row['Viscosity']) or np.isnan(row['Compressibility'])):
                    # Если каких-то параметров нет, используем базовый расчет
                    recovery_time = calculate_pressure_recovery_time(
                        row['Permeability'] if not np.isnan(row['Permeability']) else 50.0,
                        row['Porosity'] if not np.isnan(row['Porosity']) else 0.2,
                        row['Viscosity'] if not np.isnan(row['Viscosity']) else 1.0,
                        row['Skin_Factor'] if not np.isnan(row['Skin_Factor']) else 0.0,
                        row['Well_Radius'] if not np.isnan(row['Well_Radius']) else 0.1
                    )
                else:
                    # Используем уравнение линейного стока для расчета
                    # Формула из документа (раздел 5)
                    # t = 730.46 * (φ * μ * ct * r_inv²) / k
                    # где r_inv = 0.037 * sqrt(k * t / (φ * μ * ct))

                    # Константы и параметры
                    k = row['Permeability']  # [мД]
                    porosity = row['Porosity']  # [д.ед.]
                    viscosity = row['Viscosity']  # [сПз]
                    compressibility = row['Compressibility']  # [1/атм]
                    skin = row['Skin_Factor'] if not np.isnan(row['Skin_Factor']) else 0.0
                    well_radius = row['Well_Radius'] if not np.isnan(row['Well_Radius']) else 0.1  # [м]
                    height = row['Reservoir_Height'] if not np.isnan(row['Reservoir_Height']) else 10.0  # [м]
                    flow_rate = row['Initial_Flow_Rate'] if not np.isnan(row['Initial_Flow_Rate']) else 50.0  # [м³/сут]

                    # Определяем радиус исследования (r_inv)
                    # Для начального приближения используем время 30 дней
                    initial_time = 30.0  # [сут]
                    r_inv = 0.037 * np.sqrt(k * initial_time / (porosity * viscosity * compressibility))

                    # Итеративно уточняем время восстановления
                    for _ in range(10):  # 10 итераций должно быть достаточно для сходимости
                        recovery_time = 730.46 * (porosity * viscosity * compressibility * r_inv ** 2) / k
                        # Обновляем радиус исследования
                        r_inv_new = 0.037 * np.sqrt(k * recovery_time / (porosity * viscosity * compressibility))
                        # Проверяем сходимость
                        if abs(r_inv_new - r_inv) < 0.1:
                            break
                        r_inv = r_inv_new

                    # Учитываем влияние скин-фактора
                    if skin > 0:
                        # Положительный скин-фактор увеличивает время восстановления
                        recovery_time *= (1.0 + 0.5 * skin)
                    else:
                        # Отрицательный скин-фактор (ГРП) уменьшает время восстановления
                        recovery_time *= max(0.5, 1.0 + skin)

                    # Учитываем влияние дебита (больший дебит -> больше времени на восстановление)
                    flow_rate_factor = min(2.0, max(0.5, flow_rate / 50.0))
                    recovery_time *= flow_rate_factor

                linear_flow_recovery_times.append(recovery_time)

            # Добавляем результаты расчета в датафрейм
            self.results['Linear_Flow_Recovery_Time'] = linear_flow_recovery_times

            # Если ранее уже был выполнен базовый расчет, добавляем еще один столбец
            if 'Recovery_Time' in self.results.columns:
                # Рассчитываем средневзвешенное время восстановления
                weights = {
                    'basic': 0.3,  # Вес базового расчета
                    'linear_flow': 0.7  # Вес расчета по линейному стоку
                }

                self.results['Weighted_Recovery_Time'] = (
                        weights['basic'] * self.results['Recovery_Time'] +
                        weights['linear_flow'] * self.results['Linear_Flow_Recovery_Time']
                )
            else:
                # Если базового расчета не было, используем только расчет по линейному стоку
                self.results['Recovery_Time'] = self.results['Linear_Flow_Recovery_Time']

            logger.info("Расчет времени восстановления по уравнению линейного стока успешно выполнен")
            return True

        except Exception as e:
            logger.error(f"Ошибка при расчете времени восстановления по уравнению линейного стока: {str(e)}")
            return False

    def _extract_well_parameters_from_gdi(self, gdi_data):
        """
        Извлечение параметров скважин из данных ГДИС.

        Args:
            gdi_data: Данные ГДИС

        Returns:
            dict: Словарь с параметрами скважин
        """
        well_params = {}

        try:
            # Обрабатываем данные в зависимости от их структуры
            if isinstance(gdi_data, dict):
                # Если данные представлены словарем с листами
                # Ищем листы с нужными данными
                param_sheets = []

                for sheet_name, sheet_data in gdi_data.items():
                    if ('параметр' in sheet_name.lower() or
                            'проницаем' in sheet_name.lower() or
                            'skin' in sheet_name.lower() or
                            'гди' in sheet_name.lower()):
                        param_sheets.append((sheet_name, sheet_data))

                if not param_sheets:
                    logger.warning("Не найдены листы с параметрами скважин в данных ГДИС")
                    return {}

                # Обрабатываем каждый лист
                for sheet_name, sheet_data in param_sheets:
                    # Ищем колонки с идентификаторами скважин
                    well_columns = [col for col in sheet_data.columns
                                    if 'скв' in col.lower() or 'well' in col.lower()]

                    if not well_columns:
                        logger.warning(f"Не найдены колонки с идентификаторами скважин в листе {sheet_name}")
                        continue

                    well_col = well_columns[0]

                    # Для каждой скважины извлекаем параметры
                    for _, row in sheet_data.iterrows():
                        well_id = str(row[well_col])

                        if well_id not in well_params:
                            well_params[well_id] = {}

                        # Ищем параметры в строке
                        for col in sheet_data.columns:
                            col_lower = col.lower()
                            value = row[col]

                            # Пропускаем пустые значения
                            if pd.isna(value):
                                continue

                            # Проницаемость
                            if ('проницаем' in col_lower or 'permeab' in col_lower or 'пронитц' in col_lower):
                                try:
                                    well_params[well_id]['permeability'] = float(value)
                                except:
                                    pass

                            # Пористость
                            elif ('порист' in col_lower or 'porosit' in col_lower):
                                try:
                                    well_params[well_id]['porosity'] = float(value)
                                except:
                                    pass

                            # Вязкость
                            elif ('вязкость' in col_lower or 'viscosit' in col_lower):
                                try:
                                    well_params[well_id]['viscosity'] = float(value)
                                except:
                                    pass

                            # Скин-фактор
                            elif ('скин' in col_lower or 'skin' in col_lower):
                                try:
                                    well_params[well_id]['skin_factor'] = float(value)
                                except:
                                    pass

                            # Радиус скважины
                            elif ('радиус' in col_lower or 'radius' in col_lower):
                                try:
                                    well_params[well_id]['well_radius'] = float(value)
                                except:
                                    pass

                            # Сжимаемость
                            elif ('сжима' in col_lower or 'compress' in col_lower):
                                try:
                                    well_params[well_id]['compressibility'] = float(value)
                                except:
                                    pass

                            # Мощность пласта
                            elif ('мощность' in col_lower or 'толщина' in col_lower or 'height' in col_lower):
                                try:
                                    well_params[well_id]['height'] = float(value)
                                except:
                                    pass

                            # Дебит
                            elif ('дебит' in col_lower or 'rate' in col_lower or 'расход' in col_lower):
                                try:
                                    well_params[well_id]['flow_rate'] = float(value)
                                except:
                                    pass
            else:
                # Если данные представлены одним датафреймом
                # Ищем колонки с идентификаторами скважин
                well_columns = [col for col in gdi_data.columns
                                if 'скв' in col.lower() or 'well' in col.lower()]

                if not well_columns:
                    logger.warning("Не найдены колонки с идентификаторами скважин в данных ГДИС")
                    return {}

                well_col = well_columns[0]

                # Для каждой скважины извлекаем параметры
                for _, row in gdi_data.iterrows():
                    well_id = str(row[well_col])

                    if well_id not in well_params:
                        well_params[well_id] = {}

                    # Ищем параметры в строке
                    for col in gdi_data.columns:
                        col_lower = col.lower()
                        value = row[col]

                        # Пропускаем пустые значения
                        if pd.isna(value):
                            continue

                        # Проницаемость
                        if ('проницаем' in col_lower or 'permeab' in col_lower or 'пронитц' in col_lower):
                            try:
                                well_params[well_id]['permeability'] = float(value)
                            except:
                                pass

                        # Пористость
                        elif ('порист' in col_lower or 'porosit' in col_lower):
                            try:
                                well_params[well_id]['porosity'] = float(value)
                            except:
                                pass

                        # Вязкость
                        elif ('вязкость' in col_lower or 'viscosit' in col_lower):
                            try:
                                well_params[well_id]['viscosity'] = float(value)
                            except:
                                pass

                        # Скин-фактор
                        elif ('скин' in col_lower or 'skin' in col_lower):
                            try:
                                well_params[well_id]['skin_factor'] = float(value)
                            except:
                                pass

                        # Радиус скважины
                        elif ('радиус' in col_lower or 'radius' in col_lower):
                            try:
                                well_params[well_id]['well_radius'] = float(value)
                            except:
                                pass

                        # Сжимаемость
                        elif ('сжима' in col_lower or 'compress' in col_lower):
                            try:
                                well_params[well_id]['compressibility'] = float(value)
                            except:
                                pass

                        # Мощность пласта
                        elif ('мощность' in col_lower or 'толщина' in col_lower or 'height' in col_lower):
                            try:
                                well_params[well_id]['height'] = float(value)
                            except:
                                pass

                        # Дебит
                        elif ('дебит' in col_lower or 'rate' in col_lower or 'расход' in col_lower):
                            try:
                                well_params[well_id]['flow_rate'] = float(value)
                            except:
                                pass

            # Если не удалось извлечь параметры для всех скважин,
            # добавляем значения по умолчанию для отсутствующих параметров
            for well_id, params in well_params.items():
                if 'permeability' not in params:
                    params['permeability'] = 50.0  # [мД]
                if 'porosity' not in params:
                    params['porosity'] = 0.2  # [д.ед.]
                if 'viscosity' not in params:
                    params['viscosity'] = 1.0  # [сПз]
                if 'skin_factor' not in params:
                    params['skin_factor'] = 0.0
                if 'well_radius' not in params:
                    params['well_radius'] = 0.1  # [м]
                if 'compressibility' not in params:
                    params['compressibility'] = 1e-5  # [1/атм]
                if 'height' not in params:
                    params['height'] = 10.0  # [м]
                if 'flow_rate' not in params:
                    params['flow_rate'] = 50.0  # [м³/сут]

            return well_params

        except Exception as e:
            logger.error(f"Ошибка при извлечении параметров скважин из данных ГДИС: {str(e)}")
            return {}

    def plot_linear_flow_recovery_comparison(self, output_path=None):
        """
        Построение графика сравнения времени восстановления по разным методикам.

        Args:
            output_path (str, optional): Путь для сохранения графика

        Returns:
            plt.Figure: Объект фигуры matplotlib
        """
        if self.results is None or 'Linear_Flow_Recovery_Time' not in self.results.columns:
            logger.error("Нет данных для построения графика сравнения методик")
            return None

        try:
            # Построение графика
            fig, ax = plt.subplots(figsize=(12, 8))

            # Данные для графика
            wells = self.results['Well']
            basic_times = self.results['Recovery_Time']
            linear_flow_times = self.results['Linear_Flow_Recovery_Time']

            if 'Weighted_Recovery_Time' in self.results.columns:
                weighted_times = self.results['Weighted_Recovery_Time']
                has_weighted = True
            else:
                has_weighted = False

            # Сортируем скважины по времени восстановления (базовый метод)
            sorted_indices = np.argsort(basic_times.values)
            wells = wells.values[sorted_indices]
            basic_times = basic_times.values[sorted_indices]
            linear_flow_times = linear_flow_times.values[sorted_indices]
            if has_weighted:
                weighted_times = weighted_times.values[sorted_indices]

            # Строим график
            x = np.arange(len(wells))
            width = 0.35

            bar1 = ax.bar(x - width / 2, basic_times, width, label='Базовый метод', color='blue', alpha=0.7)
            bar2 = ax.bar(x + width / 2, linear_flow_times, width, label='Метод линейного стока', color='green',
                          alpha=0.7)

            if has_weighted:
                ax.plot(x, weighted_times, 'ro-', linewidth=2, label='Средневзвешенное время')

            # Добавляем подписи и заголовок
            ax.set_xlabel('Скважины')
            ax.set_ylabel('Время восстановления, сут')
            ax.set_title('Сравнение времени восстановления давления по разным методикам')
            ax.set_xticks(x)
            ax.set_xticklabels(wells, rotation=90)
            ax.legend()

            # Добавляем сетку для улучшения читаемости
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            plt.tight_layout()

            # Сохранение графика, если указан путь
            if output_path and output_path.endswith('.png'):
                compare_path = output_path.replace('.png', '_comparison.png')
                plt.savefig(compare_path, dpi=300, bbox_inches='tight')
                logger.info(f"График сравнения методик сохранен в {compare_path}")
            elif output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"График сравнения методик сохранен в {output_path}")

            return fig

        except Exception as e:
            logger.error(f"Ошибка при построении графика сравнения методик: {str(e)}")
            return None

    def get_results(self):
        """
        Получение результатов расчета.

        Returns:
            pd.DataFrame: Результаты расчета времени восстановления давления
        """
        return self.results

    def report(self):
        """
        Создание отчета о результатах расчета.

        Returns:
            str: Текстовый отчет
        """
        if self.results is None:
            return "Расчет времени восстановления давления не выполнен."

        report_text = "Результаты расчета времени восстановления давления:\n\n"

        # Статистика по всем скважинам
        report_text += "Общая статистика:\n"
        report_text += f"- Количество скважин: {len(self.results)}\n"
        report_text += f"- Минимальное время восстановления: {self.results['Recovery_Time'].min():.2f} сут.\n"
        report_text += f"- Максимальное время восстановления: {self.results['Recovery_Time'].max():.2f} сут.\n"
        report_text += f"- Среднее время восстановления: {self.results['Recovery_Time'].mean():.2f} сут.\n\n"

        # Таблица с результатами для первых 5 скважин
        report_text += "Пример результатов (первые 5 скважин):\n"
        columns_to_show = ['Well', 'Permeability', 'Porosity', 'Viscosity', 'Skin_Factor', 'Recovery_Time']
        report_text += self.results[columns_to_show].head().to_string() + "\n\n"

        # Рекомендации по интерпретации результатов
        report_text += "Интерпретация результатов:\n"
        report_text += "- Время восстановления давления зависит от проницаемости, пористости, вязкости флюида и скин-фактора.\n"
        report_text += "- Скважины с высоким скин-фактором требуют больше времени для восстановления давления.\n"
        report_text += "- Скважины с низкой проницаемостью также требуют больше времени для восстановления давления.\n"

        return report_text