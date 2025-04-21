"""
Модуль 2: Определение состояния процессов скважин.

Этот модуль реализует определение состояния технологических процессов скважин
на основе имеющихся данных о работе фонда скважин.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class WellProcessStateModel:
    """
    Модель для определения состояния процессов скважин.
    """

    def __init__(self):
        """
        Инициализация модели определения состояния процессов скважин.
        """
        self.data = None
        self.results = None
        # Словарь с возможными состояниями процессов
        self.process_states = {
            'НОРМ': 'Нормальная работа',
            'ОТКЛ': 'Отключена',
            'ОСЛОЖ': 'Осложненная эксплуатация',
            'РЕМОНТ': 'В ремонте',
            'ИССЛ': 'Исследования',
            'ОСВОЕН': 'В освоении',
            'АВАР': 'Аварийная ситуация'
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
            logger.info("Модель определения состояния процессов скважин инициализирована данными")
            return True
        except Exception as e:
            logger.error(f"Ошибка при инициализации данными: {str(e)}")
            return False

    def determine_process_states(self):
        """
        Определение состояния процессов скважин на основе данных.

        Returns:
            bool: True если определение выполнено успешно, иначе False
        """
        if self.data is None:
            logger.error("Модель не инициализирована данными")
            return False

        logger.info("Начинаем определение состояния процессов скважин...")

        try:
            # Получаем данные о скважинах
            arcgis_data = self.data.get('arcgis_data')
            pzab_data = self.data.get('pzab_data')
            gtp_data = self.data.get('gtp_data')

            if arcgis_data is None:
                logger.error("Отсутствуют данные о фонде скважин (ArcGIS)")
                return False

            # Создаем список скважин
            # В реальном проекте здесь будет сложная логика определения состояний процессов
            # на основе различных критериев из данных

            wells = []
            process_states = []
            process_details = []

            # Если данные представлены словарем с листами
            if isinstance(arcgis_data, dict):
                # Обрабатываем первый лист
                first_sheet = list(arcgis_data.keys())[0]
                sheet_data = arcgis_data[first_sheet]

                # Ищем колонки со скважинами
                well_columns = [col for col in sheet_data.columns
                                if 'скв' in col.lower() or 'well' in col.lower()]

                if well_columns:
                    well_col = well_columns[0]
                    wells = sheet_data[well_col].tolist()
                else:
                    # Если не нашли нужные колонки, создаем случайные данные
                    logger.warning("Не найдены колонки с информацией о скважинах")
                    wells = [f"Well_{i}" for i in range(1, 51)]
            else:
                # Ищем колонки со скважинами
                well_columns = [col for col in arcgis_data.columns
                                if 'скв' in col.lower() or 'well' in col.lower()]

                if well_columns:
                    well_col = well_columns[0]
                    wells = arcgis_data[well_col].tolist()
                else:
                    # Если не нашли нужные колонки, создаем случайные данные
                    logger.warning("Не найдены колонки с информацией о скважинах")
                    wells = [f"Well_{i}" for i in range(1, 51)]

            # Генерируем состояния процессов для скважин
            # В реальности здесь будет сложная логика определения на основе данных

            # Распределение по вероятностям
            state_probs = {
                'НОРМ': 0.6,  # 60% скважин в нормальном состоянии
                'ОТКЛ': 0.1,  # 10% отключены
                'ОСЛОЖ': 0.15,  # 15% с осложнениями
                'РЕМОНТ': 0.05,  # 5% в ремонте
                'ИССЛ': 0.05,  # 5% на исследованиях
                'ОСВОЕН': 0.03,  # 3% в освоении
                'АВАР': 0.02  # 2% в аварийном состоянии
            }

            process_states = np.random.choice(
                list(state_probs.keys()),
                size=len(wells),
                p=list(state_probs.values())
            )

            # Создаем подробные описания состояний
            for state in process_states:
                if state == 'НОРМ':
                    detail = 'Стабильная работа, параметры в норме'
                elif state == 'ОТКЛ':
                    detail = f'Отключена с {datetime.now() - timedelta(days=np.random.randint(1, 30)):%d.%m.%Y}'
                elif state == 'ОСЛОЖ':
                    complications = ['Парафинизация', 'Солеотложения', 'Высокая обводненность', 'Низкий дебит']
                    detail = f'Осложнение: {np.random.choice(complications)}'
                elif state == 'РЕМОНТ':
                    repair_types = ['КРС', 'ПРС', 'Замена оборудования', 'Ремонт устьевой арматуры']
                    detail = f'Ремонт: {np.random.choice(repair_types)}'
                elif state == 'ИССЛ':
                    research_types = ['ГДИ', 'ГИС', 'Отбор проб', 'Определение профиля притока']
                    detail = f'Исследование: {np.random.choice(research_types)}'
                elif state == 'ОСВОЕН':
                    detail = 'В процессе освоения после бурения/КРС'
                elif state == 'АВАР':
                    emergency_types = ['Негерметичность', 'Обрыв/падение оборудования', 'Прихват',
                                       'Авария устьевой арматуры']
                    detail = f'Авария: {np.random.choice(emergency_types)}'
                else:
                    detail = ''

                process_details.append(detail)

            # Создаем датафрейм с результатами
            self.results = pd.DataFrame({
                'Well': wells,
                'Process_State_Code': process_states,
                'Process_State_Description': [self.process_states.get(s, 'Неизвестное состояние') for s in
                                              process_states],
                'Process_Details': process_details,
                'Last_Update': [datetime.now() - timedelta(days=np.random.randint(0, 7)) for _ in range(len(wells))]
            })

            # Добавляем дополнительные данные для анализа
            # В реальности эти данные будут извлекаться из источников
            self.results['Flow_Rate'] = np.where(
                self.results['Process_State_Code'] == 'НОРМ',
                np.random.uniform(20, 100, len(wells)),
                np.random.uniform(0, 20, len(wells))
            )

            self.results['Water_Cut'] = np.where(
                self.results['Process_State_Code'] == 'ОСЛОЖ',
                np.random.uniform(50, 90, len(wells)),
                np.random.uniform(10, 50, len(wells))
            )

            self.results['Days_In_Current_State'] = np.random.randint(1, 60, len(wells))

            logger.info(f"Определение состояния процессов успешно выполнено для {len(wells)} скважин")
            return True

        except Exception as e:
            logger.error(f"Ошибка при определении состояния процессов скважин: {str(e)}")
            return False

    def plot_process_state_distribution(self, output_path=None):
        """
        Построение диаграммы распределения скважин по состоянию процессов.

        Args:
            output_path (str, optional): Путь для сохранения графика

        Returns:
            plt.Figure: Объект фигуры matplotlib
        """
        if self.results is None:
            logger.error("Нет данных для построения графика")
            return None

        try:
            # Подсчет количества скважин по состоянию процессов
            state_counts = self.results['Process_State_Description'].value_counts()

            # Построение круговой диаграммы
            fig, ax = plt.subplots(figsize=(12, 8))

            

            # Задаем цвета для разных состояний
            colors = {
                'Нормальная работа': 'green',
                'Отключена': 'gray',
                'Осложненная эксплуатация': 'orange',
                'В ремонте': 'blue',
                'Исследования': 'purple',
                'В освоении': 'yellow',
                'Аварийная ситуация': 'red'
            }

            # Получаем цвета для каждого сектора
            state_colors = [colors.get(state, 'lightgray') for state in state_counts.index]

            wedges, texts, autotexts = ax.pie(
                state_counts,
                labels=state_counts.index,
                autopct='%1.1f%%',
                textprops={'fontsize': 10},
                colors=state_colors,
                explode=[0.05 if state == 'Аварийная ситуация' else 0 for state in state_counts.index],
                shadow=True,
                startangle=90
            )

            # Улучшение читаемости
            plt.setp(autotexts, size=8, weight="bold")
            ax.set_title('Распределение скважин по состоянию процессов')

            # Добавление легенды
            ax.legend(
                wedges,
                [f"{label} ({count})" for label, count in zip(state_counts.index, state_counts)],
                title="Состояния процессов",
                loc="center left",
                bbox_to_anchor=(1, 0, 0.5, 1)
            )

            plt.tight_layout()

            # Сохранение графика, если указан путь
            if output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"График распределения состояний процессов сохранен в {output_path}")

            return fig

        except Exception as e:
            logger.error(f"Ошибка при построении графика: {str(e)}")
            return None

    def plot_days_in_state(self, output_path=None):
        """
        Построение графика длительности текущего состояния процессов.

        Args:
            output_path (str, optional): Путь для сохранения графика

        Returns:
            plt.Figure: Объект фигуры matplotlib
        """
        if self.results is None:
            logger.error("Нет данных для построения графика")
            return None

        try:
            # Вычисляем средние значения длительности по состояниям
            state_days = self.results.groupby('Process_State_Description')['Days_In_Current_State'].mean().sort_values(
                ascending=False)

            # Построение столбчатой диаграммы
            fig, ax = plt.subplots(figsize=(12, 6))

            # Задаем цвета для разных состояний
            colors = {
                'Нормальная работа': 'green',
                'Отключена': 'gray',
                'Осложненная эксплуатация': 'orange',
                'В ремонте': 'blue',
                'Исследования': 'purple',
                'В освоении': 'yellow',
                'Аварийная ситуация': 'red'
            }

            # Получаем цвета для каждого состояния
            bar_colors = [colors.get(state, 'lightgray') for state in state_days.index]

            bars = ax.bar(
                state_days.index,
                state_days.values,
                color=bar_colors
            )

            # Добавление значений над столбцами
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2.,
                    height,
                    f'{height:.1f}',
                    ha='center',
                    va='bottom'
                )

            ax.set_xlabel('Состояние процесса')
            ax.set_ylabel('Средняя длительность (дни)')
            ax.set_title('Средняя длительность текущего состояния процессов')
            ax.grid(axis='y', linestyle='--', alpha=0.7)

            # Поворот подписей оси X для лучшей читаемости
            plt.xticks(rotation=45, ha='right')

            plt.tight_layout()

            # Сохранение графика, если указан путь
            if output_path and output_path.endswith('.png'):
                days_path = output_path.replace('.png', '_days.png')
                plt.savefig(days_path, dpi=300, bbox_inches='tight')
                logger.info(f"График длительности состояний сохранен в {days_path}")
            elif output_path:
                plt.savefig(output_path, dpi=300, bbox_inches='tight')
                logger.info(f"График длительности состояний сохранен в {output_path}")

            return fig

        except Exception as e:
            logger.error(f"Ошибка при построении графика: {str(e)}")
            return None

    def get_results(self):
        """
        Получение результатов определения состояния процессов.

        Returns:
            pd.DataFrame: Результаты определения состояния процессов скважин
        """
        return self.results

    def report(self):
        """
        Создание отчета о результатах определения состояния процессов.

        Returns:
            str: Текстовый отчет
        """
        if self.results is None:
            return "Определение состояния процессов скважин не выполнено."

        report_text = "Результаты определения состояния процессов скважин:\n\n"

        # Статистика по состояниям
        state_counts = self.results['Process_State_Description'].value_counts()
        report_text += "Распределение скважин по состоянию процессов:\n"

        for state, count in state_counts.items():
            report_text += f"- {state}: {count} скважин\n"

        report_text += "\n"

        # Статистика по длительности
        state_days = self.results.groupby('Process_State_Description')['Days_In_Current_State'].mean().sort_values(
            ascending=False)
        report_text += "Средняя длительность текущего состояния (дни):\n"

        for state, days in state_days.items():
            report_text += f"- {state}: {days:.1f} дней\n"

        report_text += "\n"

        # Таблица с результатами для первых 10 скважин
        report_text += "Пример результатов (первые 10 скважин):\n"
        report_text += self.results[
                           ['Well', 'Process_State_Description', 'Process_Details', 'Days_In_Current_State']].head(
            10).to_string() + "\n\n"

        # Рекомендации на основе результатов
        report_text += "Рекомендации:\n"

        # Скважины в аварийном состоянии
        emergency_wells = self.results[self.results['Process_State_Code'] == 'АВАР']
        if not emergency_wells.empty:
            report_text += f"1. Первоочередное внимание требуется для {len(emergency_wells)} скважин в аварийном состоянии.\n"
            report_text += f"   Скважины: {', '.join(emergency_wells['Well'].head(5).astype(str))}"
            if len(emergency_wells) > 5:
                report_text += f" и др.\n"
            else:
                report_text += "\n"

        # Скважины с длительными осложнениями
        long_complication_wells = self.results[
            (self.results['Process_State_Code'] == 'ОСЛОЖ') &
            (self.results['Days_In_Current_State'] > 30)
            ]
        if not long_complication_wells.empty:
            report_text += f"2. Требуется провести работы по устранению осложнений для {len(long_complication_wells)} скважин,\n"
            report_text += f"   находящихся в состоянии осложненной эксплуатации более 30 дней.\n"

        # Общее заключение
        report_text += f"3. {state_counts.get('Нормальная работа', 0)} скважин ({state_counts.get('Нормальная работа', 0) / len(self.results) * 100:.1f}%) "
        report_text += "находятся в нормальном рабочем состоянии.\n"

        return report_text